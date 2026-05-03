from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from bearing_dt.config import read_yaml, write_yaml
from bearing_dt.data.dataset import load_processed
from bearing_dt.table import Rows, unique_values, write_rows_csv
from bearing_dt.train import train_experiment
from bearing_dt.utils import ensure_dir, read_json


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in value)


def _latest_completed_by_name(runs_dir: Path) -> dict[str, Path]:
    latest: dict[str, Path] = {}
    for metrics_path in sorted(runs_dir.glob("*/metrics.json")):
        try:
            metrics = read_json(metrics_path)
        except Exception:
            continue
        name = str(metrics.get("run", {}).get("experiment_name", metrics_path.parent.name))
        latest[name] = metrics_path.parent
    return latest


def _matrix_config(
    base_config: dict[str, Any],
    split_value: str,
    prefix: str,
    *,
    split_strategy: str,
    split_key: str,
    name_key: str,
    validation_key: str | None = None,
    validation_value: str | None = None,
    calibration_key: str | None = None,
    calibration_value: str | None = None,
    model_seed: int | None = None,
) -> dict[str, Any]:
    config = dict(base_config)
    base_name = str(base_config.get("experiment_name", "experiment"))
    seed_suffix = f"__seed_{model_seed}" if model_seed is not None else ""
    config["experiment_name"] = f"{prefix}_{_safe_name(base_name)}{seed_suffix}__{name_key}_{_safe_name(split_value)}"
    if model_seed is not None:
        config["seed"] = int(model_seed)
    split = dict(config.get("split", {}))
    split["strategy"] = split_strategy
    split[split_key] = split_value
    if validation_key and validation_value is not None:
        split[validation_key] = validation_value
    if calibration_key and calibration_value is not None:
        split[calibration_key] = calibration_value
    config["split"] = split
    return config


def _rotating_holdouts(values: list[str], test_value: str) -> tuple[str, str]:
    if len(values) < 3:
        raise ValueError("At least three split values are required for separate validation/calibration")
    start = values.index(test_value)
    val = values[(start + 1) % len(values)]
    if val == test_value:
        val = values[(start + 2) % len(values)]
    cal = values[(start + 2) % len(values)]
    offset = 3
    while cal in {test_value, val}:
        cal = values[(start + offset) % len(values)]
        offset += 1
        if offset > len(values) + 2:
            raise ValueError(f"Cannot choose distinct validation/calibration values for {test_value}")
    return val, cal


def run_split_matrix(
    *,
    processed_dir: str | Path,
    base_configs: list[str | Path],
    out: str | Path,
    split_column: str,
    split_strategy: str,
    split_key: str,
    name_key: str,
    runs_dir: str | Path = "runs",
    prefix: str = "phme_matrix",
    resume: bool = True,
    max_values: int | None = None,
    values: list[str] | None = None,
    validation_key: str | None = None,
    calibration_key: str | None = None,
    separate_calibration: bool = False,
    model_seeds: list[int] | None = None,
) -> Rows:
    _, frame, _ = load_processed(processed_dir)
    available_values = [str(v) for v in unique_values(frame, split_column)]
    if values:
        missing = sorted(set(values) - set(available_values))
        if missing:
            raise ValueError(f"Unknown {split_column} values {missing}; available={available_values}")
        split_values = values
    else:
        split_values = available_values
    if max_values is not None:
        split_values = split_values[:max_values]
    out_dir = ensure_dir(out)
    config_dir = ensure_dir(out_dir / "configs")
    manifest_path = out_dir / "matrix_manifest.csv"
    manifest: Rows = []
    seen: set[tuple[str, str]] = set()
    if manifest_path.exists():
        from bearing_dt.table import read_rows_csv

        manifest = read_rows_csv(manifest_path)
        seen = {(str(row.get("experiment_name", "")), str(row.get(f"test_{name_key}", ""))) for row in manifest}
    latest = _latest_completed_by_name(Path(runs_dir)) if resume else {}
    seeds = model_seeds or [None]
    for base_path in base_configs:
        base_path = Path(base_path)
        base_config = read_yaml(base_path)
        for split_value in split_values:
            val_value = None
            cal_value = None
            if separate_calibration:
                val_value, cal_value = _rotating_holdouts(split_values, split_value)
            for model_seed in seeds:
                config = _matrix_config(
                    base_config,
                    split_value,
                    prefix,
                    split_strategy=split_strategy,
                    split_key=split_key,
                    name_key=name_key,
                    validation_key=validation_key,
                    validation_value=val_value,
                    calibration_key=calibration_key,
                    calibration_value=cal_value,
                    model_seed=model_seed,
                )
                name = str(config["experiment_name"])
                generated_path = config_dir / f"{name}.yaml"
                write_yaml(generated_path, config)
                existing = latest.get(name)
                if existing:
                    print(f"Skipping completed {name}: {existing}")
                    run_dir = existing
                    status = "skipped_existing"
                else:
                    print(f"Running {name}")
                    run_dir = train_experiment(generated_path)
                    status = "completed"
                row_key = (name, split_value)
                row = {
                    "experiment_name": name,
                    "base_config": str(base_path),
                    f"test_{name_key}": split_value,
                    "validation_value": val_value or "",
                    "calibration_value": cal_value or "",
                    "model_seed": model_seed if model_seed is not None else "",
                    "config": str(generated_path),
                    "run_dir": str(run_dir),
                    "status": status,
                }
                if row_key in seen:
                    manifest = [
                        existing
                        for existing in manifest
                        if (str(existing.get("experiment_name", "")), str(existing.get(f"test_{name_key}", ""))) != row_key
                    ]
                manifest.append(row)
                seen.add(row_key)
                write_rows_csv(manifest_path, manifest)
    return manifest


def run_regime_matrix(
    *,
    processed_dir: str | Path,
    base_configs: list[str | Path],
    out: str | Path,
    runs_dir: str | Path = "runs",
    prefix: str = "phme_matrix",
    resume: bool = True,
    max_regimes: int | None = None,
    regimes: list[str] | None = None,
    separate_calibration: bool = False,
    model_seeds: list[int] | None = None,
) -> Rows:
    return run_split_matrix(
        processed_dir=processed_dir,
        base_configs=base_configs,
        out=out,
        split_column="operating_regime",
        split_strategy="leave_one_operating_regime_out",
        split_key="test_regime",
        name_key="regime",
        runs_dir=runs_dir,
        prefix=prefix,
        resume=resume,
        max_values=max_regimes,
        values=regimes,
        validation_key="val_regime",
        calibration_key="cal_regime",
        separate_calibration=separate_calibration,
        model_seeds=model_seeds,
    )


def run_bearing_matrix(
    *,
    processed_dir: str | Path,
    base_configs: list[str | Path],
    out: str | Path,
    runs_dir: str | Path = "runs",
    prefix: str = "phme_bearing",
    resume: bool = True,
    max_bearings: int | None = None,
    bearings: list[str] | None = None,
    separate_calibration: bool = False,
    model_seeds: list[int] | None = None,
) -> Rows:
    return run_split_matrix(
        processed_dir=processed_dir,
        base_configs=base_configs,
        out=out,
        split_column="bearing_id",
        split_strategy="leave_one_bearing_out",
        split_key="test_bearing",
        name_key="bearing",
        runs_dir=runs_dir,
        prefix=prefix,
        resume=resume,
        max_values=max_bearings,
        values=bearings,
        validation_key="val_bearing",
        calibration_key="cal_bearing",
        separate_calibration=separate_calibration,
        model_seeds=model_seeds,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m bearing_dt.matrix")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-regime-matrix")
    run.add_argument("--processed-dir", required=True)
    run.add_argument("--base-config", action="append", required=True)
    run.add_argument("--out", default="paper_artifacts/matrix/phme_regime")
    run.add_argument("--runs-dir", default="runs")
    run.add_argument("--prefix", default="phme_matrix")
    run.add_argument("--max-regimes", type=int)
    run.add_argument("--regime", action="append")
    run.add_argument("--separate-calibration", action="store_true")
    run.add_argument("--model-seed", action="append", type=int)
    bearing = sub.add_parser("run-bearing-matrix")
    bearing.add_argument("--processed-dir", required=True)
    bearing.add_argument("--base-config", action="append", required=True)
    bearing.add_argument("--out", default="paper_artifacts/matrix/phme_bearing")
    bearing.add_argument("--runs-dir", default="runs")
    bearing.add_argument("--prefix", default="phme_bearing")
    bearing.add_argument("--max-bearings", type=int)
    bearing.add_argument("--bearing", action="append")
    bearing.add_argument("--separate-calibration", action="store_true")
    bearing.add_argument("--model-seed", action="append", type=int)
    bearing.add_argument("--no-resume", action="store_true")
    run.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "run-regime-matrix":
        run_regime_matrix(
            processed_dir=args.processed_dir,
            base_configs=args.base_config,
            out=args.out,
            runs_dir=args.runs_dir,
            prefix=args.prefix,
            resume=not args.no_resume,
            max_regimes=args.max_regimes,
            regimes=args.regime,
            separate_calibration=args.separate_calibration,
            model_seeds=args.model_seed,
        )
    if args.command == "run-bearing-matrix":
        run_bearing_matrix(
            processed_dir=args.processed_dir,
            base_configs=args.base_config,
            out=args.out,
            runs_dir=args.runs_dir,
            prefix=args.prefix,
            resume=not args.no_resume,
            max_bearings=args.max_bearings,
            bearings=args.bearing,
            separate_calibration=args.separate_calibration,
            model_seeds=args.model_seed,
        )


if __name__ == "__main__":
    main()
