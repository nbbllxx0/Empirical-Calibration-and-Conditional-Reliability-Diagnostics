from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from bearing_dt.metrics import interval_metrics, regression_metrics
from bearing_dt.table import Rows, group_by, read_rows_csv, write_rows_csv
from bearing_dt.utils import ensure_dir, read_json


BEARING_RE = re.compile(r"^(B\d{2})")
LIFE_BINS = [
    ("0.00-0.25", 0.00, 0.25),
    ("0.25-0.50", 0.25, 0.50),
    ("0.50-0.75", 0.50, 0.75),
    ("0.75-1.00", 0.75, 1.01),
]


def _bearing_from_filename(filename: str) -> str:
    match = BEARING_RE.match(Path(filename).name)
    return match.group(1) if match else ""


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | str:
    return mean(values) if values else ""


def _coverage_rows(rows: Rows) -> dict[str, Any]:
    if not rows:
        return {
            "windows": 0,
            "mae": "",
            "normalized_mae": "",
            "coverage": "",
            "mean_interval_width": "",
            "normalized_interval_width": "",
            "interval_score": "",
            "normalized_interval_score": "",
        }
    y = np.asarray([float(row["rul"]) for row in rows], dtype=float)
    pred = np.asarray([float(row["y_pred_rul"]) for row in rows], dtype=float)
    norm_y = np.asarray([float(row["rul_norm"]) for row in rows], dtype=float)
    norm_pred = np.asarray([float(row["y_pred_norm"]) for row in rows], dtype=float)
    metrics = regression_metrics(y, pred)
    norm_metrics = regression_metrics(norm_y, norm_pred)
    out: dict[str, Any] = {
        "windows": len(rows),
        "mae": metrics["mae"],
        "normalized_mae": norm_metrics["mae"],
        "coverage": "",
        "mean_interval_width": "",
        "normalized_interval_width": "",
        "interval_score": "",
        "normalized_interval_score": "",
    }
    if all(str(row.get("lower_rul", "")) != "" and str(row.get("upper_rul", "")) != "" for row in rows):
        lo = np.asarray([float(row["lower_rul"]) for row in rows], dtype=float)
        hi = np.asarray([float(row["upper_rul"]) for row in rows], dtype=float)
        out.update(interval_metrics(y, lo, hi))
    return out


def _base_model_name(name: str) -> str:
    if "__regime_" in name:
        return name.split("__regime_", 1)[0]
    if "__bearing_" in name:
        return name.split("__bearing_", 1)[0]
    return name


def _split_value(name: str) -> str:
    if "__regime_" in name:
        return name.split("__regime_", 1)[1]
    if "__bearing_" in name:
        return name.split("__bearing_", 1)[1]
    return ""


def _load_feature_lookup(processed_dir: Path) -> dict[int, dict[str, Any]]:
    return {int(row["sample_id"]): row for row in read_rows_csv(processed_dir / "features.csv")}


def _prediction_runs(runs_dir: Path, prefix: str = "", model_tokens: list[str] | None = None) -> Rows:
    rows: Rows = []
    tokens = model_tokens or []
    for metrics_path in sorted(runs_dir.glob("*/metrics.json")):
        metrics = read_json(metrics_path)
        name = str(metrics.get("run", {}).get("experiment_name", metrics_path.parent.name))
        if prefix and not name.startswith(prefix):
            continue
        base_model = _base_model_name(name)
        if tokens and not any(token in base_model for token in tokens):
            continue
        pred_path = metrics_path.parent / "predictions.csv"
        if not pred_path.exists():
            continue
        rows.append(
            {
                "experiment_name": name,
                "base_model": base_model,
                "split_value": _split_value(name),
                "run_dir": str(metrics_path.parent),
                "predictions": pred_path,
            }
        )
    return rows


def _joined_test_rows(run: dict[str, Any], feature_by_sample: dict[int, dict[str, Any]]) -> Rows:
    rows: Rows = []
    for row in read_rows_csv(run["predictions"]):
        if str(row.get("split")) != "test":
            continue
        sample_id = int(row["sample_id"])
        feature = feature_by_sample.get(sample_id, {})
        rows.append(
            {
                **row,
                "experiment_name": run["experiment_name"],
                "base_model": run["base_model"],
                "held_out_split": run["split_value"],
                "operating_regime": feature.get("operating_regime", ""),
                "regime_code": feature.get("regime_code", ""),
                "load": feature.get("load", ""),
                "speed": feature.get("speed", ""),
            }
        )
    return rows


def _dominant_bearing_share(rows: Rows) -> tuple[str, int, float]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("bearing_id", ""))] += 1
    if not counts:
        return "", 0, 0.0
    bearing, count = max(counts.items(), key=lambda item: item[1])
    return bearing, count, count / max(len(rows), 1)


def phme_subset_audit(*, raw_dir: str | Path, processed_dir: str | Path, out: str | Path) -> Path:
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    out_dir = ensure_dir(out)
    file_rows = read_rows_csv(raw_dir / "file_manifest.csv")
    features = read_rows_csv(processed_dir / "features.csv")
    processed_by_bearing = group_by(features, "bearing_id")
    manifest_by_bearing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in file_rows:
        bearing = _bearing_from_filename(str(row.get("filename", "")))
        if bearing:
            manifest_by_bearing[bearing].append(row)
    all_bearings = sorted(manifest_by_bearing)
    bearing_rows: Rows = []
    for bearing in all_bearings:
        manifest_rows = manifest_by_bearing[bearing]
        local_archives = [raw_dir / str(row["filename"]) for row in manifest_rows if (raw_dir / str(row["filename"])).exists()]
        processed = processed_by_bearing.get(bearing, [])
        included = bool(processed)
        total_bytes = sum(int(row["size"]) for row in manifest_rows)
        regimes = sorted({str(row["operating_regime"]) for row in processed})
        loads = [float(row["load"]) for row in processed]
        speeds = [float(row["speed"]) for row in processed]
        if included:
            status = "parsed_qc_passed"
            reason = "Included in the frozen 10-bearing analysis set."
        elif local_archives:
            status = "downloaded_not_processed"
            reason = "Archive exists locally but is absent from the processed analysis set."
        else:
            status = "not_downloaded"
            reason = "Public archive listed in Zenodo manifest but not present in local raw storage."
        bearing_rows.append(
            {
                "bearing_id": bearing,
                "included": "yes" if included else "no",
                "archive_files": len(manifest_rows),
                "manifest_gb": total_bytes / 1e9,
                "local_archive_files": len(local_archives),
                "local_archive_gb": sum(path.stat().st_size for path in local_archives) / 1e9,
                "extracted_dir_present": "yes" if (raw_dir / bearing).exists() else "no",
                "processed_windows": len(processed),
                "processed_regimes": len(regimes),
                "load_min": min(loads) if loads else "",
                "load_mean": mean(loads) if loads else "",
                "load_max": max(loads) if loads else "",
                "speed_min": min(speeds) if speeds else "",
                "speed_mean": mean(speeds) if speeds else "",
                "speed_max": max(speeds) if speeds else "",
                "processing_status": status,
                "objective_reason": reason,
            }
        )
    write_rows_csv(out_dir / "bearing_manifest.csv", bearing_rows)

    regime_rows: Rows = []
    for regime, group in sorted(group_by(features, "operating_regime").items()):
        dominant_bearing, dominant_count, dominant_share = _dominant_bearing_share(group)
        regime_rows.append(
            {
                "operating_regime": regime,
                "windows": len(group),
                "contributing_bearings": len({row["bearing_id"] for row in group}),
                "dominant_bearing": dominant_bearing,
                "dominant_bearing_windows": dominant_count,
                "dominant_bearing_share": dominant_share,
                "load_mean": mean(float(row["load"]) for row in group),
                "speed_mean": mean(float(row["speed"]) for row in group),
                "life_fraction_mean": mean(float(row["life_fraction"]) for row in group),
            }
        )
    write_rows_csv(out_dir / "regime_bearing_counts.csv", regime_rows)

    included_rows = [row for row in bearing_rows if row["included"] == "yes"]
    excluded_rows = [row for row in bearing_rows if row["included"] == "no"]
    summary: Rows = [
        {"metric": "manifest_bearings", "value": len(bearing_rows)},
        {"metric": "included_bearings", "value": len(included_rows)},
        {"metric": "excluded_bearings", "value": len(excluded_rows)},
        {"metric": "manifest_total_gb", "value": sum(float(row["manifest_gb"]) for row in bearing_rows)},
        {"metric": "included_manifest_gb", "value": sum(float(row["manifest_gb"]) for row in included_rows)},
        {"metric": "excluded_manifest_gb", "value": sum(float(row["manifest_gb"]) for row in excluded_rows)},
        {"metric": "processed_windows", "value": len(features)},
        {"metric": "processed_regimes", "value": len({row["operating_regime"] for row in features})},
    ]
    write_rows_csv(out_dir / "subset_bias_summary.csv", summary)

    missing = ", ".join(row["bearing_id"] for row in excluded_rows)
    report = out_dir / "exclusion_bias_report.md"
    report.write_text(
        "\n".join(
            [
                "# PHME Subset Exclusion-Bias Audit",
                "",
                f"Processed subset: {len(included_rows)} of {len(bearing_rows)} PHME bearing runs.",
                f"Excluded public bearing runs: {missing or 'none'}.",
                "",
                "This audit verifies archive-level inclusion, local download status, processed-window counts, and regime occupancy from the current workspace. It cannot prove that the 10-bearing subset is unbiased because the excluded bearings have no processed load/speed trajectory rows in the local evidence tree.",
                "",
                "Required top-tier resolution: process the excluded public archives, then rerun the strict matrices on the full processed set. Until then, full-PHME performance and full-PHME regime distribution claims remain unsupported.",
                "",
                "Generated files:",
                "- `bearing_manifest.csv`",
                "- `regime_bearing_counts.csv`",
                "- `subset_bias_summary.csv`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def conditional_coverage(
    *,
    runs: str | Path,
    processed_dir: str | Path,
    out: str | Path,
    prefix: str = "",
    model_tokens: list[str] | None = None,
) -> Path:
    out_dir = ensure_dir(out)
    feature_by_sample = _load_feature_lookup(Path(processed_dir))
    test_rows: Rows = []
    for run in _prediction_runs(Path(runs), prefix, model_tokens):
        test_rows.extend(_joined_test_rows(run, feature_by_sample))
    write_rows_csv(out_dir / "joined_test_predictions.csv", test_rows)

    per_regime: Rows = []
    for base_model, model_group in sorted(group_by(test_rows, "base_model").items()):
        for regime, group in sorted(group_by(model_group, "operating_regime").items()):
            dominant_bearing, dominant_count, dominant_share = _dominant_bearing_share(group)
            metrics = _coverage_rows(group)
            per_regime.append(
                {
                    "base_model": base_model,
                    "operating_regime": regime,
                    "contributing_bearings": len({row["bearing_id"] for row in group}),
                    "dominant_bearing": dominant_bearing,
                    "dominant_bearing_windows": dominant_count,
                    "dominant_bearing_share": dominant_share,
                    **metrics,
                }
            )
    write_rows_csv(out_dir / "per_regime_coverage.csv", per_regime)

    per_bearing: Rows = []
    for base_model, model_group in sorted(group_by(test_rows, "base_model").items()):
        for bearing, group in sorted(group_by(model_group, "bearing_id").items()):
            metrics = _coverage_rows(group)
            per_bearing.append(
                {
                    "base_model": base_model,
                    "bearing_id": bearing,
                    "regimes_seen": len({row["operating_regime"] for row in group}),
                    **metrics,
                }
            )
    write_rows_csv(out_dir / "per_bearing_coverage.csv", per_bearing)

    per_life_bin: Rows = []
    for base_model, model_group in sorted(group_by(test_rows, "base_model").items()):
        for label, lower, upper in LIFE_BINS:
            group = [row for row in model_group if lower <= float(row["life_fraction"]) < upper]
            metrics = _coverage_rows(group)
            per_life_bin.append({"base_model": base_model, "life_fraction_bin": label, **metrics})
    write_rows_csv(out_dir / "per_life_bin_coverage.csv", per_life_bin)

    calibration_curve_rows: Rows = []
    nominal_levels = [0.50, 0.70, 0.80, 0.85, 0.90, 0.95]
    for run in _prediction_runs(Path(runs), prefix, model_tokens):
        predictions = read_rows_csv(run["predictions"])
        cal_rows = [row for row in predictions if str(row.get("split")) == "cal"]
        if not cal_rows:
            cal_rows = [row for row in predictions if str(row.get("split")) == "val"]
        test = [row for row in predictions if str(row.get("split")) == "test"]
        if not cal_rows or not test:
            continue
        cal_abs = np.asarray([abs(float(row["rul_norm"]) - float(row["y_pred_norm"])) for row in cal_rows], dtype=float)
        test_abs = np.asarray([abs(float(row["rul_norm"]) - float(row["y_pred_norm"])) for row in test], dtype=float)
        for nominal in nominal_levels:
            radius = float(np.quantile(cal_abs, nominal))
            calibration_curve_rows.append(
                {
                    "experiment_name": run["experiment_name"],
                    "base_model": run["base_model"],
                    "held_out_split": run["split_value"],
                    "nominal_coverage": nominal,
                    "empirical_coverage": float(np.mean(test_abs <= radius)),
                    "normalized_interval_width": 2.0 * radius,
                    "calibration_rows": len(cal_rows),
                    "test_rows": len(test),
                }
            )
    write_rows_csv(out_dir / "calibration_curve_detail.csv", calibration_curve_rows)
    curve_summary: Rows = []
    for base_model, model_group in sorted(group_by(calibration_curve_rows, "base_model").items()):
        for nominal, group in sorted(group_by(model_group, "nominal_coverage").items()):
            curve_summary.append(
                {
                    "base_model": base_model,
                    "nominal_coverage": nominal,
                    "mean_empirical_coverage": _mean([float(row["empirical_coverage"]) for row in group]),
                    "mean_normalized_interval_width": _mean([float(row["normalized_interval_width"]) for row in group]),
                    "held_out_splits": len({row["held_out_split"] for row in group}),
                }
            )
    write_rows_csv(out_dir / "calibration_curve_summary.csv", curve_summary)

    report = out_dir / "conditional_coverage_report.md"
    report.write_text(
        "\n".join(
            [
                "# Conditional Coverage Diagnostics",
                "",
                f"Run prefix: `{prefix or '<none>'}`.",
                f"Models analyzed: {len({row['base_model'] for row in test_rows})}.",
                f"Joined test rows: {len(test_rows)}.",
                "",
                "Generated files:",
                "- `joined_test_predictions.csv`",
                "- `per_regime_coverage.csv`",
                "- `per_bearing_coverage.csv`",
                "- `per_life_bin_coverage.csv`",
                "- `calibration_curve_detail.csv`",
                "- `calibration_curve_summary.csv`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m bearing_dt.diagnostics")
    sub = parser.add_subparsers(dest="command", required=True)

    subset = sub.add_parser("phme-subset-audit")
    subset.add_argument("--raw-dir", default="data/raw/phme_tvoc")
    subset.add_argument("--processed-dir", default="data/processed/phme_tvoc_10b")
    subset.add_argument("--out", default="paper_artifacts/exclusion_bias_10b")

    coverage = sub.add_parser("conditional-coverage")
    coverage.add_argument("--runs", default="runs")
    coverage.add_argument("--processed-dir", default="data/processed/phme_tvoc_10b")
    coverage.add_argument("--out", default="paper_artifacts/conditional_coverage_10b")
    coverage.add_argument("--prefix", default="phme_10b_matched")
    coverage.add_argument("--model-token", action="append")

    args = parser.parse_args(argv)
    if args.command == "phme-subset-audit":
        phme_subset_audit(raw_dir=args.raw_dir, processed_dir=args.processed_dir, out=args.out)
    elif args.command == "conditional-coverage":
        conditional_coverage(
            runs=args.runs,
            processed_dir=args.processed_dir,
            out=args.out,
            prefix=args.prefix,
            model_tokens=args.model_token,
        )


if __name__ == "__main__":
    main()
