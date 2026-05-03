from __future__ import annotations

import argparse
from pathlib import Path

from bearing_dt.metrics import interval_metrics, regression_metrics
from bearing_dt.physics import damage_monotonic_violation_rate, monotonic_violation_rate
from bearing_dt.config import read_yaml
from bearing_dt.table import Rows, group_by, read_rows_csv
from bearing_dt.utils import write_json


def _col(rows: Rows, key: str):
    import numpy as np

    return np.array([row[key] for row in rows], dtype=float)


def _has_numeric_interval(rows: Rows) -> bool:
    return bool(
        rows
        and {"lower_rul", "upper_rul"}.issubset(rows[0])
        and all(str(row.get("lower_rul", "")) != "" and str(row.get("upper_rul", "")) != "" for row in rows)
    )


def evaluate_run(run: str | Path) -> dict:
    run_dir = Path(run)
    predictions = read_rows_csv(run_dir / "predictions.csv")
    previous = {}
    if (run_dir / "metrics.json").exists():
        from bearing_dt.utils import read_json

        previous = read_json(run_dir / "metrics.json")
    run_meta = previous.get("run", {})
    alpha = 0.1
    if (not run_meta or set(run_meta) == {"path"}) and (run_dir / "config.yaml").exists():
        cfg = read_yaml(run_dir / "config.yaml")
        alpha = float(cfg.get("uncertainty", {}).get("alpha", 0.1))
        run_meta = {
            "experiment_name": cfg.get("experiment_name", run_dir.name),
            "model_type": cfg.get("model", {}).get("type", ""),
            "split": cfg.get("split", {}).get("strategy", ""),
            "path": str(run_dir),
        }
    elif (run_dir / "config.yaml").exists():
        cfg = read_yaml(run_dir / "config.yaml")
        alpha = float(cfg.get("uncertainty", {}).get("alpha", 0.1))
    else:
        run_meta = {**run_meta, "path": str(run_dir)}
    metrics = {}
    for split, group in group_by(predictions, "split").items():
        split_metrics = regression_metrics(_col(group, "rul"), _col(group, "y_pred_rul"))
        if _has_numeric_interval(group):
            split_metrics.update(interval_metrics(_col(group, "rul"), _col(group, "lower_rul"), _col(group, "upper_rul"), alpha=alpha))
        if group and all(str(row.get("damage_pred", "")) != "" for row in group):
            split_metrics["monotonic_violation_rate"] = damage_monotonic_violation_rate(group, _col(group, "damage_pred"))
            split_metrics["monotonic_metric_source"] = "damage_pred"
        else:
            split_metrics["monotonic_violation_rate"] = monotonic_violation_rate(group, _col(group, "y_pred_norm"))
            split_metrics["monotonic_metric_source"] = "rul_proxy"
        metrics[str(split)] = split_metrics
    write_json(run_dir / "metrics.json", {**metrics, "run": run_meta})
    print(f"Evaluated {run_dir}")
    if "test" in metrics:
        print(f"Test RMSE: {metrics['test']['rmse']:.4f}, MAE: {metrics['test']['mae']:.4f}")
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m bearing_dt.eval")
    parser.add_argument("--run", required=True)
    args = parser.parse_args(argv)
    evaluate_run(args.run)


if __name__ == "__main__":
    main()
