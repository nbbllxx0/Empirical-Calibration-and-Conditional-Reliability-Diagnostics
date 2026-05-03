from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from bearing_dt.table import Rows, group_by, write_rows_csv
from bearing_dt.utils import ensure_dir, read_json


def _base_model_name(name: str) -> str:
    if "__regime_" in name:
        return name.split("__regime_", 1)[0]
    if "__bearing_" in name:
        return name.split("__bearing_", 1)[0]
    return name


def _regime_name(name: str) -> str:
    return name.split("__regime_", 1)[1] if "__regime_" in name else ""


def _bearing_name(name: str) -> str:
    return name.split("__bearing_", 1)[1] if "__bearing_" in name else ""


def _metric_rows(runs_dir: Path, model_token: str) -> Rows:
    rows: Rows = []
    for metrics_path in sorted(runs_dir.glob("*/metrics.json")):
        metrics = read_json(metrics_path)
        run = metrics.get("run", {})
        name = str(run.get("experiment_name", metrics_path.parent.name))
        base_model = _base_model_name(name)
        if model_token not in base_model:
            continue
        for split, values in metrics.items():
            if split == "run" or not isinstance(values, dict) or "mae" not in values:
                continue
            rows.append(
                {
                    "experiment_name": name,
                    "base_model": base_model,
                    "test_regime": _regime_name(name),
                    "test_bearing": _bearing_name(name),
                    "metric_split": split,
                    "run_dir": str(metrics_path.parent),
                    **values,
                }
            )
    return rows


def _mean_numeric(group: Rows, key: str) -> float | str:
    values = [float(row[key]) for row in group if str(row.get(key, "")) != ""]
    return mean(values) if values else ""


def summarize_model(*, runs: str | Path, out: str | Path, model_token: str) -> Path:
    out_dir = ensure_dir(out)
    rows = _metric_rows(Path(runs), model_token)
    write_rows_csv(out_dir / "split_detail.csv", rows)
    summary: Rows = []
    for split, group in sorted(group_by(rows, "metric_split").items()):
        summary.append(
            {
                "metric_split": split,
                "regimes": len({row["test_regime"] for row in group if row.get("test_regime")}),
                "bearings": len({row["test_bearing"] for row in group if row.get("test_bearing")}),
                "mean_mae": _mean_numeric(group, "mae"),
                "mean_rmse": _mean_numeric(group, "rmse"),
                "mean_coverage": _mean_numeric(group, "coverage"),
                "mean_interval_width": _mean_numeric(group, "mean_interval_width"),
                "mean_normalized_interval_width": _mean_numeric(group, "normalized_interval_width"),
                "mean_interval_score": _mean_numeric(group, "interval_score"),
                "mean_normalized_interval_score": _mean_numeric(group, "normalized_interval_score"),
                "mean_monotonic_violation_rate": _mean_numeric(group, "monotonic_violation_rate"),
            }
        )
    write_rows_csv(out_dir / "split_summary.csv", summary)
    report = out_dir / "split_report.md"
    lines = [
        "# Split Evidence Summary",
        "",
        f"Model token: `{model_token}`.",
        "",
        "| Split | Regimes | MAE | RMSE | Coverage | Width | Monotonic Violations |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {split} | {regimes} | {mae} | {rmse} | {coverage} | {width} | {mono} |".format(
                split=row["metric_split"],
                regimes=row["regimes"],
                mae=f"{float(row['mean_mae']):.4f}" if row["mean_mae"] != "" else "",
                rmse=f"{float(row['mean_rmse']):.4f}" if row["mean_rmse"] != "" else "",
                coverage=f"{float(row['mean_coverage']):.4f}" if row["mean_coverage"] != "" else "",
                width=f"{float(row['mean_interval_width']):.4f}" if row["mean_interval_width"] != "" else "",
                mono=f"{float(row['mean_monotonic_violation_rate']):.4f}" if row["mean_monotonic_violation_rate"] != "" else "",
            )
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Split evidence report written to {report}")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m bearing_dt.evidence")
    sub = parser.add_subparsers(dest="command", required=True)
    summarize = sub.add_parser("summarize-model")
    summarize.add_argument("--runs", default="runs")
    summarize.add_argument("--out", default="paper_artifacts/evidence")
    summarize.add_argument("--model-token", required=True)
    args = parser.parse_args(argv)
    if args.command == "summarize-model":
        summarize_model(runs=args.runs, out=args.out, model_token=args.model_token)


if __name__ == "__main__":
    main()
