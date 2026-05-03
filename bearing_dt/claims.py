from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from bearing_dt.table import Rows, group_by, read_rows_csv, write_rows_csv
from bearing_dt.utils import ensure_dir, read_json


def _collect_test_metrics(runs_dir: Path, prefix: str) -> Rows:
    rows: Rows = []
    for metrics_path in sorted(runs_dir.glob("*/metrics.json")):
        metrics = read_json(metrics_path)
        run = metrics.get("run", {})
        name = str(run.get("experiment_name", metrics_path.parent.name))
        if prefix and not name.startswith(prefix):
            continue
        test = metrics.get("test")
        if not isinstance(test, dict):
            continue
        row = {
            "experiment_name": name,
            "base_model": _base_model_name(name),
            "test_regime": _regime_name(name),
            "test_bearing": _bearing_name(name),
            "run_dir": str(metrics_path.parent),
            "model_type": run.get("model_type", ""),
            **test,
        }
        rows.append(row)
    return rows


def _base_model_name(name: str) -> str:
    if "__regime_" in name:
        return name.split("__regime_", 1)[0]
    if "__bearing_" in name:
        return name.split("__bearing_", 1)[0]
    return name


def _regime_name(name: str) -> str:
    if "__regime_" in name:
        return name.split("__regime_", 1)[1]
    return ""


def _bearing_name(name: str) -> str:
    if "__bearing_" in name:
        return name.split("__bearing_", 1)[1]
    return ""


def _float(row: dict, key: str) -> float:
    return float(row[key])


def _aggregate(rows: Rows) -> Rows:
    out: Rows = []
    for model, group in group_by(rows, "base_model").items():
        coverages = [_float(row, "coverage") for row in group if str(row.get("coverage", "")) != ""]
        widths = [_float(row, "mean_interval_width") for row in group if str(row.get("mean_interval_width", "")) != ""]
        norm_widths = [_float(row, "normalized_interval_width") for row in group if str(row.get("normalized_interval_width", "")) != ""]
        interval_scores = [_float(row, "interval_score") for row in group if str(row.get("interval_score", "")) != ""]
        norm_interval_scores = [
            _float(row, "normalized_interval_score") for row in group if str(row.get("normalized_interval_score", "")) != ""
        ]
        out.append(
            {
                "base_model": model,
                "regimes": len({row["test_regime"] for row in group if row.get("test_regime")}),
                "bearings": len({row["test_bearing"] for row in group if row.get("test_bearing")}),
                "mean_mae": mean(_float(row, "mae") for row in group),
                "mean_rmse": mean(_float(row, "rmse") for row in group),
                "mean_coverage": mean(coverages) if coverages else "",
                "mean_interval_width": mean(widths) if widths else "",
                "mean_normalized_interval_width": mean(norm_widths) if norm_widths else "",
                "mean_interval_score": mean(interval_scores) if interval_scores else "",
                "mean_normalized_interval_score": mean(norm_interval_scores) if norm_interval_scores else "",
                "mean_monotonic_violation_rate": mean(_float(row, "monotonic_violation_rate") for row in group),
            }
        )
    return sorted(out, key=lambda row: float(row["mean_mae"]))


def _find_model(rows: Rows, token: str) -> dict | None:
    matches = [row for row in rows if token in str(row["base_model"]) and "weak_physics" not in str(row["base_model"])]
    if not matches:
        return None
    return sorted(matches, key=lambda row: float(row["mean_mae"]))[0]


def _audit_unit(aggregate: Rows) -> tuple[str, str, str]:
    regime_count = max((int(row.get("regimes") or 0) for row in aggregate), default=0)
    bearing_count = max((int(row.get("bearings") or 0) for row in aggregate), default=0)
    if bearing_count > 0 and regime_count == 0:
        return (
            "bearing",
            "completed PHME leave-bearing-out sanity-check runs",
            "Bearings",
        )
    return (
        "regime",
        "completed PHME operating-regime matrix runs",
        "Regimes",
    )


def _audit_text(detail: Rows, aggregate: Rows) -> str:
    proposed = _find_model(aggregate, "latent_load_speed") or _find_model(aggregate, "latent_twin")
    baselines = [row for row in aggregate if row is not proposed and "latent" not in str(row["base_model"])]
    best_baseline = min(baselines, key=lambda row: float(row["mean_mae"])) if baselines else None
    unit_key, scope, unit_label = _audit_unit(aggregate)
    lines = [
        "# Claim Audit",
        "",
        f"Scope: artifact-level audit over {scope}.",
        "",
        "## Aggregate Ranking",
        "",
        f"| Model | {unit_label} | Mean MAE | Mean RMSE | Mean Coverage | Mean Width | Mean Monotonic Violations |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            "| {base_model} | {unit_count} | {mean_mae:.4f} | {mean_rmse:.4f} | {coverage} | {width} | {mono:.4f} |".format(
                base_model=row["base_model"],
                unit_count=row[unit_key + "s"],
                mean_mae=float(row["mean_mae"]),
                mean_rmse=float(row["mean_rmse"]),
                coverage=f"{float(row['mean_coverage']):.4f}" if str(row.get("mean_coverage", "")) != "" else "",
                width=f"{float(row['mean_interval_width']):.4f}" if str(row.get("mean_interval_width", "")) != "" else "",
                mono=float(row["mean_monotonic_violation_rate"]),
            )
        )
    lines.extend(["", "## Verdict", ""])
    if proposed is None:
        lines.append("Blocked: no proposed latent-state matrix runs were found.")
    elif best_baseline is None:
        lines.append("Blocked: no baseline matrix runs were found.")
    else:
        improvement = (float(best_baseline["mean_mae"]) - float(proposed["mean_mae"])) / max(float(best_baseline["mean_mae"]), 1e-12)
        coverage = float(proposed["mean_coverage"]) if str(proposed.get("mean_coverage", "")) != "" else float("nan")
        lines.append(f"Best baseline by mean MAE: `{best_baseline['base_model']}`.")
        lines.append(f"Proposed mean MAE improvement over best baseline: {100 * improvement:.2f}%.")
        if unit_key == "bearing":
            lines.append("Leave-bearing-out sanity-check status: descriptive supporting evidence, not the primary operating-regime endpoint.")
            lines.append("This check reports bearing-identity generalization; it does not apply the primary operating-regime claim gate.")
        elif improvement >= 0.10 and 0.85 <= coverage <= 0.95:
            lines.append("Primary PHME matrix claim status: PASS for MAE and interval coverage on the completed matrix.")
        else:
            lines.append("Primary PHME matrix claim status: NOT FULLY VERIFIED.")
        lines.append(
            "Monotonicity is reported as evidence, not a pass criterion, until the latent damage state is made reliably monotone across held-out regimes."
        )
    lines.extend(
        [
            "",
            "## Scope Limits",
            "",
            "- This does not replace the primary leave-operating-regime-out endpoint.",
            "- This does not verify XJTU-SY secondary validation.",
            "- This does not verify the full 17-bearing PHME record.",
            "- If either is required, that is a dataset scale-up decision.",
        ]
    )
    return "\n".join(lines) + "\n"


def audit_claims(runs: str | Path, out: str | Path, prefix: str = "phme_matrix") -> Path:
    out_dir = ensure_dir(out)
    detail = _collect_test_metrics(Path(runs), prefix)
    write_rows_csv(out_dir / "claim_detail.csv", detail)
    aggregate = _aggregate(detail) if detail else []
    write_rows_csv(out_dir / "claim_summary.csv", aggregate)
    report = out_dir / "claim_audit.md"
    report.write_text(_audit_text(detail, aggregate), encoding="utf-8")
    print(f"Claim audit written to {report}")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m bearing_dt.claims")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--runs", default="runs")
    audit.add_argument("--out", default="paper_artifacts/claim_audit")
    audit.add_argument("--prefix", default="phme_matrix")
    args = parser.parse_args(argv)
    if args.command == "audit":
        audit_claims(args.runs, args.out, args.prefix)


if __name__ == "__main__":
    main()
