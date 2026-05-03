from __future__ import annotations

import argparse
import os
from pathlib import Path

from bearing_dt.config import read_yaml
from bearing_dt.table import Rows, group_by, read_rows_csv, write_rows_csv
from bearing_dt.utils import ensure_dir, read_json


def _pyplot():
    if os.environ.get("BEARING_DT_USE_MATPLOTLIB") != "1":
        return None
    try:
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _write_svg_placeholder(path: Path, title: str, lines: list[str]) -> Path:
    svg_path = path.with_suffix(".svg")
    text = "\n".join(
        f'<text x="24" y="{48 + 24 * i}" font-family="Arial" font-size="16">{line}</text>'
        for i, line in enumerate([title, *lines])
    )
    svg_path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="360">'
        '<rect width="100%" height="100%" fill="#ffffff"/>'
        f"{text}</svg>",
        encoding="utf-8",
    )
    return svg_path


def _collect_runs(runs_dir: Path) -> Rows:
    rows = []
    include_smoke = os.environ.get("BEARING_DT_INCLUDE_SMOKE") == "1"
    latest: dict[str, Path] = {}
    for metrics_path in sorted(runs_dir.glob("*/metrics.json")):
        metrics = read_json(metrics_path)
        run = metrics.get("run", {})
        if not include_smoke and str(run.get("experiment_name", "")).startswith("smoke_"):
            continue
        name = str(run.get("experiment_name", metrics_path.parent.name))
        latest[name] = metrics_path
    for metrics_path in latest.values():
        metrics = read_json(metrics_path)
        run = metrics.get("run", {})
        protocol = run.get("split", "")
        run = {k: v for k, v in run.items() if k != "split"}
        for split, values in metrics.items():
            if split == "run" or not isinstance(values, dict):
                continue
            if "mae" in values or "coverage" in values or "rmse" in values:
                row = {"run_dir": str(metrics_path.parent), "metric_split": split, "protocol": protocol, **run, **values}
                rows.append(row)
    return rows


def _test_rows(rows: Rows, names: set[str]) -> Rows:
    return [row for row in rows if row.get("metric_split") == "test" and str(row.get("experiment_name")) in names]


def _evidence_rows(rows: Rows, names: set[str], prefixes: tuple[str, ...]) -> Rows:
    return [
        row
        for row in rows
        if str(row.get("experiment_name")) in names
        and (row.get("metric_split") == "test" or str(row.get("metric_split", "")).startswith(prefixes))
    ]


def _dataset_rows_from_run_configs(prediction_files: list[Path]) -> Rows:
    processed_roots: dict[str, Path] = {}
    for prediction_file in prediction_files:
        config_path = prediction_file.parent / "config.yaml"
        if not config_path.exists():
            continue
        try:
            config = read_yaml(config_path)
        except Exception:
            continue
        processed_dir = Path(config.get("data", {}).get("processed_dir", ""))
        if processed_dir.exists():
            processed_roots[str(processed_dir.resolve())] = processed_dir
    rows: Rows = []
    for processed_dir in processed_roots.values():
        features_path = processed_dir / "features.csv"
        if not features_path.exists():
            continue
        frame = read_rows_csv(features_path)
        if not frame:
            continue
        rows.append(
            {
                "dataset": frame[0].get("dataset", processed_dir.name),
                "processed_dir": str(processed_dir),
                "samples": len(frame),
                "bearings": len({row["bearing_id"] for row in frame}),
                "condition_ids": len({row["condition_id"] for row in frame}),
                "operating_regimes": len({row.get("operating_regime", "") for row in frame}),
                "min_step": min(int(row["step"]) for row in frame),
                "max_step": max(int(row["step"]) for row in frame),
            }
        )
    return rows


def _plot_framework(path: Path) -> None:
    plt = _pyplot()
    if plt is None:
        _write_svg_placeholder(
            path,
            "Framework",
            ["Vibration windows -> encoders -> predictive latent state -> calibrated RUL -> stress tests"],
        )
        return
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    boxes = [
        "Vibration\nwindows",
        "Feature + context\nencoders",
        "Predictive latent\nstate",
        "RUL + calibrated\ninterval",
        "Stress tests\nand tables",
    ]
    for i, text in enumerate(boxes):
        ax.text(i, 0.5, text, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.35", fc="#eef3f8", ec="#334"))
        if i < len(boxes) - 1:
            ax.annotate("", xy=(i + 0.42, 0.5), xytext=(i + 0.58, 0.5), arrowprops=dict(arrowstyle="<-", color="#334"))
    ax.set_xlim(-0.6, len(boxes) - 0.4)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_predictions(predictions: Rows, out: Path) -> None:
    plt = _pyplot()
    if plt is None:
        _write_svg_placeholder(out, "Prediction scatter", ["Matplotlib unavailable; table data remains available."])
        return
    test = [row for row in predictions if row["split"] == "test"]
    if not test:
        test = predictions.copy()
    fig, ax = plt.subplots(figsize=(5, 4))
    true = [float(row["rul"]) for row in test]
    pred = [float(row["y_pred_rul"]) for row in test]
    ax.scatter(true, pred, s=12, alpha=0.7)
    lim = max(max(true), max(pred), 1.0)
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title("Failure cases and prediction scatter")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _plot_calibration(predictions: Rows, out: Path) -> None:
    plt = _pyplot()
    if plt is None:
        _write_svg_placeholder(out, "Uncertainty calibration", ["Matplotlib unavailable; table data remains available."])
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    if predictions and {"lower_rul", "upper_rul"}.issubset(predictions[0]):
        rows = []
        for split, group in group_by(predictions, "split").items():
            covered = sum(float(row["lower_rul"]) <= float(row["rul"]) <= float(row["upper_rul"]) for row in group) / len(group)
            rows.append((split, covered))
        names, vals = zip(*rows)
        ax.bar(names, vals, color="#476b8a")
        ax.axhline(0.9, color="k", linestyle="--", linewidth=1)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Empirical coverage")
    else:
        ax.text(0.5, 0.5, "No interval predictions available", ha="center", va="center")
        ax.axis("off")
    ax.set_title("Uncertainty calibration")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def build_report(runs: str | Path = "runs", out: str | Path = "paper_artifacts") -> Path:
    runs_dir = Path(runs)
    out_dir = ensure_dir(out)
    tables = ensure_dir(out_dir / "tables")
    figures = ensure_dir(out_dir / "figures")
    legacy_counterfactual = figures / "figure5_counterfactual.svg"
    if legacy_counterfactual.exists():
        try:
            legacy_counterfactual.unlink()
        except PermissionError:
            pass
    summary = _collect_runs(runs_dir)
    if not summary:
        summary = [{"note": "No completed runs found. Run an experiment suite before using paper tables as evidence."}]
    main_names = {
        "phme_10b_latent_load_speed",
        "phme_10b_tcn_load_speed",
        "phme_10b_attention_physics_load_speed",
        "phme_10b_gb_load_speed",
        "phme_10b_tabular_load_speed",
        "phme_10b_sklearn_rf_load_speed",
    }
    transfer_names = {"phme_10b_latent_load_speed", "phme_10b_tcn_load_speed"}
    ablation_names = {
        "phme_10b_latent_load_speed",
        "phme_10b_diag_no_context",
        "phme_10b_diag_no_features",
        "phme_10b_diag_no_raw",
        "phme_10b_diag_stress_prefix",
    }
    write_rows_csv(tables / "table2_main_benchmark.csv", _test_rows(summary, main_names) or summary)
    write_rows_csv(
        tables / "table3_transfer_and_shift.csv",
        _evidence_rows(summary, transfer_names, ("test_partial_", "test_stress_", "test_condition_response_")) or summary,
    )
    write_rows_csv(tables / "table4_ablations.csv", _test_rows(summary, ablation_names) or summary)
    _plot_framework(figures / "figure1_framework.png")
    latest_prediction_files: dict[str, Path] = {}
    for p in sorted(runs_dir.glob("*/predictions.csv")):
        metrics_path = p.parent / "metrics.json"
        if metrics_path.exists():
            run_name = read_json(metrics_path).get("run", {}).get("experiment_name", p.parent.name)
            if os.environ.get("BEARING_DT_INCLUDE_SMOKE") != "1" and str(run_name).startswith("smoke_"):
                continue
            latest_prediction_files[str(run_name)] = p
    prediction_files = list(latest_prediction_files.values())
    if prediction_files:
        predictions: Rows = []
        for p in prediction_files:
            for row in read_rows_csv(p):
                if os.environ.get("BEARING_DT_INCLUDE_SMOKE") != "1" and str(row.get("model_type", "")).startswith("smoke_"):
                    continue
                config_path = p.parent / "metrics.json"
                if config_path.exists():
                    run_name = read_json(config_path).get("run", {}).get("experiment_name", "")
                    if os.environ.get("BEARING_DT_INCLUDE_SMOKE") != "1" and str(run_name).startswith("smoke_"):
                        continue
                row["run_dir"] = str(p.parent)
                predictions.append(row)
        dataset_rows = _dataset_rows_from_run_configs(prediction_files)
        write_rows_csv(tables / "table1_dataset_splits.csv", dataset_rows)
        _plot_predictions(predictions, figures / "figure6_failure_cases.png")
        _plot_calibration(predictions, figures / "figure3_uncertainty_calibration.png")
        _write_svg_placeholder(
            figures / "figure4_robustness.png",
            "Robustness stress tests",
            ["Use table rows with split prefix test_stress_ for missing/noisy sensor evidence."],
        )
        _write_svg_placeholder(
            figures / "figure5_condition_response.png",
            "Condition-response consistency",
            ["Rows with split prefix test_condition_response_ test load/speed response without causal claims."],
        )
        damage_rows = [row for row in predictions if str(row.get("damage_pred", "")) != ""]
        if damage_rows:
            plt = _pyplot()
            if plt is None:
                _write_svg_placeholder(
                    figures / "figure2_latent_degradation.png",
                    "Latent degradation",
                    ["Matplotlib unavailable; predictions.csv contains damage_pred."],
                )
            else:
                fig, ax = plt.subplots(figsize=(6, 4))
                test_rows = [row for row in damage_rows if row["split"] == "test"]
                for _, group in group_by(test_rows, "bearing_id").items():
                    ordered = sorted(group, key=lambda r: int(r["step"]))
                    ax.plot([row["life_fraction"] for row in ordered], [row["damage_pred"] for row in ordered], alpha=0.8)
                ax.set_xlabel("Life fraction")
                ax.set_ylabel("Predicted damage")
                ax.set_title("Latent degradation trajectories")
                fig.tight_layout()
                fig.savefig(figures / "figure2_latent_degradation.png", dpi=180)
                plt.close(fig)
    else:
        write_rows_csv(tables / "table1_dataset_splits.csv", [{"note": "No prediction files found."}])
        for name in ["figure2_latent_degradation.png", "figure3_uncertainty_calibration.png", "figure6_failure_cases.png"]:
            plt = _pyplot()
            if plt is None:
                _write_svg_placeholder(figures / name, "Pending figure", ["Run experiments to populate this figure."])
            else:
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.text(0.5, 0.5, "Run experiments to populate this figure", ha="center", va="center")
                ax.axis("off")
                fig.tight_layout()
                fig.savefig(figures / name, dpi=180)
                plt.close(fig)
    readme = out_dir / "REPORT.md"
    readme.write_text(
        "# Paper Artifact Report\n\n"
        "Generated artifacts are reproducible from run directories. Tables are CSV files and figures are SVG/PNG files.\n\n"
        "- Table 2: `tables/table2_main_benchmark.csv`\n"
        "- Table 3: `tables/table3_transfer_and_shift.csv`\n"
        "- Table 4: `tables/table4_ablations.csv`\n"
        "- Matched claim audit: `claim_audit_10b_matched/claim_summary.csv`\n"
        "- Matched paired statistics: `statistics_10b_with_rf/paired_summary.csv`\n"
        "- Separate-calibration sensitivity: `claim_audit_10b_separate_calibration/claim_summary.csv`\n"
        "- Baseline fairness: `claim_audit_10b_fairness/claim_summary.csv`\n"
        "- Bearing-identity check: `claim_audit_10b_bearing/claim_summary.csv`\n"
        "- Diagnostic evidence: `claim_audit_10b_diagnostics/` and `evidence_10b_diagnostics/`\n"
        "- Final claim status: `final_claim_status.md`\n"
        "- Figures: `figures/`\n"
        "- Primary endpoint: leave-operating-regime-out on PHME/time-varying OC data.\n"
        "- Figure 5 is a condition-response consistency artifact, not a causal counterfactual claim.\n",
        encoding="utf-8",
    )
    print(f"Paper artifacts written to {out_dir}")
    return out_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m bearing_dt.paper")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-report")
    build.add_argument("--runs", default="runs")
    build.add_argument("--out", default="paper_artifacts")
    args = parser.parse_args(argv)
    if args.command == "build-report":
        build_report(args.runs, args.out)


if __name__ == "__main__":
    main()
