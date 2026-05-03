from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from statistics import mean

import numpy as np

from bearing_dt.table import Rows, write_rows_csv
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


def _collect_test_rows(runs_dir: Path) -> Rows:
    rows: Rows = []
    for metrics_path in sorted(runs_dir.glob("*/metrics.json")):
        metrics = read_json(metrics_path)
        run = metrics.get("run", {})
        name = str(run.get("experiment_name", metrics_path.parent.name))
        test = metrics.get("test")
        if not isinstance(test, dict) or "mae" not in test:
            continue
        rows.append(
            {
                "experiment_name": name,
                "base_model": _base_model_name(name),
                "test_regime": _regime_name(name),
                "test_bearing": _bearing_name(name),
                "run_dir": str(metrics_path.parent),
                **test,
            }
        )
    return rows


def _select_model(rows: Rows, token: str) -> Rows:
    selected = [row for row in rows if token in str(row["base_model"])]
    if not selected:
        raise ValueError(f"No rows matched model token: {token}")
    return selected


def _by_regime(rows: Rows) -> dict[str, dict]:
    out = {}
    for row in rows:
        split_id = str(row.get("test_regime") or row.get("test_bearing") or "")
        if split_id:
            out[split_id] = row
    return out


def _bootstrap_ci(values: np.ndarray, *, reps: int = 20000, seed: int = 123) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = []
    n = len(values)
    for _ in range(reps):
        idx = rng.integers(0, n, size=n)
        samples.append(float(np.mean(values[idx])))
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _exact_sign_flip_p(values: np.ndarray) -> float:
    observed = abs(float(np.mean(values)))
    if len(values) > 20:
        rng = np.random.default_rng(456)
        means = []
        for _ in range(200000):
            signs = rng.choice([-1.0, 1.0], size=len(values))
            means.append(abs(float(np.mean(values * signs))))
        return float(np.mean(np.asarray(means) >= observed - 1e-12))
    count = 0
    total = 0
    for signs in itertools.product([-1.0, 1.0], repeat=len(values)):
        total += 1
        if abs(float(np.mean(values * np.asarray(signs)))) >= observed - 1e-12:
            count += 1
    return count / total


def _exact_wilcoxon_p(values: np.ndarray) -> float:
    nonzero = np.asarray([v for v in values if abs(v) > 1e-12], dtype=float)
    if len(nonzero) == 0:
        return 1.0
    order = np.argsort(np.abs(nonzero))
    ranks = np.empty(len(nonzero), dtype=float)
    ranks[order] = np.arange(1, len(nonzero) + 1, dtype=float)
    observed = abs(float(np.sum(ranks * np.sign(nonzero))))
    count = 0
    total = 0
    for signs in itertools.product([-1.0, 1.0], repeat=len(nonzero)):
        total += 1
        stat = abs(float(np.sum(ranks * np.asarray(signs))))
        if stat >= observed - 1e-12:
            count += 1
    return count / total


def compare_models(
    *,
    runs: str | Path,
    out: str | Path,
    candidate: str,
    baselines: list[str],
) -> Path:
    rows = _collect_test_rows(Path(runs))
    candidate_rows = _select_model(rows, candidate)
    candidate_by_regime = _by_regime(candidate_rows)
    out_dir = ensure_dir(out)
    detail: Rows = []
    summary: Rows = []
    for baseline_token in baselines:
        baseline_by_regime = _by_regime(_select_model(rows, baseline_token))
        regimes = sorted(set(candidate_by_regime) & set(baseline_by_regime))
        if not regimes:
            raise ValueError(f"No paired regimes for {candidate} vs {baseline_token}")
        diffs = []
        rel_improvements = []
        wins = 0
        for regime in regimes:
            cand = float(candidate_by_regime[regime]["mae"])
            base = float(baseline_by_regime[regime]["mae"])
            diff = base - cand
            rel = diff / max(base, 1e-12)
            diffs.append(diff)
            rel_improvements.append(rel)
            wins += int(cand < base)
            detail.append(
                {
                    "candidate": candidate,
                    "baseline": baseline_token,
                    "test_split": regime,
                    "candidate_mae": cand,
                    "baseline_mae": base,
                    "mae_improvement": diff,
                    "relative_improvement": rel,
                    "candidate_wins": cand < base,
                }
            )
        diff_arr = np.asarray(diffs, dtype=float)
        rel_arr = np.asarray(rel_improvements, dtype=float)
        diff_ci = _bootstrap_ci(diff_arr)
        rel_ci = _bootstrap_ci(rel_arr)
        summary.append(
            {
                "candidate": candidate,
                "baseline": baseline_token,
                "paired_regimes": len(regimes),
                "candidate_mean_mae": mean(float(candidate_by_regime[r]["mae"]) for r in regimes),
                "baseline_mean_mae": mean(float(baseline_by_regime[r]["mae"]) for r in regimes),
                "mean_mae_improvement": float(np.mean(diff_arr)),
                "mae_improvement_ci_low": diff_ci[0],
                "mae_improvement_ci_high": diff_ci[1],
                "mean_relative_improvement": float(np.mean(rel_arr)),
                "relative_improvement_ci_low": rel_ci[0],
                "relative_improvement_ci_high": rel_ci[1],
                "wins": wins,
                "losses": len(regimes) - wins,
                "exact_sign_flip_p_two_sided": _exact_sign_flip_p(diff_arr),
                "exact_wilcoxon_p_two_sided": _exact_wilcoxon_p(diff_arr),
            }
        )
    write_rows_csv(out_dir / "paired_detail.csv", detail)
    write_rows_csv(out_dir / "paired_summary.csv", summary)
    report = out_dir / "statistical_report.md"
    lines = [
        "# Paired Statistical Evidence",
        "",
        f"Candidate token: `{candidate}`.",
        "",
        "| Baseline | Regimes | Candidate MAE | Baseline MAE | Mean Relative Improvement | 95% Bootstrap CI | Wins | Sign-flip p | Wilcoxon p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {baseline} | {paired_regimes} | {candidate_mean_mae:.4f} | {baseline_mean_mae:.4f} | {rel:.2f}% | [{lo:.2f}%, {hi:.2f}%] | {wins}/{paired_regimes} | {sign:.4f} | {wilcoxon:.4f} |".format(
                baseline=row["baseline"],
                paired_regimes=int(row["paired_regimes"]),
                candidate_mean_mae=float(row["candidate_mean_mae"]),
                baseline_mean_mae=float(row["baseline_mean_mae"]),
                rel=100 * float(row["mean_relative_improvement"]),
                lo=100 * float(row["relative_improvement_ci_low"]),
                hi=100 * float(row["relative_improvement_ci_high"]),
                wins=int(row["wins"]),
                sign=float(row["exact_sign_flip_p_two_sided"]),
                wilcoxon=float(row["exact_wilcoxon_p_two_sided"]),
            )
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Statistical report written to {report}")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m bearing_dt.stats")
    sub = parser.add_subparsers(dest="command", required=True)
    cmp_parser = sub.add_parser("compare")
    cmp_parser.add_argument("--runs", default="runs")
    cmp_parser.add_argument("--out", default="paper_artifacts/statistics")
    cmp_parser.add_argument("--candidate", required=True)
    cmp_parser.add_argument("--baseline", action="append", required=True)
    args = parser.parse_args(argv)
    if args.command == "compare":
        compare_models(runs=args.runs, out=args.out, candidate=args.candidate, baselines=args.baseline)


if __name__ == "__main__":
    main()
