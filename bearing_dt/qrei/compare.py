"""Matched follow-up comparisons, stopping evidence and sparse-cell summaries."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .endpoint import THERMAL_BEARINGS
from .evaluate import LEARNED
from .summarize import NAMES


def main():
    root = Path("QREI submission")
    initial = root/"results/primary"
    follow = root/"results/endpoint_v3"
    out = follow/"comparison"
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((root/"protocol_endpoint_v3.json").read_text())
    old = pd.read_csv(initial/"per_bearing_metrics.csv")
    new = pd.read_csv(follow/"per_bearing_metrics.csv")
    joined = old.merge(new, on=["model", "bearing_id"], suffixes=("_v2", "_v3"), validate="one_to_one")
    joined["delta_nMAE"] = joined.nMAE_v3-joined.nMAE_v2
    joined.to_csv(out/"paired_bearing_changes.csv", index=False)
    rng = np.random.default_rng(cfg["seeds"][0])
    draws = rng.integers(0, 8, size=(cfg["bootstrap_draws"], 8))
    pairs = []
    for name, group in joined.groupby("model"):
        g = group.set_index("bearing_id").reindex(cfg["event_bearings"])
        delta = g.delta_nMAE.to_numpy()
        boot = delta[draws].mean(axis=1)
        ci = np.quantile(boot, [.025, .975])
        pairs.append({"model": name, "nMAE_v2": g.nMAE_v2.mean(), "nMAE_v3": g.nMAE_v3.mean(),
                      "delta_v3_minus_v2": delta.mean(), "CI_low": ci[0], "CI_high": ci[1],
                      "improved_bearings": int(sum(delta < 0)), "independent_bearings": 8,
                      "inference": "descriptive paired bootstrap conditional on the fits; follow-up prompted by v2 results"})
    paired = pd.DataFrame(pairs)
    paired.to_csv(out/"paired_model_changes.csv", index=False)
    new["stopping_cause"] = np.where(new.bearing_id.isin(THERMAL_BEARINGS), "temperature", "vibration")
    strata = new.groupby(["model", "stopping_cause"]).agg(independent_bearings=("bearing_id", "nunique"),
        nMAE=("nMAE", "mean"), MAE_hours=("MAE_hours", "mean"), late_MAE_hours=("late_MAE_hours", "mean"),
        coverage=("coverage", "mean"))
    strata.to_csv(out/"stopping_cause_metrics.csv")
    preds = pd.read_csv(follow/"joined_predictions.csv")
    rows = []
    for (model, b, cell), g in preds.groupby(["model", "bearing_id", "physical_regime"]):
        rows.append({"model": model, "bearing_id": b, "physical_regime": cell, "records": len(g),
            "nMAE": np.mean(np.abs(g.pred_hours-g.rul_hours))/float(g.observed_duration_hours.iloc[0]),
            "coverage": np.mean((g.rul_hours >= g.lower_hours)&(g.rul_hours <= g.upper_hours))})
    cells = pd.DataFrame(rows)
    cells.to_csv(out/"bearing_regime_metrics.csv", index=False)
    cellrows = []
    for (model, cell), g in cells.groupby(["model", "physical_regime"]):
        n = len(g)
        largest = g.records.max()/g.records.sum()
        boot = None
        if n >= 2:
            draw = rng.integers(0, n, size=(10000, n))
            boot = g.coverage.to_numpy()[draw].mean(axis=1)
        ci = np.quantile(boot, [.025, .975]) if boot is not None else (np.nan, np.nan)
        cellrows.append({"model": model, "physical_regime": cell, "independent_bearings": n,
            "records": g.records.sum(), "largest_bearing_window_fraction": largest,
            "bearing_equal_coverage": g.coverage.mean(), "bearing_equal_nMAE": g.nMAE.mean(),
            "coverage_CI_low": ci[0], "coverage_CI_high": ci[1],
            "uncertainty": "bearing bootstrap conditional on fits; very sparse" if n >= 2 else "NOT ESTIMABLE: one independent bearing"})
    pd.DataFrame(cellrows).to_csv(out/"conditional_regime_summary.csv", index=False)
    features = pd.read_csv("data/processed/phme_tvoc_10b_endpoint_v3/features.csv")
    rows = []
    for b, g in features.groupby("bearing_id"):
        for col, thresholds in (("ep_temperature_max_C", [110]), ("A_rms_g", [6, 8, 10]), ("C_rms_g", [6, 8, 10])):
            for threshold in thresholds:
                hits = np.flatnonzero(g[col].to_numpy() >= threshold)
                rows.append({"bearing_id": b, "quantity": col, "threshold": threshold,
                    "crossed": bool(len(hits)), "crossing_count": len(hits),
                    "first_crossing_elapsed_hours": g.elapsed_hours.iloc[hits[0]] if len(hits) else np.nan,
                    "first_crossing_remaining_hours": g.observed_duration_hours.iloc[0]-g.elapsed_hours.iloc[hits[0]] if len(hits) else np.nan,
                    "interpretation": "acquisition mean/RMS and common candidate limits; not reconstruction of controller filtering/persistence"})
    pd.DataFrame(rows).to_csv(out/"threshold_crossing_diagnostics.csv", index=False)
    support = []
    for b in cfg["event_bearings"]:
        split = json.loads((root/"results/endpoint_neural"/b/"split.json").read_text())
        fit = split["training_bearings"]
        support.append({"test_bearing": b, "training_bearings": ",".join(fit),
                        "thermal_fitting_bearings": sum(t in THERMAL_BEARINGS for t in fit),
                        "vibration_fitting_bearings": sum(t not in THERMAL_BEARINGS for t in fit),
                        "test_cause_is_only_a_retrospective_stratum": True})
    pd.DataFrame(support).to_csv(out/"fitting_cause_support.csv", index=False)
    plots(features, paired, out, follow/"figures")
    print(paired[["model", "nMAE_v2", "nMAE_v3", "delta_v3_minus_v2", "CI_low", "CI_high"]].to_string(index=False))


def plots(features, paired, out, figures):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    def save(fig, name):
        fig.savefig(figures/(name+".pdf"), bbox_inches="tight")
        fig.savefig(figures/(name+".png"), bbox_inches="tight", dpi=300)
        plt.close(fig)
    fig, axs = plt.subplots(5, 2, figsize=(11, 12), constrained_layout=True)
    for ax, (b, g) in zip(axs.flat, features.groupby("bearing_id")):
        ax.plot(g.elapsed_hours, g.temperature_1_C, color="#2962a3", label="Position T1")
        ax.plot(g.elapsed_hours, g.temperature_2_C, color="#c56f20", label="Position T2")
        ax.plot(g.elapsed_hours, g.ep_room_temperature_C, color="#606a75", lw=.8, label="Ambient")
        ax.axhline(110, color="#b03f42", ls="--", label="Documented thermal limit")
        ax.set_title(b+(" (diagnostic only)" if b in ("B01", "B05") else " (thermal stop)" if b in THERMAL_BEARINGS else " (vibration stop)"))
        ax.set_xlabel("Elapsed hours"); ax.set_ylabel("Temperature (C)")
    axs[0, 0].legend(fontsize=7, loc="lower right")
    save(fig, "temperature_trajectories")
    g = paired.set_index("model").reindex(LEARNED)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for j, name in enumerate(g.index):
        row = g.loc[name]
        ax.plot([row.CI_low, row.CI_high], [j, j], color="#2962a3", lw=2)
        ax.scatter([row.delta_v3_minus_v2], [j], color="#2962a3", s=42)
    ax.axvline(0, color="#686d75", ls="--")
    ax.set_yticks(range(len(g)), [NAMES[n] for n in g.index])
    ax.set_xlabel("Change in mean normalized error: endpoint-aware minus vibration-only")
    ax.invert_yaxis()
    save(fig, "endpoint_paired_changes")


if __name__ == "__main__":
    main()
