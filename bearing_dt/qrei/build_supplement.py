"""Create supplementary tables from frozen aggregate evidence."""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from .build_revision import esc, minus, table
from .summarize import NAMES


def feature_group(name):
    """Map a stored feature name to its reader-facing group (checked by audit_tables.py)."""
    if name == "elapsed_hours":
        return "Elapsed time"
    if name.startswith("hi_"):
        return "Vibration indicator"
    if name.startswith("ep_"):
        thermal = any(k in name for k in ("temperature", "thermal", "room"))
        return "Temperature history" if thermal else "Vibration-threshold history"
    return ("Smoothed descriptor, channel " if name.endswith("_ema") else "Current descriptor, channel ") + name[0]


GROUP_TEXT = {
    "Current descriptor, channel A": "RMS, peak, peak-to-peak value, kurtosis, crest factor, skewness; envelope RMS and "
                                     "kurtosis; power in five fixed bands; envelope power and power fraction at the "
                                     "outer-race, inner-race, ball and cage orders",
    "Current descriptor, channel C": "Same 21 descriptors for channel C",
    "Smoothed descriptor, channel A": "Causal averages (5 min half-life) of the channel A descriptors except skewness",
    "Smoothed descriptor, channel C": "Same 20 causal averages for channel C",
    "Vibration indicator": "Causal average of the larger channel RMS; its logarithmic slope over the last 15 min",
    "Temperature history": "Larger of T1 and T2; their difference; ambient temperature; rise above ambient; causal "
                           "average; running maximum; margin to 110 $^{\\circ}$C; slopes over 15, 30 and 60 min",
    "Vibration-threshold history": "RMS slopes over 15, 30 and 60 min; margins to 6, 8 and 10 g; number of crossings "
                                   "of each level so far",
    "Elapsed time": "Hours since the first acquisition",
}


def main():
    root = Path("QREI submission")
    paper = root/"manuscript"
    out = paper/"tables"
    results = root/"results/endpoint_v3"
    cfg = json.loads((root/"protocol_endpoint_v3.json").read_text())
    names = cfg["learned_models"]+cfg["controls"]
    seed = pd.read_csv(results/"comparison/latent_seed_summary.csv")
    rows = [[int(v.seed), "Yes" if v.primary else "No", f"{v.nMAE:.3f}", f"{v.late_MAE_hours:.3f}", f"{v.median_slope:.3f}",
             f"{v.coverage:.3f}", int(v['bearings_nMAE_le_0.20'])] for _, v in seed.iterrows()]
    table(out/"seeds.tex", ["Seed", "Reported", "Norm.\\ error", "Late error (h)", "Median slope", "Coverage",
                            "Bearings $\\le 0.20$"], rows, "@{}rlrrrrr@{}",
          "Latent-state network trained with the three seeds listed before fitting. The main text reports the first seed. "
          "Each row uses the same eight held-out bearings.", "tab:seeds", placement="htbp")
    per = pd.read_csv(results/"per_bearing_metrics.csv")
    source = [r"\begin{longtable}{@{}llrrrrr@{}}",
              r"\caption{Results for every method and held-out bearing with the endpoint-aware inputs. Coverage refers to "
              r"empirical 90\% intervals; slope is the least-squares slope of the forecast against elapsed time. Method codes "
              r"are defined in Table~\ref{tab:codes}.}\label{tab:allmetrics}\\",
              r"\toprule Method & Test & Norm.\ error & Error (h) & Late error (h) & Coverage & Slope\\\midrule\endfirsthead",
              r"\toprule Method & Test & Norm.\ error & Error (h) & Late error (h) & Coverage & Slope\\\midrule\endhead",
              r"\bottomrule\endfoot"]
    for i, name in enumerate(names, 1):
        for _, v in per[per.model == name].sort_values("bearing_id").iterrows():
            cells = [f"M{i}", v.bearing_id] + [f"{x:.3f}" for x in (v.nMAE, v.MAE_hours, v.late_MAE_hours, v.coverage, v.slope)]
            source.append(" & ".join(minus(c) for c in cells)+r"\\")
    source.append(r"\end{longtable}")
    (out/"allmetrics.tex").write_text("\n".join(source)+"\n", encoding="utf-8")
    table(out/"codes.tex", ["Code", "Method"], [[f"M{i}", esc(NAMES[name])] for i, name in enumerate(names, 1)], "@{}ll@{}",
          "Method codes used in Table~\\ref{tab:allmetrics}. M1--M8 are learned predictors; M9--M14 are reference controls.",
          "tab:codes", placement="h")
    support = pd.read_csv(results/"comparison/fitting_cause_support.csv").set_index("test_bearing")
    rows = []
    for b in cfg["event_bearings"]:
        split = json.loads((root/"results/endpoint_neural"/b/"split.json").read_text())
        rows.append([b, ", ".join(split["training_bearings"]), split["validation_bearing"], split["calibration_bearing"],
                     int(support.loc[b, "thermal_fitting_bearings"]), int(support.loc[b, "vibration_fitting_bearings"])])
    table(out/"roles.tex", ["Test", "Fitting bearings", "Validation", "Calibration", "Temperature stops", "Vibration stops"],
          rows, "@{}lp{48mm}llrr@{}",
          "Bearing roles in the eight folds and the stop causes among the five fitting bearings. In the B03 fold, the only "
          "other temperature-stopped test (B08) is the calibration bearing, so no temperature stop is available for fitting.",
          "tab:roles", placement="htbp")
    conditional = pd.read_csv(results/"comparison/conditional_regime_summary.csv")
    conditional = conditional[conditional.model == "representation"].sort_values("physical_regime")
    load = ("Low", "Low", "Low", "Medium", "Medium", "Medium", "High", "High", "High")
    speed = ("low", "medium", "high")*3
    rows = [[int(v.physical_regime), f"{load[int(v.physical_regime)]} load, {speed[int(v.physical_regime)]} speed",
             int(v.independent_bearings), int(v.records), f"{v.largest_bearing_window_fraction:.3f}",
             f"{v.bearing_equal_nMAE:.3f}", f"{v.bearing_equal_coverage:.3f}",
             "Not estimable" if pd.isna(v.coverage_CI_low) else f"[{v.coverage_CI_low:.3f}, {v.coverage_CI_high:.3f}]"]
            for _, v in conditional.iterrows()]
    table(out/"conditional.tex", ["Cell", "Operating condition", "Bearings", "Records", "Largest share", "Norm.\\ error",
                                  "Coverage", "Coverage interval"], rows, "@{}rlrrrrrl@{}",
          "Latent-state results by operating cell. Load bands use static plus dynamic load with cut points 3,500 and "
          "4,250~N; speed bands use 1,800 and 2,500~rpm. Each bearing in a cell has equal weight. Largest share is the "
          "fraction of the cell's records from one bearing. Intervals resample the contributing bearings.", "tab:conditional",
          placement="htbp", size=r"\footnotesize")
    diag = pd.read_csv(results/"comparison/threshold_crossing_diagnostics.csv")
    rows = []
    for b in cfg["raw_bearings"]:
        g = diag[diag.bearing_id == b]
        t = g[(g.quantity == "ep_temperature_max_C") & (g.threshold == 110)].iloc[0]
        v = g[g.quantity.isin(["A_rms_g", "C_rms_g"]) & (g.threshold == 8)]
        crossed = v[v.crossed]
        first = crossed.first_crossing_elapsed_hours.min() if len(crossed) else float("nan")
        fmt = lambda value: "No crossing" if pd.isna(value) else f"{value:.3f}"
        rows.append([b, fmt(t.first_crossing_elapsed_hours), fmt(t.first_crossing_remaining_hours), fmt(first)])
    table(out/"crossings.tex", ["Test", r"First 110~$^{\circ}$C (h)", "Remaining time (h)", "First 8 g RMS (h)"], rows,
          "@{}llll@{}",
          "First acquisition at which the higher of the two bearing temperatures (T1, T2) reaches 110~$^{\\circ}$C or a "
          "channel RMS reaches 8~g. These "
          "acquisition statistics do not reproduce the controller's filtering or persistence rules.", "tab:crossings",
          placement="htbp")
    manifest = json.loads(Path("data/processed/phme_tvoc_10b_endpoint_v3/manifest.json").read_text())
    counts = Counter(feature_group(n) for n in manifest["feature_columns"])
    assert sum(counts.values()) == 104 and set(counts) == set(GROUP_TEXT), counts
    rows = [[g, counts[g], GROUP_TEXT[g]] for g in GROUP_TEXT]
    rows.append([r"\textbf{Total}", r"\textbf{"+str(sum(counts.values()))+"}", ""])
    table(out/"features.tex", ["Group", "Count", "Contents"], rows, r"@{}p{46mm}rp{96mm}@{}",
          "Composition of the 104-value engineered input of the endpoint-aware comparison. The vibration-only comparison "
          "omits the 19 temperature and vibration-threshold history values. Operating context (static load, dynamic load, "
          "shaft speed) is supplied separately.", "tab:features", placement="htbp", size=r"\small")
    print(f"Generated supplement tables: {len(per)} method/bearing rows, {len(manifest['feature_columns'])} features in "
          f"{len(counts)} groups, {len(seed)} seeds.", flush=True)


if __name__ == "__main__":
    main()
