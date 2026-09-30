"""Build paper tables/macros from aggregate evidence, without retraining."""
import json
import re
from pathlib import Path
import pandas as pd

from .summarize import NAMES
from .evaluate import LEARNED


ROOT = Path("QREI submission")
PAPER = ROOT/"manuscript"

# Reader-facing labels for the bootstrap ranking criteria (also used by audit_tables.py).
CRITERION_LABELS = {
    "MAE_hours": "Mean absolute error (h)",
    "nMAE": "Normalized error",
    "asymmetric_5": "Asymmetric error, 5:1",
    "asymmetric_10": "Asymmetric error, 10:1",
    "alpha_lambda_accuracy": r"Accuracy within $\pm$20\%",
    "coverage_error": "Coverage gap",
    "interval_score_hours": "Interval score (h)",
    "normalized_interval_score": "Normalized interval score",
    "prognostic_horizon_fraction": "Prognostic horizon",
}
TRIGGER_LABELS = {"point": "Point forecast", "lower_bound": "Lower bound", "control": "Control"}
CONTROL_NAMES = {"always_action": "Always act at first record", "never_action": "Never act"}
DEFECTS = {"IR": "inner race", "OR": "outer race", "B": "ball"}


NUMERIC = re.compile(r"^[\[\]\s,+\-0-9.]+$")


def esc(s):
    return str(s).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def minus(cell):
    """Typeset the hyphen of a purely numeric cell as a true minus sign."""
    text = str(cell)
    return text.replace("-", "$-$") if NUMERIC.match(text) else text


def table(path, columns, rows, spec, caption, label, placement="tbp", size=r"\small", note=None):
    """Write a booktabs table. A string row is copied verbatim (for group headings)."""
    source = [r"\begin{table}["+placement+"]", r"\centering"+size, r"\caption{"+caption+"}", r"\label{"+label+"}",
              r"\begin{tabular}{"+spec+"}", r"\toprule", " & ".join(columns)+r"\\", r"\midrule"]
    for row in rows:
        source.append(row if isinstance(row, str) else " & ".join(minus(v) for v in row)+r"\\")
    source += [r"\bottomrule", r"\end{tabular}"]
    if note:
        source += [r"\par\smallskip\parbox{\linewidth}{\footnotesize "+note+"}"]
    source += [r"\end{table}"]
    path.write_text("\n".join(source)+"\n", encoding="utf-8")


def group(title, ncol):
    return r"\multicolumn{"+str(ncol)+r"}{@{}l}{\textit{"+title+r"}}\\"


def sensitivity_macros(sens, s, policy):
    """Macros for the registered secondary analyses (protocol_sensitivity_addendum.json)."""
    top = pd.read_csv(sens/"rank_scenarios_top.csv")
    paired = pd.read_csv(sens/"paired_differences.csv").set_index(["metric", "comparison"])
    pop = pd.read_csv(sens/"population_controls_per_bearing.csv").groupby("control")[["nMAE", "late_MAE_hours"]].mean()
    phase = pd.read_csv(sens/"phase_split_per_bearing.csv").groupby("model")[["early_error_over_life", "final40_error_over_life"]].mean()
    horizon = pd.read_csv(sens/"horizon_support.csv")
    gaps = pd.read_csv(sens/"calendar_gaps.csv").set_index("bearing_id")
    first = pd.read_csv(sens/"lower_bound_first_record.csv").set_index("model")
    roles = pd.read_csv(sens/"roles_model_summary.csv").set_index(["scenario", "model"])
    roles_m = pd.read_csv(sens/"roles_maintenance_one_hour.csv").set_index(["scenario", "model", "policy"])
    seeds = pd.read_csv(sens/"seed_model_summary.csv").set_index(["seed", "model"])
    t = top.set_index(["scenario", "criterion"])
    def best(scenario):
        g = top[top.scenario == scenario]
        return g.loc[g.first_rank_fraction.idxmax()]
    deleted = top[top.scenario.str.startswith("endpoint_all8_without_")]
    dmax = deleted.loc[deleted.first_rank_fraction.idxmax()]
    # Facts the text states in words; the build stops if the evidence no longer supports them.
    assert t.loc[("endpoint_common7", "MAE_hours")].model == "representation"
    assert len(horizon[horizon.inputs == "endpoint_aware"]) == 0
    vib_h = horizon[horizon.inputs == "vibration_only"]
    assert len(vib_h) == 1 and vib_h.iloc[0].model == "representation" and vib_h.iloc[0].bearing_id == "B12", vib_h
    assert dmax.scenario == "endpoint_all8_without_B11" and dmax.model == "attention", dmax
    assert set(gaps.index[gaps.gaps_over_5_min_total_minutes > 0]) == {"B08", "B10"}
    k = lambda key: paired.loc[key]
    macros = {
        "CommonHour": (t.loc[("endpoint_common7", "MAE_hours")].first_rank_fraction, ".2f"),
        "CommonTop": (best("endpoint_common7").first_rank_fraction, ".2f"),
        "VibCommonHorizon": (t.loc[("vibration_common7", "prognostic_horizon_fraction")].first_rank_fraction, ".2f"),
        "HorizonSeconds": (vib_h.iloc[0].residual_seconds_at_start, ".0f"),
        "HorizonRecords": (vib_h.iloc[0].sustained_records, "d"),
        "DeleteTop": (dmax.first_rank_fraction, ".2f"),
        "LateDiffClock": (k(("late_MAE_hours", "representation minus time_only"))["mean"], ".2f"),
        "LateDiffClockLow": (k(("late_MAE_hours", "representation minus time_only")).CI_low, ".2f"),
        "LateDiffClockHigh": (k(("late_MAE_hours", "representation minus time_only")).CI_high, ".2f"),
        "LateWinsClock": (k(("late_MAE_hours", "representation minus time_only")).bearings_lower, "d"),
        "LateGainClock": (-k(("late_MAE_hours", "representation minus time_only"))["mean"], ".1f"),
        "LateDiffContext": (k(("late_MAE_hours", "representation minus context_only"))["mean"], ".2f"),
        "LateDiffContextLow": (k(("late_MAE_hours", "representation minus context_only")).CI_low, ".2f"),
        "LateDiffContextHigh": (k(("late_MAE_hours", "representation minus context_only")).CI_high, ".2f"),
        "LateWinsContext": (k(("late_MAE_hours", "representation minus context_only")).bearings_lower, "d"),
        "AttnDiff": (k(("nMAE", "representation minus attention"))["mean"], ".3f"),
        "AttnDiffLow": (k(("nMAE", "representation minus attention")).CI_low, ".3f"),
        "AttnDiffHigh": (k(("nMAE", "representation minus attention")).CI_high, ".3f"),
        "AttnWins": (k(("nMAE", "representation minus attention")).bearings_lower, "d"),
        "MrlError": (pop.loc["conditional mean residual life", "nMAE"], ".3f"),
        "MrlLate": (pop.loc["conditional mean residual life", "late_MAE_hours"], ".2f"),
        "MedrlError": (pop.loc["conditional median residual life", "nMAE"], ".3f"),
        "MedrlLate": (pop.loc["conditional median residual life", "late_MAE_hours"], ".2f"),
        "FinalLatent": (phase.loc["representation", "final40_error_over_life"], ".3f"),
        "EarlyLatent": (phase.loc["representation", "early_error_over_life"], ".3f"),
        "FinalClock": (phase.loc["time_only", "final40_error_over_life"], ".3f"),
        "EarlyElapsed": (phase.loc["elapsed_clock", "early_error_over_life"], ".3f"),
        "GapBeight": (gaps.loc["B08", "gaps_over_5_min_total_minutes"], ".0f"),
        "GapBten": (gaps.loc["B10", "gaps_over_5_min_total_minutes"], ".0f"),
        "LowerFirst": (first.loc["representation", "bearings_triggered_at_first_record"], "d"),
        "AttnLowerUnused": (policy.loc[("attention", "lower_bound"), "unused_life_fraction"], ".3f"),
        "AlwaysUnused": (policy.loc[("always_action", "control"), "unused_life_fraction"], ".3f"),
        "LowerGain": (policy.loc[("always_action", "control"), "unused_life_fraction"]
                      - policy.loc[("representation", "lower_bound"), "unused_life_fraction"], ".3f"),
    }
    for sc, tag in (("swap", "Swap"), ("reverse", "Reverse")):
        macros.update({
            tag+"Error": (roles.loc[(sc, "representation"), "nMAE"], ".3f"),
            tag+"Count": (int(roles.loc[(sc, "representation"), "bearings_nMAE_le_0.20"]), "d"),
            tag+"Late": (roles.loc[(sc, "representation"), "late_MAE_hours"], ".2f"),
            tag+"ClockLate": (roles.loc[(sc, "time_only"), "late_MAE_hours"], ".2f"),
            tag+"Coverage": (roles.loc[(sc, "representation"), "coverage"], ".3f"),
            tag+"WorstCoverage": (roles.loc[(sc, "representation"), "worst_coverage"], ".3f"),
            tag+"Top": (best("endpoint_all8_roles_"+sc).first_rank_fraction, ".2f"),
            tag+"LowerUnused": (roles_m.loc[(sc, "representation", "lower_bound"), "unused_life_fraction"], ".3f"),
            tag+"LowerLate": (roles_m.loc[(sc, "representation", "lower_bound"), "too_late"]*8, ".0f"),
        })
    for sc, tag in (("swap", "Swap"), ("reverse", "Reverse")):
        macros[tag+"AttnError"] = (roles.loc[(sc, "attention"), "nMAE"], ".3f")
    # Words in the text: attention is lower than the latent-state network only under the reversed roles.
    assert roles.loc[("reverse", "attention"), "nMAE"] < roles.loc[("reverse", "representation"), "nMAE"]
    assert roles.loc[("swap", "representation"), "nMAE"] < roles.loc[("swap", "attention"), "nMAE"]
    for seed, tag in ((20260930, "SeedB"), (20260931, "SeedC")):
        macros.update({tag+"Top": (best(f"endpoint_all8_seed_{seed}").first_rank_fraction, ".2f"),
                       tag+"AttnError": (seeds.loc[(seed, "attention"), "nMAE"], ".3f")})
    # Seed 20260930 meets target R (two different stable winners); the other seeds do not.
    b_stable = top[(top.scenario == "endpoint_all8_seed_20260930") & (top.first_rank_fraction >= .7)]
    c_stable = top[(top.scenario == "endpoint_all8_seed_20260931") & (top.first_rank_fraction >= .7)]
    assert set(b_stable.model) == {"representation", "attention"}, b_stable
    assert set(c_stable.model) == {"representation"}, c_stable
    macros.update({"SeedBHour": (t.loc[("endpoint_all8_seed_20260930", "MAE_hours")].first_rank_fraction, ".2f"),
                   "SeedBInterval": (t.loc[("endpoint_all8_seed_20260930", "interval_score_hours")].first_rank_fraction, ".2f"),
                   "SeedCNorm": (t.loc[("endpoint_all8_seed_20260931", "nMAE")].first_rank_fraction, ".2f"),
                   "SwapAccuracy": (t.loc[("endpoint_all8_roles_swap", "alpha_lambda_accuracy")].first_rank_fraction, ".2f")})
    lowest = seeds.reset_index().loc[lambda d: d.groupby("seed").nMAE.idxmin()]
    assert set(lowest.model) == {"representation"}, lowest
    macros["PrimaryAttnError"] = (seeds.loc[(20260929, "attention"), "nMAE"], ".3f")
    return macros


def build():
    (PAPER/"tables").mkdir(parents=True, exist_ok=True)
    (PAPER/"figures").mkdir(exist_ok=True)
    r = ROOT/"results/endpoint_v3"
    s = pd.read_csv(r/"model_summary.csv").set_index("model")
    p = pd.read_csv(r/"per_bearing_metrics.csv")
    old = pd.read_csv(ROOT/"results/primary/model_summary.csv").set_index("model")
    b = pd.read_csv(r/"cluster_bootstrap_intervals.csv")
    pair = pd.read_csv(r/"comparison/paired_model_changes.csv").set_index("model")
    ledger = pd.read_csv(ROOT/"evidence/endpoint_ledger.csv")
    rank = pd.read_csv(r/"rank_probabilities.csv")
    ms = pd.read_csv(r/"maintenance_summary.csv")
    seeds = pd.read_csv(r/"comparison/latent_seed_summary.csv")
    rep = p[p.model == "representation"].set_index("bearing_id")
    policy = ms[(ms.required_lead_hours == 1) & (ms.failure_cost_ratio == 10)].set_index(["model", "policy"])
    top = rank.loc[rank.groupby("criterion").P_rank_1.idxmax()].set_index("criterion")
    other = rank[(rank.model != "representation") & (rank.criterion != "prognostic_horizon_fraction")]
    best_other = other.loc[other.P_rank_1.idxmax()]
    vib_rank = pd.read_csv(ROOT/"results/primary/rank_probabilities.csv")
    vib_top = vib_rank.loc[vib_rank.groupby("criterion").P_rank_1.idxmax()].set_index("criterion")
    vals = {
        "BestError": (s.loc["representation", "nMAE"], ".3f"),
        "OldError": (old.loc["representation", "nMAE"], ".3f"),
        "BestHourError": (s.loc["representation", "MAE_hours"], ".2f"),
        "BestLateError": (s.loc["representation", "late_MAE_hours"], ".2f"),
        "ClockLateError": (s.loc["time_only", "late_MAE_hours"], ".2f"),
        "ClockError": (s.loc["time_only", "nMAE"], ".3f"),
        "ElapsedError": (s.loc["elapsed_clock", "nMAE"], ".3f"),
        "BestSlope": (s.loc["representation", "median_slope"], ".2f"),
        "BestCoverage": (s.loc["representation", "coverage"], ".3f"),
        "WorstCoverage": (s.loc["representation", "worst_bearing_coverage"], ".3f"),
        "BestCount": (s.loc["representation", "bearings_nMAE_le_0.20"], "d"),
        "PairDelta": (pair.loc["representation", "delta_v3_minus_v2"], ".3f"),
        "PairLow": (pair.loc["representation", "CI_low"], ".3f"),
        "PairHigh": (pair.loc["representation", "CI_high"], ".3f"),
        "CompetingError": (s.loc["competing_stop", "nMAE"], ".3f"),
        "ThresholdError": (s.loc["competing_threshold", "nMAE"], ".3f"),
        "DegradationError": (s.loc["degradation", "nMAE"], ".3f"),
        "TcnError": (s.loc["tcn", "nMAE"], ".3f"),
        "AttentionError": (s.loc["attention", "nMAE"], ".3f"),
        "RankHour": (top.loc["MAE_hours", "P_rank_1"], ".2f"),
        "RankNorm": (top.loc["nMAE", "P_rank_1"], ".2f"),
        "RankOther": (best_other.P_rank_1, ".2f"),
        "PointLoss": (policy.loc[("representation", "point"), "loss"], ".2f"),
        "PointLate": (policy.loc[("representation", "point"), "too_late"]*8, ".0f"),
        "LowerLoss": (policy.loc[("representation", "lower_bound"), "loss"], ".2f"),
        "LowerUnused": (policy.loc[("representation", "lower_bound"), "unused_life_fraction"], ".3f"),
        "AlwaysLoss": (policy.loc[("always_action", "control"), "loss"], ".2f"),
        "SeedTwo": (seeds.loc[seeds.seed == 20260930, "nMAE"].iloc[0], ".3f"),
        "SeedThree": (seeds.loc[seeds.seed == 20260931, "nMAE"].iloc[0], ".3f"),
        "ErrorBthree": (rep.loc["B03", "nMAE"], ".3f"),
        "ErrorBtwelve": (rep.loc["B12", "nMAE"], ".3f"),
        "CoverageBten": (rep.loc["B10", "coverage"], ".3f"),
        "CoverageBtwelve": (rep.loc["B12", "coverage"], ".3f"),
        "MissCount": (8 - int(s.loc["representation", "bearings_nMAE_le_0.20"]), "d"),
        "SeedTwoCount": (int(seeds.loc[seeds.seed == 20260930, "bearings_nMAE_le_0.20"].iloc[0]), "d"),
        "SeedThreeCount": (int(seeds.loc[seeds.seed == 20260931, "bearings_nMAE_le_0.20"].iloc[0]), "d"),
        "ContextError": (s.loc["context_only", "nMAE"], ".3f"),
        "TemperatureContextError": (s.loc["temperature_only", "nMAE"], ".3f"),
        "RidgeError": (s.loc["subspace_ridge", "nMAE"], ".3f"),
        "RankTop": (rank.P_rank_1.max(), ".2f"),
        "VibRankNorm": (vib_top.loc["nMAE", "P_rank_1"], ".2f"),
        "VibRankAsym": (vib_top.loc["asymmetric_5", "P_rank_1"], ".2f"),
        "VibRankHorizon": (vib_top.loc["prognostic_horizon_fraction", "P_rank_1"], ".2f"),
    }
    # The text names these winners of the vibration-only comparison (target R met there).
    assert vib_top.loc[["nMAE", "asymmetric_5", "asymmetric_10"], "model"].eq("attention").all(), vib_top
    assert vib_top.loc["prognostic_horizon_fraction", "model"] == "representation", vib_top
    assert (vib_top.P_rank_1 >= .70).sum() == 4, vib_top
    # The text states that no criterion gives a first-rank fraction of 0.70 or more.
    assert rank.P_rank_1.max() < .70, rank.P_rank_1.max()
    assert best_other.model == "attention" and best_other.criterion == "interval_score_hours", best_other  # named in the text
    ci = b[(b.model == "representation")&(b.metric == "nMAE")].iloc[0]
    vals.update({"BestErrorLow": (ci.CI_low, ".3f"), "BestErrorHigh": (ci.CI_high, ".3f")})
    vals.update(sensitivity_macros(r/"sensitivity", s, policy))
    # Long tests: the three longest scored tests and their share of the scored hours (Table 6, Table 1).
    life = rep.duration_hours.sort_values()
    longest = list(life.index[-3:])
    assert longest == ["B08", "B17", "B10"], longest
    vals.update({"SlopeBeight": (rep.loc["B08", "slope"], ".2f"), "SlopeBten": (rep.loc["B10", "slope"], ".2f"),
                 "SlopeBseventeen": (rep.loc["B17", "slope"], ".2f"),
                 "LongHours": (life.iloc[-3:].sum(), ".0f"), "ScoredHours": (life.sum(), ".0f")})
    vals = {k: ((int(round(v)) if f == "d" else v), f) for k, (v, f) in vals.items()}
    macros = ["\\newcommand{\\"+k+"}{"+format(v, f).replace("-", "$-$")+"}" for k, (v, f) in vals.items()]
    (PAPER/"numbers.tex").write_text("\n".join(macros)+"\n", encoding="utf-8")
    pd.DataFrame([{"macro": k, "data_value": v, "printed_value": format(v, f), "source": "aggregate CSV read in build_revision.py"}
                  for k, (v, f) in vals.items()]).to_csv(ROOT/"evidence/claims_ledger.csv", index=False)

    rows = []
    for _, row in ledger.iterrows():
        status = {"B03": "Temperature", "B08": "Temperature", "B01": "Uncertain", "B05": "Interrupted"}.get(row.bearing_id, "Vibration")
        defect = ", ".join(DEFECTS[d] for d in str(row.postmortem_defect).split("/"))
        rows.append([row.bearing_id, int(row.records), f"{row.duration_hours:.3f}", int(round(row.raw_fs_Hz/1000)), status,
                     defect, "Yes" if row.event_observed else "No"])
    table(PAPER/"tables/assets.tex", ["Test", "Records", "Hours", "Rate (kHz)", "Stop cause", "Damage found", "Scored"], rows,
          "@{}lrrrlll@{}",
          "The ten analyzed tests. Hours is the calendar time from the first to the last acquisition. Stop cause and damage "
          "found after dismantling follow the archive documentation. B01 and B05 are not scored.", "tab:assets")

    names = list(LEARNED)+["competing_stop"]
    controls = ["time_only", "elapsed_clock", "context_only", "temperature_only", "degradation", "competing_threshold"]

    def result_row(n):
        v = s.loc[n]
        return [esc(NAMES[n]), f"{v.nMAE:.3f}", f"{v.MAE_hours:.2f}", f"{v.late_MAE_hours:.2f}", f"{v.coverage:.3f}",
                f"{v.median_slope:.2f}".replace("-", "$-$"), int(v['bearings_nMAE_le_0.20'])]
    rows = [group("Learned predictors", 7)] + [result_row(n) for n in names] + [r"\addlinespace", group("Reference controls", 7)] \
        + [result_row(n) for n in controls]
    table(PAPER/"tables/results.tex", ["Method", "Norm.\\ error", "Error (h)", "Late (h)", "Coverage", "Slope",
                                       "$n_{\\le 0.20}$"], rows, "@{}lrrrrrr@{}",
          "Forecast accuracy on the eight held-out bearings with the endpoint-aware inputs. Each bearing has equal weight. "
          "Late is the mean absolute error over the second half of each observed life. Coverage refers to empirical 90\\% "
          "intervals. Slope is the median over bearings of the forecast slope against elapsed time ($-1$ for a perfect "
          "forecast). $n_{\\le 0.20}$ counts bearings with normalized error at most 0.20.", "tab:results")

    rows = []
    for n in LEARNED:
        v = pair.loc[n]
        rows.append([esc(NAMES[n]), f"{v.nMAE_v2:.3f}", f"{v.nMAE_v3:.3f}", f"{v.delta_v3_minus_v2:+.3f}",
                     f"[{v.CI_low:+.3f}, {v.CI_high:+.3f}]", int(v.improved_bearings)])
    table(PAPER/"tables/paired.tex", ["Method", "Vibration only", "Endpoint aware", "Change", "95\\% interval", "Improved"],
          rows, "@{}lrrrrr@{}",
          "Matched change in mean normalized error when causal temperature and threshold histories are added. Negative changes favor "
          "the endpoint-aware inputs. Intervals resample the eight test bearings with the fitted models held fixed. Improved "
          "counts the bearings whose error decreased.", "tab:paired",
          placement="htbp", size=r"\footnotesize")

    rows = []
    for _, v in rep.reset_index().iterrows():
        rows.append([v.bearing_id, {"B03": "Temperature", "B08": "Temperature"}.get(v.bearing_id, "Vibration"),
                     f"{v.nMAE:.3f}", f"{v.MAE_hours:.2f}", f"{v.coverage:.3f}", f"{v.slope:.2f}".replace("-", "$-$")])
    table(PAPER/"tables/bearings.tex", ["Test", "Stop cause", "Norm.\\ error", "Error (h)", "Coverage", "Slope"], rows,
          "@{}llrrrr@{}",
          "Results of the latent-state network for each held-out bearing. Slope is the least-squares slope of the forecast against "
          "elapsed time.", "tab:bearings")

    rows = []
    for c in CRITERION_LABELS:
        v = top.loc[c]
        winner = "Tied (all zero)" if c == "prognostic_horizon_fraction" else esc(NAMES[v.model])
        rows.append([CRITERION_LABELS[c], winner, f"{v.P_rank_1:.4f}"])
    table(PAPER/"tables/ranks.tex", ["Criterion", "Most frequent first rank", "Fraction"], rows, "@{}llr@{}",
          "Most frequent first-ranked learned predictor for each criterion over 10,000 bearing resamples. Ties receive "
          "fractional credit. The coverage gap of a bearing is the absolute difference between its interval coverage and the "
          "nominal 0.90. These fractions describe the fixed fits; they are not probabilities that a model is best in general.",
          "tab:ranks", placement="htbp")

    rows = []
    for n in ["representation", "attention", "random_forest", "competing_stop", "competing_threshold", "time_only",
              "always_action", "never_action"]:
        for pol in ("point", "lower_bound", "control"):
            if (n, pol) not in policy.index:
                continue
            v = policy.loc[(n, pol)]
            rows.append([esc(CONTROL_NAMES.get(n, NAMES.get(n, n))), TRIGGER_LABELS[pol], f"{v.loss:.3f}", f"{v.too_late:.3f}",
                         f"{v.unused_life_fraction:.3f}"])
    table(PAPER/"tables/policy.tex", ["Method", "Trigger", "Loss", "Late fraction", "Unused life"], rows, "@{}llrrr@{}",
          "First-trigger maintenance rule with a one-hour required lead time and missed-action penalty $c=10$. One action is "
          "allowed per bearing. Late fraction is the share of the eight bearings with no timely action; unused life is the mean "
          "fraction of observed life left after the required lead time. The loss is dimensionless and excludes real costs.",
          "tab:policy")
    print("Generated paper macros and seven tables from verified aggregates.", flush=True)


if __name__ == "__main__":
    build()
