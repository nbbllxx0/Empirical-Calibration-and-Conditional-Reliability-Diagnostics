"""Secondary analyses registered in protocol_sensitivity_addendum.json.

Primary results are read, never rewritten. Output: results/endpoint_v3/sensitivity/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .summarize import metrics, decision_rows

ROOT = Path("QREI submission")
RESULTS = ROOT/"results"
OUT = RESULTS/"endpoint_v3"/"sensitivity"
CRITERIA = {"nMAE": False, "MAE_hours": False, "normalized_interval_score": False, "interval_score_hours": False,
            "asymmetric_5": False, "asymmetric_10": False, "coverage_error": False,
            "alpha_lambda_accuracy": True, "prognostic_horizon_fraction": True}


def config():
    return json.loads((ROOT/"protocol_endpoint_v3.json").read_text(encoding="utf-8"))


def draws(n_units, cfg):
    return np.random.default_rng(cfg["seeds"][0]).integers(0, n_units, size=(cfg["bootstrap_draws"], n_units))


def rank_fractions(per, models, units, label, cfg):
    """First-rank fractions with fractional tie credit, as in summarize.py."""
    d = draws(len(units), cfg)
    names = sorted(models)
    rows = []
    for criterion, higher in CRITERIA.items():
        matrix = per.pivot(index="bearing_id", columns="model", values=criterion).reindex(index=units, columns=names).to_numpy()
        if np.isnan(matrix).any():
            raise ValueError(f"Missing values for {label}/{criterion}")
        scores = matrix[d].mean(axis=1)
        best = scores.max(axis=1, keepdims=True) if higher else scores.min(axis=1, keepdims=True)
        ties = np.isclose(scores, best, rtol=1e-10, atol=1e-12)
        fractions = (ties/ties.sum(axis=1, keepdims=True)).mean(axis=0)
        for name, value in zip(names, fractions):
            rows.append({"scenario": label, "model": name, "criterion": criterion, "first_rank_fraction": float(value)})
    return pd.DataFrame(rows)


def per_bearing(predictions):
    rows = []
    for (model, bearing), g in predictions.groupby(["model", "bearing_id"]):
        rows.append({"model": model, "bearing_id": bearing, **metrics(g.sort_values("elapsed_hours"))})
    return pd.DataFrame(rows)


def read_joined(folder):
    """joined_predictions.csv, or its .gz copy as published in the public repository."""
    path = Path(folder)/"joined_predictions.csv"
    return pd.read_csv(path if path.exists() else path.with_name(path.name+".gz"))


def load_predictions(folders):
    """Per-fold forecast files, or the run's joined_predictions.csv.gz as published in the public repository."""
    parts = []
    for folder in map(Path, folders):
        files = sorted(folder.rglob("predictions.csv"))
        parts += [pd.read_csv(p) for p in files] if files else [pd.read_csv(folder/"joined_predictions.csv.gz")]
    frame = pd.concat(parts, ignore_index=True)
    if frame.duplicated(["model", "bearing_id", "record_id"]).any():
        raise ValueError("Duplicate forecast rows in " + ", ".join(map(str, folders)))
    return frame


def check_complete(predictions, models, expected_rows, label):
    counts = predictions.groupby("model").size()
    missing = set(models)-set(counts.index)
    if missing or any(counts[m] != expected_rows for m in models):
        raise ValueError(f"{label}: incomplete forecasts {counts.to_dict()}, expected {expected_rows} for {sorted(models)}")


def radius_table(predictions, label):
    return (predictions.groupby(["model", "bearing_id"]).calibration_radius_hours.first()
            .rename("half_width_hours").reset_index().assign(scenario=label))


def main():
    cfg = config()
    addendum = json.loads((ROOT/"protocol_sensitivity_addendum.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    bearings = cfg["event_bearings"]
    learned = cfg["learned_models"]
    common = [m for m in learned if m != "competing_stop"]
    primary_pred = read_joined(RESULTS/"endpoint_v3")
    v3 = pd.read_csv(RESULTS/"endpoint_v3"/"per_bearing_metrics.csv")
    v2 = pd.read_csv(RESULTS/"primary"/"per_bearing_metrics.csv")
    expected_rows = int(primary_pred.groupby("model").size().iloc[0])
    ranks = [rank_fractions(v2, common, bearings, "vibration_common7", cfg),
             rank_fractions(v3, common, bearings, "endpoint_common7", cfg),
             rank_fractions(v3, learned, bearings, "endpoint_all8", cfg)]
    saved = pd.read_csv(RESULTS/"endpoint_v3"/"rank_probabilities.csv")
    check = ranks[-1].merge(saved, on=["model", "criterion"])
    reproduction = float((check.first_rank_fraction-check.P_rank_1).abs().max())
    for b in bearings:
        ranks.append(rank_fractions(v3, learned, [u for u in bearings if u != b], f"endpoint_all8_without_{b}", cfg))

    # Seeds: every stochastic learned predictor refitted with the same seed.
    seed_summary, seed_per = [], []
    for seed in addendum["seed_sensitivity"]["seeds"]:
        refit = load_predictions([RESULTS/"sensitivity"/f"seed_{seed}_tabular", RESULTS/"sensitivity"/f"seed_{seed}_neural"])
        latent = read_joined(RESULTS/f"endpoint_seed_{seed}")
        stop = primary_pred[primary_pred.model == "competing_stop"]
        pred = pd.concat([refit, latent, stop], ignore_index=True)
        check_complete(pred, learned, expected_rows, f"seed {seed}")
        per = per_bearing(pred[pred.model.isin(learned)])
        ranks.append(rank_fractions(per, learned, bearings, f"endpoint_all8_seed_{seed}", cfg))
        seed_per.append(per.assign(seed=seed))
    seed_per.append(v3[v3.model.isin(learned)].assign(seed=cfg["seeds"][0]))
    seed_per = pd.concat(seed_per, ignore_index=True)
    for (seed, model), g in seed_per.groupby(["seed", "model"]):
        seed_summary.append({"seed": int(seed), "primary": int(seed) == cfg["seeds"][0], "model": model,
                             "nMAE": g.nMAE.mean(), "late_MAE_hours": g.late_MAE_hours.mean(),
                             "bearings_nMAE_le_0.20": int((g.nMAE <= .2).sum()), "coverage": g.coverage.mean()})
    pd.DataFrame(seed_summary).to_csv(OUT/"seed_model_summary.csv", index=False)

    # Alternative fold roles: all methods refitted.
    role_summary, radii, maint = [], [radius_table(primary_pred, "forward")], []
    primary_dec = pd.read_csv(RESULTS/"endpoint_v3"/"maintenance_decisions.csv").assign(scenario="forward")
    maint.append(primary_dec)
    all_models = sorted(primary_pred.model.unique())
    for roles in addendum["role_allocation_sensitivity"]["allocations"]:
        pred = load_predictions([RESULTS/"sensitivity"/f"roles_{roles}_tabular", RESULTS/"sensitivity"/f"roles_{roles}_neural"])
        check_complete(pred, all_models, expected_rows, f"roles {roles}")
        per = per_bearing(pred)
        ranks.append(rank_fractions(per, learned, bearings, f"endpoint_all8_roles_{roles}", cfg))
        per.assign(scenario=roles).to_csv(OUT/f"roles_{roles}_per_bearing.csv", index=False)
        for model, g in per.groupby("model"):
            role_summary.append({"scenario": roles, "model": model, "nMAE": g.nMAE.mean(),
                                 "late_MAE_hours": g.late_MAE_hours.mean(), "median_slope": g.slope.median(),
                                 "bearings_nMAE_le_0.20": int((g.nMAE <= .2).sum()), "coverage": g.coverage.mean(),
                                 "worst_coverage": g.coverage.min()})
        radii.append(radius_table(pred, roles))
        maint.append(decision_rows(pred).assign(scenario=roles))
    for model, g in v3.groupby("model"):
        role_summary.append({"scenario": "forward", "model": model, "nMAE": g.nMAE.mean(),
                             "late_MAE_hours": g.late_MAE_hours.mean(), "median_slope": g.slope.median(),
                             "bearings_nMAE_le_0.20": int((g.nMAE <= .2).sum()), "coverage": g.coverage.mean(),
                             "worst_coverage": g.coverage.min()})
    pd.DataFrame(role_summary).to_csv(OUT/"roles_model_summary.csv", index=False)
    pd.concat(radii, ignore_index=True).to_csv(OUT/"roles_half_widths.csv", index=False)
    maint = pd.concat(maint, ignore_index=True)
    one_hour = maint[(maint.required_lead_hours == 1) & (maint.failure_cost_ratio == 10)]
    one_hour.groupby(["scenario", "model", "policy"])[["loss", "too_late", "unused_life_fraction"]].mean().reset_index() \
        .to_csv(OUT/"roles_maintenance_one_hour.csv", index=False)

    rank = pd.concat(ranks, ignore_index=True)
    rank.to_csv(OUT/"rank_scenarios.csv", index=False)
    top = rank.loc[rank.groupby(["scenario", "criterion"]).first_rank_fraction.idxmax()].sort_values(["scenario", "criterion"])
    top.to_csv(OUT/"rank_scenarios_top.csv", index=False)

    # Support of nonzero prognostic horizons in both input sets.
    horizon = []
    for label, per, folder in (("vibration_only", v2, RESULTS/"primary"), ("endpoint_aware", v3, RESULTS/"endpoint_v3")):
        raw = read_joined(folder)
        for row in per[per.model.isin(learned) & (per.prognostic_horizon_fraction > 0)].itertuples():
            g = raw[(raw.model == row.model) & (raw.bearing_id == row.bearing_id)].sort_values("elapsed_hours")
            g = g[g.rul_hours > 0]
            accurate = np.abs(g.pred_hours.to_numpy()-g.rul_hours.to_numpy()) <= .2*g.rul_hours.to_numpy()
            sustained = np.logical_and.accumulate(accurate[::-1])[::-1]
            start = g.iloc[int(np.flatnonzero(sustained)[0])]
            horizon.append({"inputs": label, "model": row.model, "bearing_id": row.bearing_id,
                            "horizon_fraction": row.prognostic_horizon_fraction, "sustained_records": int(sustained.sum()),
                            "residual_seconds_at_start": float(start.rul_hours*3600)})
    pd.DataFrame(horizon, columns=["inputs", "model", "bearing_id", "horizon_fraction", "sustained_records",
                                   "residual_seconds_at_start"]).to_csv(OUT/"horizon_support.csv", index=False)

    # Paired bearing differences (fixed fits, primary draws).
    d = draws(len(bearings), cfg)
    paired = []
    for metric, other in (("late_MAE_hours", "time_only"), ("late_MAE_hours", "elapsed_clock"),
                          ("late_MAE_hours", "context_only"), ("late_MAE_hours", "temperature_only"),
                          ("nMAE", "attention"), ("MAE_hours", "attention"), ("late_MAE_hours", "attention")):
        p = v3.pivot(index="bearing_id", columns="model", values=metric).reindex(bearings)
        delta = (p["representation"]-p[other]).to_numpy()
        boot = delta[d].mean(axis=1)
        paired.append({"metric": metric, "comparison": "representation minus " + other, "mean": float(delta.mean()),
                       "CI_low": float(np.quantile(boot, .025)), "CI_high": float(np.quantile(boot, .975)),
                       "bearings_lower": int((delta < 0).sum())})
    pd.DataFrame(paired).to_csv(OUT/"paired_differences.csv", index=False)

    # Population controls from the five fitting lifetimes of each fold.
    life = primary_pred.groupby("bearing_id").observed_duration_hours.first()
    from .roles import fold_roles
    rows = []
    base = primary_pred[primary_pred.model == "time_only"]
    for b in bearings:
        train, _, _ = fold_roles(bearings, b)
        lives = np.array([life[t] for t in train])
        g = base[base.bearing_id == b].sort_values("elapsed_hours")
        t, y, L = g.elapsed_hours.to_numpy(), g.rul_hours.to_numpy(), float(life[b])
        late = t >= .5*L
        for kind in ("mean", "median"):
            pred = np.array([(np.mean if kind == "mean" else np.median)(lives[lives > x]-x) if (lives > x).any() else 0.0 for x in t])
            err = np.abs(pred-y)
            rows.append({"control": f"conditional {kind} residual life", "bearing_id": b, "nMAE": err.mean()/L,
                         "late_MAE_hours": err[late].mean()})
    pop = pd.DataFrame(rows)
    pop.to_csv(OUT/"population_controls_per_bearing.csv", index=False)

    # Exploratory: error over the last 40% of each observed life, divided by that life.
    phase = []
    for (model, b), g in primary_pred.groupby(["model", "bearing_id"]):
        L = float(g.observed_duration_hours.iloc[0])
        err = np.abs(g.pred_hours-g.rul_hours).to_numpy()
        tail = g.elapsed_hours.to_numpy() >= .6*L
        phase.append({"model": model, "bearing_id": b, "early_error_over_life": err[~tail].mean()/L,
                      "final40_error_over_life": err[tail].mean()/L})
    phase = pd.DataFrame(phase)
    phase.to_csv(OUT/"phase_split_per_bearing.csv", index=False)

    # Calendar gaps and lower-bound timing.
    gaps = []
    for b, g in base.groupby("bearing_id"):
        step = np.diff(g.sort_values("elapsed_hours").elapsed_hours.to_numpy())*60
        gaps.append({"bearing_id": b, "max_gap_minutes": step.max(), "gaps_over_5_min_total_minutes": step[step > 5].sum()})
    pd.DataFrame(gaps).to_csv(OUT/"calendar_gaps.csv", index=False)
    first = []
    lb = primary_dec[(primary_dec.required_lead_hours == 1) & (primary_dec.failure_cost_ratio == 10) & (primary_dec.policy == "lower_bound")]
    for model, g in lb.groupby("model"):
        at_first = [np.isclose(r.actual_lead_hours, life[r.bearing_id]) for r in g.itertuples()]
        first.append({"model": model, "bearings_triggered_at_first_record": int(sum(at_first)),
                      "unused_life_fraction": g.unused_life_fraction.mean(), "too_late": g.too_late.mean()})
    pd.DataFrame(first).to_csv(OUT/"lower_bound_first_record.csv", index=False)

    summary = {"rank_reproduction_max_delta": reproduction,
               "top_first_rank": {s: g.set_index("criterion")[["model", "first_rank_fraction"]].to_dict("index")
                                  for s, g in top.groupby("scenario")},
               "paired": paired,
               "population_controls": pop.groupby("control")[["nMAE", "late_MAE_hours"]].mean().to_dict("index"),
               "phase_split_mean": phase.groupby("model")[["early_error_over_life", "final40_error_over_life"]].mean().to_dict("index"),
               "horizon_support": horizon, "gaps": gaps, "lower_bound_first_record": first,
               "note": "Secondary analyses; primary results unchanged. Fixed-fit bootstrap unless a scenario states a refit."}
    (OUT/"sensitivity_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("rank_reproduction_max_delta", "paired", "population_controls")}, indent=1, default=float))


if __name__ == "__main__":
    main()
