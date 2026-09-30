"""Causal endpoint features and exploratory competing-stop comparators.

Test-bearing stopping labels are never predictor inputs. The parametric model is
a penalized landmark working model, not an identified physical failure law.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import gamma

from .signal import causal_history

THERMAL_BEARINGS = {"B03", "B08"}


def trailing_slope(values, hours, width=.5):
    """Ordinary slope over an available prefix, with no future padding."""
    values = np.asarray(values, dtype=float)
    hours = np.asarray(hours, dtype=float)
    result = np.zeros(len(values))
    for i in range(2, len(values)):
        lo = np.searchsorted(hours, hours[i] - width)
        t = hours[lo:i+1]
        if len(t) >= 3 and np.ptp(t) > 0:
            result[i] = np.polyfit(t, values[lo:i+1], 1)[0]
    return result


def endpoint_history(t1, t2, room, rms_a, rms_c, hours):
    tmax = np.maximum(t1, t2)
    rms = np.maximum(rms_a, rms_c)
    smoothed_t = causal_history(tmax, hours)
    smoothed_rms = causal_history(rms, hours)
    result = {
        "ep_temperature_max_C": tmax,
        "ep_temperature_difference_C": np.asarray(t1)-t2,
        "ep_room_temperature_C": room,
        "ep_temperature_rise_C": tmax-room,
        "ep_temperature_ema_C": smoothed_t,
        "ep_thermal_margin_C": 110-tmax,
        "ep_temperature_running_max_C": np.maximum.accumulate(tmax),
    }
    for width in (.25, .5, 1.0):
        tag = str(int(width*60))
        result["ep_thermal_rate_"+tag+"min_C_per_hour"] = trailing_slope(smoothed_t, hours, width)
        result["ep_vibration_rate_"+tag+"min_g_per_hour"] = trailing_slope(smoothed_rms, hours, width)
    for threshold in (6, 8, 10):
        result[f"ep_vibration_margin_{threshold}_g"] = threshold-rms
        result[f"ep_vibration_crossings_{threshold}"] = np.cumsum(rms >= threshold)
    return result


def prepare_endpoint(source, raw, out, protocol, evidence):
    import pandas as pd
    cfg = json.loads(protocol.read_text(encoding="utf-8"))
    manifest = json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(source/"features.csv")
    frames = []
    columns = None
    checks = []
    for b, group in frame.groupby("bearing_id", sort=False):
        g = group.copy()
        temp = pd.read_csv(raw/b/b/(b+"_meanTemperatures.csv"))
        stamp = pd.to_datetime(temp.iloc[:, 0], format="%d-%b-%Y %H:%M:%S")
        assert len(temp) == len(g)
        assert np.array_equal(stamp.values, pd.to_datetime(g.timestamp).values)
        inputs = [temp.iloc[:, j].to_numpy() for j in (1, 2, 3)]
        inputs += [g.A_rms_g.to_numpy(), g.C_rms_g.to_numpy(), g.elapsed_hours.to_numpy()]
        added = endpoint_history(*inputs)
        columns = list(added)
        for c, values in added.items():
            g[c] = values
        # Two independently truncated histories reproduce all prefix features.
        for stop in (max(3, len(g)//3), max(3, 2*len(g)//3)):
            prefix = endpoint_history(*(a[:stop] for a in inputs))
            assert all(np.allclose(prefix[c], added[c][:stop], atol=1e-12) for c in columns)
        checks.append({"bearing_id": b, "timestamp_alignment": "PASS", "prefix_invariance": "PASS", "records": len(g)})
        frames.append(g)
    frame = pd.concat(frames, ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out/"features.csv", index=False)
    manifest.update({"feature_columns": manifest["feature_columns"]+columns,
                     "temperature_primary": True, "endpoint_feature_columns": columns,
                     "signals_file": str((source/"signals.npy").resolve()),
                     "source_features_sha256": hashlib.sha256((source/"features.csv").read_bytes()).hexdigest(),
                     "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest()})
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    evidence.mkdir(parents=True, exist_ok=True)
    verify = {"checks": checks, "finite_predictors": bool(np.isfinite(frame[manifest["feature_columns"]]).all().all()),
              "added_features": columns, "test_stopping_cause_in_predictors": False,
              "features_sha256": hashlib.sha256((out/"features.csv").read_bytes()).hexdigest(),
              "protocol_sha256": manifest["protocol_sha256"], "original_processed_data_changed": False}
    assert verify["finite_predictors"]
    (evidence/"endpoint_preparation_verification.json").write_text(json.dumps(verify, indent=2), encoding="utf-8")
    print(json.dumps(verify, indent=2), flush=True)


def threshold_predictions(frame, train, idx):
    """Minimum of two causal local extrapolations to common documented limits."""
    source = frame.iloc[train]
    cap = 2*float(source.observed_duration_hours.max())
    age_clock = np.maximum(float(source.groupby("bearing_id").observed_duration_hours.first().median())-frame.iloc[idx].elapsed_hours.to_numpy(), 0)
    positive_t = source.ep_thermal_rate_30min_C_per_hour
    positive_v = source.ep_vibration_rate_30min_g_per_hour
    thermal_prior = float(positive_t[positive_t > 0].median())
    vibration_prior = float(positive_v[positive_v > 0].median())
    g = frame.iloc[idx]
    t_rate = np.maximum(.1, .75*np.maximum(g.ep_thermal_rate_30min_C_per_hour.to_numpy(), 0)+.25*thermal_prior)
    v_rate = np.maximum(.01, .75*np.maximum(g.ep_vibration_rate_30min_g_per_hour.to_numpy(), 0)+.25*vibration_prior)
    thermal = np.maximum(g.ep_thermal_margin_C.to_numpy(), 0)/t_rate
    vibration = np.maximum(g.ep_vibration_margin_8_g.to_numpy(), 0)/v_rate
    prediction = np.minimum(thermal, vibration)
    # Before enough history exists, use the same training-life clock as control.
    prediction = np.where(g.elapsed_hours.to_numpy() < .1, age_clock, prediction)
    return np.clip(prediction, 0, cap), {"thermal_threshold_C": 110, "vibration_threshold_g": 8,
        "threshold_basis": "documented global thermal limit and midpoint of vibration range; no held-out per-test setting",
        "thermal_rate_prior": thermal_prior, "vibration_rate_prior": vibration_prior,
        "local_rate_weight": .75, "local_window_hours": .5, "minimum_history_hours": .1}


class CompetingStopModel:
    """Two cause-specific Weibull landmark hazards, with fixed shape two.

    Each source landmark contributes an observed event density for its logged
    stop cause and survival for the alternative cause. Coefficients are shrunk;
    repeated landmarks do not create more independent experimental units.
    No claim of independent physical latent lifetimes is made.
    """
    columns = ["elapsed_hours", "ep_temperature_rise_C", "ep_thermal_margin_C",
               "ep_thermal_rate_30min_C_per_hour", "hi_rms_g",
               "ep_vibration_rate_30min_g_per_hour", "static_load_N", "dynamic_load_N", "speed_rpm"]

    def fit(self, frame, train):
        g = frame.iloc[train]
        z = g[self.columns].to_numpy(dtype=float)
        self.mean = z.mean(axis=0)
        self.std = np.maximum(z.std(axis=0), 1e-6)
        z = np.column_stack([np.ones(len(z)), np.clip((z-self.mean)/self.std, -10, 10)])
        times = np.maximum(g.rul_hours.to_numpy(), 1/60)
        cause = g.bearing_id.isin(THERMAL_BEARINGS).to_numpy().astype(int)
        counts = g.bearing_id.value_counts()
        weight = np.asarray([1/counts[b] for b in g.bearing_id])
        weight /= weight.sum()
        lives = g.groupby("bearing_id").observed_duration_hours.first()
        self.cap = 2*float(lives.max())
        prior = np.log(2*float(lives.median()))
        logt = np.log(times)
        observed = np.eye(2)[cause]
        shape = 2.0
        def objective(flat):
            coef = flat.reshape(2, z.shape[1])
            eta = z@coef.T
            exponent = shape*(logt[:, None]-eta)
            hazard = np.exp(np.clip(exponent, -40, 40))
            event_log = np.log(shape)+(shape-1)*logt[:, None]-shape*eta
            loss = np.sum(weight*(hazard.sum(axis=1)-(observed*event_log).sum(axis=1)))
            # Fixed penalties; not selected on test-bearing performance.
            loss += .1*np.sum(coef[:, 1:]**2)+.02*np.sum((coef[:, 0]-prior)**2)
            active = (exponent > -40)&(exponent < 40)
            grad = ((shape*(observed-hazard*active))*weight[:, None]).T@z
            grad[:, 1:] += .2*coef[:, 1:]
            grad[:, 0] += .04*(coef[:, 0]-prior)
            return float(loss), grad.ravel()
        initial = np.zeros((2, z.shape[1])); initial[:, 0] = prior
        fit = minimize(objective, initial.ravel(), jac=True, method="L-BFGS-B",
                       bounds=[(-5, 6)]+[(-3, 3)]*(z.shape[1]-1)+[(-5, 6)]+[(-3, 3)]*(z.shape[1]-1),
                       options={"maxiter": 500, "ftol": 1e-10})
        if not fit.success:
            raise RuntimeError("Competing-stop fit did not converge: "+str(fit.message))
        self.coefficients = fit.x.reshape(2, z.shape[1])
        self.metadata = {"working_family": "cause-specific Weibull landmark; fixed shape 2",
            "columns": self.columns, "coefficient_penalty": .1, "intercept_prior_penalty": .02,
            "intercept_prior": prior, "mean": self.mean.tolist(), "std": self.std.tolist(),
            "coefficients": self.coefficients.tolist(), "fit_objective": float(fit.fun),
            "cause_event_bearings": {"vibration": int(sum(b not in THERMAL_BEARINGS for b in lives.index)),
                                     "thermal": int(sum(b in THERMAL_BEARINGS for b in lives.index))},
            "test_cause_used": False, "intervals": "common independent calibration residuals, not parametric quantiles"}
        return self

    def predict(self, frame):
        z = frame[self.columns].to_numpy(dtype=float)
        z = np.column_stack([np.ones(len(z)), np.clip((z-self.mean)/self.std, -10, 10)])
        eta = np.clip(z@self.coefficients.T, -20, 20)
        # S(r)=exp[-r^2 sum(exp(-2 eta_k))]; its mean is explicit.
        combined_scale = 1/np.sqrt(np.exp(-2*eta).sum(axis=1))
        return np.clip(gamma(1.5)*combined_scale, 0, self.cap)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=Path("data/processed/phme_tvoc_10b_v2"))
    p.add_argument("--raw", type=Path, default=Path("data/raw/phme_tvoc"))
    p.add_argument("--out", type=Path, default=Path("data/processed/phme_tvoc_10b_endpoint_v3"))
    p.add_argument("--protocol", type=Path, default=Path("QREI submission/protocol_endpoint_v3.json"))
    p.add_argument("--evidence", type=Path, default=Path("QREI submission/evidence"))
    a = p.parse_args()
    prepare_endpoint(a.source, a.raw, a.out, a.protocol, a.evidence)
