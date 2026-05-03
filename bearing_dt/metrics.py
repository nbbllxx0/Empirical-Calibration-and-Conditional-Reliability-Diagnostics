from __future__ import annotations

import numpy as np


def phm_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    early = np.clip(-err / 13.0, None, 700.0)
    late = np.clip(err / 10.0, None, 700.0)
    score = np.where(err < 0, np.exp(early) - 1.0, np.exp(late) - 1.0)
    return float(np.sum(score))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    err = pred - true
    metrics = {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "mape": float(np.mean(np.abs(err) / np.maximum(np.abs(true), 1.0))),
        "bias": float(np.mean(err)),
        "phm_score": phm_score(true, pred),
    }
    if len(true) >= 3:
        rng = np.random.default_rng(123)
        mae_samples = []
        abs_err = np.abs(err)
        for _ in range(300):
            idx = rng.integers(0, len(abs_err), size=len(abs_err))
            mae_samples.append(float(np.mean(abs_err[idx])))
        metrics["mae_ci_low"] = float(np.quantile(mae_samples, 0.025))
        metrics["mae_ci_high"] = float(np.quantile(mae_samples, 0.975))
    return metrics


def interval_metrics(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float = 0.1) -> dict[str, float]:
    true = np.asarray(y_true, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    covered = (true >= lo) & (true <= hi)
    width = hi - lo
    below = np.maximum(lo - true, 0.0)
    above = np.maximum(true - hi, 0.0)
    interval_score = width + (2.0 / max(alpha, 1e-12)) * below + (2.0 / max(alpha, 1e-12)) * above
    scale = np.maximum(np.mean(np.abs(true)), 1.0)
    return {
        "coverage": float(np.mean(covered)),
        "mean_interval_width": float(np.mean(width)),
        "normalized_interval_width": float(np.mean(width) / scale),
        "interval_score": float(np.mean(interval_score)),
        "normalized_interval_score": float(np.mean(interval_score) / scale),
    }
