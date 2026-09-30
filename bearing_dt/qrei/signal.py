from __future__ import annotations

import numpy as np
from scipy import signal, stats

CHANNELS = ("accHorizRear_A", "accHorizFrontal_C")
BANDS = ((0.5, 200), (200, 1000), (1000, 3000), (3000, 6000), (6000, 10000))


def defect_orders(angle_degrees: float = 0.0) -> dict[str, float]:
    """Ideal kinematic orders, with no claim of measured slip/contact angle."""
    ratio = 3.18 / 35.5
    c = np.cos(np.deg2rad(angle_degrees))
    return {"BPFO": 19 / 2 * (1-ratio*c), "BPFI": 19 / 2 * (1+ratio*c),
            "BSF": 35.5 / (2*3.18) * (1-(ratio*c)**2), "FTF": .5*(1-ratio*c)}


def harmonize_record(mat: dict, g_per_volt: float = 10.0) -> tuple[np.ndarray, float]:
    t = np.asarray(mat["measTime"], dtype=float).ravel()
    dt = np.diff(t)
    fs = 1.0 / np.median(dt)
    if len(t) not in (102400, 204800) or not np.allclose(dt, 1/fs, rtol=1e-4, atol=1e-10):
        raise ValueError("Unexpected duration or nonuniform measurement time base")
    if min(abs(fs-64000), abs(fs-128000)) > .1:
        raise ValueError(f"Unsupported true sampling rate {fs}")
    x = np.stack([np.asarray(mat[c], dtype=float).ravel() for c in CHANNELS]) * g_per_volt
    if x.shape[1] != len(t) or not np.isfinite(x).all():
        raise ValueError("Invalid named-channel samples")
    # Remove DC only; never standardize individual acquisition amplitude.
    x -= x.mean(axis=1, keepdims=True)
    if fs > 100000:
        x = signal.resample_poly(x, 1, 2, axis=1)
    if x.shape != (2, 102400):
        raise ValueError(f"Unexpected harmonized shape {x.shape}")
    return x, fs


def physical_features(x: np.ndarray, speed_rpm: float) -> tuple[dict[str, float], np.ndarray]:
    fs = 64000
    sos = signal.butter(4, [6000, 10000], btype="bandpass", fs=fs, output="sos")
    env = np.abs(signal.hilbert(signal.sosfiltfilt(sos, x, axis=1), axis=1))
    f, p = signal.periodogram(x, fs=fs, window="hann", axis=1, scaling="density")
    ef, ep = signal.periodogram(env-env.mean(axis=1, keepdims=True), fs=fs,
                                window="hann", axis=1, scaling="density")
    df = f[1]-f[0]
    features = {}
    for j,label in enumerate(("A", "C")):
        y = x[j]
        rms = float(np.sqrt(np.mean(y*y)))
        peak = float(np.max(np.abs(y)))
        vals = {"rms_g": rms, "peak_g": peak, "ptp_g": float(np.ptp(y)),
                "kurtosis": float(stats.kurtosis(y, fisher=False)),
                "crest": peak/max(rms, 1e-12), "skew": float(stats.skew(y)),
                "env_rms_g": float(np.sqrt(np.mean(env[j]**2))),
                "env_kurtosis": float(stats.kurtosis(env[j], fisher=False))}
        for lo,hi in BANDS:
            vals[f"power_{lo:g}_{hi:g}_g2"] = float(p[j, (f>=lo)&(f<hi)].sum()*df)
        denominator = max(float(ep[j, (ef>=1)&(ef<=2000)].sum()*df), 1e-12)
        for name,order in defect_orders().items():
            energy = 0.0
            if speed_rpm > 1:
                for harmonic in (1, 2, 3):
                    center = order*speed_rpm/60*harmonic
                    halfwidth = max(1.25, .03*center)
                    energy += float(ep[j, (ef>=center-halfwidth)&(ef<=center+halfwidth)].sum()*df)
            vals[f"{name}_power_g2"] = energy
            vals[f"{name}_fraction"] = energy/denominator
        features.update({label+"_"+k: v for k,v in vals.items()})
    waveform = signal.resample_poly(x, 1, 50, axis=1).astype(np.float32)
    assert waveform.shape == (2, 2048)
    return features, waveform


def causal_history(values: np.ndarray, elapsed_hours: np.ndarray, half_life_hours: float = 1/12) -> np.ndarray:
    """Time-aware exponential average; future measurements cannot change a prefix."""
    out = np.asarray(values, dtype=float).copy()
    for i in range(1, len(out)):
        decay = np.exp(-np.log(2)*max(0.0,elapsed_hours[i]-elapsed_hours[i-1])/half_life_hours)
        out[i] = decay*out[i-1] + (1-decay)*values[i]
    return out
