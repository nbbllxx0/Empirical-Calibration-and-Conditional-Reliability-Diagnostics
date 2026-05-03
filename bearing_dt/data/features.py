from __future__ import annotations

import numpy as np


EPS = 1e-8


def _safe_kurtosis(x: np.ndarray) -> float:
    centered = x - np.mean(x)
    var = np.mean(centered**2) + EPS
    return float(np.mean(centered**4) / (var**2))


def _safe_skewness(x: np.ndarray) -> float:
    centered = x - np.mean(x)
    std = np.sqrt(np.mean(centered**2)) + EPS
    return float(np.mean(centered**3) / (std**3))


def _spectral_features(x: np.ndarray, sample_rate: float) -> dict[str, float]:
    spectrum = np.abs(np.fft.rfft(x))
    power = spectrum**2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sample_rate)
    total = float(np.sum(power) + EPS)
    probs = power / total
    centroid = float(np.sum(freqs * probs))
    entropy = float(-np.sum(probs * np.log(probs + EPS)) / np.log(len(probs) + EPS))
    bands: dict[str, float] = {}
    edges = np.linspace(0, len(power), 5, dtype=int)
    for idx in range(4):
        start, end = edges[idx], edges[idx + 1]
        bands[f"band_energy_{idx}"] = float(np.sum(power[start:end]) / total)
    return {"spectral_centroid": centroid, "spectral_entropy": entropy, **bands}


def _envelope_features(x: np.ndarray) -> dict[str, float]:
    envelope = np.abs(x)
    return {
        "envelope_mean": float(np.mean(envelope)),
        "envelope_std": float(np.std(envelope)),
        "envelope_peak": float(np.max(envelope)),
    }


def _channel_features(x: np.ndarray, sample_rate: float) -> dict[str, float]:
    abs_x = np.abs(x)
    rms = float(np.sqrt(np.mean(x**2) + EPS))
    mean_abs = float(np.mean(abs_x) + EPS)
    peak = float(np.max(abs_x))
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "rms": rms,
        "kurtosis": _safe_kurtosis(x),
        "skewness": _safe_skewness(x),
        "peak_to_peak": float(np.ptp(x)),
        "crest_factor": float(peak / (rms + EPS)),
        "shape_factor": float(rms / mean_abs),
        "impulse_factor": float(peak / mean_abs),
        **_spectral_features(x, sample_rate),
        **_envelope_features(x),
    }


def normalize_signal(signal: np.ndarray, window_size: int) -> np.ndarray:
    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] < window_size:
        pad = np.zeros((window_size - arr.shape[0], arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    elif arr.shape[0] > window_size:
        arr = arr[:window_size]
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True) + EPS
    return ((arr - mean) / std).astype(np.float32)


def extract_feature_row(signal: np.ndarray, sample_rate: float = 25_600.0) -> dict[str, float]:
    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    rows: dict[str, float] = {}
    for channel_idx in range(arr.shape[1]):
        feats = _channel_features(arr[:, channel_idx], sample_rate)
        rows.update({f"ch{channel_idx}_{name}": value for name, value in feats.items()})
    stacked = np.mean(arr, axis=1)
    rows.update({f"fused_{name}": value for name, value in _channel_features(stacked, sample_rate).items()})
    return rows
