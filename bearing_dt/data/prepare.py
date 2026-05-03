from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from bearing_dt.data.features import extract_feature_row, normalize_signal
from bearing_dt.data.loaders import RawWindow, load_raw_windows
from bearing_dt.table import Rows, write_rows_csv
from bearing_dt.utils import ensure_dir, write_json


METADATA_COLUMNS = {
    "sample_id",
    "dataset",
    "bearing_id",
    "bearing_code",
    "condition_id",
    "condition_code",
    "operating_regime",
    "regime_code",
    "step",
    "max_step",
    "life_fraction",
    "rul",
    "rul_norm",
    "load",
    "speed",
}


def _regime_labels(loads: list[float], speeds: list[float], bins: int = 3) -> list[str]:
    if bins < 2:
        raise ValueError("regime_bins must be at least 2")
    labels_by_bins = {
        2: ["low", "high"],
        3: ["low", "mid", "high"],
    }
    labels = labels_by_bins.get(bins, [f"q{i + 1}" for i in range(bins)])
    quantile_grid = np.arange(1, bins, dtype=float) / bins
    load_q = np.quantile(np.array(loads, dtype=float), quantile_grid)
    speed_q = np.quantile(np.array(speeds, dtype=float), quantile_grid)

    def bin_name(value: float, q: np.ndarray) -> str:
        return labels[int(np.searchsorted(q, value, side="right"))]

    return [f"L{bin_name(l, load_q)}_S{bin_name(s, speed_q)}" for l, s in zip(loads, speeds)]


def _add_labels(dataset: str, windows: list[RawWindow], regime_bins: int = 3) -> tuple[np.ndarray, Rows]:
    max_steps = {}
    for w in windows:
        max_steps[w.bearing_id] = max(max_steps.get(w.bearing_id, 0), w.step)
    bearing_codes = {bid: idx for idx, bid in enumerate(sorted(max_steps))}
    condition_ids = sorted({w.condition_id for w in windows})
    condition_codes = {cid: idx for idx, cid in enumerate(condition_ids)}
    regimes = _regime_labels([w.load for w in windows], [w.speed for w in windows], bins=regime_bins)
    regime_codes = {rid: idx for idx, rid in enumerate(sorted(set(regimes)))}
    rows: Rows = []
    for idx, (w, regime) in enumerate(zip(windows, regimes)):
        max_step = max_steps[w.bearing_id]
        denom = max(1, max_step)
        rul = float(max_step - w.step)
        rows.append(
            {
                "sample_id": idx,
                "dataset": dataset,
                "bearing_id": w.bearing_id,
                "bearing_code": bearing_codes[w.bearing_id],
                "condition_id": w.condition_id,
                "condition_code": condition_codes[w.condition_id],
                "operating_regime": regime,
                "regime_code": regime_codes[regime],
                "step": int(w.step),
                "max_step": int(max_step),
                "life_fraction": float(w.step / denom),
                "rul": rul,
                "rul_norm": float(rul / denom),
                "load": float(w.load),
                "speed": float(w.speed),
            }
        )
    return np.array([w.signal for w in windows], dtype=object), rows


def prepare_dataset(
    dataset: str,
    out: str | Path,
    raw: str | Path | None = None,
    window_size: int = 512,
    sample_rate: float = 25_600.0,
    seed: int = 7,
    synthetic_bearings: int = 8,
    synthetic_steps: int = 48,
    max_windows: int | None = None,
    regime_bins: int = 3,
) -> dict[str, Any]:
    out_path = ensure_dir(out)
    raw_windows = load_raw_windows(
        dataset,
        raw,
        synthetic_bearings=synthetic_bearings,
        synthetic_steps=synthetic_steps,
        window_size=window_size,
        seed=seed,
        max_windows=max_windows,
    )
    raw_signals, metadata = _add_labels(dataset, raw_windows, regime_bins=regime_bins)
    normalized = []
    feature_rows = []
    for signal in raw_signals:
        norm = normalize_signal(signal, window_size)
        normalized.append(norm)
        feature_rows.append(extract_feature_row(norm, sample_rate=sample_rate))
    signals = np.stack(normalized).astype(np.float32)
    features: Rows = []
    for meta, feats in zip(metadata, feature_rows):
        features.append({**meta, **feats})
    feature_columns = [c for c in features[0] if c not in METADATA_COLUMNS]
    np.save(out_path / "signals.npy", signals)
    write_rows_csv(out_path / "features.csv", features)
    manifest = {
        "dataset": dataset,
        "raw": str(raw) if raw is not None else None,
        "samples": int(len(features)),
        "bearings": int(len({row["bearing_id"] for row in features})),
        "conditions": sorted({str(row["condition_id"]) for row in features}),
        "window_size": int(window_size),
        "channels": int(signals.shape[-1]),
        "sample_rate": float(sample_rate),
        "regime_bins": int(regime_bins),
        "feature_columns": feature_columns,
        "metadata_columns": sorted(METADATA_COLUMNS),
    }
    write_json(out_path / "manifest.json", manifest)
    return manifest
