from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from bearing_dt.data.prepare import METADATA_COLUMNS
from bearing_dt.table import Rows, read_rows_csv
from bearing_dt.utils import read_json


CONTEXT_COLUMNS_BY_MODE = {
    "none": [],
    "load_speed": ["load", "speed"],
    "load_speed_regime": ["load", "speed", "regime_code"],
    "load_speed_condition": ["load", "speed", "condition_code"],
}


def normalize_context_mode(context_mode: str | None) -> str:
    mode = str(context_mode or "load_speed").strip().lower()
    aliases = {
        "": "load_speed",
        "load-speed": "load_speed",
        "load_speed_only": "load_speed",
        "no_context": "none",
        "load_speed_regime_code": "load_speed_regime",
        "legacy": "load_speed_regime",
    }
    mode = aliases.get(mode, mode)
    if mode not in CONTEXT_COLUMNS_BY_MODE:
        raise ValueError(f"Unsupported context_mode {context_mode!r}; expected one of {sorted(CONTEXT_COLUMNS_BY_MODE)}")
    return mode


def context_columns(context_mode: str | None = None) -> list[str]:
    return list(CONTEXT_COLUMNS_BY_MODE[normalize_context_mode(context_mode)])


def load_processed(processed_dir: str | Path) -> tuple[np.ndarray, Rows, dict[str, Any]]:
    root = Path(processed_dir)
    signals_path = root / "signals.npy"
    features_path = root / "features.csv"
    manifest_path = root / "manifest.json"
    if not signals_path.exists() or not features_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Processed dataset is incomplete: {root}")
    signals = np.load(signals_path).astype(np.float32)
    frame = read_rows_csv(features_path)
    manifest = read_json(manifest_path)
    return signals, frame, manifest


def feature_columns(frame: Rows) -> list[str]:
    if not frame:
        return []
    return [c for c in frame[0] if c not in METADATA_COLUMNS]


class BearingDataset(Dataset):
    def __init__(
        self,
        signals: np.ndarray,
        frame: Rows,
        indices: np.ndarray | list[int],
        feature_cols: list[str] | None = None,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        context_mean: np.ndarray | None = None,
        context_std: np.ndarray | None = None,
        context_mode: str = "load_speed",
    ) -> None:
        self.signals = signals
        self.frame = list(frame)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.feature_cols = feature_cols or feature_columns(frame)
        self.context_mode = normalize_context_mode(context_mode)
        self.context_cols = context_columns(self.context_mode)
        feats = np.array([[float(row[c]) for c in self.feature_cols] for row in self.frame], dtype=np.float32)
        self.feature_mean = feature_mean if feature_mean is not None else np.nanmean(feats[self.indices], axis=0, keepdims=True)
        self.feature_std = feature_std if feature_std is not None else np.nanstd(feats[self.indices], axis=0, keepdims=True) + 1e-6
        contexts = np.array([self._raw_context(row, self.context_mode) for row in self.frame], dtype=np.float32)
        if len(self.context_cols) == 0:
            self.context_mean = context_mean if context_mean is not None else np.zeros((1, 0), dtype=np.float32)
            self.context_std = context_std if context_std is not None else np.ones((1, 0), dtype=np.float32)
        else:
            self.context_mean = context_mean if context_mean is not None else np.nanmean(contexts[self.indices], axis=0, keepdims=True)
            self.context_std = context_std if context_std is not None else np.nanstd(contexts[self.indices], axis=0, keepdims=True) + 1e-6

    @staticmethod
    def _raw_context(row: dict[str, Any], context_mode: str = "load_speed") -> list[float]:
        return [float(row[col]) for col in context_columns(context_mode)]

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[item])
        row = self.frame[idx]
        features = np.array([float(row[c]) for c in self.feature_cols], dtype=np.float32)
        features = np.nan_to_num((features - self.feature_mean.squeeze(0)) / self.feature_std.squeeze(0))
        context = np.array(self._raw_context(row, self.context_mode), dtype=np.float32)
        context = np.nan_to_num((context - self.context_mean.squeeze(0)) / self.context_std.squeeze(0))
        return {
            "signal": torch.from_numpy(self.signals[idx].T.copy()),
            "features": torch.from_numpy(features),
            "context": torch.from_numpy(context),
            "target": torch.tensor(float(row["rul_norm"]), dtype=torch.float32),
            "rul": torch.tensor(float(row["rul"]), dtype=torch.float32),
            "bearing_code": torch.tensor(int(row["bearing_code"]), dtype=torch.long),
            "condition_code": torch.tensor(int(row["condition_code"]), dtype=torch.long),
            "step": torch.tensor(int(row["step"]), dtype=torch.long),
            "life_fraction": torch.tensor(float(row["life_fraction"]), dtype=torch.float32),
            "sample_id": torch.tensor(int(row["sample_id"]), dtype=torch.long),
        }
