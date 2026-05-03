from __future__ import annotations

import re
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bearing_dt.data.mat5 import loadmat_first_numeric
from bearing_dt.data.synthetic import generate_synthetic_windows


NUMERIC_EXTENSIONS = {".csv", ".txt", ".tsv", ".dat", ".mat"}
METADATA_NAMES = {"metadata.csv", "conditions.csv", "operating_conditions.csv", "load_speed.csv"}


def _is_metadata_file(path: Path) -> bool:
    lower = path.name.lower()
    return (
        lower in METADATA_NAMES
        or "manifest" in lower
        or "operatingconditions" in lower
        or "mean" in lower and "temperature" in lower
        or lower.endswith("_log.pdf")
    )


@dataclass(frozen=True)
class RawWindow:
    bearing_id: str
    condition_id: str
    step: int
    load: float
    speed: float
    signal: np.ndarray


def _natural_key(path: Path) -> list[int | str]:
    parts: list[int | str] = []
    for chunk in re.split(r"(\d+)", path.as_posix().lower()):
        if chunk.isdigit():
            parts.append(int(chunk))
        elif chunk:
            parts.append(chunk)
    return parts


def _read_numeric_file(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".mat":
        arr = loadmat_first_numeric(path)
        if arr.shape[1] > 4:
            arr = arr[:, :4]
        return arr
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".dat"} else ","
    arr = np.genfromtxt(path, delimiter=delimiter, dtype=np.float32, invalid_raise=False)
    if arr.ndim == 0 or np.isnan(arr).all():
        arr = np.genfromtxt(path, dtype=np.float32, invalid_raise=False)
    if arr.ndim == 1:
        arr = arr[:, None]
    arr = arr[~np.isnan(arr).all(axis=1)]
    valid_cols = ~np.isnan(arr).all(axis=0)
    arr = arr[:, valid_cols]
    arr = np.nan_to_num(arr)
    if arr.size == 0:
        raise ValueError(f"No numeric signal data found in {path}")
    if arr.shape[1] > 4:
        arr = arr[:, :4]
    return arr


def _infer_ids(path: Path, raw_root: Path) -> tuple[str, str]:
    rel = path.relative_to(raw_root)
    parts = list(rel.parts)
    bearing_match = re.search(r"(B\d{2})", path.name, re.IGNORECASE)
    if bearing_match:
        bearing_id = bearing_match.group(1).upper()
    elif len(parts) >= 2:
        bearing_id = parts[-2]
    else:
        bearing_id = path.stem
    condition_id = "unknown"
    for part in parts[:-1]:
        lower = part.lower()
        if any(token in lower for token in ["condition", "load", "speed", "rpm", "hz"]):
            condition_id = part
            break
    if condition_id == "unknown" and len(parts) >= 3:
        condition_id = parts[-3]
    return bearing_id, condition_id


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except ValueError:
        return None


def _infer_load_speed_from_name(path: Path) -> tuple[float | None, float | None]:
    text = path.as_posix().lower()
    load = None
    speed = None
    load_match = re.search(r"(?:load|force|f)[_\-= ]?([0-9]+(?:\.[0-9]+)?)", text)
    speed_match = re.search(r"(?:speed|rpm|hz|n)[_\-= ]?([0-9]+(?:\.[0-9]+)?)", text)
    if load_match:
        load = float(load_match.group(1))
    if speed_match:
        speed = float(speed_match.group(1))
    return load, speed


def _preferred_numeric_by_name(row: dict[str, str], candidates: list[str]) -> object | None:
    normalized = {key.lower().replace(" ", "").replace("_", ""): value for key, value in row.items()}
    for token in candidates:
        for key, value in normalized.items():
            if token in key:
                return value
    return None


def _read_condition_metadata(raw_root: Path) -> dict[tuple[str, int], tuple[str | None, float | None, float | None]]:
    mapping: dict[tuple[str, int], tuple[str | None, float | None, float | None]] = {}
    for path in raw_root.rglob("*.csv"):
        lower = path.name.lower()
        if not (lower in METADATA_NAMES or "operatingconditions" in lower):
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        inferred_bearing = None
        match = re.search(r"(B\d{2})", path.name, re.IGNORECASE)
        if match:
            inferred_bearing = match.group(1).upper()
        for row_idx, row in enumerate(rows):
            bearing = row.get("bearing_id") or row.get("bearing") or row.get("experiment") or row.get("id") or inferred_bearing
            step_value = row.get("step") or row.get("window") or row.get("file_index") or row.get("time_index")
            if bearing is None:
                continue
            step = int(float(step_value)) if step_value is not None else row_idx
            condition = row.get("condition_id") or row.get("condition") or row.get("regime")
            load_value = _preferred_numeric_by_name(
                row,
                [
                    "meanabsstatload",
                    "setstatload",
                    "statload",
                    "equivalentload",
                    "load",
                    "force",
                ],
            )
            speed_value = _preferred_numeric_by_name(row, ["meanabsspeed", "setspeed", "speed", "rpm", "hz"])
            load = _float_or_none(row.get("load") or row.get("force") or load_value)
            speed = _float_or_none(row.get("speed") or row.get("rpm") or row.get("hz") or speed_value)
            mapping[(str(bearing), step)] = (str(condition) if condition else None, load, speed)
    return mapping


def load_generic_bearing_windows(raw: str | Path, max_windows: int | None = None) -> list[RawWindow]:
    raw_root = Path(raw)
    if not raw_root.exists():
        raise FileNotFoundError(raw_root)
    metadata = _read_condition_metadata(raw_root)
    files = sorted(
        [
            p
            for p in raw_root.rglob("*")
            if p.is_file() and p.suffix.lower() in NUMERIC_EXTENSIONS and not _is_metadata_file(p)
        ],
        key=_natural_key,
    )
    if max_windows is not None:
        files = files[:max_windows]
    grouped_counts: dict[str, int] = {}
    windows: list[RawWindow] = []
    for path in files:
        bearing_id, condition_id = _infer_ids(path, raw_root)
        step = grouped_counts.get(bearing_id, 0)
        grouped_counts[bearing_id] = step + 1
        signal = _read_numeric_file(path)
        meta_condition, meta_load, meta_speed = metadata.get((bearing_id, step), (None, None, None))
        name_load, name_speed = _infer_load_speed_from_name(path)
        load = meta_load if meta_load is not None else name_load if name_load is not None else 1.0
        speed = meta_speed if meta_speed is not None else name_speed if name_speed is not None else 1.0
        if meta_condition:
            condition_id = meta_condition
        windows.append(
            RawWindow(
                bearing_id=bearing_id,
                condition_id=condition_id,
                step=step,
                load=load,
                speed=speed,
                signal=signal,
            )
        )
    if not windows:
        raise ValueError(f"No numeric bearing windows found under {raw_root}")
    return windows


def load_time_varying_oc(raw: str | Path, max_windows: int | None = None) -> list[RawWindow]:
    raw_root = Path(raw)
    csv_files = sorted(raw_root.rglob("*.csv"), key=_natural_key)
    if len(csv_files) == 1:
        with csv_files[0].open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows or not {"bearing_id", "step"}.issubset(rows[0]):
            raise ValueError("Single-file time_varying_oc format requires bearing_id and step columns")
        signal_cols = [c for c in rows[0] if c.startswith("signal_")]
        feature_cols = [c for c in rows[0] if c.startswith("vib_")]
        value_cols = signal_cols or feature_cols
        if not value_cols:
            raise ValueError("Expected signal_* or vib_* numeric columns for time_varying_oc")
        windows = []
        rows.sort(key=lambda r: (str(r["bearing_id"]), int(float(r["step"]))))
        for row in rows:
            signal = np.array([float(row[c]) for c in value_cols], dtype=np.float32)
            windows.append(
                RawWindow(
                    bearing_id=str(row["bearing_id"]),
                    condition_id=str(row.get("condition_id") or "time_varying"),
                    step=int(row["step"]),
                    load=float(row.get("load") or 1.0),
                    speed=float(row.get("speed") or 1.0),
                    signal=signal[:, None],
                )
            )
        return windows[:max_windows] if max_windows is not None else windows
    return load_generic_bearing_windows(raw_root, max_windows=max_windows)


def load_raw_windows(
    dataset: str,
    raw: str | Path | None = None,
    *,
    synthetic_bearings: int = 8,
    synthetic_steps: int = 48,
    window_size: int = 512,
    seed: int = 7,
    max_windows: int | None = None,
) -> list[RawWindow]:
    dataset_key = dataset.lower()
    if dataset_key == "synthetic":
        return [
            RawWindow(w.bearing_id, w.condition_id, w.step, w.load, w.speed, w.signal)
            for w in generate_synthetic_windows(
                bearings=synthetic_bearings,
                steps=synthetic_steps,
                window_size=window_size,
                seed=seed,
            )
        ]
    if raw is None:
        raise ValueError(f"--raw is required for dataset {dataset}")
    if dataset_key in {"xjtu_sy", "nasa_ims", "femto", "generic_bearing"}:
        return load_generic_bearing_windows(raw, max_windows=max_windows)
    if dataset_key in {"time_varying_oc", "phme_tvoc", "phme", "paderborn_tvoc"}:
        return load_time_varying_oc(raw, max_windows=max_windows)
    raise ValueError(f"Unsupported dataset: {dataset}")
