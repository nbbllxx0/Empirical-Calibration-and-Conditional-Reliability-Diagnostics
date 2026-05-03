from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bearing_dt.table import Rows, indices_where, unique_values


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    description: str
    cal: np.ndarray | None = None


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _validate_no_leakage(frame: Rows, split: SplitIndices) -> None:
    train_bearings = {str(frame[int(i)]["bearing_id"]) for i in split.train}
    test_bearings = {str(frame[int(i)]["bearing_id"]) for i in split.test}
    if split.description.startswith("leave_one_bearing") and train_bearings & test_bearings:
        raise AssertionError("Bearing leakage detected in leave-one-bearing split")


def leave_one_bearing_out(
    frame: Rows,
    test_bearing: str | None = None,
    seed: int = 7,
    val_bearing: str | None = None,
    cal_bearing: str | None = None,
) -> SplitIndices:
    bearings = [str(v) for v in unique_values(frame, "bearing_id")]
    if not bearings:
        raise ValueError("No bearing_id values found")
    if test_bearing is None:
        test_bearing = bearings[-1]
    if test_bearing not in bearings:
        raise ValueError(f"Unknown test bearing {test_bearing}; available={bearings}")
    remaining = [b for b in bearings if b != test_bearing]
    if val_bearing is None:
        val_bearing = remaining[_rng(seed).integers(0, len(remaining))] if remaining else test_bearing
    if val_bearing not in bearings:
        raise ValueError(f"Unknown validation bearing {val_bearing}; available={bearings}")
    if val_bearing == test_bearing and len(bearings) > 1:
        raise ValueError("Validation bearing cannot match the test bearing")
    if cal_bearing is not None:
        if cal_bearing not in bearings:
            raise ValueError(f"Unknown calibration bearing {cal_bearing}; available={bearings}")
        if cal_bearing in {test_bearing, val_bearing}:
            raise ValueError("Calibration bearing must be distinct from test and validation bearings")
    test = np.array(indices_where(frame, lambda r: str(r["bearing_id"]) == test_bearing), dtype=np.int64)
    val = np.array(indices_where(frame, lambda r: str(r["bearing_id"]) == val_bearing), dtype=np.int64)
    held_out = {test_bearing, val_bearing}
    cal = None
    description = f"leave_one_bearing:{test_bearing}|val_bearing:{val_bearing}"
    if cal_bearing is not None:
        held_out.add(cal_bearing)
        cal = np.array(indices_where(frame, lambda r: str(r["bearing_id"]) == cal_bearing), dtype=np.int64)
        description += f"|cal_bearing:{cal_bearing}"
    train = np.array(indices_where(frame, lambda r: str(r["bearing_id"]) not in held_out), dtype=np.int64)
    split = SplitIndices(train=train, val=val, test=test, description=description, cal=cal)
    _validate_no_leakage(frame, split)
    return split


def leave_one_condition_out(
    frame: Rows,
    test_condition: str | None = None,
    seed: int = 7,
    val_condition: str | None = None,
    cal_condition: str | None = None,
) -> SplitIndices:
    conditions = [str(v) for v in unique_values(frame, "condition_id")]
    if not conditions:
        raise ValueError("No condition_id values found")
    if test_condition is None:
        test_condition = conditions[-1]
    if test_condition not in conditions:
        raise ValueError(f"Unknown test condition {test_condition}; available={conditions}")
    remaining = [c for c in conditions if c != test_condition]
    if val_condition is None:
        val_condition = remaining[_rng(seed).integers(0, len(remaining))] if remaining else test_condition
    if val_condition not in conditions:
        raise ValueError(f"Unknown validation condition {val_condition}; available={conditions}")
    if val_condition == test_condition and len(conditions) > 1:
        raise ValueError("Validation condition cannot match the test condition")
    if cal_condition is not None:
        if cal_condition not in conditions:
            raise ValueError(f"Unknown calibration condition {cal_condition}; available={conditions}")
        if cal_condition in {test_condition, val_condition}:
            raise ValueError("Calibration condition must be distinct from test and validation conditions")
    test = np.array(indices_where(frame, lambda r: str(r["condition_id"]) == test_condition), dtype=np.int64)
    val = np.array(indices_where(frame, lambda r: str(r["condition_id"]) == val_condition), dtype=np.int64)
    held_out = {test_condition, val_condition}
    cal = None
    description = f"leave_one_condition:{test_condition}|val_condition:{val_condition}"
    if cal_condition is not None:
        held_out.add(cal_condition)
        cal = np.array(indices_where(frame, lambda r: str(r["condition_id"]) == cal_condition), dtype=np.int64)
        description += f"|cal_condition:{cal_condition}"
    train = np.array(indices_where(frame, lambda r: str(r["condition_id"]) not in held_out), dtype=np.int64)
    return SplitIndices(train=train, val=val, test=test, description=description, cal=cal)


def leave_one_operating_regime_out(
    frame: Rows,
    test_regime: str | None = None,
    seed: int = 7,
    val_regime: str | None = None,
    cal_regime: str | None = None,
) -> SplitIndices:
    regimes = [str(v) for v in unique_values(frame, "operating_regime")]
    if len(regimes) < 3:
        return leave_one_condition_out(frame, test_regime, seed, val_regime, cal_regime)
    if test_regime is None:
        test_regime = regimes[-1]
    if test_regime not in regimes:
        raise ValueError(f"Unknown operating regime {test_regime}; available={regimes}")
    remaining = [r for r in regimes if r != test_regime]
    if val_regime is None:
        val_regime = remaining[_rng(seed).integers(0, len(remaining))]
    if val_regime not in regimes:
        raise ValueError(f"Unknown validation operating regime {val_regime}; available={regimes}")
    if val_regime == test_regime:
        raise ValueError("Validation operating regime cannot match the test regime")
    if cal_regime is not None:
        if cal_regime not in regimes:
            raise ValueError(f"Unknown calibration operating regime {cal_regime}; available={regimes}")
        if cal_regime in {test_regime, val_regime}:
            raise ValueError("Calibration operating regime must be distinct from test and validation regimes")
    test = np.array(indices_where(frame, lambda r: str(r["operating_regime"]) == test_regime), dtype=np.int64)
    val = np.array(indices_where(frame, lambda r: str(r["operating_regime"]) == val_regime), dtype=np.int64)
    held_out = {test_regime, val_regime}
    cal = None
    description = f"leave_one_operating_regime:{test_regime}|val_regime:{val_regime}"
    if cal_regime is not None:
        held_out.add(cal_regime)
        cal = np.array(indices_where(frame, lambda r: str(r["operating_regime"]) == cal_regime), dtype=np.int64)
        description += f"|cal_regime:{cal_regime}"
    train = np.array(indices_where(frame, lambda r: str(r["operating_regime"]) not in held_out), dtype=np.int64)
    return SplitIndices(train=train, val=val, test=test, description=description, cal=cal)


def by_bearing_random(frame: Rows, seed: int = 7, val_fraction: float = 0.2, test_fraction: float = 0.2) -> SplitIndices:
    bearings = np.array([str(v) for v in unique_values(frame, "bearing_id")])
    if len(bearings) < 3:
        idx = np.arange(len(frame))
        _rng(seed).shuffle(idx)
        n_test = max(1, int(len(idx) * test_fraction))
        n_val = max(1, int(len(idx) * val_fraction))
        return SplitIndices(idx[n_test + n_val :], idx[n_test : n_test + n_val], idx[:n_test], "sample_random")
    shuffled = bearings.copy()
    _rng(seed).shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_fraction))
    n_val = max(1, int(len(shuffled) * val_fraction))
    test_b = set(shuffled[:n_test])
    val_b = set(shuffled[n_test : n_test + n_val])
    train_b = set(shuffled[n_test + n_val :])
    return SplitIndices(
        train=np.array(indices_where(frame, lambda r: str(r["bearing_id"]) in train_b), dtype=np.int64),
        val=np.array(indices_where(frame, lambda r: str(r["bearing_id"]) in val_b), dtype=np.int64),
        test=np.array(indices_where(frame, lambda r: str(r["bearing_id"]) in test_b), dtype=np.int64),
        description="by_bearing_random",
    )


def make_split(frame: Rows, config: dict, seed: int = 7) -> SplitIndices:
    strategy = config.get("strategy", "by_bearing_random")
    split_seed = int(config.get("seed", seed))
    if strategy == "leave_one_bearing_out":
        return leave_one_bearing_out(
            frame,
            config.get("test_bearing"),
            split_seed,
            config.get("val_bearing") or config.get("validation_bearing"),
            config.get("cal_bearing") or config.get("calibration_bearing"),
        )
    if strategy == "leave_one_condition_out":
        return leave_one_condition_out(
            frame,
            config.get("test_condition"),
            split_seed,
            config.get("val_condition") or config.get("validation_condition"),
            config.get("cal_condition") or config.get("calibration_condition"),
        )
    if strategy in {"leave_one_operating_regime_out", "leave_one_regime_out"}:
        return leave_one_operating_regime_out(
            frame,
            config.get("test_regime"),
            split_seed,
            config.get("val_regime") or config.get("validation_regime"),
            config.get("cal_regime") or config.get("calibration_regime"),
        )
    if strategy == "by_bearing_random":
        return by_bearing_random(
            frame,
            seed=split_seed,
            val_fraction=float(config.get("val_fraction", 0.2)),
            test_fraction=float(config.get("test_fraction", 0.2)),
        )
    if strategy == "all":
        idx = np.arange(len(frame))
        return SplitIndices(train=idx, val=idx, test=idx, description="all")
    raise ValueError(f"Unsupported split strategy: {strategy}")
