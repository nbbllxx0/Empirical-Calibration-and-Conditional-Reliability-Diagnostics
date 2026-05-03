from pathlib import Path

import numpy as np
import yaml

from bearing_dt.data.prepare import prepare_dataset
from bearing_dt.data.dataset import BearingDataset, context_columns, load_processed
from bearing_dt.data.splits import leave_one_bearing_out, leave_one_operating_regime_out, make_split
from bearing_dt.matrix import _matrix_config, _rotating_holdouts
from bearing_dt.table import group_by, read_rows_csv
from bearing_dt.train import train_experiment


def test_prepare_labels_and_leave_one_bearing_split(tmp_path: Path):
    out = tmp_path / "processed"
    prepare_dataset("synthetic", out, synthetic_bearings=5, synthetic_steps=12, window_size=128, seed=5)
    signals, frame, manifest = load_processed(out)
    assert signals.shape[0] == 60
    assert manifest["bearings"] == 5
    assert "operating_regime" in frame[0]
    for group in group_by(frame, "bearing_id").values():
        ordered = sorted(group, key=lambda r: int(r["step"]))
        assert np.all(np.diff([row["rul"] for row in ordered]) <= 0)
    split = leave_one_bearing_out(frame, seed=4)
    train_bearings = {frame[int(i)]["bearing_id"] for i in split.train}
    test_bearings = {frame[int(i)]["bearing_id"] for i in split.test}
    assert train_bearings.isdisjoint(test_bearings)
    regime_split = leave_one_operating_regime_out(frame, seed=4)
    train_regimes = {frame[int(i)]["operating_regime"] for i in regime_split.train}
    test_regimes = {frame[int(i)]["operating_regime"] for i in regime_split.test}
    assert train_regimes.isdisjoint(test_regimes)


def test_random_split_has_all_parts(tmp_path: Path):
    out = tmp_path / "processed"
    prepare_dataset("synthetic", out, synthetic_bearings=6, synthetic_steps=8, window_size=96, seed=2)
    _, frame, _ = load_processed(out)
    split = make_split(frame, {"strategy": "by_bearing_random"}, seed=7)
    assert len(split.train) > 0
    assert len(split.val) > 0
    assert len(split.test) > 0


def test_leave_regime_split_can_share_validation_seed_across_model_seeds(tmp_path: Path):
    out = tmp_path / "processed"
    prepare_dataset("synthetic", out, synthetic_bearings=6, synthetic_steps=14, window_size=96, seed=9)
    _, frame, _ = load_processed(out)
    regimes = sorted({str(row["operating_regime"]) for row in frame})
    test_regime = regimes[-1]

    split_a = make_split(
        frame,
        {"strategy": "leave_one_operating_regime_out", "test_regime": test_regime, "seed": 101},
        seed=1,
    )
    split_b = make_split(
        frame,
        {"strategy": "leave_one_operating_regime_out", "test_regime": test_regime, "seed": 101},
        seed=999,
    )

    assert set(split_a.val) == set(split_b.val)
    assert set(split_a.test) == set(split_b.test)
    assert "val_regime:" in split_a.description


def test_leave_regime_split_accepts_explicit_validation_regime(tmp_path: Path):
    out = tmp_path / "processed"
    prepare_dataset("synthetic", out, synthetic_bearings=6, synthetic_steps=14, window_size=96, seed=10)
    _, frame, _ = load_processed(out)
    regimes = sorted({str(row["operating_regime"]) for row in frame})
    test_regime = regimes[-1]
    val_regime = regimes[0]
    if val_regime == test_regime:
        val_regime = regimes[1]

    split = make_split(
        frame,
        {
            "strategy": "leave_one_operating_regime_out",
            "test_regime": test_regime,
            "val_regime": val_regime,
            "seed": 101,
        },
        seed=7,
    )

    val_regimes = {frame[int(i)]["operating_regime"] for i in split.val}
    test_regimes = {frame[int(i)]["operating_regime"] for i in split.test}
    assert val_regimes == {val_regime}
    assert test_regimes == {test_regime}


def test_leave_regime_split_accepts_separate_calibration_regime(tmp_path: Path):
    out = tmp_path / "processed"
    prepare_dataset("synthetic", out, synthetic_bearings=7, synthetic_steps=16, window_size=96, seed=11)
    _, frame, _ = load_processed(out)
    regimes = sorted({str(row["operating_regime"]) for row in frame})
    test_regime, val_regime, cal_regime = regimes[-1], regimes[0], regimes[1]
    if val_regime == test_regime:
        val_regime = regimes[2]
    if cal_regime in {test_regime, val_regime}:
        cal_regime = next(regime for regime in regimes if regime not in {test_regime, val_regime})

    split = make_split(
        frame,
        {
            "strategy": "leave_one_operating_regime_out",
            "test_regime": test_regime,
            "val_regime": val_regime,
            "cal_regime": cal_regime,
            "seed": 101,
        },
        seed=7,
    )

    assert split.cal is not None
    assert {frame[int(i)]["operating_regime"] for i in split.val} == {val_regime}
    assert {frame[int(i)]["operating_regime"] for i in split.cal} == {cal_regime}
    assert {frame[int(i)]["operating_regime"] for i in split.test} == {test_regime}
    train_regimes = {frame[int(i)]["operating_regime"] for i in split.train}
    assert train_regimes.isdisjoint({test_regime, val_regime, cal_regime})


def test_training_uses_separate_calibration_split_when_configured(tmp_path: Path):
    processed = tmp_path / "processed"
    runs = tmp_path / "runs"
    prepare_dataset("synthetic", processed, synthetic_bearings=7, synthetic_steps=16, window_size=96, seed=12)
    _, frame, _ = load_processed(processed)
    regimes = sorted({str(row["operating_regime"]) for row in frame})
    test_regime, val_regime, cal_regime = regimes[-1], regimes[0], regimes[1]
    if cal_regime in {test_regime, val_regime}:
        cal_regime = next(regime for regime in regimes if regime not in {test_regime, val_regime})
    config = {
        "experiment_name": "pytest_separate_cal",
        "seed": 12,
        "data": {"processed_dir": str(processed)},
        "context": {"mode": "load_speed"},
        "run": {"runs_dir": str(runs)},
        "split": {
            "strategy": "leave_one_operating_regime_out",
            "test_regime": test_regime,
            "val_regime": val_regime,
            "cal_regime": cal_regime,
            "seed": 101,
        },
        "model": {"type": "random_forest", "n_estimators": 3},
        "uncertainty": {"alpha": 0.1},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    run_dir = train_experiment(config_path)
    preds = read_rows_csv(run_dir / "predictions.csv")
    manifest = yaml.safe_load((run_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert {"train", "val", "cal", "test"}.issubset({row["split"] for row in preds})
    assert manifest["split"]["cal"] is not None
    assert manifest["split"]["description"].endswith(f"|cal_regime:{cal_regime}")


def test_matrix_config_can_emit_seeded_separate_calibration_config():
    values = ["Lhigh_Shigh", "Lhigh_Slow", "Llow_Shigh"]
    val_regime, cal_regime = _rotating_holdouts(values, "Lhigh_Shigh")
    config = _matrix_config(
        {"experiment_name": "base", "seed": 21, "split": {"strategy": "leave_one_operating_regime_out"}},
        "Lhigh_Shigh",
        "prefix",
        split_strategy="leave_one_operating_regime_out",
        split_key="test_regime",
        name_key="regime",
        validation_key="val_regime",
        validation_value=val_regime,
        calibration_key="cal_regime",
        calibration_value=cal_regime,
        model_seed=42,
    )

    assert config["experiment_name"] == "prefix_base__seed_42__regime_Lhigh_Shigh"
    assert config["seed"] == 42
    assert config["split"]["test_regime"] == "Lhigh_Shigh"
    assert config["split"]["val_regime"] == val_regime
    assert config["split"]["cal_regime"] == cal_regime


def test_context_modes_exclude_regime_from_canonical_input(tmp_path: Path):
    out = tmp_path / "processed"
    prepare_dataset("synthetic", out, synthetic_bearings=4, synthetic_steps=8, window_size=96, seed=3)
    signals, frame, _ = load_processed(out)
    feat_cols = ["ch0_rms", "fused_rms"]

    canonical = BearingDataset(signals, frame, [0, 1, 2], feat_cols, context_mode="load_speed")
    regime = BearingDataset(signals, frame, [0, 1, 2], feat_cols, context_mode="load_speed_regime")
    none = BearingDataset(signals, frame, [0, 1, 2], feat_cols, context_mode="none")

    assert context_columns("load_speed") == ["load", "speed"]
    assert canonical[0]["context"].shape[0] == 2
    assert regime[0]["context"].shape[0] == 3
    assert none[0]["context"].shape[0] == 0


def test_prepare_supports_coarse_regime_bins(tmp_path: Path):
    out = tmp_path / "processed"
    prepare_dataset("synthetic", out, synthetic_bearings=5, synthetic_steps=12, window_size=96, seed=8, regime_bins=2)
    _, frame, manifest = load_processed(out)
    assert manifest["regime_bins"] == 2
    regimes = {row["operating_regime"] for row in frame}
    assert regimes
    assert all("mid" not in regime for regime in regimes)
