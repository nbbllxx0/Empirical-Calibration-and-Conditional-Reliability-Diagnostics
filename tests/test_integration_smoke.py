from pathlib import Path

import yaml

from bearing_dt.data.prepare import prepare_dataset
from bearing_dt.paper import build_report
from bearing_dt.table import read_rows_csv
from bearing_dt.train import train_experiment


def test_tiny_experiment_trains_and_reports(tmp_path: Path):
    processed = tmp_path / "processed"
    runs = tmp_path / "runs"
    prepare_dataset("synthetic", processed, synthetic_bearings=5, synthetic_steps=10, window_size=96, seed=9)
    config = {
        "experiment_name": "pytest_tiny_dt",
        "seed": 9,
        "data": {"processed_dir": str(processed)},
        "context": {"mode": "load_speed"},
        "run": {"runs_dir": str(runs)},
        "split": {"strategy": "by_bearing_random", "val_fraction": 0.2, "test_fraction": 0.2},
        "model": {"type": "digital_twin", "hidden": 24, "ensemble_size": 1},
        "training": {"epochs": 2, "batch_size": 8, "lr": 0.001, "patience": 2, "physics_weight": 0.05, "cuda": False},
        "uncertainty": {"alpha": 0.1},
        "eval_subsets": [{"name": "partial_40", "max_life_fraction": 0.4}],
        "stress_tests": [{"name": "noise_05", "signal_noise_std": 0.05}],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_dir = train_experiment(config_path)
    preds = read_rows_csv(run_dir / "predictions.csv")
    assert {"test", "test_partial_40", "test_stress_noise_05"}.issubset({row["split"] for row in preds})
    assert (run_dir / "metrics.json").exists()
    manifest = yaml.safe_load((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["context"]["context_mode"] == "load_speed"
    assert manifest["context"]["regime_used_as_input"] is False
    assert manifest["runtime"]["deterministic_cuda_kernels"] is False
    assert manifest["training_selection"]["reported_state"] == "best_validation_loss"
    assert manifest["training_selection"]["members"][0]["selection_metric"] == "validation_mse"
    assert manifest["training_selection"]["members"][0]["best_epoch"] >= 1
    out = build_report(runs, tmp_path / "paper_artifacts")
    assert (out / "tables" / "table2_main_benchmark.csv").exists()
