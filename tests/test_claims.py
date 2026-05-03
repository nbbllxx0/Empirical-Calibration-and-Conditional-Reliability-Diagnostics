import json
from pathlib import Path

from bearing_dt.claims import audit_claims
from bearing_dt.table import read_rows_csv


def _write_metric(run_dir: Path, experiment_name: str, mae: float) -> None:
    run_dir.mkdir(parents=True)
    payload = {
        "run": {
            "experiment_name": experiment_name,
            "model_type": "digital_twin",
        },
        "test": {
            "mae": mae,
            "rmse": mae + 10.0,
            "coverage": 0.9,
            "mean_interval_width": 100.0,
            "normalized_interval_width": 1.0,
            "interval_score": 120.0,
            "normalized_interval_score": 1.2,
            "monotonic_violation_rate": 0.4,
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def test_claim_audit_labels_bearing_sanity_check(tmp_path: Path):
    runs = tmp_path / "runs"
    _write_metric(
        runs / "latent_b01",
        "phme_10b_bearing_phme_10b_latent_load_speed__bearing_B01",
        100.0,
    )
    _write_metric(
        runs / "gb_b01",
        "phme_10b_bearing_phme_10b_gb_load_speed__bearing_B01",
        110.0,
    )

    out = tmp_path / "claim_audit"
    audit_claims(runs, out, prefix="phme_10b_bearing")

    report = (out / "claim_audit.md").read_text(encoding="utf-8")
    assert "leave-bearing-out sanity-check" in report
    assert "| Model | Bearings |" in report
    assert "Primary PHME matrix claim status: NOT FULLY VERIFIED" not in report

    summary = read_rows_csv(out / "claim_summary.csv")
    assert summary[0]["bearings"] == 1
