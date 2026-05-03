from pathlib import Path

from bearing_dt.diagnostics import conditional_coverage, phme_subset_audit
from bearing_dt.table import write_rows_csv
from bearing_dt.utils import write_json


def test_phme_subset_audit_reports_excluded_manifest_bearings(tmp_path: Path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()
    write_rows_csv(
        raw / "file_manifest.csv",
        [
            {"filename": "B01.zip", "size": 1000, "checksum": "a", "download_url": "u1"},
            {"filename": "B02_part1.zip", "size": 2000, "checksum": "b", "download_url": "u2"},
        ],
    )
    write_rows_csv(
        processed / "features.csv",
        [
            {
                "sample_id": 0,
                "bearing_id": "B01",
                "operating_regime": "Llow_Slow",
                "load": 1.0,
                "speed": 2.0,
                "life_fraction": 0.2,
            }
        ],
    )

    report = phme_subset_audit(raw_dir=raw, processed_dir=processed, out=tmp_path / "out")

    assert report.exists()
    rows = (tmp_path / "out" / "bearing_manifest.csv").read_text(encoding="utf-8")
    assert "B01,yes" in rows
    assert "B02,no" in rows


def test_conditional_coverage_joins_predictions_to_processed_features(tmp_path: Path):
    processed = tmp_path / "processed"
    run = tmp_path / "runs" / "run1"
    processed.mkdir()
    run.mkdir(parents=True)
    write_rows_csv(
        processed / "features.csv",
        [
            {"sample_id": 1, "bearing_id": "B01", "operating_regime": "Llow_Slow", "regime_code": 0, "load": 1, "speed": 2},
            {"sample_id": 2, "bearing_id": "B02", "operating_regime": "Llow_Slow", "regime_code": 0, "load": 1, "speed": 2},
            {"sample_id": 3, "bearing_id": "B03", "operating_regime": "Lhigh_Shigh", "regime_code": 1, "load": 3, "speed": 4},
        ],
    )
    write_json(run / "metrics.json", {"run": {"experiment_name": "demo_model__regime_Llow_Slow"}})
    write_rows_csv(
        run / "predictions.csv",
        [
            {
                "split": "cal",
                "sample_id": 1,
                "bearing_id": "B01",
                "life_fraction": 0.2,
                "rul": 10,
                "rul_norm": 0.5,
                "y_pred_rul": 8,
                "y_pred_norm": 0.4,
            },
            {
                "split": "test",
                "sample_id": 2,
                "bearing_id": "B02",
                "life_fraction": 0.4,
                "rul": 10,
                "rul_norm": 0.5,
                "y_pred_rul": 9,
                "y_pred_norm": 0.45,
                "lower_rul": 7,
                "upper_rul": 11,
            },
            {
                "split": "test",
                "sample_id": 3,
                "bearing_id": "B03",
                "life_fraction": 0.8,
                "rul": 20,
                "rul_norm": 0.8,
                "y_pred_rul": 16,
                "y_pred_norm": 0.64,
                "lower_rul": 14,
                "upper_rul": 19,
            },
        ],
    )

    report = conditional_coverage(runs=tmp_path / "runs", processed_dir=processed, out=tmp_path / "coverage", prefix="demo")

    assert report.exists()
    per_regime = (tmp_path / "coverage" / "per_regime_coverage.csv").read_text(encoding="utf-8")
    assert "Llow_Slow" in per_regime
    assert "Lhigh_Shigh" in per_regime
