import numpy as np

from bearing_dt.metrics import phm_score
from bearing_dt.uncertainty import ConformalInterval


def test_conformal_interval_covers_calibration_points():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.1, 1.8, 2.9, 4.2])
    interval = ConformalInterval(alpha=0.1).fit(y, pred)
    lo, hi = interval.predict(pred)
    assert np.mean((y >= lo) & (y <= hi)) >= 0.75
    assert np.all(hi >= lo)


def test_phm_score_handles_extreme_errors_without_overflow():
    score = phm_score(np.array([0.0, 1.0]), np.array([10_000.0, -10_000.0]))
    assert np.isfinite(score)
