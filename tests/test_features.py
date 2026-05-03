import numpy as np

from bearing_dt.data.features import extract_feature_row, normalize_signal


def test_feature_extraction_is_deterministic():
    rng = np.random.default_rng(3)
    signal = rng.normal(size=(128, 2))
    norm = normalize_signal(signal, 128)
    a = extract_feature_row(norm)
    b = extract_feature_row(norm)
    assert a == b
    assert "ch0_rms" in a
    assert "fused_spectral_entropy" in a
