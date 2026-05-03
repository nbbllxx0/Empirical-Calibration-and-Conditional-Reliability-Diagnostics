import torch

from bearing_dt.models import build_model
from bearing_dt.physics import normalized_first_passage_rul


def _batch():
    return {
        "signal": torch.randn(4, 2, 128),
        "features": torch.randn(4, 12),
        "context": torch.randn(4, 3),
        "target": torch.rand(4),
        "life_fraction": torch.rand(4),
        "bearing_code": torch.arange(4),
        "step": torch.arange(4),
    }


def test_digital_twin_forward_outputs_physics_terms():
    model = build_model("digital_twin", channels=2, feature_dim=12, hidden=24)
    out = model(_batch())
    assert out["rul"].shape == (4,)
    assert out["damage"].shape == (4,)
    assert out["damage_rate"].shape == (4,)
    expected = normalized_first_passage_rul(out["damage"], out["damage_rate"])
    assert torch.allclose(out["rul"], expected)


def test_feature_only_forward():
    model = build_model("attnpinn", channels=2, feature_dim=12, hidden=24)
    out = model(_batch())
    assert out["rul"].shape == (4,)


def test_fused_direct_forward_has_no_damage_semantics():
    model = build_model("fused_direct", channels=2, feature_dim=12, hidden=24)
    out = model(_batch())
    assert out["rul"].shape == (4,)
    assert "damage" not in out
    assert "damage_rate" not in out
