import torch

from bearing_dt.physics import physics_regularization


def test_physics_loss_penalizes_negative_rate_and_nonmonotonic_damage():
    batch = {
        "target": torch.tensor([0.9, 0.6, 0.3]),
        "life_fraction": torch.tensor([0.1, 0.4, 0.7]),
        "context": torch.tensor([[1.0, 1.0, 0.0], [1.1, 1.0, 0.0], [1.2, 1.0, 0.0]]),
        "bearing_code": torch.tensor([0, 0, 0]),
        "step": torch.tensor([0, 1, 2]),
    }
    outputs = {
        "damage": torch.tensor([0.7, 0.5, 0.9]),
        "damage_rate": torch.tensor([-0.2, 0.1, 0.1]),
    }
    losses = physics_regularization(outputs, batch)
    assert losses["non_negative_rate"] > 0
    assert losses["monotonic"] > 0
    assert losses["threshold"] > 0
