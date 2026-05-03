from __future__ import annotations

import torch
import torch.nn.functional as F


def normalized_first_passage_rul(damage: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
    threshold_gap = F.relu(1.0 - damage)
    raw_time = threshold_gap / (rate + 1e-4)
    return raw_time / (1.0 + raw_time)


def physics_regularization(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    damage = outputs.get("damage")
    rate = outputs.get("damage_rate")
    target = batch["target"].float().view(-1)
    life_fraction = batch["life_fraction"].float().view(-1)
    if batch["context"].shape[1] >= 2:
        load = batch["context"][:, 0].float().view(-1)
        speed = batch["context"][:, 1].float().view(-1)
    else:
        load = torch.ones_like(target)
        speed = torch.ones_like(target)
    device = target.device
    if damage is None or rate is None:
        zero = torch.zeros((), device=device)
        return {
            "non_negative_rate": zero,
            "monotonic": zero,
            "threshold": zero,
            "damage_life_alignment": zero,
            "smoothness": zero,
            "load_speed_prior": zero,
        }
    damage = damage.view(-1)
    rate = rate.view(-1)
    non_negative_rate = F.relu(-rate).mean()
    first_passage_rul = normalized_first_passage_rul(damage, rate)
    threshold = F.mse_loss(first_passage_rul, target)
    damage_life_alignment = F.mse_loss(damage, life_fraction.clamp(0.0, 1.0))
    pair_penalties = []
    smooth_penalties = []
    for bearing in torch.unique(batch["bearing_code"]):
        mask = batch["bearing_code"] == bearing
        if int(mask.sum()) < 2:
            continue
        order = torch.argsort(batch["step"][mask])
        d = damage[mask][order]
        pair_penalties.append(F.relu(d[:-1] - d[1:]).mean())
        if len(d) >= 3:
            smooth_penalties.append(((d[2:] - 2 * d[1:-1] + d[:-2]) ** 2).mean())
    monotonic = torch.stack(pair_penalties).mean() if pair_penalties else torch.zeros((), device=device)
    smoothness = torch.stack(smooth_penalties).mean() if smooth_penalties else torch.zeros((), device=device)
    if len(rate) >= 2:
        load_order = torch.argsort(load)
        speed_order = torch.argsort(speed)
        load_rate = rate[load_order]
        speed_rate = rate[speed_order]
        load_speed_prior = F.relu(load_rate[:-1] - load_rate[1:]).mean() + F.relu(speed_rate[:-1] - speed_rate[1:]).mean()
    else:
        load_speed_prior = torch.zeros((), device=device)
    return {
        "non_negative_rate": non_negative_rate,
        "monotonic": monotonic,
        "threshold": threshold,
        "damage_life_alignment": damage_life_alignment,
        "smoothness": smoothness,
        "load_speed_prior": load_speed_prior,
    }


def monotonic_violation_rate(frame, predictions) -> float:
    import numpy as np

    values = np.asarray(predictions, dtype=float)
    total = 0
    violations = 0
    rows = []
    if hasattr(frame, "to_dict"):
        rows = frame.to_dict("records")
    else:
        rows = list(frame)
    grouped = {}
    for row, pred in zip(rows, values):
        grouped.setdefault(row["bearing_id"], []).append({**row, "_pred": float(pred)})
    for group in grouped.values():
        ordered = sorted(group, key=lambda r: int(r["step"]))
        if len(ordered) < 2:
            continue
        damage_proxy = 1.0 - np.array([row["_pred"] for row in ordered], dtype=float)
        diffs = np.diff(damage_proxy)
        total += len(diffs)
        violations += int(np.sum(diffs < -1e-8))
    return float(violations / max(1, total))


def damage_monotonic_violation_rate(frame, damage_predictions) -> float:
    import numpy as np

    values = np.asarray(damage_predictions, dtype=float)
    total = 0
    violations = 0
    rows = frame.to_dict("records") if hasattr(frame, "to_dict") else list(frame)
    grouped = {}
    for row, pred in zip(rows, values):
        grouped.setdefault(row["bearing_id"], []).append({**row, "_damage": float(pred)})
    for group in grouped.values():
        ordered = sorted(group, key=lambda r: int(r["step"]))
        if len(ordered) < 2:
            continue
        diffs = np.diff([row["_damage"] for row in ordered])
        total += len(diffs)
        violations += int(np.sum(diffs < -1e-8))
    return float(violations / max(1, total))
