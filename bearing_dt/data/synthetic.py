from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticWindow:
    bearing_id: str
    condition_id: str
    step: int
    load: float
    speed: float
    signal: np.ndarray


def generate_synthetic_windows(
    bearings: int = 8,
    steps: int = 48,
    window_size: int = 512,
    seed: int = 7,
    channels: int = 2,
    sample_rate: float = 25_600.0,
) -> list[SyntheticWindow]:
    rng = np.random.default_rng(seed)
    windows: list[SyntheticWindow] = []
    t = np.arange(window_size, dtype=np.float32) / sample_rate
    condition_grid = [
        ("low", 0.75, 0.75),
        ("nominal", 1.0, 1.0),
        ("high", 1.25, 1.2),
        ("variable", 1.0, 1.35),
    ]
    for bearing_idx in range(bearings):
        condition_id, base_load, base_speed = condition_grid[bearing_idx % len(condition_grid)]
        phase = rng.uniform(0, 2 * np.pi)
        damage_accel = 0.8 + 0.35 * base_load + 0.2 * base_speed + rng.normal(0, 0.03)
        fault_freq = 130 + 8 * bearing_idx
        shaft_freq = 40 * base_speed
        for step in range(steps):
            frac = step / max(1, steps - 1)
            if condition_id == "variable":
                load = base_load + 0.18 * np.sin(2 * np.pi * frac)
                speed = base_speed + 0.16 * np.cos(3 * np.pi * frac)
            else:
                load = base_load + rng.normal(0, 0.015)
                speed = base_speed + rng.normal(0, 0.015)
            damage = np.clip((frac**1.7) * damage_accel, 0, 1.8)
            impulse_train = (np.sin(2 * np.pi * fault_freq * t + phase) > 0.985).astype(np.float32)
            carrier = np.sin(2 * np.pi * shaft_freq * t + phase)
            noise = rng.normal(0, 0.16 + 0.05 * damage, size=(window_size, channels))
            sig = []
            for ch in range(channels):
                ch_phase = phase + 0.3 * ch
                base = 0.45 * np.sin(2 * np.pi * shaft_freq * t + ch_phase)
                fault = (0.05 + 0.95 * damage) * impulse_train * np.sin(2 * np.pi * 2_200 * t)
                modulation = 1.0 + 0.25 * load * carrier
                sig.append(base + modulation * fault + noise[:, ch])
            signal = np.stack(sig, axis=1).astype(np.float32)
            windows.append(
                SyntheticWindow(
                    bearing_id=f"B{bearing_idx:03d}",
                    condition_id=condition_id,
                    step=step,
                    load=float(load),
                    speed=float(speed),
                    signal=signal,
                )
            )
    return windows
