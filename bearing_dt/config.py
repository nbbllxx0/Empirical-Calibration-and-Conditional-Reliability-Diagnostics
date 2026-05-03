from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at {path}")
    return data


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def require_keys(config: dict[str, Any], keys: list[str], context: str) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise KeyError(f"{context} missing required keys: {missing}")


@dataclass(frozen=True)
class RuntimePaths:
    processed_dir: Path
    runs_dir: Path

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RuntimePaths":
        data = config.get("data", {})
        run = config.get("run", {})
        processed = Path(data.get("processed_dir", "data/processed/synthetic_tiny"))
        runs = Path(run.get("runs_dir", "runs"))
        return cls(processed_dir=processed, runs_dir=runs)
