"""Utility helpers used across the trading models project."""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Any) -> None:
    """Persist json serialisable payload to ``path``."""
    ensure_parent(path)
    if is_dataclass(payload):
        payload = asdict(payload)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True, default=str)


def load_json(path: Path) -> Any:
    """Load JSON from ``path`` if it exists."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def rolling_window(series, window: int):
    """Simple rolling window generator."""
    for idx in range(window - 1, len(series)):
        yield series[idx - window + 1 : idx + 1]
