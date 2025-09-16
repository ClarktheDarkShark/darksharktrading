"""Visualisation helpers for the day trading model."""
from __future__ import annotations

from typing import Any

import pandas as pd


def history_to_plot(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return {"epochs": [], "metrics": {}}
    epochs = [entry["epoch"] for entry in history]
    metric_keys = [k for k in history[0].keys() if k != "epoch"]
    metrics = {key: [entry.get(key) for entry in history] for key in metric_keys}
    return {"epochs": epochs, "metrics": metrics}


def stream_points_to_plot(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "timestamps": df["timestamp"].astype(str).tolist(),
        "prices": df["price"].tolist(),
        "probabilities": df["probability"].tolist(),
        "signals": df["signal"].tolist(),
    }
