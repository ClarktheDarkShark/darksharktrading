"""Configuration for the day trading model."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta


@dataclass
class DayTradingConfig:
    symbol: str = "AAPL"
    interval: str = "1m"
    lookback_days: int = 10
    feature_windows: tuple[int, ...] = (5, 15, 30, 60)
    rsi_window: int = 14
    validation_size: float = 0.2
    epochs: int = 12
    threshold: float = 0.0005
    random_state: int = 42
    model_filename: str = "day_trading_sgd.joblib"
    metrics_filename: str = "metrics.json"
    history_filename: str = "training_history.json"
    refresh_interval: timedelta = timedelta(minutes=1)
    max_stream_points: int = 300
