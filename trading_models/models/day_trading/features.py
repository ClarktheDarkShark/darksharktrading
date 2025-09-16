"""Feature engineering helpers for the day trading model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DayTradingConfig


def compute_rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def engineer_features(df: pd.DataFrame, cfg: DayTradingConfig) -> pd.DataFrame:
    data = df.copy()
    data["return"] = data["Close"].pct_change()
    data["log_return"] = np.log1p(data["return"])
    data["price_change"] = data["Close"].diff()

    for window in cfg.feature_windows:
        data[f"sma_{window}"] = data["Close"].rolling(window).mean()
        data[f"ema_{window}"] = data["Close"].ewm(span=window, adjust=False).mean()
        data[f"momentum_{window}"] = data["Close"].pct_change(window)
        data[f"volatility_{window}"] = data["return"].rolling(window).std()
        data[f"volume_sma_{window}"] = data["Volume"].rolling(window).mean()

    data["rsi"] = compute_rsi(data["Close"], cfg.rsi_window)
    data["target_return"] = data["Close"].pct_change().shift(-1)
    data["target"] = (data["target_return"] > cfg.threshold).astype(int)

    data.dropna(inplace=True)
    feature_cols = [
        col
        for col in data.columns
        if col
        not in {
            "timestamp",
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
            "target",
            "target_return",
        }
    ]
    data = data[["timestamp", "Close", "Volume", *feature_cols, "target", "target_return"]]
    return data, feature_cols
