"""Data ingestion utilities for the day trading model."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from ...config import AppConfig
from .config import DayTradingConfig


def _cache_path(config: AppConfig, model_cfg: DayTradingConfig) -> Path:
    return config.data_dir / "day_trading" / f"{model_cfg.symbol}_{model_cfg.interval}.parquet"


def download_data(app_config: AppConfig, model_cfg: DayTradingConfig) -> pd.DataFrame:
    """Download recent intraday data from Yahoo Finance via yfinance."""
    path = _cache_path(app_config, model_cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = yf.download(
        model_cfg.symbol,
        interval=model_cfg.interval,
        period=f"{model_cfg.lookback_days}d",
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        raise RuntimeError(
            "No data returned from yfinance. Try a different symbol or reduce lookback." 
        )
    data.reset_index(inplace=True)
    data.rename(columns={"Datetime": "timestamp"}, inplace=True)
    data.sort_values("timestamp", inplace=True)
    data.to_parquet(path, index=False)
    return data


def load_or_download(app_config: AppConfig, model_cfg: DayTradingConfig, force: bool = False) -> pd.DataFrame:
    path = _cache_path(app_config, model_cfg)
    if not force and path.exists():
        return pd.read_parquet(path)
    return download_data(app_config, model_cfg)


def describe_data(df: pd.DataFrame) -> dict[str, float]:
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "start": df["timestamp"].min().isoformat(),
        "end": df["timestamp"].max().isoformat(),
        "symbol": df.get("Symbol", "N/A"),
    }
