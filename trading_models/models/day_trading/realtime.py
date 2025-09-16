"""Real-time execution helpers for the day trading model."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from ...config import AppConfig
from .config import DayTradingConfig
from .data import load_or_download
from .features import engineer_features
from .pipeline import DayTradingPipeline


@dataclass
class StreamPoint:
    timestamp: datetime
    price: float
    probability: float
    signal: int


class DayTradingStreamer:
    def __init__(self, pipeline: DayTradingPipeline):
        self.pipeline = pipeline
        self.app_cfg = pipeline.app_cfg
        self.cfg = pipeline.model_cfg
        self.pipeline.model.load()

    def latest_points(self) -> pd.DataFrame:
        raw = load_or_download(self.app_cfg, self.cfg, force=True)
        features_df, feature_cols = engineer_features(raw, self.cfg)
        if features_df.empty:
            return pd.DataFrame(columns=["timestamp", "price", "probability", "signal"])
        X = features_df[feature_cols].values
        probs = self.pipeline.model.predict_proba(X)[:, 1]
        signals = (probs > 0.5).astype(int)
        return pd.DataFrame(
            {
                "timestamp": features_df["timestamp"].values,
                "price": features_df["Close"].values,
                "probability": probs,
                "signal": signals,
            }
        ).tail(self.cfg.max_stream_points)

    def to_stream_points(self) -> list[StreamPoint]:
        df = self.latest_points()
        return [
            StreamPoint(
                timestamp=pd.to_datetime(row.timestamp),
                price=float(row.price),
                probability=float(row.probability),
                signal=int(row.signal),
            )
            for row in df.itertuples()
        ]
