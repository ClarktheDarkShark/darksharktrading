"""Training and evaluation pipeline for the day trading model."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...config import AppConfig
from ...utils import load_json, save_json
from .config import DayTradingConfig
from .data import describe_data, load_or_download
from .features import engineer_features
from .model import DayTradingModel


class DayTradingPipeline:
    def __init__(self, app_cfg: AppConfig, model_cfg: DayTradingConfig | None = None):
        self.app_cfg = app_cfg
        self.model_cfg = model_cfg or DayTradingConfig(symbol=app_cfg.default_symbol)
        self.storage_dir = app_cfg.day_trading_storage
        self.model = DayTradingModel(self.model_cfg, storage_dir=self.storage_dir)

    def load_data(self, force_download: bool = False) -> pd.DataFrame:
        return load_or_download(self.app_cfg, self.model_cfg, force=force_download)

    def prepare_datasets(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        features_df, feature_cols = engineer_features(df, self.model_cfg)
        X = features_df[feature_cols].values
        y = features_df["target"].values.astype(int)
        split_idx = int(len(features_df) * (1 - self.model_cfg.validation_size))
        split_idx = max(1, min(len(features_df) - 1, split_idx))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        return X_train, X_val, y_train, y_val, feature_cols

    def train(self, force_download: bool = False) -> dict[str, Any]:
        df = self.load_data(force_download=force_download)
        X_train, X_val, y_train, y_val, feature_cols = self.prepare_datasets(df)
        history = self.model.fit(X_train, y_train, X_val, y_val)
        evaluation = self.model.evaluate(X_val, y_val)
        pred_val = self.model.predict(X_val)
        final_report = self.model.inference_metrics(y_val, pred_val)
        metadata = {
            "config": self.model_cfg.__dict__,
            "data": describe_data(df),
            "features": feature_cols,
        }
        self.model.save()
        save_json(self.storage_dir / self.model_cfg.metrics_filename, {
            "evaluation": evaluation,
            "metadata": metadata,
            "history": history,
        })
        save_json(self.storage_dir / self.model_cfg.history_filename, history)
        return {
            "evaluation": evaluation,
            "history": history,
            "metadata": metadata,
            "report": final_report,
        }

    def load_metrics(self) -> dict[str, Any] | None:
        return load_json(self.storage_dir / self.model_cfg.metrics_filename)

    def load_history(self) -> list[dict[str, Any]] | None:
        data = load_json(self.storage_dir / self.model_cfg.history_filename)
        if not data:
            return []
        return data

    def latest_model_path(self) -> Path:
        return self.storage_dir / self.model_cfg.model_filename
