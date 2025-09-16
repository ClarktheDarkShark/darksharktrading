"""Model definition for the day trading strategy."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from .config import DayTradingConfig


class DayTradingModel:
    def __init__(self, cfg: DayTradingConfig, storage_dir: Path):
        self.cfg = cfg
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.model = self._create_model()
        self.classes_ = np.array([0, 1])

    # Training loop
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, epochs: int | None = None) -> list[dict[str, Any]]:
        epochs = epochs or self.cfg.epochs
        history: list[dict[str, Any]] = []
        self.model = self._create_model()
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        # Initial call to establish classes
        self.model.partial_fit(X_train_scaled, y_train, classes=self.classes_)
        for epoch in range(1, epochs + 1):
            self.model.partial_fit(X_train_scaled, y_train)
            metrics = self._evaluate_scaled(X_val_scaled, y_val)
            metrics["epoch"] = epoch
            history.append(metrics)
        return history

    def _evaluate_scaled(self, X_scaled: np.ndarray, y: np.ndarray) -> dict[str, float]:
        preds = self.model.predict(X_scaled)
        proba = self.model.predict_proba(X_scaled)[:, 1]
        metrics = {
            "accuracy": float(accuracy_score(y, preds)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "f1": float(f1_score(y, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else None,
        }
        return metrics

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        X_scaled = self.scaler.transform(X)
        return self._evaluate_scaled(X_scaled, y)

    def _create_model(self) -> SGDClassifier:
        return SGDClassifier(
            loss="log_loss",
            penalty="l2",
            learning_rate="optimal",
            random_state=self.cfg.random_state,
            tol=None,
        )

    def inference_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
        return {
            "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(X))

    def save(self) -> None:
        path = self.storage_dir / self.cfg.model_filename
        joblib.dump({"model": self.model, "scaler": self.scaler, "config": self.cfg}, path)

    def load(self) -> None:
        path = self.storage_dir / self.cfg.model_filename
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        saved_cfg = bundle.get("config")
        if saved_cfg:
            self.cfg = saved_cfg
