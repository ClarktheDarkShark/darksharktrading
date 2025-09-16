"""Flask blueprint for the day trading model."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from flask import Blueprint, Response, current_app, jsonify, render_template, request

from ...config import AppConfig
from .config import DayTradingConfig
from .pipeline import DayTradingPipeline
from .realtime import DayTradingStreamer
from .viz import history_to_plot, stream_points_to_plot


TEMPLATE_FOLDER = Path(__file__).resolve().parents[2] / "webapp" / "templates"

day_trading_bp = Blueprint(
    "day_trading",
    __name__,
    template_folder=str(TEMPLATE_FOLDER),
)


def _app_config() -> AppConfig:
    return current_app.config["APP_CONFIG"]


@day_trading_bp.route("/")
def dashboard() -> str:
    pipeline = DayTradingPipeline(_app_config())
    metrics = pipeline.load_metrics()
    history = pipeline.load_history()
    return render_template(
        "day_trading.html",
        metrics=metrics or {},
        history_plot=history_to_plot(history or []),
        model_config=pipeline.model_cfg,
    )


@day_trading_bp.post("/train")
def train_endpoint() -> Response:
    payload = request.get_json(silent=True) or {}
    cfg_kwargs = {
        field: payload[field]
        for field in DayTradingConfig.__dataclass_fields__.keys()
        if field in payload
    }
    cfg = DayTradingConfig(**cfg_kwargs) if cfg_kwargs else DayTradingConfig()
    app_cfg = _app_config()
    pipeline = DayTradingPipeline(app_cfg, cfg)
    try:
        result = pipeline.train(force_download=payload.get("force_download", False))
        return jsonify({"status": "success", **result})
    except Exception as exc:  # pragma: no cover - best effort reporting
        return jsonify({"status": "error", "message": str(exc)}), 500


@day_trading_bp.get("/status")
def status_endpoint() -> Response:
    pipeline = DayTradingPipeline(_app_config())
    metrics = pipeline.load_metrics() or {}
    history = pipeline.load_history() or []
    return jsonify(
        {
            "status": "ok" if metrics else "not_trained",
            "metrics": metrics,
            "history_plot": history_to_plot(history),
            "model_config": asdict(pipeline.model_cfg),
        }
    )


@day_trading_bp.get("/stream")
def stream_endpoint() -> Response:
    pipeline = DayTradingPipeline(_app_config())
    try:
        streamer = DayTradingStreamer(pipeline)
    except FileNotFoundError:
        return jsonify({"status": "not_trained"}), 404
    df = streamer.latest_points()
    return jsonify(
        {
            "status": "ok",
            "stream": stream_points_to_plot(df),
        }
    )
