"""Command line interface for managing trading models."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .config import AppConfig
from .models.day_trading.config import DayTradingConfig
from .models.day_trading.pipeline import DayTradingPipeline
from .models.day_trading.realtime import DayTradingStreamer


def _day_trading_config_from_args(args: argparse.Namespace) -> DayTradingConfig:
    kwargs = {}
    for field in DayTradingConfig.__dataclass_fields__:
        value = getattr(args, field, None)
        if value is not None:
            kwargs[field] = value
    return DayTradingConfig(**kwargs) if kwargs else DayTradingConfig()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Utilities for training and monitoring trading models")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a trading model")
    train_parser.add_argument("model", choices=["day_trading"], help="Model identifier")
    train_parser.add_argument("--symbol", dest="symbol")
    train_parser.add_argument("--lookback-days", dest="lookback_days", type=int)
    train_parser.add_argument("--epochs", dest="epochs", type=int)
    train_parser.add_argument("--force-download", dest="force_download", action="store_true")

    status_parser = subparsers.add_parser("status", help="Show model metrics")
    status_parser.add_argument("model", choices=["day_trading"], help="Model identifier")

    stream_parser = subparsers.add_parser("stream", help="Show latest stream points")
    stream_parser.add_argument("model", choices=["day_trading"], help="Model identifier")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    app_cfg = AppConfig()
    app_cfg.ensure_directories()

    if args.command == "train":
        cfg = _day_trading_config_from_args(args)
        pipeline = DayTradingPipeline(app_cfg, cfg)
        result = pipeline.train(force_download=args.force_download)
        print(json.dumps(result, indent=2, default=str))
    elif args.command == "status":
        pipeline = DayTradingPipeline(app_cfg)
        metrics = pipeline.load_metrics()
        if not metrics:
            print("No metrics found. Train the model first.")
        else:
            print(json.dumps(metrics, indent=2, default=str))
    elif args.command == "stream":
        pipeline = DayTradingPipeline(app_cfg)
        try:
            streamer = DayTradingStreamer(pipeline)
        except FileNotFoundError:
            print("Model artefacts missing. Train the model first.")
            return
        for point in streamer.to_stream_points():
            print(json.dumps(asdict(point), default=str))


if __name__ == "__main__":
    main()
