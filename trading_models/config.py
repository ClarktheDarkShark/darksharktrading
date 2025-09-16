"""Global configuration helpers for the trading models project."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class AppConfig:
    """Configuration shared across the app and training pipelines."""

    data_dir: Path = field(default_factory=lambda: Path(os.getenv("TRADING_DATA_DIR", "data")))
    artifacts_dir: Path = field(default_factory=lambda: Path(os.getenv("TRADING_ARTIFACTS_DIR", "artifacts")))
    default_symbol: str = os.getenv("TRADING_DEFAULT_SYMBOL", "AAPL")
    broker_api_key: str | None = os.getenv("BROKER_API_KEY")
    broker_api_secret: str | None = os.getenv("BROKER_API_SECRET" )
    broker_base_url: str | None = os.getenv("BROKER_BASE_URL", "https://paper-api.alpaca.markets")

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    @property
    def day_trading_storage(self) -> Path:
        path = self.artifacts_dir / "day_trading"
        path.mkdir(parents=True, exist_ok=True)
        return path
