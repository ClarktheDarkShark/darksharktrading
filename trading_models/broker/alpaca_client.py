"""Minimal Alpaca broker client stub for live trading integration."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class AlpacaCredentials:
    api_key: str
    api_secret: str
    base_url: str = "https://paper-api.alpaca.markets"

    @classmethod
    def from_env(cls) -> "AlpacaCredentials":
        api_key = os.getenv("BROKER_API_KEY")
        api_secret = os.getenv("BROKER_API_SECRET")
        base_url = os.getenv("BROKER_BASE_URL", cls.base_url)
        if not api_key or not api_secret:
            raise RuntimeError("Broker credentials missing: set BROKER_API_KEY and BROKER_API_SECRET")
        return cls(api_key=api_key, api_secret=api_secret, base_url=base_url)


class AlpacaBrokerClient:
    """Lightweight wrapper around Alpaca's REST API for order placement."""

    def __init__(self, credentials: AlpacaCredentials | None = None):
        self.credentials = credentials or AlpacaCredentials.from_env()

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.credentials.api_key,
            "APCA-API-SECRET-KEY": self.credentials.api_secret,
            "Content-Type": "application/json",
        }

    def account(self) -> dict[str, Any]:
        response = requests.get(f"{self.credentials.base_url}/v2/account", headers=self._headers(), timeout=10)
        response.raise_for_status()
        return response.json()

    def submit_market_order(self, symbol: str, qty: int, side: str = "buy", time_in_force: str = "day") -> dict[str, Any]:
        payload = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": "market",
            "time_in_force": time_in_force,
        }
        response = requests.post(
            f"{self.credentials.base_url}/v2/orders",
            headers=self._headers(),
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    def close_position(self, symbol: str) -> dict[str, Any]:
        response = requests.delete(
            f"{self.credentials.base_url}/v2/positions/{symbol}",
            headers=self._headers(),
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
