"""Application entry point for trading models web app."""
from __future__ import annotations

from pathlib import Path
from flask import Flask, render_template

from .config import AppConfig
from .models.day_trading.routes import day_trading_bp


config = AppConfig()
config.ensure_directories()


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "webapp" / "templates"),
        static_folder=str(Path(__file__).parent / "webapp" / "static"),
    )

    @app.route("/")
    def index() -> str:
        return render_template("index.html", models=[
            {
                "name": "Day Trading",
                "slug": "day_trading",
                "description": "Intraday momentum strategy with streaming inference"
            }
        ])

    app.config["APP_CONFIG"] = config

    # Register model specific blueprints
    app.register_blueprint(day_trading_bp, url_prefix="/day_trading")

    return app


app = create_app()
