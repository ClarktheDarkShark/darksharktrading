# DarkShark Trading Models

DarkShark Trading is a collection of production-ready machine learning models for algorithmic trading. The initial release ships with a fully featured **day trading** strategy that can be trained locally or deployed to Heroku for web-based monitoring and streaming inference. The project is structured so that additional models can be added easily.

## Features

- Flask web dashboard with separate routes per model (`/day_trading`) and Chart.js visualisations for training progress and live signals.
- Training and evaluation pipeline powered by scikit-learn's `SGDClassifier` to support incremental updates.
- Automated data ingestion using free [Yahoo Finance intraday data](https://github.com/ranaroussi/yfinance) (1 minute candles, up to the past 30 days).
- Command line interface for batch retraining and monitoring.
- Real-time streaming helper that reuses the trained model to score the latest bars.
- Broker integration stub for Alpaca (paper/live trading) with environment-driven credentials.
- Standalone Jupyter notebook under `notebooks/day_trading_model.ipynb` for experimentation without the web UI.

## Project layout

```
trading_models/
├── app.py                  # Flask app factory / entry point
├── cli.py                  # Command line interface
├── config.py               # Global configuration (paths, broker env vars)
├── broker/
│   └── alpaca_client.py    # Broker integration stub
├── models/
│   ├── __init__.py
│   ├── interfaces.py       # Base artefact dataclasses
│   └── day_trading/
│       ├── config.py       # Model hyper-parameters
│       ├── data.py         # Dataset download + caching helpers
│       ├── features.py     # Feature engineering utilities
│       ├── model.py        # SGDClassifier wrapper
│       ├── pipeline.py     # Training & evaluation orchestration
│       ├── realtime.py     # Streaming inference helpers
│       ├── routes.py       # Flask blueprint for /day_trading
│       └── viz.py          # Helpers for plotting data
└── webapp/
    ├── templates/          # HTML templates for Flask
    └── static/             # CSS/JS served by Flask
```

The `data/` and `artifacts/` directories are created automatically when training.

## Quick start (local)

1. **Install dependencies**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Train the day trading model** using the CLI:
   ```bash
   python -m trading_models.cli train day_trading --symbol AAPL --lookback-days 10
   ```
   The script downloads 1-minute candles from Yahoo Finance via `yfinance`, engineers features (SMA/EMA, momentum, RSI, volatility) and trains the `SGDClassifier` for the configured number of epochs. Metrics and the fitted scaler/model are stored in `artifacts/day_trading`.

3. **Launch the Flask app locally**:
   ```bash
   export FLASK_APP=trading_models.app
   flask run --reload
   ```
   Visit `http://127.0.0.1:5000/day_trading` to view metrics, retrain the model, and watch the live signal charts update every minute.

4. **Experiment in the notebook**:
   Open `notebooks/day_trading_model.ipynb` in JupyterLab or VS Code to run the full training workflow step-by-step without starting the web server.

## Deploy to Heroku

1. Ensure the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli) is installed and you are logged in.
2. Create an app and push the code:
   ```bash
   heroku create your-darkshark-app
   heroku config:set BROKER_API_KEY=... BROKER_API_SECRET=... # optional for broker integration
   git push heroku main
   ```
3. Heroku uses the provided `runtime.txt`, `requirements.txt`, and `Procfile` to build the slug. The web process runs `gunicorn trading_models.app:app`.
4. Visit `https://your-darkshark-app.herokuapp.com/day_trading` to trigger training and view monitoring dashboards. Training runs synchronously; expect a minute or two for a fresh download and 12 epochs of SGD training.

## Real-time streaming & broker integration

- `DayTradingStreamer` reloads the persisted model, fetches the latest minute bars from Yahoo Finance, and computes probabilities. The `/day_trading/stream` endpoint returns JSON suitable for dashboards or external automation.
- The `trading_models/broker/alpaca_client.py` stub demonstrates how to call the Alpaca REST API using environment variables (`BROKER_API_KEY`, `BROKER_API_SECRET`, `BROKER_BASE_URL`). Replace the stub with risk-managed order logic before enabling live trading.
- You can extend the streamer loop to call the broker when `probability` exceeds your thresholds, add position sizing, and manage orders (e.g. with stop-loss rules).

## Adding new models

1. Create a new package inside `trading_models/models/<model_name>` mirroring the day trading structure (`config.py`, `data.py`, `features.py`, etc.).
2. Register a new blueprint in `trading_models/app.py` and add CLI handlers in `trading_models/cli.py`.
3. Add templates/static assets under `trading_models/webapp` and expose the route in `index.html`.
4. Update `README.md` and optionally create a dedicated notebook.

## Useful commands

- Retrain with fresh data: `python -m trading_models.cli train day_trading --force-download`
- Inspect current metrics: `python -m trading_models.cli status day_trading`
- Stream latest predictions in the console: `python -m trading_models.cli stream day_trading`

## Environment variables

| Variable | Purpose |
| --- | --- |
| `TRADING_DATA_DIR` | Override the default data cache directory (`data`). |
| `TRADING_ARTIFACTS_DIR` | Override the directory for persisted models (`artifacts`). |
| `TRADING_DEFAULT_SYMBOL` | Symbol used when no override is provided (default `AAPL`). |
| `BROKER_API_KEY` / `BROKER_API_SECRET` | Credentials for the Alpaca broker client. |
| `BROKER_BASE_URL` | Base URL for the Alpaca API (paper trading by default). |

## Dataset notes

Yahoo Finance allows free download of intraday bars via `yfinance`. The free tier limits 1-minute bars to the most recent 30 calendar days, which is suitable for day-trading backtests and ongoing retraining. For extended historical coverage consider commercial data providers and update `data.py` accordingly.

## Tests and linting

Currently there is no automated test suite. Use the CLI commands and the notebook to validate behaviour before deploying.
