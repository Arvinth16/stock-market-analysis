"""
config.py — Central configuration for the India Stock Research Agent.

Reads settings from environment variables / .env file with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if it exists
load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = os.getenv("DB_PATH", str(PROJECT_ROOT / "data" / "stocks.db"))
MODEL_DIR = os.getenv("MODEL_DIR", str(PROJECT_ROOT / "models"))

# ─── API Keys (optional) ─────────────────────────────────────────────────────
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# ─── Data Parameters ─────────────────────────────────────────────────────────
HISTORICAL_YEARS = 2              # How many years of OHLCV history to fetch
NEWS_LOOKBACK_DAYS = 7            # How many days of news to fetch

# ─── Feature / Model Parameters ──────────────────────────────────────────────
PREDICTION_HORIZON = 20           # Predict N-trading-day forward return
POSITIVE_RETURN_THRESHOLD = 0.02  # Binary label threshold (2%)
TRAIN_TEST_RATIO = 0.85

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "eval_metric": "logloss",
}

# ─── Scoring Weights ─────────────────────────────────────────────────────────
WEIGHT_MODEL = 0.55       # ML predicted probability
WEIGHT_SENTIMENT = 0.30   # News sentiment score
WEIGHT_MOMENTUM = 0.15    # Recent price momentum

# ─── Output ───────────────────────────────────────────────────────────────────
TOP_N_RECOMMENDATIONS = 10

# Ensure directories exist
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
