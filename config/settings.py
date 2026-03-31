"""
config/settings.py — Central configuration.
All modules import from here. Edit .env, not this file.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

# ── MT5 ──────────────────────────────────────────────────────
MT5_LOGIN    = int(os.getenv("MT5_LOGIN", 0))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER   = os.getenv("MT5_SERVER", "")

# ── Market ───────────────────────────────────────────────────
SYMBOL       = os.getenv("SYMBOL", "XAUUSD")
MAGIC_NUMBER = int(os.getenv("MAGIC_NUMBER", 20240101))

# ── Risk Management (DO NOT TOUCH THESE LIGHTLY) ─────────────
RISK_PERCENT    = float(os.getenv("RISK_PERCENT", 1.0))
MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", 2))
ATR_MULTIPLIER  = 1.5    # stop loss = ATR * this value
MIN_RR_RATIO    = 1.5    # minimum reward:risk ratio to take a trade

# ── API Keys ─────────────────────────────────────────────────
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# ── Data Paths ───────────────────────────────────────────────
DATA_DIR      = BASE_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR  = DATA_DIR / "features"
MODELS_DIR    = BASE_DIR / "models"
LOGS_DIR      = BASE_DIR / "logs"

# Create dirs if missing
for d in [RAW_DIR, PROCESSED_DIR, FEATURES_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Data Config ──────────────────────────────────────────────
HISTORY_BARS   = 5000
START_DATE     = "2018-01-01"
TIMEFRAMES     = ["M15", "H1", "H4", "D1"]

# ── Model Config ─────────────────────────────────────────────
LOOKAHEAD_BARS = 4        # predict H1 price direction 4 bars ahead
MIN_CONFIDENCE = 0.60     # minimum model probability to act on signal
TRAIN_SPLIT    = 0.70     # 70% train, 30% walk-forward test

# ── News Blackout (minutes before/after high-impact news) ────
NEWS_BLACKOUT_MINUTES = 30
