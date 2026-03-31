# Gold Bot — XAU/USD AI Trading System

Personal algorithmic trading bot for Gold (XAU/USD) using
XGBoost + LightGBM on MetaTrader 5. Built for Windows + VSCode.

---

## Quick Start (Follow This Order)

### Step 1 — Clone and open in VSCode
```
Open the gold_bot_complete folder in VSCode
```

### Step 2 — Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3 — Set up your credentials
```bash
copy .env.example .env
```
Open `.env` and fill in:
- `MT5_LOGIN` — your MT5 account number
- `MT5_PASSWORD` — your MT5 password
- `MT5_SERVER` — found in MT5 → File → Open Account (e.g. `MetaQuotes-Demo`)
- `FRED_API_KEY` — free at https://fred.stlouisfed.org/docs/api/api_key.html

### Step 4 — Open MetaTrader 5
- Log in to your **demo account**
- Make sure `XAUUSD` is visible in Market Watch
- Leave MT5 running in the background

### Step 5 — Run the bot
```bash
python main.py
```
Select options from the menu, **run them in order 1 → 2 → 3 → 4**.

Or use VSCode's Run & Debug panel (F5) and select any configuration.

---

## Project Structure

```
gold_bot_complete/
│
├── main.py                    ← START HERE — interactive menu
├── requirements.txt
├── .env.example               ← copy to .env and fill in credentials
├── .env                       ← YOUR credentials (never commit this)
│
├── .vscode/
│   ├── settings.json          ← Python interpreter + env settings
│   └── launch.json            ← F5 run configs for each phase
│
├── config/
│   └── settings.py            ← All config loaded from .env
│
├── utils/
│   ├── logger.py              ← Shared logger (console + file)
│   ├── mt5_client.py          ← MT5 connection, data fetch, order execution
│   ├── macro_data.py          ← FRED + yfinance (DXY, VIX, rates, CPI)
│   └── cot_data.py            ← CFTC COT report (institutional positioning)
│
├── data/
│   ├── pipeline.py            ← Phase 1: fetch all data, save master_h1.parquet
│   ├── features.py            ← Phase 2: build all features, save features.parquet
│   ├── raw/                   ← Raw parquet files (auto-created, gitignored)
│   ├── processed/             ← Merged master dataset (auto-created, gitignored)
│   └── features/              ← Engineered features (auto-created, gitignored)
│
├── models/
│   └── train.py               ← Phase 3: walk-forward training, saves models
│
├── backtest/
│   └── engine.py              ← Phase 4: realistic P&L simulation + equity chart
│
├── execution/
│   └── live_bot.py            ← Phase 5: live signal loop on MT5 (DEMO only)
│
└── logs/                      ← Rotating daily logs + backtest charts
```

---

## The 5 Phases

| Phase | Command | What it does |
|-------|---------|--------------|
| 1 — Data Pipeline | Menu option 1 | Fetches 5000 bars M15/H1/H4/D1 from MT5, macro from FRED, COT from CFTC |
| 2 — Feature Engineering | Menu option 2 | Builds 80+ technical, macro, session, and S/R features |
| 3 — Model Training | Menu option 3 | Walk-forward trains XGBoost + LightGBM, saves best models |
| 4 — Backtest | Menu option 4 | Simulates trades on unseen test data, shows equity curve |
| 5 — Live Bot | Menu option 5 | Runs signal loop every hour, places real orders on MT5 |

**Always run phases in order. Never skip to phase 5 without a successful backtest.**

---

## Features Built (Phase 2)

**Trend:** EMA 8/13/21/50/200, EMA crossovers, MACD, ADX  
**Momentum:** RSI 7/14, Stochastic, Rate of Change  
**Volatility:** ATR 7/14/21, ATR ratio, Bollinger Bands  
**Volume:** OBV, volume ratio, volume spike detector  
**Price Action:** Body/wick size, candle direction, returns 1-10h  
**Session:** London, NY, overlap, Asian session flags, hour/day  
**Support/Resistance:** Rolling 20/50 highs/lows, distance from S/R  
**Macro:** Real rates, CPI, DXY, VIX, GLD ETF, S&P500  
**COT:** Net speculative position, COT index (52-week), week-over-week change  

---

## Risk Management Rules (Hardcoded)

- **1% account risk per trade** — position sized by ATR stop loss
- **ATR × 1.5 stop loss** — adapts to current market volatility
- **Minimum 1:1.5 reward:risk** — no low RR trades accepted
- **Max 2 open trades** — prevents overexposure
- **News blackout** — configurable in `execution/live_bot.py`
- **Min 60% model confidence** — filters low-conviction signals

---

## Important Rules

1. **Use demo account for at least 4 weeks** before any real money
2. **Never modify risk settings** to take bigger positions early
3. **A losing streak is normal** — judge over 50+ trades, not 10
4. **Push to GitHub immediately** — you already lost this once
5. **Do not go live** unless backtest shows: win rate >52%, profit factor >1.2, max drawdown <20%

---

## Getting Your FRED API Key (Free)

1. Go to https://fred.stlouisfed.org/docs/api/api_key.html
2. Create a free account
3. Request an API key (instant approval)
4. Paste it in your `.env` file

---

## Troubleshooting

**MT5 connection fails:**
- Make sure MetaTrader 5 is open and you are logged in
- Check your MT5_SERVER name exactly matches what's in MT5

**No XAUUSD data:**
- Right-click Market Watch in MT5 → Show All → find XAUUSD
- Make sure your broker supports XAUUSD on demo

**Import errors:**
- Make sure your virtual environment is activated: `.venv\Scripts\activate`
- Re-run: `pip install -r requirements.txt`

**FRED data returns empty:**
- Your FRED_API_KEY might be wrong — double check `.env`
- The bot still works without FRED (macro features will be NaN-filled)
