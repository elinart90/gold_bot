"""
execution/live_bot.py — Live Trading Bot (Phase 5)
Runs on a schedule, generates signals every H1 candle close,
and places orders on MT5 demo account.

RULES:
  - Only runs on DEMO account until you manually change settings
  - Max 2 open trades at once
  - ATR-based SL/TP always enforced
  - No trading during high-impact news blackout window

Run: python -m execution.live_bot
"""
import json
import time
import joblib
import schedule
import pandas as pd
from datetime import datetime, timezone, timedelta
from utils.logger import log
from utils.mt5_client import (
    connect, disconnect, get_ohlcv, get_latest_tick,
    get_account_info, get_open_positions, send_order
)
from utils.macro_data import fetch_yf_data
from data.features import (
    add_trend_features, add_momentum_features, add_volatility_features,
    add_volume_features, add_price_action_features,
    add_session_features, add_support_resistance
)
from config.settings import (
    SYMBOL, MODELS_DIR, MIN_CONFIDENCE,
    ATR_MULTIPLIER, MIN_RR_RATIO,
    RISK_PERCENT, MAX_OPEN_TRADES,
    NEWS_BLACKOUT_MINUTES
)

# ── High-impact news times (UTC) — update weekly from forexfactory.com ──
# Format: (month, day, hour, minute)
HIGH_IMPACT_EVENTS: list[tuple] = [
    # Example: (1, 10, 13, 30)  ← NFP first Friday of month at 13:30 UTC
    # Add real events here each week
]


def load_models():
    try:
        xgb   = joblib.load(MODELS_DIR / "xgboost_model.pkl")
        lgb   = joblib.load(MODELS_DIR / "lightgbm_model.pkl")
        with open(MODELS_DIR / "feature_cols.json") as f:
            feature_cols = json.load(f)
        log.info("Models loaded successfully")
        return xgb, lgb, feature_cols
    except Exception as e:
        log.error(f"Failed to load models: {e}")
        return None, None, None


def is_news_blackout(now: datetime) -> bool:
    """Return True if we're within NEWS_BLACKOUT_MINUTES of a high-impact event."""
    for month, day, hour, minute in HIGH_IMPACT_EVENTS:
        event_time = now.replace(month=month, day=day, hour=hour, minute=minute, second=0)
        diff = abs((now - event_time).total_seconds() / 60)
        if diff <= NEWS_BLACKOUT_MINUTES:
            log.warning(f"News blackout active — skipping trade signal")
            return True
    return False


def build_live_features(feature_cols: list) -> pd.DataFrame | None:
    """
    Fetch latest bars from MT5, run feature engineering,
    return the most recent row ready for model inference.
    """
    df = get_ohlcv(symbol=SYMBOL, timeframe="H1", n_bars=300)
    if df.empty:
        log.error("Failed to fetch live OHLCV")
        return None

    # Add H4 and D1 context
    h4 = get_ohlcv(symbol=SYMBOL, timeframe="H4", n_bars=100)
    d1 = get_ohlcv(symbol=SYMBOL, timeframe="D1", n_bars=60)

    if not h4.empty:
        h4_renamed = h4[["open", "high", "low", "close"]].rename(
            columns=lambda c: f"h4_{c}"
        )
        df = df.merge(h4_renamed.resample("1h").ffill(), left_index=True, right_index=True, how="left")
        df.ffill(inplace=True)

    if not d1.empty:
        d1_renamed = d1[["open", "high", "low", "close"]].rename(
            columns=lambda c: f"d1_{c}"
        )
        df = df.merge(d1_renamed.resample("1h").ffill(), left_index=True, right_index=True, how="left")
        df.ffill(inplace=True)

    # Run feature engineering pipeline
    try:
        df = add_trend_features(df)
        df = add_momentum_features(df)
        df = add_volatility_features(df)
        df = add_volume_features(df)
        df = add_price_action_features(df)
        df = add_session_features(df)
        df = add_support_resistance(df)
    except Exception as e:
        log.error(f"Feature engineering failed: {e}")
        return None

    df.dropna(inplace=True)
    if df.empty:
        log.warning("No rows after feature engineering")
        return None

    # Add any missing feature columns with 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    last_row = df.iloc[[-1]][feature_cols]
    return last_row


def get_ensemble_signal(xgb_model, lgb_model, X_row: pd.DataFrame) -> tuple[int, float]:
    xgb_prob  = xgb_model.predict_proba(X_row)[0]
    lgb_prob  = lgb_model.predict_proba(X_row)[0]
    avg_prob  = (xgb_prob + lgb_prob) / 2
    buy_conf  = float(avg_prob[1])
    sell_conf = float(avg_prob[0])

    if buy_conf >= MIN_CONFIDENCE:
        return 1, buy_conf
    elif sell_conf >= MIN_CONFIDENCE:
        return -1, sell_conf
    return 0, max(buy_conf, sell_conf)


def trading_cycle(xgb_model, lgb_model, feature_cols: list):
    """One full decision cycle — runs every hour at candle close."""
    now = datetime.now(tz=timezone.utc)
    log.info(f"─── Trading cycle | {now.strftime('%Y-%m-%d %H:%M UTC')} ───")

    # ── Connect ───────────────────────────────────────────────
    if not connect():
        log.error("MT5 connection failed — skipping cycle")
        return
    
    try:
        # ── Account check ─────────────────────────────────────
        account = get_account_info()
        log.info(f"Balance: ${account['balance']:.2f} | Equity: ${account['equity']:.2f}")

        # ── News blackout check ───────────────────────────────
        if is_news_blackout(now):
            return

        # ── Check open positions ──────────────────────────────
        open_pos = get_open_positions(SYMBOL)
        n_open   = len(open_pos)
        log.info(f"Open positions: {n_open}/{MAX_OPEN_TRADES}")

        if n_open >= MAX_OPEN_TRADES:
            log.info("Max positions reached — waiting for exits")
            return

        # ── Build features and get signal ─────────────────────
        X_row = build_live_features(feature_cols)
        if X_row is None:
            return

        direction, confidence = get_ensemble_signal(xgb_model, lgb_model, X_row)

        if direction == 0:
            log.info(f"No signal | Confidence: {confidence:.3f} (threshold: {MIN_CONFIDENCE})")
            return

        direction_str = "BUY" if direction == 1 else "SELL"
        log.info(f"Signal: {direction_str} | Confidence: {confidence:.3f}")

        # ── ATR-based SL/TP calculation ───────────────────────
        tick = get_latest_tick(SYMBOL)
        if tick is None:
            return

        atr     = float(X_row["atr_14"].iloc[0])
        sl_pips = atr * ATR_MULTIPLIER * 10   # convert to pips
        tp_pips = sl_pips * MIN_RR_RATIO

        # ── Place order ───────────────────────────────────────
        result = send_order(
            direction=direction_str,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            symbol=SYMBOL,
            comment=f"GoldBot {direction_str} {confidence:.2f}",
        )

        if result["success"]:
            log.info(
                f"ORDER PLACED | {direction_str} | "
                f"Ticket: #{result['ticket']} | "
                f"Price: {result['price']} | "
                f"Lots: {result['lots']}"
            )
        else:
            log.error(f"Order failed: {result.get('error')}")

    finally:
        disconnect()


def run():
    log.info("=" * 55)
    log.info("GOLD BOT — Live Trading (DEMO MODE)")
    log.info(f"Symbol: {SYMBOL}")
    log.info(f"Risk per trade: {RISK_PERCENT}%")
    log.info(f"Max open trades: {MAX_OPEN_TRADES}")
    log.info(f"Min confidence: {MIN_CONFIDENCE}")
    log.info("=" * 55)

    xgb_model, lgb_model, feature_cols = load_models()
    if xgb_model is None:
        log.error("Cannot start — models not loaded. Run training first.")
        return

    log.info("Bot starting. First cycle runs immediately, then every hour at :02")
    log.info("Press Ctrl+C to stop.")

    # Run immediately on start
    trading_cycle(xgb_model, lgb_model, feature_cols)

    # Then schedule every hour at :02 (just after H1 candle close)
    schedule.every().hour.at(":02").do(
        trading_cycle, xgb_model, lgb_model, feature_cols
    )

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        log.info("Bot stopped by user.")
        disconnect()


if __name__ == "__main__":
    run()
