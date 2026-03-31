"""
backtest/engine.py — Backtesting Engine (Phase 4)
Simulates trades on historical data using model signals.
Accounts for spread, ATR-based SL/TP, and risk management rules.

Run: python -m backtest.engine
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from utils.logger import log
from config.settings import (
    FEATURES_DIR, MODELS_DIR, LOGS_DIR,
    MIN_CONFIDENCE, ATR_MULTIPLIER, MIN_RR_RATIO,
    RISK_PERCENT, MAX_OPEN_TRADES
)


def load_model_and_features():
    xgb_path  = MODELS_DIR / "xgboost_model.pkl"
    lgb_path  = MODELS_DIR / "lightgbm_model.pkl"
    feat_path = MODELS_DIR / "feature_cols.json"

    if not all(p.exists() for p in [xgb_path, lgb_path, feat_path]):
        log.error("Model files not found. Run training first.")
        return None, None, None

    xgb_model    = joblib.load(xgb_path)
    lgb_model    = joblib.load(lgb_path)
    with open(feat_path) as f:
        feature_cols = json.load(f)

    log.info("Models loaded for backtesting")
    return xgb_model, lgb_model, feature_cols


def get_ensemble_signal(xgb_model, lgb_model, X_row: pd.DataFrame) -> tuple[int, float]:
    """
    Average XGBoost + LightGBM probabilities (ensemble).
    Returns (direction, confidence):
      direction  1=BUY, -1=SELL, 0=no trade
      confidence 0.0-1.0
    """
    xgb_prob = xgb_model.predict_proba(X_row)[0]  # [prob_0, prob_1]
    lgb_prob = lgb_model.predict_proba(X_row)[0]

    avg_prob   = (xgb_prob + lgb_prob) / 2
    buy_conf   = avg_prob[1]
    sell_conf  = avg_prob[0]

    if buy_conf >= MIN_CONFIDENCE:
        return 1, float(buy_conf)
    elif sell_conf >= MIN_CONFIDENCE:
        return -1, float(sell_conf)
    return 0, float(max(buy_conf, sell_conf))


class Trade:
    def __init__(self, entry_time, direction, entry_price, sl, tp, risk_pct, balance):
        self.entry_time  = entry_time
        self.direction   = direction   # 1=BUY, -1=SELL
        self.entry_price = entry_price
        self.sl          = sl
        self.tp          = tp
        self.risk_pct    = risk_pct
        self.risk_amount = balance * (risk_pct / 100)
        self.exit_time   = None
        self.exit_price  = None
        self.pnl         = None
        self.outcome     = None        # "WIN", "LOSS", "OPEN"

    def check_exit(self, high: float, low: float, current_time) -> bool:
        """Check if SL or TP was hit on this bar."""
        if self.direction == 1:   # BUY
            if low <= self.sl:
                self.close(self.sl, current_time, "LOSS")
                return True
            if high >= self.tp:
                self.close(self.tp, current_time, "WIN")
                return True
        else:                     # SELL
            if high >= self.sl:
                self.close(self.sl, current_time, "LOSS")
                return True
            if low <= self.tp:
                self.close(self.tp, current_time, "WIN")
                return True
        return False

    def close(self, price: float, time, outcome: str):
        self.exit_price = price
        self.exit_time  = time
        self.outcome    = outcome
        pip_diff = (price - self.entry_price) * self.direction
        # For XAUUSD: 1 pip ≈ $0.1 per 0.01 lot
        sl_pips  = abs(self.entry_price - self.sl)
        self.pnl = self.risk_amount * (pip_diff / sl_pips) if sl_pips > 0 else 0


def run_backtest(df: pd.DataFrame, xgb_model, lgb_model, feature_cols: list,
                 initial_balance: float = 1000.0) -> dict:
    """
    Walk forward through the test set bar by bar,
    generate signals, simulate entries/exits with ATR-based SL/TP.
    """
    balance      = initial_balance
    equity_curve = [balance]
    trades       = []
    open_trades  = []
    skipped      = 0

    log.info(f"Starting backtest | Balance: ${balance:.2f} | Bars: {len(df)}")

    for i in range(len(df)):
        row = df.iloc[i]
        ts  = df.index[i]

        # ── Check open trades for exit ─────────────────────
        still_open = []
        for trade in open_trades:
            exited = trade.check_exit(row["high"], row["low"], ts)
            if exited:
                balance += trade.pnl
                trades.append(trade)
                log.debug(f"{ts} | {trade.outcome} | PnL: ${trade.pnl:.2f} | Balance: ${balance:.2f}")
            else:
                still_open.append(trade)
        open_trades = still_open

        # ── Generate signal ────────────────────────────────
        if len(open_trades) >= MAX_OPEN_TRADES:
            skipped += 1
            equity_curve.append(balance)
            continue

        X_row = pd.DataFrame([row[feature_cols]])
        direction, confidence = get_ensemble_signal(xgb_model, lgb_model, X_row)

        if direction == 0:
            equity_curve.append(balance)
            continue

        # ── ATR-based SL/TP ────────────────────────────────
        atr    = row.get("atr_14", 2.0)
        price  = row["close"]
        sl_dist = atr * ATR_MULTIPLIER
        tp_dist = sl_dist * MIN_RR_RATIO

        if direction == 1:   # BUY
            sl = price - sl_dist
            tp = price + tp_dist
        else:                 # SELL
            sl = price + sl_dist
            tp = price - tp_dist

        trade = Trade(ts, direction, price, sl, tp, RISK_PERCENT, balance)
        open_trades.append(trade)
        equity_curve.append(balance)

    # Close any remaining open trades at last price
    last_price = df.iloc[-1]["close"]
    last_time  = df.index[-1]
    for trade in open_trades:
        trade.close(last_price, last_time, "OPEN")
        balance += trade.pnl
        trades.append(trade)

    return {
        "trades":       trades,
        "equity_curve": equity_curve,
        "final_balance": balance,
        "initial_balance": initial_balance,
    }


def compute_stats(result: dict) -> dict:
    trades        = result["trades"]
    closed        = [t for t in trades if t.outcome in ("WIN", "LOSS")]
    wins          = [t for t in closed if t.outcome == "WIN"]
    losses        = [t for t in closed if t.outcome == "LOSS"]

    if not closed:
        log.warning("No closed trades in backtest.")
        return {}

    pnls          = [t.pnl for t in closed]
    equity        = result["equity_curve"]
    total_return  = (result["final_balance"] - result["initial_balance"]) / result["initial_balance"] * 100

    # Max drawdown
    peak      = result["initial_balance"]
    max_dd    = 0
    running   = result["initial_balance"]
    for p in equity:
        if p > peak:
            peak = p
        dd = (peak - p) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Profit factor
    gross_profit = sum(t.pnl for t in wins)
    gross_loss   = abs(sum(t.pnl for t in losses)) or 1e-9
    profit_factor = gross_profit / gross_loss

    # Sharpe (simplified, assumes 252 trading days)
    pnl_series = pd.Series(pnls)
    sharpe     = (pnl_series.mean() / pnl_series.std()) * np.sqrt(252) if pnl_series.std() > 0 else 0

    stats = {
        "total_trades":   len(closed),
        "wins":           len(wins),
        "losses":         len(losses),
        "win_rate":       round(len(wins) / len(closed) * 100, 2),
        "total_return":   round(total_return, 2),
        "profit_factor":  round(profit_factor, 3),
        "max_drawdown":   round(max_dd, 2),
        "sharpe_ratio":   round(sharpe, 3),
        "avg_win":        round(np.mean([t.pnl for t in wins]), 2) if wins else 0,
        "avg_loss":       round(np.mean([t.pnl for t in losses]), 2) if losses else 0,
        "final_balance":  round(result["final_balance"], 2),
    }
    return stats


def plot_equity_curve(result: dict, stats: dict):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Gold Bot — Backtest Results", fontsize=14, fontweight="bold")

    # Equity curve
    ax1 = axes[0]
    equity = result["equity_curve"]
    ax1.plot(equity, color="#FFD700", linewidth=1.5, label="Equity")
    ax1.axhline(result["initial_balance"], color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Balance ($)")
    ax1.set_title(
        f"Win Rate: {stats.get('win_rate','N/A')}% | "
        f"Return: {stats.get('total_return','N/A')}% | "
        f"Max DD: {stats.get('max_drawdown','N/A')}% | "
        f"Sharpe: {stats.get('sharpe_ratio','N/A')}"
    )
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Trade PnL distribution
    ax2 = axes[1]
    closed = [t for t in result["trades"] if t.outcome in ("WIN", "LOSS")]
    pnls   = [t.pnl for t in closed]
    colors = ["#2ecc71" if p > 0 else "#e74c3c" for p in pnls]
    ax2.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
    ax2.axhline(0, color="white", linewidth=0.5)
    ax2.set_xlabel("Trade #")
    ax2.set_ylabel("PnL ($)")
    ax2.set_title("Individual Trade PnL")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = LOGS_DIR / "backtest_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    log.info(f"Equity curve saved: {out}")
    plt.show()


def run():
    log.info("=" * 55)
    log.info("GOLD BOT — Backtesting Engine")
    log.info("=" * 55)

    xgb_model, lgb_model, feature_cols = load_model_and_features()
    if xgb_model is None:
        return

    features_path = FEATURES_DIR / "features.parquet"
    if not features_path.exists():
        log.error("features.parquet not found.")
        return

    df = pd.read_parquet(features_path)

    # Use only the test portion (last 30%) — never backtest on training data
    split_idx = int(len(df) * 0.70)
    test_df   = df.iloc[split_idx:].copy()
    log.info(f"Test set: {len(test_df)} bars | {test_df.index[0]} → {test_df.index[-1]}")

    result = run_backtest(test_df, xgb_model, lgb_model, feature_cols)
    stats  = compute_stats(result)

    log.info("=" * 55)
    log.info("BACKTEST RESULTS")
    log.info("=" * 55)
    for k, v in stats.items():
        log.info(f"  {k:<20} {v}")
    log.info("=" * 55)

    # Save stats
    import json
    with open(LOGS_DIR / "backtest_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    plot_equity_curve(result, stats)

    if stats.get("win_rate", 0) >= 52 and stats.get("profit_factor", 0) >= 1.2:
        log.info("Results acceptable. Proceed to demo trading.")
        log.info("Next step → run: python -m execution.live_bot")
    else:
        log.warning("Results below threshold. Revisit features or model config before going live.")


if __name__ == "__main__":
    run()
