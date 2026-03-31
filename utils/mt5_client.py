"""
utils/mt5_client.py
All MetaTrader 5 interactions: connect, fetch data, account info, orders.
"""
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone
from utils.logger import log
from config.settings import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, SYMBOL, MAGIC_NUMBER

TF_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
    "W1":  mt5.TIMEFRAME_W1,
}


# ── Connection ────────────────────────────────────────────────

def connect() -> bool:
    if not mt5.initialize():
        log.error(f"MT5 initialize() failed: {mt5.last_error()}")
        return False
    authorized = mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    if not authorized:
        log.error(f"MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        return False
    info = mt5.account_info()
    log.info(f"MT5 connected | #{info.login} | Balance: {info.balance:.2f} {info.currency} | {info.server}")
    return True


def disconnect():
    mt5.shutdown()
    log.info("MT5 disconnected")


# ── Data Fetching ─────────────────────────────────────────────

def get_ohlcv(symbol: str = SYMBOL, timeframe: str = "H1", n_bars: int = 5000) -> pd.DataFrame:
    """Fetch historical OHLCV bars from MT5."""
    tf = TF_MAP.get(timeframe.upper())
    if tf is None:
        raise ValueError(f"Unknown timeframe '{timeframe}'. Valid: {list(TF_MAP)}")
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, n_bars)
    if rates is None or len(rates) == 0:
        log.error(f"No data for {symbol} {timeframe}: {mt5.last_error()}")
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)
    log.info(f"Fetched {len(df)} bars | {symbol} {timeframe} | {df.index[0]} → {df.index[-1]}")
    return df


def get_latest_tick(symbol: str = SYMBOL) -> dict | None:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        log.error(f"Failed to get tick for {symbol}")
        return None
    return {
        "time":   datetime.fromtimestamp(tick.time, tz=timezone.utc),
        "bid":    tick.bid,
        "ask":    tick.ask,
        "spread": round(tick.ask - tick.bid, 5),
    }


def get_account_info() -> dict:
    info = mt5.account_info()
    if info is None:
        return {}
    return {
        "balance":      info.balance,
        "equity":       info.equity,
        "margin":       info.margin,
        "free_margin":  info.margin_free,
        "currency":     info.currency,
        "leverage":     info.leverage,
    }


def get_open_positions(symbol: str = SYMBOL) -> pd.DataFrame:
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return pd.DataFrame()
    df = pd.DataFrame([p._asdict() for p in positions])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df


# ── Order Execution ───────────────────────────────────────────

def get_lot_size(account_balance: float, stop_loss_pips: float, risk_pct: float = 1.0) -> float:
    """
    Calculate position size based on fixed % risk.
    For XAUUSD: 1 pip = $1 per 0.01 lot (micro lot).
    Returns lot size rounded to 2 decimal places.
    """
    risk_amount   = account_balance * (risk_pct / 100)
    pip_value     = 1.0   # USD per pip per 0.01 lot for XAUUSD
    lots          = risk_amount / (stop_loss_pips * pip_value * 100)
    return round(max(0.01, min(lots, 10.0)), 2)  # clamp between 0.01 and 10


def send_order(
    direction: str,           # "BUY" or "SELL"
    sl_pips: float,           # stop loss in pips
    tp_pips: float,           # take profit in pips
    symbol: str = SYMBOL,
    comment: str = "GoldBot",
) -> dict:
    """
    Place a market order on MT5.
    Returns dict with success status and order ticket.
    """
    tick = get_latest_tick(symbol)
    if tick is None:
        return {"success": False, "error": "Could not get tick"}

    account = get_account_info()
    lot = get_lot_size(account["balance"], sl_pips)

    price = tick["ask"] if direction == "BUY" else tick["bid"]
    sl    = (price - sl_pips * 0.1) if direction == "BUY" else (price + sl_pips * 0.1)
    tp    = (price + tp_pips * 0.1) if direction == "BUY" else (price - tp_pips * 0.1)

    request = {
        "action":    mt5.TRADE_ACTION_DEAL,
        "symbol":    symbol,
        "volume":    lot,
        "type":      mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL,
        "price":     price,
        "sl":        round(sl, 2),
        "tp":        round(tp, 2),
        "deviation": 20,
        "magic":     MAGIC_NUMBER,
        "comment":   comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        log.info(f"Order placed | {direction} {lot} lots {symbol} @ {price} | Ticket #{result.order}")
        return {"success": True, "ticket": result.order, "price": price, "lots": lot}
    else:
        log.error(f"Order failed | retcode={result.retcode} | {result.comment}")
        return {"success": False, "retcode": result.retcode, "error": result.comment}


def close_position(ticket: int) -> bool:
    """Close a specific position by ticket number."""
    position = mt5.positions_get(ticket=ticket)
    if not position:
        log.warning(f"Position #{ticket} not found")
        return False

    pos   = position[0]
    price = mt5.symbol_info_tick(pos.symbol).bid if pos.type == 0 else mt5.symbol_info_tick(pos.symbol).ask
    close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       pos.symbol,
        "volume":       pos.volume,
        "type":         close_type,
        "position":     ticket,
        "price":        price,
        "deviation":    20,
        "magic":        MAGIC_NUMBER,
        "comment":      "GoldBot close",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        log.info(f"Position #{ticket} closed")
        return True
    else:
        log.error(f"Close failed #{ticket}: {result.comment}")
        return False
