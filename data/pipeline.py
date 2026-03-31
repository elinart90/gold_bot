"""
data/pipeline.py — Master data pipeline.
Run first: python -m data.pipeline
Fetches MT5 OHLCV + FRED macro + COT, merges into master_h1.parquet
"""
import pandas as pd
from utils.logger import log
from utils.mt5_client import connect, disconnect, get_ohlcv
from utils.macro_data import fetch_all_macro
from utils.cot_data import fetch_cot_data
from config.settings import (
    SYMBOL, HISTORY_BARS, TIMEFRAMES,
    START_DATE, RAW_DIR, PROCESSED_DIR
)


def fetch_all_ohlcv() -> dict[str, pd.DataFrame]:
    data = {}
    for tf in TIMEFRAMES:
        df = get_ohlcv(symbol=SYMBOL, timeframe=tf, n_bars=HISTORY_BARS)
        if not df.empty:
            data[tf] = df
    return data


def save_raw(ohlcv: dict, macro: pd.DataFrame, cot: pd.DataFrame):
    for tf, df in ohlcv.items():
        df.to_parquet(RAW_DIR / f"ohlcv_{tf}.parquet")
    if not macro.empty:
        macro.to_parquet(RAW_DIR / "macro.parquet")
    if not cot.empty:
        cot.to_parquet(RAW_DIR / "cot.parquet")
    log.info("Raw data saved to data/raw/")


def merge_master(ohlcv: dict, macro: pd.DataFrame, cot: pd.DataFrame) -> pd.DataFrame:
    h1 = ohlcv.get("H1")
    if h1 is None:
        log.error("H1 data missing — cannot build master")
        return pd.DataFrame()

    master = h1.copy()

    # Merge macro (daily → hourly forward fill)
    if not macro.empty:
        master = master.merge(
            macro.resample("1h").ffill(),
            left_index=True, right_index=True, how="left"
        )
        master.ffill(inplace=True)

    # Merge COT (weekly → hourly forward fill)
    if not cot.empty:
        master = master.merge(
            cot.resample("1h").ffill(),
            left_index=True, right_index=True, how="left"
        )
        master.ffill(inplace=True)

    # Add H4 and D1 OHLC context columns
    for tf in ["H4", "D1"]:
        df = ohlcv.get(tf)
        if df is None:
            continue
        renamed = df[["open", "high", "low", "close"]].rename(
            columns=lambda c: f"{tf.lower()}_{c}"
        )
        master = master.merge(
            renamed.resample("1h").ffill(),
            left_index=True, right_index=True, how="left"
        )
        master.ffill(inplace=True)

    master.dropna(subset=["open", "high", "low", "close"], inplace=True)
    log.info(f"Master frame: {master.shape[0]} rows × {master.shape[1]} cols")
    return master


def run():
    log.info("=" * 55)
    log.info("GOLD BOT — Data Pipeline")
    log.info("=" * 55)

    # Step 1: MT5
    log.info("Step 1/4 — Fetching OHLCV from MT5...")
    if not connect():
        log.error("MT5 connection failed. Is MT5 open and logged in?")
        return
    ohlcv = fetch_all_ohlcv()
    disconnect()
    if not ohlcv:
        log.error("No OHLCV data. Aborting.")
        return

    # Step 2: Macro
    log.info("Step 2/4 — Fetching macro data (FRED + yfinance)...")
    macro = fetch_all_macro(start_date=START_DATE)

    # Step 3: COT
    log.info("Step 3/4 — Fetching COT data (CFTC)...")
    cot = fetch_cot_data(start_year=2018)

    # Step 4: Merge and save
    log.info("Step 4/4 — Merging and saving...")
    save_raw(ohlcv, macro, cot)
    master = merge_master(ohlcv, macro, cot)
    if master.empty:
        log.error("Merge failed.")
        return

    out = PROCESSED_DIR / "master_h1.parquet"
    master.to_parquet(out)

    log.info("=" * 55)
    log.info(f"Pipeline complete!")
    log.info(f"Rows: {master.shape[0]}  |  Columns: {master.shape[1]}")
    log.info(f"Range: {master.index[0]} → {master.index[-1]}")
    log.info(f"Saved: {out}")
    log.info("=" * 55)
    log.info("Next step → run: python -m data.features")


if __name__ == "__main__":
    run()
