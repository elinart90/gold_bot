"""
utils/macro_data.py
Fetches macro data that drives Gold price:
  FRED  → real rates, CPI, fed funds
  yfinance → DXY, VIX, GLD ETF, S&P500
"""
import pandas as pd
import yfinance as yf
from fredapi import Fred
from utils.logger import log
from config.settings import FRED_API_KEY, START_DATE

FRED_SERIES = {
    "real_rate":   "DFII10",    # 10Y TIPS real yield — strongest Gold driver
    "us10y_yield": "DGS10",     # Nominal 10Y yield
    "fed_funds":   "FEDFUNDS",  # Fed Funds Rate
    "cpi_yoy":     "CPIAUCSL",  # CPI (we compute YoY change)
}

YF_TICKERS = {
    "dxy":   "DX-Y.NYB",   # US Dollar Index (inverse to Gold)
    "gld":   "GLD",         # Gold ETF (volume/flow signal)
    "sp500": "^GSPC",       # Risk-on/off
    "vix":   "^VIX",        # Fear index
}


def fetch_fred_data(start_date: str = START_DATE) -> pd.DataFrame:
    if not FRED_API_KEY:
        log.warning("FRED_API_KEY not set — skipping FRED data")
        return pd.DataFrame()
    fred   = Fred(api_key=FRED_API_KEY)
    frames = {}
    for name, series_id in FRED_SERIES.items():
        try:
            s = fred.get_series(series_id, observation_start=start_date)
            frames[name] = s
            log.info(f"FRED | {name}: {len(s)} obs")
        except Exception as e:
            log.error(f"FRED {series_id} failed: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index, utc=True)
    df.sort_index(inplace=True)
    if "cpi_yoy" in df.columns:
        df["cpi_yoy"] = df["cpi_yoy"].pct_change(12) * 100
    df.ffill(inplace=True)
    return df


def fetch_yf_data(start_date: str = START_DATE) -> pd.DataFrame:
    frames = {}
    for name, ticker in YF_TICKERS.items():
        try:
            raw = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
            if raw.empty:
                continue
            frames[name] = raw["Close"].rename(name)
            log.info(f"yfinance | {name}: {len(raw)} rows")
        except Exception as e:
            log.error(f"yfinance {ticker} failed: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames.values(), axis=1)
    df.index = pd.to_datetime(df.index, utc=True)
    df.sort_index(inplace=True)
    df.ffill(inplace=True)
    if "dxy" in df.columns:
        df["dxy_return_1d"] = df["dxy"].pct_change() * 100
    if "vix" in df.columns:
        df["vix_spike"] = (df["vix"] > df["vix"].rolling(20).mean() * 1.2).astype(int)
    return df


def fetch_all_macro(start_date: str = START_DATE) -> pd.DataFrame:
    fred_df = fetch_fred_data(start_date)
    yf_df   = fetch_yf_data(start_date)
    if fred_df.empty and yf_df.empty:
        return pd.DataFrame()
    if fred_df.empty:
        return yf_df
    if yf_df.empty:
        return fred_df
    merged = pd.merge(fred_df, yf_df, left_index=True, right_index=True, how="outer")
    merged.ffill(inplace=True)
    merged.dropna(how="all", inplace=True)
    log.info(f"Macro merged: {merged.shape[0]} rows × {merged.shape[1]} cols")
    return merged
