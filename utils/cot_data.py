"""
utils/cot_data.py
CFTC Commitment of Traders data for Gold futures.
Shows institutional positioning — strong 2-4 week directional signal.
"""
import io
import zipfile
import requests
import pandas as pd
from datetime import datetime
from utils.logger import log

CFTC_URL = "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"


def fetch_cot_year(year: int) -> pd.DataFrame:
    url = CFTC_URL.format(year=year)
    try:
        log.info(f"COT | Downloading {year}...")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            with z.open(z.namelist()[0]) as f:
                return pd.read_csv(f, low_memory=False)
    except Exception as e:
        log.error(f"COT {year} failed: {e}")
        return pd.DataFrame()


def parse_gold_cot(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    gold = df[df["Market_and_Exchange_Names"].str.contains("GOLD", na=False)].copy()
    if gold.empty:
        return pd.DataFrame()
    cols = {
        "Report_Date_as_MM_DD_YYYY":   "date",
        "NonComm_Positions_Long_All":  "nc_long",
        "NonComm_Positions_Short_All": "nc_short",
        "Comm_Positions_Long_All":     "comm_long",
        "Comm_Positions_Short_All":    "comm_short",
        "Open_Interest_All":           "open_interest",
    }
    available = {k: v for k, v in cols.items() if k in gold.columns}
    gold = gold[list(available)].rename(columns=available)
    gold["date"] = pd.to_datetime(gold["date"], utc=True)
    gold.set_index("date", inplace=True)
    gold.sort_index(inplace=True)
    for col in gold.columns:
        gold[col] = pd.to_numeric(gold[col], errors="coerce")
    return gold


def build_cot_features(cot: pd.DataFrame) -> pd.DataFrame:
    df = cot.copy()
    df["net_speculative"]  = df["nc_long"] - df["nc_short"]
    df["net_spec_change"]  = df["net_speculative"].diff()
    df["spec_long_pct"]    = df["nc_long"] / df["open_interest"] * 100
    roll_min = df["net_speculative"].rolling(52).min()
    roll_max = df["net_speculative"].rolling(52).max()
    df["cot_index"]        = (df["net_speculative"] - roll_min) / (roll_max - roll_min + 1e-9) * 100
    df.dropna(subset=["net_speculative"], inplace=True)
    log.info(f"COT features built: {len(df)} weekly observations")
    return df


def fetch_cot_data(start_year: int = 2018) -> pd.DataFrame:
    current_year = datetime.now().year
    frames = []
    for year in range(start_year, current_year + 1):
        raw = fetch_cot_year(year)
        if not raw.empty:
            parsed = parse_gold_cot(raw)
            if not parsed.empty:
                frames.append(parsed)
    if not frames:
        log.error("COT | No data fetched")
        return pd.DataFrame()
    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)
    return build_cot_features(combined)
