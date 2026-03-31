"""
data/features.py — Feature Engineering (Phase 2)
Loads master_h1.parquet, adds all technical + session features,
creates the target variable, saves features.parquet

Run: python -m data.features
"""
import pandas as pd
import numpy as np
import ta
from utils.logger import log
from config.settings import PROCESSED_DIR, FEATURES_DIR, LOOKAHEAD_BARS


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    h = df["high"]
    l = df["low"]

    for period in [8, 13, 21, 50, 200]:
        df[f"ema_{period}"] = ta.trend.ema_indicator(c, window=period)

    df["ema_8_21_cross"]  = (df["ema_8"]  > df["ema_21"]).astype(int)
    df["ema_21_50_cross"] = (df["ema_21"] > df["ema_50"]).astype(int)

    macd = ta.trend.MACD(c)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()
    df["macd_cross"]  = (df["macd"] > df["macd_signal"]).astype(int)

    adx = ta.trend.ADXIndicator(h, l, c)
    df["adx"]        = adx.adx()
    df["adx_pos"]    = adx.adx_pos()
    df["adx_neg"]    = adx.adx_neg()
    df["adx_strong"] = (df["adx"] > 25).astype(int)

    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    h = df["high"]
    l = df["low"]

    df["rsi_14"]          = ta.momentum.RSIIndicator(c, window=14).rsi()
    df["rsi_7"]           = ta.momentum.RSIIndicator(c, window=7).rsi()
    df["rsi_overbought"]  = (df["rsi_14"] > 70).astype(int)
    df["rsi_oversold"]    = (df["rsi_14"] < 30).astype(int)

    stoch = ta.momentum.StochasticOscillator(h, l, c)
    df["stoch_k"]     = stoch.stoch()
    df["stoch_d"]     = stoch.stoch_signal()
    df["stoch_cross"] = (df["stoch_k"] > df["stoch_d"]).astype(int)

    df["roc_5"]  = ta.momentum.ROCIndicator(c, window=5).roc()
    df["roc_10"] = ta.momentum.ROCIndicator(c, window=10).roc()

    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    h = df["high"]
    l = df["low"]

    for period in [7, 14, 21]:
        df[f"atr_{period}"] = ta.volatility.AverageTrueRange(h, l, c, window=period).average_true_range()

    df["atr_ratio"] = df["atr_14"] / df["atr_14"].rolling(50).mean()

    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"]   = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_pct"]   = bb.bollinger_pband()

    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    v = df["tick_volume"].astype(float)

    df["vol_ma_20"] = v.rolling(20).mean()
    df["vol_ratio"] = v / df["vol_ma_20"]
    df["vol_spike"] = (df["vol_ratio"] > 2.0).astype(int)
    df["obv"]       = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()

    return df


def add_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]

    df["body_size"]  = (c - o).abs() / df["atr_14"]
    df["upper_wick"] = (h - pd.concat([c, o], axis=1).max(axis=1)) / df["atr_14"]
    df["lower_wick"] = (pd.concat([c, o], axis=1).min(axis=1) - l) / df["atr_14"]
    df["is_bullish"] = (c > o).astype(int)
    df["is_doji"]    = (df["body_size"] < 0.1).astype(int)

    for n in [1, 2, 3, 5, 10]:
        df[f"return_{n}h"] = c.pct_change(n) * 100

    df["dist_ema_21"]  = (c - df["ema_21"])  / df["atr_14"]
    df["dist_ema_50"]  = (c - df["ema_50"])  / df["atr_14"]
    df["dist_ema_200"] = (c - df["ema_200"]) / df["atr_14"]

    if "h4_close" in df.columns:
        df["h4_trend"] = (c > df["h4_close"]).astype(int)
    if "d1_close" in df.columns:
        df["d1_trend"] = (c > df["d1_close"]).astype(int)

    return df


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    hour = df.index.hour

    df["session_london"]  = ((hour >= 7)  & (hour < 16)).astype(int)
    df["session_ny"]      = ((hour >= 13) & (hour < 21)).astype(int)
    df["session_overlap"] = ((hour >= 13) & (hour < 16)).astype(int)
    df["session_asian"]   = ((hour >= 0)  & (hour < 7)).astype(int)
    df["hour_of_day"]     = hour
    df["day_of_week"]     = df.index.dayofweek
    df["is_monday"]       = (df["day_of_week"] == 0).astype(int)
    df["is_friday"]       = (df["day_of_week"] == 4).astype(int)

    return df


def add_support_resistance(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    h = df["high"]
    l = df["low"]

    df["rolling_high_20"] = h.rolling(20).max()
    df["rolling_low_20"]  = l.rolling(20).min()
    df["rolling_high_50"] = h.rolling(50).max()
    df["rolling_low_50"]  = l.rolling(50).min()

    df["dist_from_high_20"] = (df["rolling_high_20"] - c) / df["atr_14"]
    df["dist_from_low_20"]  = (c - df["rolling_low_20"]) / df["atr_14"]
    df["range_compression"] = (df["rolling_high_20"] - df["rolling_low_20"]) / df["atr_14"]

    return df


def create_target(df: pd.DataFrame, lookahead: int = LOOKAHEAD_BARS) -> pd.DataFrame:
    future_close        = df["close"].shift(-lookahead)
    df["target"]        = (future_close > df["close"]).astype(int)
    df["future_return"] = (future_close - df["close"]) / df["close"] * 100
    df = df.iloc[:-lookahead].copy()
    return df


def run():
    log.info("=" * 55)
    log.info("GOLD BOT — Feature Engineering")
    log.info("=" * 55)

    master_path = PROCESSED_DIR / "master_h1.parquet"
    if not master_path.exists():
        log.error("master_h1.parquet not found. Run data pipeline first.")
        return

    df = pd.read_parquet(master_path)
    log.info(f"Loaded master: {df.shape[0]} rows x {df.shape[1]} cols")

    log.info("Building features...")
    df = add_trend_features(df)
    df = add_momentum_features(df)
    df = add_volatility_features(df)
    df = add_volume_features(df)
    df = add_price_action_features(df)
    df = add_session_features(df)
    df = add_support_resistance(df)
    df = create_target(df)

    # Step 1: Fill ALL non-OHLCV columns with 0
    core_cols = ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
    fill_cols = [c for c in df.columns if c not in core_cols]
    df[fill_cols] = df[fill_cols].fillna(0)

    # Step 2: Forward fill remaining NaN from rolling windows
    df.ffill(inplace=True)

    # Step 3: Drop only rows where critical features are still NaN
    critical = ["target", "atr_14", "rsi_14", "ema_21", "macd", "adx"]
    before = len(df)
    df.dropna(subset=critical, inplace=True)
    log.info(f"Dropped {before - len(df)} rows with NaN | Remaining: {len(df)}")

    if len(df) == 0:
        log.error("Still 0 rows after NaN handling. Check your master_h1.parquet data.")
        return

    out = FEATURES_DIR / "features.parquet"
    df.to_parquet(out)

    log.info("=" * 55)
    log.info("Features complete!")
    log.info(f"Rows: {df.shape[0]}  |  Features: {df.shape[1]}")
    log.info(f"Target distribution: {df['target'].value_counts().to_dict()}")
    log.info(f"Saved: {out}")
    log.info("=" * 55)
    log.info("Next step → run: python -m models.train")


if __name__ == "__main__":
    run()