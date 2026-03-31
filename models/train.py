"""
models/train.py — Model Training (Phase 3)
Walk-forward validation to avoid data leakage.
Trains XGBoost + LightGBM, saves best model to models/

Run: python -m models.train
"""
import numpy as np
import pandas as pd
import joblib
import json
from sklearn.metrics import (
    classification_report, precision_score,
    recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from utils.logger import log
from config.settings import FEATURES_DIR, MODELS_DIR, TRAIN_SPLIT

# Columns that are NOT features (raw OHLCV, meta, target)
EXCLUDE_COLS = [
    "open", "high", "low", "close", "tick_volume", "spread",
    "real_volume", "target", "future_return",
    # Higher TF raw prices (we already derived features from them)
    "h4_open", "h4_high", "h4_low", "h4_close",
    "d1_open", "d1_high", "d1_low", "d1_close",
]


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def walk_forward_split(df: pd.DataFrame, n_splits: int = 5):
    """
    Time-series walk-forward splits.
    Each fold: train on past data, test on next unseen window.
    NEVER shuffles. This is critical to avoid look-ahead bias.
    """
    total = len(df)
    min_train = int(total * 0.5)
    step = (total - min_train) // n_splits

    splits = []
    for i in range(n_splits):
        train_end = min_train + i * step
        test_end  = min(train_end + step, total)
        splits.append((
            df.iloc[:train_end],
            df.iloc[train_end:test_end]
        ))
    return splits


def train_xgboost(X_train, y_train, X_val, y_val) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=1,
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def train_lightgbm(X_train, y_train, X_val, y_val) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=5,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    return model


def evaluate(model, X_test, y_test, name: str) -> dict:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model":     name,
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall":    round(recall_score(y_test, preds, zero_division=0), 4),
        "f1":        round(f1_score(y_test, preds, zero_division=0), 4),
        "auc_roc":   round(roc_auc_score(y_test, proba), 4),
    }
    log.info(f"{name} | Precision: {metrics['precision']} | "
             f"Recall: {metrics['recall']} | F1: {metrics['f1']} | "
             f"AUC: {metrics['auc_roc']}")
    return metrics


def run():
    log.info("=" * 55)
    log.info("GOLD BOT — Model Training (Walk-Forward)")
    log.info("=" * 55)

    features_path = FEATURES_DIR / "features.parquet"
    if not features_path.exists():
        log.error("features.parquet not found. Run feature engineering first.")
        return

    df = pd.read_parquet(features_path)
    log.info(f"Loaded features: {df.shape[0]} rows × {df.shape[1]} cols")

    feature_cols = get_feature_cols(df)
    log.info(f"Using {len(feature_cols)} features")

    X = df[feature_cols]
    y = df["target"]

    # ── Walk-forward validation ───────────────────────────────
    splits    = walk_forward_split(df, n_splits=5)
    xgb_scores = []
    lgb_scores = []

    for i, (train_df, test_df) in enumerate(splits):
        log.info(f"Fold {i+1}/5 | Train: {len(train_df)} | Test: {len(test_df)}")
        X_train = train_df[feature_cols]
        y_train = train_df["target"]
        X_test  = test_df[feature_cols]
        y_test  = test_df["target"]

        # Use last 20% of train as validation for early stopping
        val_split  = int(len(X_train) * 0.8)
        X_tr, X_val = X_train.iloc[:val_split], X_train.iloc[val_split:]
        y_tr, y_val = y_train.iloc[:val_split], y_train.iloc[val_split:]

        xgb_model = train_xgboost(X_tr, y_tr, X_val, y_val)
        lgb_model = train_lightgbm(X_tr, y_tr, X_val, y_val)

        xgb_scores.append(evaluate(xgb_model, X_test, y_test, f"XGBoost fold {i+1}"))
        lgb_scores.append(evaluate(lgb_model, X_test, y_test, f"LightGBM fold {i+1}"))

    # ── Average scores across folds ───────────────────────────
    def avg_scores(scores, name):
        avg = {k: round(np.mean([s[k] for s in scores if isinstance(s[k], float)]), 4)
               for k in ["precision", "recall", "f1", "auc_roc"]}
        log.info(f"\n{name} Average across folds: {avg}")
        return avg

    xgb_avg = avg_scores(xgb_scores, "XGBoost")
    lgb_avg = avg_scores(lgb_scores, "LightGBM")

    # ── Train final model on ALL data ─────────────────────────
    log.info("Training final models on full dataset...")
    val_idx = int(len(X) * 0.8)
    X_tr_full, X_val_full = X.iloc[:val_idx], X.iloc[val_idx:]
    y_tr_full, y_val_full = y.iloc[:val_idx], y.iloc[val_idx:]

    final_xgb = train_xgboost(X_tr_full, y_tr_full, X_val_full, y_val_full)
    final_lgb = train_lightgbm(X_tr_full, y_tr_full, X_val_full, y_val_full)

    # ── Save models and metadata ──────────────────────────────
    joblib.dump(final_xgb, MODELS_DIR / "xgboost_model.pkl")
    joblib.dump(final_lgb, MODELS_DIR / "lightgbm_model.pkl")

    # Save feature list (needed for inference)
    with open(MODELS_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    # Save performance summary
    summary = {
        "xgboost_avg":  xgb_avg,
        "lightgbm_avg": lgb_avg,
        "n_features":   len(feature_cols),
        "n_samples":    len(df),
    }
    with open(MODELS_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("=" * 55)
    log.info("Training complete!")
    log.info(f"XGBoost avg precision:  {xgb_avg['precision']}")
    log.info(f"LightGBM avg precision: {lgb_avg['precision']}")
    log.info(f"Models saved to: {MODELS_DIR}")
    log.info("=" * 55)
    log.info("Next step → run: python -m backtest.engine")


if __name__ == "__main__":
    run()
