"""
trainer.py — Train XGBoost classifier with walk-forward cross-validation.

Uses time-based splits to prevent lookahead bias.
"""

import os
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

from src.core.config import XGBOOST_PARAMS, MODEL_DIR
from src.core.database import get_db


def load_training_data() -> pd.DataFrame:
    """Load all feature rows from the database for training."""
    with get_db() as conn:
        df = pd.read_sql_query(
            """SELECT symbol, date, sma_20, sma_50, sma_200, rsi_14,
                      macd, macd_signal, macd_hist, bb_width, atr_14,
                      volatility_20, volume_ratio, dist_52w_high, dist_52w_low,
                      return_5d, return_10d, return_20d, label
               FROM features
               WHERE label IS NOT NULL
               ORDER BY date""",
            conn, parse_dates=["date"],
        )

    # We need to recompute price_vs_sma columns — they're derived features
    # stored in the DB as sma values. For the model, we use the stored raw features.
    # Adjust FEATURE_COLUMNS for what's actually in DB
    return df


# Features available in the DB (subset of FEATURE_COLUMNS)
DB_FEATURE_COLS = [
    "sma_20", "sma_50", "sma_200",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_width", "atr_14", "volatility_20", "volume_ratio",
    "dist_52w_high", "dist_52w_low",
    "return_5d", "return_10d", "return_20d",
]


def walk_forward_train(df: pd.DataFrame, n_splits: int = 5):
    """
    Walk-forward cross-validation.

    Splits the time-ordered data into expanding training windows
    with fixed-size test windows.
    """
    print("━" * 60)
    print("🤖 TRAINING MODEL (Walk-Forward Cross-Validation)")
    print("━" * 60)

    # Drop rows with NaN features
    df_clean = df.dropna(subset=DB_FEATURE_COLS + ["label"])
    if len(df_clean) < 500:
        print("  ⚠  Not enough data for training. Need at least 500 clean rows.")
        return None

    dates = df_clean["date"].sort_values().unique()
    split_size = len(dates) // (n_splits + 1)

    metrics_list = []
    best_auc = 0

    for fold in range(n_splits):
        train_end_idx = split_size * (fold + 2)
        test_end_idx = min(train_end_idx + split_size, len(dates))

        if test_end_idx <= train_end_idx:
            continue

        train_dates = dates[:train_end_idx]
        test_dates = dates[train_end_idx:test_end_idx]

        train_mask = df_clean["date"].isin(train_dates)
        test_mask = df_clean["date"].isin(test_dates)

        X_train = df_clean.loc[train_mask, DB_FEATURE_COLS].values
        y_train = df_clean.loc[train_mask, "label"].values
        X_test = df_clean.loc[test_mask, DB_FEATURE_COLS].values
        y_test = df_clean.loc[test_mask, "label"].values

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        model = XGBClassifier(**XGBOOST_PARAMS)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        metrics_list.append({
            "fold": fold + 1, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "auc": auc,
            "train_size": len(X_train), "test_size": len(X_test),
        })

        print(f"\n  Fold {fold+1}: AUC={auc:.4f} | Prec={prec:.4f} | "
              f"Rec={rec:.4f} | F1={f1:.4f} | Acc={acc:.4f}")

        if auc > best_auc:
            best_auc = auc

    if not metrics_list:
        print("  ✗ No valid folds produced. Check data quality.")
        return None

    # Print average metrics
    avg = pd.DataFrame(metrics_list).mean(numeric_only=True)
    print("\n  ─── Average Across Folds ───")
    print(f"  AUC:       {avg['auc']:.4f}")
    print(f"  Precision: {avg['precision']:.4f}")
    print(f"  Recall:    {avg['recall']:.4f}")
    print(f"  F1:        {avg['f1']:.4f}")
    print(f"  Accuracy:  {avg['accuracy']:.4f}")

    # Train final model on ALL data
    print("\n  Training final model on full dataset...")
    X_all = df_clean[DB_FEATURE_COLS].values
    y_all = df_clean["label"].values
    final_model = XGBClassifier(**XGBOOST_PARAMS)
    final_model.fit(X_all, y_all, verbose=False)

    # Save model
    model_path = os.path.join(MODEL_DIR, "xgb_model.joblib")
    joblib.dump(final_model, model_path)
    print(f"  ✅ Model saved to {model_path}\n")

    # Save feature importance
    importances = pd.DataFrame({
        "feature": DB_FEATURE_COLS,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("  ─── Feature Importance (Top 10) ───")
    for _, row in importances.head(10).iterrows():
        bar = "█" * int(row["importance"] * 50)
        print(f"  {row['feature']:>20s}  {bar} {row['importance']:.4f}")

    return final_model


def run_training():
    """Main entry point for training."""
    df = load_training_data()
    if df.empty:
        print("  ✗ No training data found. Run data fetcher and feature computation first.")
        return None
    print(f"  Loaded {len(df)} feature rows from database.\n")
    return walk_forward_train(df)


if __name__ == "__main__":
    run_training()
