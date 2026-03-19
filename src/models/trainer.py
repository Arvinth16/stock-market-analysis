"""
trainer.py — Train ensemble models with walk-forward cross-validation.

V2: 34 features, ensemble (XGBoost + LightGBM + LogReg), bootstrap uncertainty.
"""

import os
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import uniform, randint

from src.core.config import MODEL_DIR
from src.core.database import get_db
from src.data.features import FEATURE_COLUMNS
from src.models.ensemble import (
    EnsembleModel, EnsembleRegressor, save_ensemble,
)

# Use the canonical feature column list from features.py
DB_FEATURE_COLS = FEATURE_COLUMNS


def load_training_data() -> pd.DataFrame:
    """Load all feature rows from the database for training."""
    col_str = ", ".join(DB_FEATURE_COLS)
    query = f"""SELECT symbol, date, {col_str},
                       label, label_reg, label_60d, label_vol_scaled, label_bucket
                FROM features
                WHERE label IS NOT NULL AND label_reg IS NOT NULL
                ORDER BY date"""
    with get_db() as conn:
        df = pd.read_sql_query(query, conn, parse_dates=["date"])
    return df


def tune_hyperparameters(X, y):
    """Tune XGBoost hyperparameters using TimeSeriesSplit."""
    print("  Searching for best XGBoost hyperparameters...")
    param_dist = {
        "n_estimators": randint(100, 500),
        "max_depth": randint(3, 8),
        "learning_rate": uniform(0.01, 0.2),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "reg_alpha": uniform(0, 2),
        "reg_lambda": uniform(1, 4),
    }

    base_model = XGBClassifier(random_state=42, eval_metric="logloss")
    tscv = TimeSeriesSplit(n_splits=3)

    search = RandomizedSearchCV(
        base_model, param_distributions=param_dist,
        n_iter=20, scoring="roc_auc", cv=tscv,
        random_state=42, n_jobs=-1, verbose=0,
    )

    search.fit(X, y)
    print(f"  Best CV AUC: {search.best_score_:.4f}")
    return search.best_params_


def tune_hyperparameters_regressor(X, y):
    """Tune XGBoost Regressor hyperparameters."""
    print("  Searching for best regressor hyperparameters...")
    param_dist = {
        "n_estimators": randint(100, 500),
        "max_depth": randint(3, 8),
        "learning_rate": uniform(0.01, 0.2),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "reg_alpha": uniform(0, 2),
        "reg_lambda": uniform(1, 4),
    }

    base_model = XGBRegressor(random_state=42)
    tscv = TimeSeriesSplit(n_splits=3)

    search = RandomizedSearchCV(
        base_model, param_distributions=param_dist,
        n_iter=15, scoring="neg_mean_squared_error", cv=tscv,
        random_state=42, n_jobs=-1, verbose=0,
    )

    search.fit(X, y)
    print(f"  Best CV MSE: {-search.best_score_:.4f}")
    return search.best_params_


def walk_forward_train(df: pd.DataFrame, n_splits: int = 5):
    """Walk-forward cross-validation with ensemble training."""
    print("━" * 60)
    print("🤖 TRAINING V2 ENSEMBLE (Walk-Forward Cross-Validation)")
    print("━" * 60)

    # Handle NaN fundamentals by filling with median
    for col in DB_FEATURE_COLS:
        if col.startswith("fund_"):
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0.0)

    df_clean = df.dropna(subset=[c for c in DB_FEATURE_COLS
                                 if not c.startswith("fund_")] + ["label"]).copy()

    # Also fill remaining NaN in non-fund columns
    for col in DB_FEATURE_COLS:
        df_clean[col] = df_clean[col].fillna(0.0)

    if len(df_clean) < 500:
        print("  ⚠  Not enough data for training. Need at least 500 clean rows.")
        return None

    df_clean = df_clean.sort_values(by="date").reset_index(drop=True)

    X_all = df_clean[DB_FEATURE_COLS].values
    y_all = df_clean["label"].values
    y_reg_all = df_clean["label_reg"].values

    # Tune XGBoost component
    best_params = tune_hyperparameters(X_all, y_all)
    best_params_reg = tune_hyperparameters_regressor(X_all, y_reg_all)

    # Walk-forward evaluation
    dates = df_clean["date"].unique()
    split_size = len(dates) // (n_splits + 1)

    metrics_list = []

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

        # Train ensemble for this fold
        ensemble = EnsembleModel(xgb_params=best_params)
        ensemble.fit(X_train, y_train)

        y_prob = ensemble.predict_proba(X_test)
        y_pred = (y_prob > 0.5).astype(int)

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

    if not metrics_list:
        print("  ✗ No valid folds produced.")
        return None

    avg = pd.DataFrame(metrics_list).mean(numeric_only=True)
    print("\n  ─── Average Walk-Forward Metrics ───")
    print(f"  AUC:       {avg['auc']:.4f}")
    print(f"  Precision: {avg['precision']:.4f}")
    print(f"  Recall:    {avg['recall']:.4f}")
    print(f"  F1:        {avg['f1']:.4f}")
    print(f"  Accuracy:  {avg['accuracy']:.4f}")

    # Train final ensemble on ALL data
    print("\n  Training final ENSEMBLE classifier on full dataset...")
    final_clf = EnsembleModel(xgb_params=best_params)
    final_clf.fit(X_all, y_all)

    print("  Training final ENSEMBLE regressor on full dataset...")
    final_reg = EnsembleRegressor(xgb_params=best_params_reg)
    final_reg.fit(X_all, y_reg_all)

    # Save
    save_ensemble(final_clf, final_reg)

    # Also save standalone XGBoost for SHAP compatibility
    final_xgb = XGBClassifier(**best_params, random_state=42, eval_metric="logloss")
    final_xgb.fit(X_all, y_all, verbose=False)
    joblib.dump(final_xgb, os.path.join(MODEL_DIR, "xgb_model.joblib"))

    final_xgb_reg = XGBRegressor(**best_params_reg, random_state=42)
    final_xgb_reg.fit(X_all, y_reg_all, verbose=False)
    joblib.dump(final_xgb_reg, os.path.join(MODEL_DIR, "xgb_regressor.joblib"))

    print(f"  ✅ All models saved to {MODEL_DIR}\n")

    # Feature importance (from XGBoost component)
    importances = pd.DataFrame({
        "feature": DB_FEATURE_COLS,
        "importance": final_xgb.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("  ─── Feature Importance (Top 15) ───")
    for _, row in importances.head(15).iterrows():
        bar = "█" * int(row["importance"] * 50)
        print(f"  {row['feature']:>25s}  {bar} {row['importance']:.4f}")

    return final_clf


def run_training():
    """Main entry point for training."""
    df = load_training_data()
    if df.empty:
        print("  ✗ No training data. Run: india-stock update")
        return None
    print(f"  Loaded {len(df)} feature rows ({len(DB_FEATURE_COLS)} features).\n")
    return walk_forward_train(df)


if __name__ == "__main__":
    run_training()
