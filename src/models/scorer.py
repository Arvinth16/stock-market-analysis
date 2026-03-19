"""
scorer.py — Score all symbols using ensemble models with regime awareness.

V2: Ensemble predictions, SHAP explainability, confidence intervals, regime overlay.
"""

import os
import numpy as np
import pandas as pd
import joblib
import shap

from src.core.config import MODEL_DIR, WEIGHT_MODEL, WEIGHT_SENTIMENT, WEIGHT_MOMENTUM
from src.core.database import get_db
from src.data.universe import get_symbols, get_symbol_name, get_symbol_sector
from src.data.features import FEATURE_COLUMNS
from src.models.ensemble import load_ensemble, bootstrap_predictions
from src.models.regime import get_current_regime, apply_regime_adjustment

DB_FEATURE_COLS = FEATURE_COLUMNS


def load_xgb_model():
    """Load standalone XGBoost for SHAP explainability."""
    path = os.path.join(MODEL_DIR, "xgb_model.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return None


def get_latest_features(symbol: str) -> dict | None:
    """Get the most recent feature row for a symbol from the DB."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM features WHERE symbol = ? ORDER BY date DESC LIMIT 1",
            (symbol,),
        ).fetchone()
    if row is None:
        return None
    return dict(row)


def get_latest_sentiment(symbol: str) -> float:
    """Get average sentiment from recent news."""
    with get_db() as conn:
        result = conn.execute(
            """SELECT AVG(sentiment) as avg_sent FROM news
               WHERE symbol = ? AND date >= date('now', '-7 days')""",
            (symbol,),
        ).fetchone()
    if result and result["avg_sent"] is not None:
        return float(result["avg_sent"])
    return 0.0


def score_all_stocks() -> pd.DataFrame:
    """
    Score all stocks using ensemble with regime-aware adjustments.
    Returns ranked DataFrame with confidence intervals.
    """
    # Load models
    try:
        ensemble_clf, ensemble_reg = load_ensemble()
    except FileNotFoundError:
        print("  ⚠ Ensemble not found, falling back to standalone XGBoost.")
        ensemble_clf = None
        ensemble_reg = None

    xgb_model = load_xgb_model()

    if ensemble_clf is None and xgb_model is None:
        raise FileNotFoundError("No models found. Run 'india-stock train' first.")

    symbols = get_symbols()

    # Get current regime
    regime_data = get_current_regime()
    regime = regime_data["regime"]

    results = []

    for symbol in symbols:
        feat = get_latest_features(symbol)
        if feat is None:
            continue

        # Extract feature values
        feature_vals = []
        skip = False
        for col in DB_FEATURE_COLS:
            val = feat.get(col)
            if val is None:
                # Use 0.0 for missing fundamentals
                if col.startswith("fund_"):
                    feature_vals.append(0.0)
                else:
                    skip = True
                    break
            else:
                feature_vals.append(float(val))

        if skip:
            continue

        X = np.array([feature_vals])

        # Ensemble predictions
        if ensemble_clf is not None:
            prob = float(ensemble_clf.predict_proba(X)[0])
            pred_return = float(ensemble_reg.predict(X)[0])
            median, low, high = bootstrap_predictions(ensemble_reg, X)
            pred_low = float(low[0])
            pred_high = float(high[0])
        else:
            # Fallback to standalone XGBoost
            prob = float(xgb_model.predict_proba(X)[0][1])
            xgb_reg = joblib.load(os.path.join(MODEL_DIR, "xgb_regressor.joblib"))
            pred_return = float(xgb_reg.predict(X)[0])
            pred_low = pred_return - 0.03
            pred_high = pred_return + 0.03

        # Sentiment
        sentiment = get_latest_sentiment(symbol)

        # Momentum
        momentum = feat.get("return_20d", 0.0) or 0.0

        # Combined score
        final_score = (
            WEIGHT_MODEL * prob
            + WEIGHT_SENTIMENT * ((sentiment + 1) / 2)
            + WEIGHT_MOMENTUM * max(0, min(1, momentum + 0.5))
        )

        # Get latest price
        with get_db() as conn:
            last_price_row = conn.execute(
                "SELECT close FROM ohlcv WHERE symbol = ? ORDER BY date DESC LIMIT 1",
                (symbol,),
            ).fetchone()

        last_price = float(last_price_row["close"]) if last_price_row else 0.0

        # Apply regime adjustment
        adj_return, adj_low, adj_high = apply_regime_adjustment(
            pred_return, pred_low, pred_high, regime,
        )

        target_price = last_price * (1 + adj_return)
        target_low = last_price * (1 + adj_low)
        target_high = last_price * (1 + adj_high)

        # SHAP explainability (using standalone XGBoost)
        top_bullish = []
        top_bearish = []
        if xgb_model is not None:
            explainer = shap.TreeExplainer(xgb_model)
            shap_vals = explainer.shap_values(X)[0]
            contributions = list(zip(DB_FEATURE_COLS, shap_vals))
            sorted_contribs = sorted(contributions, key=lambda x: x[1], reverse=True)
            top_bullish = [
                {"feature": f, "impact": float(v)}
                for f, v in sorted_contribs if v > 0
            ][:3]
            top_bearish = [
                {"feature": f, "impact": float(v)}
                for f, v in reversed(sorted_contribs) if v < 0
            ][:3]

        results.append({
            "symbol": symbol,
            "name": get_symbol_name(symbol),
            "sector": get_symbol_sector(symbol),
            "model_score": round(prob, 4),
            "predicted_return": round(float(adj_return), 4),
            "pred_return_low": round(float(adj_low), 4),
            "pred_return_high": round(float(adj_high), 4),
            "target_price": round(target_price, 2),
            "target_low": round(target_low, 2),
            "target_high": round(target_high, 2),
            "last_price": round(last_price, 2),
            "sentiment": round(sentiment, 4),
            "momentum_20d": round(float(momentum) * 100, 2),
            "final_score": round(final_score, 4),
            "rsi": round(float(feat.get("rsi_14", 0) or 0), 2),
            "volatility_20": round(float(feat.get("volatility_20", 0) or 0), 4),
            "macro_nifty": float(feat.get("macro_nifty_drawdown", 0) or 0),
            "macro_vix": float(feat.get("macro_vix_percentile", 0) or 0),
            "regime": regime,
            "shap_bullish": top_bullish,
            "shap_bearish": top_bearish,
            "date": feat.get("date", ""),
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values(
        "final_score", ascending=False,
    ).reset_index(drop=True)
    return df


def store_signals(df: pd.DataFrame):
    """Store signal scores in the database."""
    if df.empty:
        return
    with get_db() as conn:
        for _, row in df.iterrows():
            conn.execute(
                """INSERT OR REPLACE INTO signals
                   (symbol, date, model_score, sentiment_score,
                    momentum_score, final_score, pred_return,
                    pred_return_low, pred_return_high, regime)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (row["symbol"], row["date"], row["model_score"],
                 row["sentiment"], row["momentum_20d"], row["final_score"],
                 row["predicted_return"], row["pred_return_low"],
                 row["pred_return_high"], row["regime"]),
            )


def run_scoring():
    """Score all stocks and print rankings."""
    print("━" * 60)
    print("📈 SCORING ALL STOCKS (V2 Ensemble)")
    print("━" * 60)

    df = score_all_stocks()
    if df.empty:
        print("  ✗ No stocks scored.")
        return df

    store_signals(df)

    regime = df.iloc[0]["regime"] if not df.empty else "normal"
    if regime == "crash":
        print("\n  🚨 MARKET REGIME: CRASH — All bullish forecasts suppressed!")
    elif regime == "stressed":
        print("\n  ⚠️  MARKET REGIME: STRESSED — Bullish forecasts dampened.")
    else:
        print("\n  ✅ MARKET REGIME: NORMAL")

    print(f"\n  Scored {len(df)} stocks. Top 10:\n")
    top = df.head(10)
    for i, (_, row) in enumerate(top.iterrows(), 1):
        ci = f"[{row['pred_return_low']*100:+.1f}%, {row['pred_return_high']*100:+.1f}%]"
        print(
            f"  {i:>2}. {row['name']:>30s} ({row['symbol']:>15s}) | "
            f"Score: {row['final_score']:.4f} | "
            f"Return: {row['predicted_return']*100:+.1f}% {ci}"
        )
    print()
    return df


if __name__ == "__main__":
    run_scoring()
