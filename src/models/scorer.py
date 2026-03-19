"""
scorer.py — Score all symbols for a given date using the trained model.
"""

import os
import pandas as pd
import numpy as np
import joblib

from src.core.config import MODEL_DIR, WEIGHT_MODEL, WEIGHT_SENTIMENT, WEIGHT_MOMENTUM
from src.core.database import get_db
from src.data.universe import get_symbols, get_symbol_name, get_symbol_sector
from src.models.trainer import DB_FEATURE_COLS


def load_model():
    """Load the saved XGBoost model."""
    model_path = os.path.join(MODEL_DIR, "xgb_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}. Run trainer first.")
    return joblib.load(model_path)


def load_regressor():
    """Load the saved XGBoost regressor model."""
    model_path = os.path.join(MODEL_DIR, "xgb_regressor.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No regressor found at {model_path}. Run trainer first.")
    return joblib.load(model_path)


def get_latest_features(symbol: str) -> dict | None:
    """Get the most recent feature row for a symbol from the DB."""
    with get_db() as conn:
        row = conn.execute(
            """SELECT * FROM features WHERE symbol = ?
               ORDER BY date DESC LIMIT 1""",
            (symbol,),
        ).fetchone()
    if row is None:
        return None
    return dict(row)


def get_latest_sentiment(symbol: str) -> float:
    """Get average sentiment from recent news for symbol."""
    with get_db() as conn:
        result = conn.execute(
            """SELECT AVG(sentiment) as avg_sent FROM news
               WHERE symbol = ? AND date >= date('now', '-7 days')""",
            (symbol,),
        ).fetchone()
    if result and result["avg_sent"] is not None:
        return float(result["avg_sent"])
    return 0.0  # neutral default


def score_all_stocks() -> pd.DataFrame:
    """
    Score all stocks in the universe and return a ranked DataFrame.
    """
    model = load_model()
    regressor = load_regressor()
    symbols = get_symbols()

    results = []

    for symbol in symbols:
        feat = get_latest_features(symbol)
        if feat is None:
            continue

        # Extract feature values in correct order
        feature_vals = []
        skip = False
        for col in DB_FEATURE_COLS:
            val = feat.get(col)
            if val is None:
                skip = True
                break
            feature_vals.append(float(val))

        if skip:
            continue

        # Model prediction
        X = np.array([feature_vals])
        prob = model.predict_proba(X)[0][1]  # Probability of positive class
        pred_return = regressor.predict(X)[0] # Predicted percentage return

        # Sentiment
        sentiment = get_latest_sentiment(symbol)

        # Momentum (20-day return)
        momentum = feat.get("return_20d", 0.0) or 0.0

        # Combined score
        final_score = (
            WEIGHT_MODEL * prob +
            WEIGHT_SENTIMENT * ((sentiment + 1) / 2) +  # normalize -1..1 to 0..1
            WEIGHT_MOMENTUM * max(0, min(1, momentum + 0.5))  # rough normalization
        )
        
        # Get latest price to calculate target
        with get_db() as conn:
            last_price_row = conn.execute(
                "SELECT close FROM ohlcv WHERE symbol = ? ORDER BY date DESC LIMIT 1",
                (symbol,)
            ).fetchone()
            
        last_price = float(last_price_row["close"]) if last_price_row else 0.0
        target_price = last_price * (1 + pred_return)

        # SHAP Explainability (Why does the AI think this?)
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)[0]
        
        contributions = list(zip(DB_FEATURE_COLS, shap_vals))
        sorted_contribs = sorted(contributions, key=lambda x: x[1], reverse=True)
        
        top_bullish = [{"feature": f, "impact": v} for f, v in sorted_contribs if v > 0][:3]
        top_bearish = [{"feature": f, "impact": v} for f, v in reversed(sorted_contribs) if v < 0][:3]

        results.append({
            "symbol": symbol,
            "name": get_symbol_name(symbol),
            "sector": get_symbol_sector(symbol),
            "model_score": round(prob, 4),
            "predicted_return": round(float(pred_return), 4),
            "target_price": round(target_price, 2),
            "last_price": round(last_price, 2),
            "sentiment": round(sentiment, 4),
            "momentum_20d": round(float(momentum) * 100, 2),
            "final_score": round(final_score, 4),
            "rsi": round(float(feat.get("rsi_14", 0) or 0), 2),
            "volatility_20": round(float(feat.get("volatility_20", 0) or 0), 4),
            "macro_nifty": float(feat.get("macro_nifty_drawdown", 0) or 0),
            "macro_vix": float(feat.get("macro_vix_percentile", 0) or 0),
            "shap_bullish": top_bullish,
            "shap_bearish": top_bearish,
            "date": feat.get("date", ""),
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values("final_score", ascending=False).reset_index(drop=True)
    return df


def store_signals(df: pd.DataFrame):
    """Store signal scores in the database."""
    if df.empty:
        return
    with get_db() as conn:
        for _, row in df.iterrows():
            conn.execute(
                """INSERT OR REPLACE INTO signals
                   (symbol, date, model_score, sentiment_score, momentum_score, final_score)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (row["symbol"], row["date"], row["model_score"],
                 row["sentiment"], row["momentum_20d"], row["final_score"]),
            )


def run_scoring():
    """Score all stocks and print rankings."""
    print("━" * 60)
    print("📈 SCORING ALL STOCKS")
    print("━" * 60)

    df = score_all_stocks()
    if df.empty:
        print("  ✗ No stocks scored. Ensure data and model are ready.")
        return df

    store_signals(df)

    print(f"\n  ✅ Scored {len(df)} stocks. Top 10:\n")
    top = df.head(10)
    for i, (_, row) in enumerate(top.iterrows(), 1):
        print(f"  {i:>2}. {row['name']:>30s} ({row['symbol']:>15s}) | "
              f"Score: {row['final_score']:.4f} | "
              f"Model: {row['model_score']:.3f} | "
              f"Sent: {row['sentiment']:+.3f} | "
              f"Mom: {row['momentum_20d']:+.1f}%")
    print()
    return df


if __name__ == "__main__":
    run_scoring()
