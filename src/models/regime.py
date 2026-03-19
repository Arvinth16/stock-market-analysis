"""
regime.py — Market regime classifier: Normal / Stressed / Crash.

Uses Nifty drawdown, VIX levels, VIX acceleration, and market breadth
to classify the current market state and adjust model behavior.
"""

from src.core.database import get_db
from src.data.features import load_macro_data


# Regime thresholds (calibrated on Indian market history)
REGIME_THRESHOLDS = {
    "crash": {
        "nifty_drawdown": -10.0,    # >10% drawdown from peak
        "vix_percentile": 80.0,     # VIX in top 20% of range
    },
    "stressed": {
        "nifty_drawdown": -5.0,     # >5% drawdown
        "vix_percentile": 60.0,     # VIX elevated
    },
}


def classify_regime(nifty_drawdown: float, vix_percentile: float,
                    vix_change_5d: float = 0.0,
                    market_breadth: float = 50.0) -> tuple:
    """
    Classify current market regime.

    Returns: (regime_label, confidence)
        regime_label: 'crash', 'stressed', or 'normal'
        confidence: 0.0 to 1.0
    """
    score = 0.0

    # Drawdown component
    if nifty_drawdown < REGIME_THRESHOLDS["crash"]["nifty_drawdown"]:
        score += 0.4
    elif nifty_drawdown < REGIME_THRESHOLDS["stressed"]["nifty_drawdown"]:
        score += 0.2

    # VIX component
    if vix_percentile > REGIME_THRESHOLDS["crash"]["vix_percentile"]:
        score += 0.3
    elif vix_percentile > REGIME_THRESHOLDS["stressed"]["vix_percentile"]:
        score += 0.15

    # VIX acceleration (fast-rising VIX)
    if vix_change_5d > 20:
        score += 0.15
    elif vix_change_5d > 10:
        score += 0.08

    # Market breadth (low breadth = most stocks falling)
    if market_breadth < 30:
        score += 0.15
    elif market_breadth < 50:
        score += 0.07

    # Classify
    if score >= 0.6:
        return "crash", min(score, 1.0)
    elif score >= 0.3:
        return "stressed", score
    else:
        return "normal", 1.0 - score


def get_current_regime() -> dict:
    """
    Compute the current market regime from latest macro data.

    Returns: dict with regime, confidence, and underlying metrics.
    """
    macro_df = load_macro_data()
    if macro_df.empty:
        return {
            "regime": "normal",
            "confidence": 0.5,
            "nifty_drawdown": 0.0,
            "vix_level": 0.0,
            "vix_percentile": 50.0,
            "vix_change_5d": 0.0,
            "breadth": 50.0,
        }

    # Get latest values
    nifty = macro_df['nifty_close'].dropna()
    vix = macro_df['vix_close'].dropna()

    if nifty.empty or vix.empty:
        return {
            "regime": "normal", "confidence": 0.5,
            "nifty_drawdown": 0.0, "vix_level": 0.0,
            "vix_percentile": 50.0, "vix_change_5d": 0.0,
            "breadth": 50.0,
        }

    # Nifty drawdown from 252-day peak
    nifty_peak = nifty.rolling(252, min_periods=20).max().iloc[-1]
    nifty_current = nifty.iloc[-1]
    nifty_drawdown = ((nifty_current - nifty_peak) / nifty_peak) * 100

    # VIX percentile (252-day range)
    vix_min = vix.rolling(252, min_periods=20).min().iloc[-1]
    vix_max = vix.rolling(252, min_periods=20).max().iloc[-1]
    vix_range = vix_max - vix_min
    vix_current = vix.iloc[-1]
    vix_pct = ((vix_current - vix_min) / vix_range * 100) if vix_range > 0 else 50.0

    # VIX 5-day change
    vix_5d_ago = vix.iloc[-6] if len(vix) > 5 else vix.iloc[0]
    vix_change = ((vix_current - vix_5d_ago) / vix_5d_ago * 100) if vix_5d_ago > 0 else 0.0

    regime, confidence = classify_regime(nifty_drawdown, vix_pct, vix_change)

    result = {
        "regime": regime,
        "confidence": round(confidence, 3),
        "nifty_drawdown": round(nifty_drawdown, 2),
        "vix_level": round(vix_current, 2),
        "vix_percentile": round(vix_pct, 1),
        "vix_change_5d": round(vix_change, 1),
        "breadth": 50.0,  # Placeholder — computed during scoring
    }

    # Store to history
    store_regime(result)

    return result


def store_regime(regime_data: dict):
    """Store regime classification to DB."""
    with get_db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO regime_history
               (date, regime, confidence, nifty_drawdown, vix_level, breadth)
               VALUES (date('now'), ?, ?, ?, ?, ?)""",
            (regime_data["regime"], regime_data["confidence"],
             regime_data["nifty_drawdown"], regime_data["vix_level"],
             regime_data.get("breadth", 50.0)),
        )


def apply_regime_adjustment(predicted_return: float, confidence_low: float,
                            confidence_high: float,
                            regime: str) -> tuple:
    """
    Adjust predictions based on market regime.
    In crash/stressed regimes: dampen bullish forecasts, widen intervals.
    """
    if regime == "crash":
        if predicted_return > 0:
            predicted_return *= 0.3   # Heavy suppression of bullish calls
        else:
            predicted_return *= 1.3   # Amplify bearish calls
        width = confidence_high - confidence_low
        confidence_low -= width * 0.5
        confidence_high += width * 0.3

    elif regime == "stressed":
        if predicted_return > 0:
            predicted_return *= 0.6
        width = confidence_high - confidence_low
        confidence_low -= width * 0.3
        confidence_high += width * 0.2

    return predicted_return, confidence_low, confidence_high
