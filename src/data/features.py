"""
features.py — Technical indicator computation and label generation.

All features use ONLY data available at or before the given date (no future leakage).
Labels use strictly future data (forward returns).
"""

import numpy as np
import pandas as pd

from src.core.config import PREDICTION_HORIZON, POSITIVE_RETURN_THRESHOLD
from src.core.database import get_db
from src.data.fetcher import load_ohlcv
from src.data.universe import get_symbols


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=window, min_periods=window).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series):
    """MACD line, signal line, and histogram."""
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_band_width(series: pd.Series, window: int = 20) -> pd.Series:
    """Bollinger Band width as percentage of middle band."""
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return ((upper - lower) / sma) * 100


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators from OHLCV data.

    Args:
        df: DataFrame with columns [open, high, low, close, volume], indexed by date.

    Returns:
        DataFrame with computed feature columns, indexed by date.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    features = pd.DataFrame(index=df.index)

    # ─── Trend: Moving Averages ───────────────────────────────────────
    sma_20 = compute_sma(close, 20)
    sma_50 = compute_sma(close, 50)
    sma_200 = compute_sma(close, 200)

    # Price relative to SMAs (%)
    features["price_vs_sma20"] = ((close - sma_20) / sma_20) * 100
    features["price_vs_sma50"] = ((close - sma_50) / sma_50) * 100
    features["price_vs_sma200"] = ((close - sma_200) / sma_200) * 100

    # Z-scores
    std_20 = close.rolling(20).std()
    std_50 = close.rolling(50).std()
    features["close_zscore_20"] = (close - sma_20) / std_20.replace(0, np.nan)
    features["close_zscore_50"] = (close - sma_50) / std_50.replace(0, np.nan)

    # ─── Momentum: RSI, MACD ─────────────────────────────────────────
    features["rsi_14"] = compute_rsi(close, 14)
    macd, macd_sig, macd_hist = compute_macd(close)
    
    # Normalize MACD by price so it works across different price scales
    features["macd_norm"] = (macd / close) * 100
    features["macd_signal_norm"] = (macd_sig / close) * 100
    features["macd_hist_norm"] = (macd_hist / close) * 100

    # ─── Volatility ──────────────────────────────────────────────────
    features["bb_width"] = compute_bollinger_band_width(close, 20)
    atr_14 = compute_atr(high, low, close, 14)
    features["atr_percent"] = (atr_14 / close) * 100
    features["volatility_20"] = close.pct_change().rolling(20).std() * np.sqrt(252)  # annualized

    # ─── Volume ──────────────────────────────────────────────────────
    avg_vol_20 = volume.rolling(20).mean()
    features["volume_ratio"] = volume / avg_vol_20.replace(0, np.nan)

    # ─── Price Action ────────────────────────────────────────────────
    rolling_high_252 = high.rolling(252, min_periods=50).max()
    rolling_low_252 = low.rolling(252, min_periods=50).min()
    features["dist_52w_high"] = ((close - rolling_high_252) / rolling_high_252) * 100
    features["dist_52w_low"] = ((close - rolling_low_252) / rolling_low_252) * 100

    # Recent returns (backward-looking — no leakage)
    features["return_5d"] = close.pct_change(5)
    features["return_10d"] = close.pct_change(10)
    features["return_20d"] = close.pct_change(20)

    return features


def compute_labels(df: pd.DataFrame, horizon: int = PREDICTION_HORIZON,
                   threshold: float = POSITIVE_RETURN_THRESHOLD) -> pd.Series:
    """
    Generate binary labels from forward returns.

    Label = 1 if the forward return over `horizon` days > `threshold`, else 0.
    This uses FUTURE data and is only for training labels.
    """
    forward_return = df["close"].pct_change(horizon).shift(-horizon)
    label = (forward_return > threshold).astype(int)
    label_reg = forward_return  # The exact continuous forward return
    
    return pd.DataFrame({
        "label": label,
        "label_reg": label_reg
    })


def store_features(symbol: str, features_df: pd.DataFrame, labels_df: pd.DataFrame):
    """Store computed features and labels to the database."""
    if features_df.empty:
        return
    with get_db() as conn:
        for date in features_df.index:
            row = features_df.loc[date]
            
            lbl = labels_df.loc[date, "label"] if date in labels_df.index else None
            lbl_val = int(lbl) if pd.notna(lbl) else None
            
            lbl_reg = labels_df.loc[date, "label_reg"] if date in labels_df.index else None
            lbl_reg_val = float(lbl_reg) if pd.notna(lbl_reg) else None

            conn.execute(
                """INSERT OR REPLACE INTO features
                   (symbol, date, close_zscore_20, close_zscore_50,
                    price_vs_sma20, price_vs_sma50, price_vs_sma200,
                    rsi_14, macd_norm, macd_signal_norm, macd_hist_norm,
                    bb_width, atr_percent, volatility_20, volume_ratio,
                    dist_52w_high, dist_52w_low, return_5d, return_10d, return_20d, label, label_reg)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (symbol, date.strftime("%Y-%m-%d"),
                 _safe(row, "close_zscore_20"), _safe(row, "close_zscore_50"),
                 _safe(row, "price_vs_sma20"), _safe(row, "price_vs_sma50"), _safe(row, "price_vs_sma200"),
                 _safe(row, "rsi_14"), _safe(row, "macd_norm"), _safe(row, "macd_signal_norm"),
                 _safe(row, "macd_hist_norm"), _safe(row, "bb_width"), _safe(row, "atr_percent"),
                 _safe(row, "volatility_20"), _safe(row, "volume_ratio"),
                 _safe(row, "dist_52w_high"), _safe(row, "dist_52w_low"),
                 _safe(row, "return_5d"), _safe(row, "return_10d"), _safe(row, "return_20d"),
                 lbl_val, lbl_reg_val),
            )


def _safe(row, col):
    """Safely get a value, returning None for NaN."""
    val = row.get(col, None)
    if val is not None and pd.notna(val):
        return float(val)
    return None


FEATURE_COLUMNS = [
    "close_zscore_20", "close_zscore_50",
    "price_vs_sma20", "price_vs_sma50", "price_vs_sma200",
    "rsi_14", "macd_norm", "macd_signal_norm", "macd_hist_norm",
    "bb_width", "atr_percent", "volatility_20", "volume_ratio",
    "dist_52w_high", "dist_52w_low",
    "return_5d", "return_10d", "return_20d",
]



def compute_and_store_all():
    """Compute features and labels for all stocks and store in DB."""
    symbols = get_symbols()
    total = len(symbols)

    print("━" * 60)
    print("🔧 COMPUTING TECHNICAL FEATURES")
    print("━" * 60)

    success = 0
    for i, symbol in enumerate(symbols, 1):
        print(f"  [{i:>2}/{total}] {symbol}...", end=" ", flush=True)
        df = load_ohlcv(symbol)
        if df.empty or len(df) < 200:
            print("✗ insufficient data")
            continue

        features = compute_features(df)
        labels = compute_labels(df)
        store_features(symbol, features, labels)
        valid = features.dropna().shape[0]
        print(f"✓ {valid} feature rows")
        success += 1

    print(f"\n  ✅ Features computed for {success}/{total} stocks.\n")
    return success


if __name__ == "__main__":
    compute_and_store_all()
