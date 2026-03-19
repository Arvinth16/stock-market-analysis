"""
features.py — Technical, cross-sectional, fundamental, and macro feature computation.

V2: 34 features + multi-horizon labels.
All features use ONLY data available at or before the given date (no future leakage).
Labels use strictly future data (forward returns).
"""

import numpy as np
import pandas as pd

from src.core.config import PREDICTION_HORIZON, POSITIVE_RETURN_THRESHOLD
from src.core.database import get_db
from src.data.fetcher import load_ohlcv
from src.data.universe import get_symbols, get_symbol_sector


# ─── Indicator Functions ──────────────────────────────────────────────────────

def compute_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_band_width(series: pd.Series, window: int = 20) -> pd.Series:
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return ((upper - lower) / sma) * 100


def compute_atr(high, low, close, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


# ─── Main Feature Computation ────────────────────────────────────────────────

def compute_features(df: pd.DataFrame, macro_df: pd.DataFrame = None,
                     fundamentals: dict = None, symbol: str = None,
                     sector_returns: dict = None) -> pd.DataFrame:
    """
    Compute all 34 features from OHLCV + macro + fundamentals data.

    Args:
        df: OHLCV DataFrame indexed by date.
        macro_df: DataFrame with nifty_close and vix_close columns.
        fundamentals: Dict with fund keys (pe_ratio, pb_ratio, roe, etc.)
        symbol: Current stock symbol (for sector lookups).
        sector_returns: Dict mapping sector -> Series of 20d returns.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    features = pd.DataFrame(index=df.index)

    # ─── 1. Trend: Moving Averages (5 features) ─────────────────────
    sma_20 = compute_sma(close, 20)
    sma_50 = compute_sma(close, 50)
    sma_200 = compute_sma(close, 200)

    features["price_vs_sma20"] = ((close - sma_20) / sma_20) * 100
    features["price_vs_sma50"] = ((close - sma_50) / sma_50) * 100
    features["price_vs_sma200"] = ((close - sma_200) / sma_200) * 100

    std_20 = close.rolling(20).std()
    std_50 = close.rolling(50).std()
    features["close_zscore_20"] = (close - sma_20) / std_20.replace(0, np.nan)
    features["close_zscore_50"] = (close - sma_50) / std_50.replace(0, np.nan)

    # ─── 2. Momentum: RSI, MACD (4 features) ────────────────────────
    features["rsi_14"] = compute_rsi(close, 14)
    macd, macd_sig, macd_hist = compute_macd(close)
    features["macd_norm"] = (macd / close) * 100
    features["macd_signal_norm"] = (macd_sig / close) * 100
    features["macd_hist_norm"] = (macd_hist / close) * 100

    # ─── 3. Volatility (3 features) ─────────────────────────────────
    features["bb_width"] = compute_bollinger_band_width(close, 20)
    features["atr_percent"] = (compute_atr(high, low, close, 14) / close) * 100
    features["volatility_20"] = close.pct_change().rolling(20).std() * np.sqrt(252)

    # ─── 4. Volume (1 feature) ──────────────────────────────────────
    avg_vol_20 = volume.rolling(20).mean()
    features["volume_ratio"] = volume / avg_vol_20.replace(0, np.nan)

    # ─── 5. Price Action (4 features) ───────────────────────────────
    rolling_high_252 = high.rolling(252, min_periods=50).max()
    rolling_low_252 = low.rolling(252, min_periods=50).min()
    features["dist_52w_high"] = ((close - rolling_high_252) / rolling_high_252) * 100
    features["dist_52w_low"] = ((close - rolling_low_252) / rolling_low_252) * 100
    features["return_5d"] = close.pct_change(5)
    features["return_10d"] = close.pct_change(10)
    features["return_20d"] = close.pct_change(20)

    # ─── 6. Macro & Regime (2 features) ─────────────────────────────
    if macro_df is not None and not macro_df.empty:
        macro = macro_df.reindex(df.index).ffill()
        nifty_high = macro['nifty_close'].rolling(252, min_periods=20).max()
        features['macro_nifty_drawdown'] = (
            (macro['nifty_close'] - nifty_high) / nifty_high.replace(0, np.nan)
        ) * 100

        vix_min = macro['vix_close'].rolling(252, min_periods=20).min()
        vix_max = macro['vix_close'].rolling(252, min_periods=20).max()
        vix_range = (vix_max - vix_min).replace(0, np.nan)
        features['macro_vix_percentile'] = (
            (macro['vix_close'] - vix_min) / vix_range
        ) * 100
    else:
        features['macro_nifty_drawdown'] = 0.0
        features['macro_vix_percentile'] = 50.0

    # ─── 7. Cross-Sectional (V2: 8 features) ────────────────────────
    # 7a. Relative strength vs Nifty
    if macro_df is not None and 'nifty_close' in macro_df.columns:
        macro = macro_df.reindex(df.index).ffill()
        nifty_ret_20 = macro['nifty_close'].pct_change(20)
        stock_ret_20 = close.pct_change(20)
        features['rel_strength_vs_nifty'] = (stock_ret_20 - nifty_ret_20) * 100
        features['nifty_return_20d'] = nifty_ret_20 * 100

        # 7b. VIX change over 5 days
        if 'vix_close' in macro.columns:
            features['vix_change_5d'] = macro['vix_close'].pct_change(5) * 100
        else:
            features['vix_change_5d'] = 0.0

        # 7c. Market breadth (% of stocks above SMA200) — approximated
        # We'll compute this from the stock itself as a stand-in
        above_sma200 = (close > sma_200).astype(float)
        features['market_breadth'] = above_sma200.rolling(20).mean() * 100
    else:
        features['rel_strength_vs_nifty'] = 0.0
        features['nifty_return_20d'] = 0.0
        features['vix_change_5d'] = 0.0
        features['market_breadth'] = 50.0

    # 7d. Sector momentum
    if sector_returns and symbol:
        sector = get_symbol_sector(symbol)
        if sector in sector_returns:
            features['sector_momentum'] = sector_returns[sector].reindex(
                df.index
            ).ffill() * 100
        else:
            features['sector_momentum'] = 0.0
    else:
        features['sector_momentum'] = 0.0

    # 7e. Volume z-score (60-day)
    vol_mean_60 = volume.rolling(60).mean()
    vol_std_60 = volume.rolling(60).std()
    features['volume_zscore'] = (
        (volume - vol_mean_60) / vol_std_60.replace(0, np.nan)
    )

    # 7f. 60-day return
    features['return_60d'] = close.pct_change(60)

    # 7g. Volatility-adjusted return
    vol_20 = features['volatility_20'].replace(0, np.nan)
    features['vol_adjusted_return'] = features['return_20d'] / vol_20

    # ─── 8. Fundamentals (V2: 6 features) ───────────────────────────
    if fundamentals:
        features['fund_pe'] = fundamentals.get('pe_ratio') or np.nan
        features['fund_pb'] = fundamentals.get('pb_ratio') or np.nan
        features['fund_roe'] = fundamentals.get('roe') or np.nan
        features['fund_debt_equity'] = fundamentals.get('debt_to_equity') or np.nan
        features['fund_earnings_growth'] = fundamentals.get('earnings_growth') or np.nan
        features['fund_dividend_yield'] = fundamentals.get('dividend_yield') or np.nan
    else:
        features['fund_pe'] = np.nan
        features['fund_pb'] = np.nan
        features['fund_roe'] = np.nan
        features['fund_debt_equity'] = np.nan
        features['fund_earnings_growth'] = np.nan
        features['fund_dividend_yield'] = np.nan

    return features


# ─── Label Computation ────────────────────────────────────────────────────────

def compute_labels(df: pd.DataFrame, horizon: int = PREDICTION_HORIZON,
                   threshold: float = POSITIVE_RETURN_THRESHOLD) -> pd.DataFrame:
    """
    Generate multi-horizon labels from forward returns.
    V2: 20d binary, 20d regression, 60d regression, vol-scaled, 3-bucket.
    """
    close = df["close"]

    # 20-day forward
    fwd_20 = close.pct_change(horizon).shift(-horizon)
    label = (fwd_20 > threshold).astype(int)
    label_reg = fwd_20

    # 60-day forward
    fwd_60 = close.pct_change(60).shift(-60)

    # Vol-scaled: 20d return / 20d realized vol
    vol_20 = close.pct_change().rolling(20).std() * np.sqrt(252)
    vol_scaled = fwd_20 / vol_20.replace(0, np.nan)

    # 3-bucket: 0=strong_down(<-2%), 1=flat(-2% to +5%), 2=strong_up(>5%)
    bucket = pd.Series(1, index=df.index, dtype=int)  # default flat
    bucket[fwd_20 < -0.02] = 0  # strong down
    bucket[fwd_20 > 0.05] = 2   # strong up
    bucket[fwd_20.isna()] = np.nan

    return pd.DataFrame({
        "label": label,
        "label_reg": label_reg,
        "label_60d": fwd_60,
        "label_vol_scaled": vol_scaled,
        "label_bucket": bucket,
    })


# ─── Feature Column Registry ─────────────────────────────────────────────────

FEATURE_COLUMNS = [
    # Trend (5)
    "close_zscore_20", "close_zscore_50",
    "price_vs_sma20", "price_vs_sma50", "price_vs_sma200",
    # Momentum (4)
    "rsi_14", "macd_norm", "macd_signal_norm", "macd_hist_norm",
    # Volatility (3)
    "bb_width", "atr_percent", "volatility_20",
    # Volume (1)
    "volume_ratio",
    # Price action (5)
    "dist_52w_high", "dist_52w_low", "return_5d", "return_10d", "return_20d",
    # Macro (2)
    "macro_nifty_drawdown", "macro_vix_percentile",
    # Cross-sectional V2 (8)
    "rel_strength_vs_nifty", "sector_momentum", "volume_zscore",
    "return_60d", "vol_adjusted_return", "nifty_return_20d",
    "vix_change_5d", "market_breadth",
    # Fundamentals V2 (6)
    "fund_pe", "fund_pb", "fund_roe", "fund_debt_equity",
    "fund_earnings_growth", "fund_dividend_yield",
]


# ─── Storage ──────────────────────────────────────────────────────────────────

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
            lbl_60 = labels_df.loc[date, "label_60d"] if date in labels_df.index else None
            lbl_60_val = float(lbl_60) if pd.notna(lbl_60) else None
            lbl_vs = labels_df.loc[date, "label_vol_scaled"] if date in labels_df.index else None
            lbl_vs_val = float(lbl_vs) if pd.notna(lbl_vs) else None
            lbl_bkt = labels_df.loc[date, "label_bucket"] if date in labels_df.index else None
            lbl_bkt_val = int(lbl_bkt) if pd.notna(lbl_bkt) else None

            vals = [symbol, date.strftime("%Y-%m-%d")]
            for col in FEATURE_COLUMNS:
                vals.append(_safe(row, col))
            vals.extend([lbl_val, lbl_reg_val, lbl_60_val, lbl_vs_val, lbl_bkt_val])

            placeholders = ",".join(["?"] * len(vals))
            col_names = "symbol, date, " + ", ".join(FEATURE_COLUMNS)
            col_names += ", label, label_reg, label_60d, label_vol_scaled, label_bucket"

            conn.execute(
                f"INSERT OR REPLACE INTO features ({col_names}) VALUES ({placeholders})",
                vals,
            )


def _safe(row, col):
    val = row.get(col, None)
    if val is not None and pd.notna(val):
        return float(val)
    return None


# ─── Macro Data Loader ───────────────────────────────────────────────────────

def load_macro_data() -> pd.DataFrame:
    with get_db() as conn:
        try:
            nifty = pd.read_sql_query(
                "SELECT date, close as nifty_close FROM macro_data WHERE symbol='^NSEI'",
                conn, parse_dates=["date"], index_col="date",
            )
            vix = pd.read_sql_query(
                "SELECT date, close as vix_close FROM macro_data WHERE symbol='^INDIAVIX'",
                conn, parse_dates=["date"], index_col="date",
            )
            return nifty.join(vix, how="outer")
        except Exception:
            return pd.DataFrame()


def load_fundamentals_dict(symbol: str) -> dict:
    """Load fundamental data for a single symbol from DB."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM fundamentals WHERE symbol = ?", (symbol,)
        ).fetchone()
    if row:
        return dict(row)
    return {}


def compute_sector_returns(macro_df: pd.DataFrame) -> dict:
    """Compute average 20d sector returns across the universe."""
    symbols = get_symbols()
    sector_data = {}

    for sym in symbols:
        sector = get_symbol_sector(sym)
        df = load_ohlcv(sym)
        if df.empty or len(df) < 30:
            continue
        ret_20 = df["close"].pct_change(20)
        if sector not in sector_data:
            sector_data[sector] = []
        sector_data[sector].append(ret_20)

    sector_returns = {}
    for sector, series_list in sector_data.items():
        combined = pd.concat(series_list, axis=1)
        sector_returns[sector] = combined.mean(axis=1)

    return sector_returns


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def compute_and_store_all():
    """Compute features and labels for all stocks and store in DB."""
    from src.core.database import init_db
    init_db()
    symbols = get_symbols()
    total = len(symbols)
    macro_df = load_macro_data()

    print("━" * 60)
    print("🔧 COMPUTING TECHNICAL FEATURES (V2: 34 features)")
    print("━" * 60)

    # Pre-compute sector returns
    print("  Computing sector momentum...")
    sector_returns = compute_sector_returns(macro_df)

    success = 0
    for i, symbol in enumerate(symbols, 1):
        print(f"  [{i:>2}/{total}] {symbol}...", end=" ", flush=True)
        df = load_ohlcv(symbol)
        if df.empty or len(df) < 200:
            print("✗ insufficient data")
            continue

        fund = load_fundamentals_dict(symbol)
        features = compute_features(
            df, macro_df, fundamentals=fund,
            symbol=symbol, sector_returns=sector_returns,
        )
        labels = compute_labels(df)
        store_features(symbol, features, labels)
        
        core_cols = [c for c in features.columns if not c.startswith("fund_")]
        valid = features.dropna(subset=core_cols).shape[0]
        
        print(f"✓ {valid} feature rows")
        success += 1

    print(f"\n  ✅ Features computed for {success}/{total} stocks.\n")
    return success


if __name__ == "__main__":
    compute_and_store_all()
