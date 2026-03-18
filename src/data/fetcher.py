"""
fetcher.py — Download historical OHLCV data for Indian stocks from Yahoo Finance.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from src.core.config import HISTORICAL_YEARS
from src.core.database import get_db, init_db
from src.data.universe import get_symbols, get_symbol_name


def fetch_ohlcv(symbol: str, years: int = HISTORICAL_YEARS) -> pd.DataFrame:
    """
    Download daily OHLCV data for a single NSE ticker.

    Args:
        symbol: Yahoo Finance ticker (e.g. 'RELIANCE.NS').
        years: Number of years of history.

    Returns:
        DataFrame with columns [open, high, low, close, volume] indexed by date.
    """
    try:
        end = datetime.now()
        start = end - timedelta(days=years * 365)
        stock = yf.Ticker(symbol)
        df = stock.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))

        if df.empty:
            print(f"  ⚠  No data for {symbol}")
            return pd.DataFrame()

        # Normalize columns
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "date"
        df.dropna(inplace=True)
        return df

    except Exception as e:
        print(f"  ✗  Error fetching {symbol}: {e}")
        return pd.DataFrame()


def store_ohlcv(symbol: str, df: pd.DataFrame):
    """Insert or replace OHLCV rows into the database."""
    if df.empty:
        return
    with get_db() as conn:
        for date, row in df.iterrows():
            conn.execute(
                """INSERT OR REPLACE INTO ohlcv (symbol, date, open, high, low, close, volume)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (symbol, date.strftime("%Y-%m-%d"),
                 row["open"], row["high"], row["low"], row["close"], int(row["volume"])),
            )


def load_ohlcv(symbol: str) -> pd.DataFrame:
    """Load OHLCV data for a symbol from the database."""
    with get_db() as conn:
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume FROM ohlcv WHERE symbol = ? ORDER BY date",
            conn, params=(symbol,), parse_dates=["date"], index_col="date",
        )
    return df


def fetch_and_store_all():
    """Download OHLCV for every stock in the universe and store in DB."""
    init_db()
    symbols = get_symbols()
    total = len(symbols)

    print("━" * 60)
    print("📊 DOWNLOADING HISTORICAL PRICE DATA")
    print("━" * 60)

    success = 0
    for i, symbol in enumerate(symbols, 1):
        name = get_symbol_name(symbol)
        print(f"  [{i:>2}/{total}] {name} ({symbol})...", end=" ", flush=True)
        df = fetch_ohlcv(symbol)
        if not df.empty:
            store_ohlcv(symbol, df)
            print(f"✓ {len(df)} rows")
            success += 1
        else:
            print("✗ skipped")

    print(f"\n  ✅ Loaded {success}/{total} stocks into database.\n")
    return success


if __name__ == "__main__":
    fetch_and_store_all()
