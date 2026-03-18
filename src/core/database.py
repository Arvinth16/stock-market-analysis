"""
database.py — SQLite database layer for storing stock data, features, and signals.
"""

import sqlite3
from contextlib import contextmanager
from src.core.config import DB_PATH


def get_connection() -> sqlite3.Connection:
    """Get a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create all required tables if they don't exist."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol    TEXT NOT NULL,
                date      TEXT NOT NULL,
                open      REAL,
                high      REAL,
                low       REAL,
                close     REAL,
                volume    INTEGER,
                PRIMARY KEY (symbol, date)
            );

            CREATE TABLE IF NOT EXISTS features (
                symbol    TEXT NOT NULL,
                date      TEXT NOT NULL,
                sma_20    REAL,
                sma_50    REAL,
                sma_200   REAL,
                rsi_14    REAL,
                macd      REAL,
                macd_signal REAL,
                macd_hist REAL,
                bb_width  REAL,
                atr_14    REAL,
                volatility_20 REAL,
                volume_ratio  REAL,
                dist_52w_high REAL,
                dist_52w_low  REAL,
                return_5d  REAL,
                return_10d REAL,
                return_20d REAL,
                label      INTEGER,
                PRIMARY KEY (symbol, date)
            );

            CREATE TABLE IF NOT EXISTS signals (
                symbol    TEXT NOT NULL,
                date      TEXT NOT NULL,
                model_score REAL,
                sentiment_score REAL,
                momentum_score REAL,
                final_score REAL,
                PRIMARY KEY (symbol, date)
            );

            CREATE TABLE IF NOT EXISTS news (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol    TEXT NOT NULL,
                date      TEXT NOT NULL,
                title     TEXT,
                source    TEXT,
                url       TEXT,
                sentiment REAL,
                sentiment_label TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol ON ohlcv(symbol);
            CREATE INDEX IF NOT EXISTS idx_features_symbol ON features(symbol);
            CREATE INDEX IF NOT EXISTS idx_signals_date ON signals(date);
            CREATE INDEX IF NOT EXISTS idx_news_symbol ON news(symbol);
        """)


if __name__ == "__main__":
    init_db()
    print(f"✅ Database initialized at {DB_PATH}")
