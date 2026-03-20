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

            CREATE TABLE IF NOT EXISTS macro_data (
                symbol    TEXT NOT NULL,
                date      TEXT NOT NULL,
                close     REAL,
                PRIMARY KEY (symbol, date)
            );

            CREATE TABLE IF NOT EXISTS fundamentals (
                symbol    TEXT NOT NULL,
                latest_date TEXT NOT NULL,
                pe_ratio  REAL,
                forward_pe REAL,
                pb_ratio  REAL,
                roe       REAL,
                roa       REAL,
                debt_to_equity REAL,
                revenue_growth REAL,
                earnings_growth REAL,
                profit_margin REAL,
                dividend_yield REAL,
                PRIMARY KEY (symbol)
            );

            CREATE TABLE IF NOT EXISTS features (
                symbol    TEXT NOT NULL,
                date      TEXT NOT NULL,
                -- Trend
                close_zscore_20 REAL,
                close_zscore_50 REAL,
                price_vs_sma20  REAL,
                price_vs_sma50  REAL,
                price_vs_sma200 REAL,
                -- Momentum
                rsi_14    REAL,
                macd_norm REAL,
                macd_signal_norm REAL,
                macd_hist_norm   REAL,
                -- Volatility
                bb_width  REAL,
                atr_percent REAL,
                volatility_20 REAL,
                -- Volume
                volume_ratio  REAL,
                -- Price action
                dist_52w_high REAL,
                dist_52w_low  REAL,
                return_5d  REAL,
                return_10d REAL,
                return_20d REAL,
                -- Macro
                macro_nifty_drawdown REAL,
                macro_vix_percentile REAL,
                -- Cross-sectional (V2)
                rel_strength_vs_nifty REAL,
                sector_momentum REAL,
                volume_zscore REAL,
                return_60d REAL,
                vol_adjusted_return REAL,
                nifty_return_20d REAL,
                vix_change_5d REAL,
                market_breadth REAL,
                -- Fundamentals (V2)
                fund_pe REAL,
                fund_pb REAL,
                fund_roe REAL,
                fund_debt_equity REAL,
                fund_earnings_growth REAL,
                fund_dividend_yield REAL,
                -- Labels
                label      INTEGER,
                label_reg  REAL,
                label_60d  REAL,
                label_vol_scaled REAL,
                label_bucket INTEGER,
                PRIMARY KEY (symbol, date)
            );

            CREATE TABLE IF NOT EXISTS signals (
                symbol    TEXT NOT NULL,
                date      TEXT NOT NULL,
                model_score REAL,
                sentiment_score REAL,
                momentum_score REAL,
                final_score REAL,
                pred_return REAL,
                pred_return_low REAL,
                pred_return_high REAL,
                regime TEXT,
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

            CREATE TABLE IF NOT EXISTS discoveries (
                symbol    TEXT NOT NULL,
                date      TEXT NOT NULL,
                source    TEXT,
                reason    TEXT,
                sentiment REAL,
                momentum  REAL,
                onboarded INTEGER DEFAULT 0,
                PRIMARY KEY (symbol, date)
            );

            CREATE TABLE IF NOT EXISTS regime_history (
                date      TEXT NOT NULL,
                regime    TEXT NOT NULL,
                confidence REAL,
                nifty_drawdown REAL,
                vix_level REAL,
                breadth   REAL,
                PRIMARY KEY (date)
            );

            CREATE TABLE IF NOT EXISTS strategy_performance (
                date      TEXT NOT NULL,
                regime    TEXT NOT NULL,
                hit_rate  REAL,
                mae       REAL,
                pnl       REAL,
                benchmark_pnl REAL,
                PRIMARY KEY (date)
            );

            CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol ON ohlcv(symbol);
            CREATE INDEX IF NOT EXISTS idx_features_symbol ON features(symbol);
            CREATE INDEX IF NOT EXISTS idx_signals_date ON signals(date);
            CREATE INDEX IF NOT EXISTS idx_news_symbol ON news(symbol);
            CREATE INDEX IF NOT EXISTS idx_discoveries_date ON discoveries(date);
        """)

        # Execute Auto-Migrations for V2 and V3 columns
        features_columns = [row["name"] for row in conn.execute("PRAGMA table_info(features)").fetchall()]
        signals_columns = [row["name"] for row in conn.execute("PRAGMA table_info(signals)").fetchall()]

        features_migrations = {
            "rel_strength_vs_nifty": "REAL",
            "sector_momentum": "REAL",
            "volume_zscore": "REAL",
            "return_60d": "REAL",
            "vol_adjusted_return": "REAL",
            "nifty_return_20d": "REAL",
            "vix_change_5d": "REAL",
            "market_breadth": "REAL",
            "fund_pe": "REAL",
            "fund_pb": "REAL",
            "fund_roe": "REAL",
            "fund_debt_equity": "REAL",
            "fund_earnings_growth": "REAL",
            "fund_dividend_yield": "REAL",
            "label_60d": "REAL",
            "label_vol_scaled": "REAL",
            "label_bucket": "INTEGER",
        }

        signals_migrations = {
            "pred_return": "REAL",
            "pred_return_low": "REAL",
            "pred_return_high": "REAL",
            "regime": "TEXT",
            "model_breakdown": "TEXT",  # V3
            "shap_top_features": "TEXT", # V3
        }

        for col, dtype in features_migrations.items():
            if col not in features_columns:
                try:
                    conn.execute(f"ALTER TABLE features ADD COLUMN {col} {dtype}")
                except sqlite3.OperationalError:
                    pass

        for col, dtype in signals_migrations.items():
            if col not in signals_columns:
                try:
                    conn.execute(f"ALTER TABLE signals ADD COLUMN {col} {dtype}")
                except sqlite3.OperationalError:
                    pass


if __name__ == "__main__":
    init_db()
    print(f"✅ Database initialized at {DB_PATH}")
