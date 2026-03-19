"""
fundamentals.py — Scrapes and stores core valuation ratios and financial health metrics from Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
from src.core.database import get_db, init_db
from src.data.universe import get_symbols, get_symbol_name

def fetch_fundamentals(symbol: str) -> dict:
    """Download fundamental ratios for a single ticker."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # We handle None gracefully. If a key is missing, it gets None.
        return {
            "symbol": symbol,
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "pb_ratio": info.get("priceToBook"),
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            "debt_to_equity": info.get("debtToEquity"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "profit_margin": info.get("profitMargins"),
            "dividend_yield": info.get("dividendYield")
        }
    except Exception as e:
        print(f"  ✗ Error fetching fundamentals for {symbol}: {e}")
        return None


def store_fundamentals(symbol: str, data: dict):
    """Store the extracted fundamental dict to the DB."""
    if not data:
        return
        
    with get_db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO fundamentals
               (symbol, latest_date, pe_ratio, forward_pe, pb_ratio, roe, roa, 
                debt_to_equity, revenue_growth, earnings_growth, profit_margin, dividend_yield)
               VALUES (?, date('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                symbol,
                data.get("pe_ratio"), data.get("forward_pe"), data.get("pb_ratio"),
                data.get("roe"), data.get("roa"), data.get("debt_to_equity"),
                data.get("revenue_growth"), data.get("earnings_growth"),
                data.get("profit_margin"), data.get("dividend_yield")
            )
        )


def fetch_and_store_all_fundamentals():
    """Batch process fundamental extraction for the whole universe."""
    init_db()
    symbols = get_symbols()
    total = len(symbols)
    
    print("━" * 60)
    print("🏦 DOWNLOADING QUARTERLY FUNDAMENTALS")
    print("━" * 60)

    success = 0
    for i, symbol in enumerate(symbols, 1):
        name = get_symbol_name(symbol)
        print(f"  [{i:>2}/{total}] {name} ({symbol})...", end=" ", flush=True)
        
        data = fetch_fundamentals(symbol)
        if data:
            store_fundamentals(symbol, data)
            print("✓ stored")
            success += 1
        else:
            print("✗ failed")

    print(f"\n  ✅ Loaded {success}/{total} stock fundamentals into database.\n")
    return success

if __name__ == "__main__":
    fetch_and_store_all_fundamentals()
