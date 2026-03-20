"""
discovery.py — News-driven stock discovery outside the tracked universe.

Scans Google News for trending Indian stocks, identifies those with bullish
sentiment/momentum signals, and auto-onboards promising candidates.
"""

import re
from datetime import datetime

from src.core.database import get_db, init_db
from src.news.fetcher import fetch_news
from src.news.sentiment import analyze_sentiment_batch
from src.data.universe import get_symbols


# Keywords that signal potential upside
BULLISH_KEYWORDS = [
    "upgrade", "raises target", "all-time high", "record profit",
    "record revenue", "strong quarter", "beats estimate", "outperform",
    "top pick", "breakout", "52-week high", "strong buy", "re-rating",
    "order win", "contract", "expansion", "growth", "rally",
    "surges", "soars", "jumps", "gains",
]

# General Indian market news queries to discover new stocks
DISCOVERY_QUERIES = [
    "Indian stock market top picks today",
    "NSE stocks to buy recommendation",
    "India multibagger stocks 2026",
    "best performing Indian stocks this week",
    "NSE BSE breakout stocks momentum",
    "Indian stock market upgrade target price",
    "India small cap mid cap rising stocks",
    "NSE stocks record high volume surge",
]


def extract_stock_symbols_from_text(text: str) -> list:
    """
    Extract potential NSE stock symbols from news headlines.
    Looks for patterns like company names and tries to map them.
    """
    # Common NSE stock name patterns
    matches = set()

    # Look for explicit ticker mentions (e.g., "TATASTEEL", "RELIANCE")
    ticker_pattern = re.findall(r'\b([A-Z]{3,15})\b', text)
    for t in ticker_pattern:
        sym = f"{t}.NS"
        # Skip common English words
        if t not in {"THE", "AND", "FOR", "ARE", "NOT", "BUT", "HAS",
                     "WAS", "HIS", "HER", "ITS", "TOP", "NEW", "ALL",
                     "CAN", "HAD", "MAY", "GET", "SET", "NOW", "HOW",
                     "BIG", "LOW", "NSE", "BSE", "IPO", "ETF", "GDP",
                     "RBI", "FII", "DII", "NIFTY", "SENSEX", "SEBI",
                     "THIS", "THAT", "WITH", "FROM", "INTO", "OVER",
                     "ALSO", "BEST", "MOST", "SOME", "WEEK", "YEAR",
                     "WILL", "AMID", "NEAR", "HIGH", "RISE", "FALL",
                     "HERE", "WHAT", "LIVE", "MARCH", "APRIL", "JUNE",
                     "STOCK", "SHARE", "PRICE", "TODAY", "INDIA",
                     "MARKET", "RALLY", "WATCH", "FOCUS", "AMONG",
                     "AFTER", "BELOW", "ABOVE", "ABOUT", "WHICH"}:
            matches.add(sym)

    return list(matches)


def scan_news_for_discoveries(max_queries: int = 8) -> list:
    """
    Scan general market news for stocks not in our current universe.
    Returns list of discovery dicts with symbol, reason, sentiment.
    """
    tracked = set(get_symbols())
    discoveries = {}

    for query in DISCOVERY_QUERIES[:max_queries]:
        articles = fetch_news(query, max_results=10)
        if not articles:
            continue

        titles = [a["title"] for a in articles]
        sentiments = analyze_sentiment_batch(titles)

        for article, sent in zip(articles, sentiments):
            title = article["title"]
            # Check for bullish keywords
            has_bullish = any(kw in title.lower() for kw in BULLISH_KEYWORDS)

            if not has_bullish and sent["compound"] <= 0.05:
                continue

            # Extract potential symbols
            symbols = extract_stock_symbols_from_text(title)

            for sym in symbols:
                if sym in tracked:
                    continue
                if sym not in discoveries:
                    discoveries[sym] = {
                        "symbol": sym,
                        "mentions": 0,
                        "total_sentiment": 0.0,
                        "reasons": [],
                        "bullish_count": 0,
                    }

                discoveries[sym]["mentions"] += 1
                discoveries[sym]["total_sentiment"] += sent["compound"]
                if has_bullish:
                    discoveries[sym]["bullish_count"] += 1

                reason = title[:120]
                if reason not in discoveries[sym]["reasons"]:
                    discoveries[sym]["reasons"].append(reason)

    # Score and rank discoveries
    results = []
    for sym, data in discoveries.items():
        if data["mentions"] < 2:
            continue  # Need at least 2 mentions

        avg_sentiment = data["total_sentiment"] / data["mentions"]
        score = (
            data["mentions"] * 0.3
            + data["bullish_count"] * 0.4
            + max(0, avg_sentiment) * 0.3
        )

        results.append({
            "symbol": sym,
            "mentions": data["mentions"],
            "avg_sentiment": round(avg_sentiment, 3),
            "bullish_mentions": data["bullish_count"],
            "score": round(score, 3),
            "reasons": data["reasons"][:5],
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:20]  # Top 20 discoveries


def store_discoveries(discoveries: list):
    """Store discovered stocks in the database."""
    today = datetime.now().strftime("%Y-%m-%d")
    with get_db() as conn:
        for d in discoveries:
            reason = " | ".join(d["reasons"][:3])
            conn.execute(
                """INSERT OR REPLACE INTO discoveries
                   (symbol, date, source, reason, sentiment, momentum, onboarded)
                   VALUES (?, ?, 'news', ?, ?, 0.0, 0)""",
                (d["symbol"], today, reason, d["avg_sentiment"]),
            )


def onboard_stock(symbol: str) -> bool:
    """
    Auto-onboard a discovered stock:
    1. Download OHLCV history
    2. Compute features
    3. Mark as onboarded in discoveries table
    """
    from src.data.fetcher import fetch_and_store_ohlcv
    from src.data.features import (
        compute_features, compute_labels, store_features,
        load_macro_data, load_fundamentals_dict,
    )
    from src.data.fetcher import load_ohlcv

    print(f"  📥 Onboarding {symbol}...", end=" ", flush=True)

    try:
        # Download history
        fetch_and_store_ohlcv(symbol)
        df = load_ohlcv(symbol)

        if df.empty or len(df) < 250:
            print("✗ insufficient history (need ≥ 250 days)")
            return False

        # Apply Hard Filters for Liquidity and Price
        recent = df.tail(20)
        avg_price = recent["close"].mean()
        avg_volume = recent["volume"].mean()
        adv = avg_price * avg_volume

        if avg_price < 20.0:
            print(f"✗ price too low (₹{avg_price:.2f} < ₹20)")
            return False

        if adv < 50_000_000:  # ₹5 Crore
            print(f"✗ liquidity too low (ADV ₹{adv/10000000:.2f}Cr < ₹5Cr)")
            return False

        # Compute features
        macro_df = load_macro_data()
        fund = load_fundamentals_dict(symbol)
        features = compute_features(df, macro_df, fundamentals=fund, symbol=symbol)
        labels = compute_labels(df)
        store_features(symbol, features, labels)

        # Mark onboarded
        with get_db() as conn:
            conn.execute(
                "UPDATE discoveries SET onboarded = 1 WHERE symbol = ?",
                (symbol,),
            )

        core_cols = [c for c in features.columns if not c.startswith("fund_")]
        valid = features.dropna(subset=core_cols).shape[0]
        print(f"✓ {valid} feature rows, {len(df)} days of history")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def run_discovery():
    """Main discovery pipeline."""
    init_db()

    print("━" * 60)
    print("🔍 NEWS-DRIVEN STOCK DISCOVERY")
    print("━" * 60)

    print("  Scanning market news for trending stocks...")
    discoveries = scan_news_for_discoveries()

    if not discoveries:
        print("  No new stocks discovered today.\n")
        return []

    store_discoveries(discoveries)

    print(f"\n  🔎 Found {len(discoveries)} candidate stocks:\n")
    for i, d in enumerate(discoveries, 1):
        sent_icon = "🟢" if d["avg_sentiment"] > 0.1 else "🔴" if d["avg_sentiment"] < -0.1 else "⚪"
        print(f"  {i:>2}. {d['symbol']:<20s} | "
              f"Mentions: {d['mentions']} | "
              f"Sentiment: {sent_icon} {d['avg_sentiment']:+.3f} | "
              f"Score: {d['score']:.2f}")
        for r in d["reasons"][:2]:
            print(f"      → {r}")

    # Auto-onboard top candidates
    print(f"\n  📦 Auto-onboarding top {min(5, len(discoveries))} candidates...")
    onboarded = 0
    for d in discoveries[:5]:
        if onboard_stock(d["symbol"]):
            onboarded += 1

    print(f"\n  ✅ Onboarded {onboarded} new stocks.\n")
    return discoveries


if __name__ == "__main__":
    run_discovery()
