"""
fetcher.py — Fetch recent news articles for Indian stocks via Google News RSS.

Uses requests + feedparser for reliable RSS parsing with proper SSL handling.
"""

import urllib.parse
from datetime import datetime

import feedparser
import requests

from src.core.database import get_db
from src.data.universe import get_universe


def fetch_news(company_name: str, max_results: int = 10,
               period: str = None) -> list[dict]:
    """
    Fetch recent news articles about a company from Google News RSS.

    Args:
        company_name: Human-readable company name.
        max_results: Maximum articles to return.
        period: Lookback period (e.g. '7d') — used for context but RSS returns recent.

    Returns:
        List of dicts with keys: title, description, published_date, url, source.
    """
    articles = []
    try:
        query = urllib.parse.quote(f"{company_name} stock")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

        # Use requests for reliable SSL handling, then parse with feedparser
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (compatible; StockResearchBot/1.0)"
        })
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)

        for entry in feed.entries[:max_results]:
            # Extract source from title (Google News format: "Title - Source")
            title = entry.get("title", "")
            source = "Unknown"
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                title = parts[0]
                source = parts[1] if len(parts) > 1 else "Unknown"

            published = entry.get("published", "")
            try:
                dt = datetime.strptime(published, "%a, %d %b %Y %H:%M:%S %Z")
                published = dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                pass

            articles.append({
                "title": title,
                "description": entry.get("summary", ""),
                "published_date": published,
                "url": entry.get("link", ""),
                "source": source,
            })
    except Exception as e:
        print(f"    ⚠  News fetch error for {company_name}: {e}")

    return articles


def store_news(symbol: str, articles: list[dict], sentiments: list[dict]):
    """Store fetched news articles with sentiment into the database."""
    with get_db() as conn:
        for article, sent in zip(articles, sentiments):
            conn.execute(
                """INSERT INTO news (symbol, date, title, source, url, sentiment, sentiment_label)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (symbol,
                 article.get("published_date", ""),
                 article.get("title", ""),
                 article.get("source", ""),
                 article.get("url", ""),
                 sent.get("compound", 0.0),
                 sent.get("label", "neutral")),
            )


def fetch_all_news(with_sentiment: bool = True):
    """Fetch and store news for all stocks in the universe."""
    from src.news.sentiment import analyze_sentiment_batch

    universe = get_universe()
    total = len(universe)

    print("━" * 60)
    print("📰 FETCHING RECENT NEWS")
    print("━" * 60)

    success = 0
    for i, (symbol, info) in enumerate(universe.items(), 1):
        name = info["name"]
        print(f"  [{i:>2}/{total}] {name}...", end=" ", flush=True)

        articles = fetch_news(name)
        if not articles:
            print("no articles found")
            continue

        if with_sentiment:
            sentiments = analyze_sentiment_batch([a["title"] for a in articles])
            store_news(symbol, articles, sentiments)
        else:
            neutral = [{"compound": 0.0, "label": "neutral"}] * len(articles)
            store_news(symbol, articles, neutral)

        print(f"✓ {len(articles)} articles")
        success += 1

    print(f"\n  ✅ News fetched for {success}/{total} stocks.\n")


if __name__ == "__main__":
    fetch_all_news()
