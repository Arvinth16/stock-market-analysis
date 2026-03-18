"""
fetcher.py — Fetch recent news articles for Indian stocks using Google News.
"""

from gnews import GNews
from src.core.config import NEWS_LOOKBACK_DAYS
from src.core.database import get_db
from src.data.universe import get_universe, get_symbol_name


def fetch_news(company_name: str, max_results: int = 10,
               period: str = None) -> list[dict]:
    """
    Fetch recent news articles about a company from Google News.

    Args:
        company_name: Human-readable company name.
        max_results: Maximum articles to return.
        period: Lookback period (e.g. '7d').

    Returns:
        List of dicts with keys: title, description, published_date, url, source.
    """
    if period is None:
        period = f"{NEWS_LOOKBACK_DAYS}d"

    articles = []
    try:
        gn = GNews(language="en", country="IN", max_results=max_results, period=period)
        query = f"{company_name} stock"
        raw = gn.get_news(query)

        for article in raw:
            articles.append({
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "published_date": article.get("published date", ""),
                "url": article.get("url", ""),
                "source": article.get("publisher", {}).get("title", "Unknown"),
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

    print(f"\n  ✅ News fetched for all stocks.\n")


if __name__ == "__main__":
    fetch_all_news()
