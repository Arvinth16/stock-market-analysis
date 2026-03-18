"""
server.py — MCP server exposing Indian stock research tools for Claude.

Tools:
  - get_ohlcv: Retrieve historical OHLCV data for a symbol
  - get_fundamentals: Get basic fundamentals (sector, market cap proxy)
  - get_recent_news: Fetch and score recent news articles
  - get_signals: Get model-scored stock rankings
  - run_backtest: Run a simple backtest of signal-based strategy
"""

import json
from datetime import datetime, timedelta

import pandas as pd
from mcp.server.fastmcp import FastMCP

from src.core.database import get_db, init_db
from src.data.universe import get_universe, get_symbol_name, get_symbol_sector
from src.data.fetcher import load_ohlcv
from src.models.scorer import score_all_stocks
from src.news.fetcher import fetch_news
from src.news.sentiment import analyze_sentiment_batch, get_aggregate_sentiment

mcp = FastMCP("india-stock-research")


@mcp.tool()
def get_ohlcv(symbol: str, start_date: str = "", end_date: str = "") -> str:
    """
    Get historical OHLCV (Open, High, Low, Close, Volume) data for an Indian stock.

    Args:
        symbol: NSE ticker with .NS suffix (e.g., 'RELIANCE.NS')
        start_date: Start date in YYYY-MM-DD format (default: 30 days ago)
        end_date: End date in YYYY-MM-DD format (default: today)

    Returns:
        JSON array of daily OHLCV records.
    """
    df = load_ohlcv(symbol)
    if df.empty:
        return json.dumps({"error": f"No data found for {symbol}"})

    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]

    if not start_date and not end_date:
        df = df.tail(30)

    records = []
    for date, row in df.iterrows():
        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "open": round(row["open"], 2),
            "high": round(row["high"], 2),
            "low": round(row["low"], 2),
            "close": round(row["close"], 2),
            "volume": int(row["volume"]),
        })

    return json.dumps({"symbol": symbol, "records": records, "count": len(records)})


@mcp.tool()
def get_fundamentals(symbol: str) -> str:
    """
    Get basic fundamental info for an Indian stock (name, sector, recent price stats).

    Args:
        symbol: NSE ticker with .NS suffix (e.g., 'TCS.NS')

    Returns:
        JSON object with fundamental data.
    """
    universe = get_universe()
    info = universe.get(symbol)
    if not info:
        return json.dumps({"error": f"Unknown symbol: {symbol}"})

    df = load_ohlcv(symbol)
    fundamentals = {
        "symbol": symbol,
        "name": info["name"],
        "sector": info["sector"],
    }

    if not df.empty:
        latest = df.iloc[-1]
        fundamentals.update({
            "latest_close": round(latest["close"], 2),
            "latest_date": df.index[-1].strftime("%Y-%m-%d"),
            "52w_high": round(df["high"].tail(252).max(), 2),
            "52w_low": round(df["low"].tail(252).min(), 2),
            "avg_volume_20d": int(df["volume"].tail(20).mean()),
            "return_1m": round(float(df["close"].pct_change(20).iloc[-1] * 100), 2),
            "return_3m": round(float(df["close"].pct_change(60).iloc[-1] * 100), 2),
        })

    return json.dumps(fundamentals)


@mcp.tool()
def get_recent_news(symbol: str, days: int = 7) -> str:
    """
    Fetch and analyze recent news for an Indian stock.

    Args:
        symbol: NSE ticker with .NS suffix (e.g., 'INFY.NS')
        days: Number of days to look back (default: 7)

    Returns:
        JSON object with news articles and sentiment analysis.
    """
    name = get_symbol_name(symbol)
    articles = fetch_news(name, max_results=10, period=f"{days}d")

    if not articles:
        return json.dumps({"symbol": symbol, "articles": [], "sentiment": "no data"})

    # Analyze sentiment
    titles = [a["title"] for a in articles]
    sentiments = analyze_sentiment_batch(titles)
    agg = get_aggregate_sentiment(sentiments)

    # Combine articles with their sentiment
    enriched = []
    for article, sent in zip(articles, sentiments):
        enriched.append({
            "title": article["title"],
            "source": article.get("source", ""),
            "date": article.get("published_date", ""),
            "url": article.get("url", ""),
            "sentiment": sent["label"],
            "sentiment_score": sent["compound"],
        })

    return json.dumps({
        "symbol": symbol,
        "name": name,
        "articles": enriched,
        "aggregate_sentiment": agg,
    })


@mcp.tool()
def get_signals(top_n: int = 10, sector: str = "") -> str:
    """
    Get AI model-scored stock rankings for today.

    Args:
        top_n: Number of top stocks to return (default: 10)
        sector: Optional sector filter (e.g., 'IT', 'Banking', 'Pharma')

    Returns:
        JSON array of ranked stocks with scores and rationale.
    """
    try:
        df = score_all_stocks()
    except FileNotFoundError:
        return json.dumps({"error": "Model not trained yet. Run the trainer first."})

    if df.empty:
        return json.dumps({"error": "No scores available."})

    if sector:
        df = df[df["sector"].str.lower() == sector.lower()]

    top = df.head(top_n)
    results = []
    for _, row in top.iterrows():
        results.append({
            "rank": len(results) + 1,
            "symbol": row["symbol"],
            "name": row["name"],
            "sector": row["sector"],
            "final_score": row["final_score"],
            "model_probability": row["model_score"],
            "sentiment": row["sentiment"],
            "momentum_20d_pct": row["momentum_20d"],
            "rsi": row["rsi"],
            "caveat": "This is a research signal, NOT investment advice.",
        })

    return json.dumps({"date": datetime.now().strftime("%Y-%m-%d"), "signals": results})


@mcp.tool()
def run_backtest(start_date: str, end_date: str, top_n: int = 5) -> str:
    """
    Run a simple backtest: buy top_n model-scored stocks each month.

    Args:
        start_date: Backtest start in YYYY-MM-DD
        end_date: Backtest end in YYYY-MM-DD
        top_n: Number of top stocks to hold each period

    Returns:
        JSON with backtest results (returns, win rate).
    """
    with get_db() as conn:
        signals = pd.read_sql_query(
            """SELECT s.symbol, s.date, s.final_score, o.close
               FROM signals s
               JOIN ohlcv o ON s.symbol = o.symbol AND s.date = o.date
               WHERE s.date BETWEEN ? AND ?
               ORDER BY s.date, s.final_score DESC""",
            conn, params=(start_date, end_date),
        )

    if signals.empty:
        return json.dumps({"error": "No signal data for the given period."})

    # Group by date, take top N, compute forward returns
    results = []
    dates = signals["date"].unique()

    for date in dates:
        day_signals = signals[signals["date"] == date].head(top_n)
        for _, row in day_signals.iterrows():
            # Look up price 20 days later
            future_price = signals[
                (signals["symbol"] == row["symbol"]) &
                (signals["date"] > date)
            ].head(1)

            if not future_price.empty:
                ret = (future_price.iloc[0]["close"] - row["close"]) / row["close"]
                results.append({
                    "symbol": row["symbol"],
                    "entry_date": date,
                    "entry_price": row["close"],
                    "exit_price": future_price.iloc[0]["close"],
                    "return_pct": round(ret * 100, 2),
                })

    if not results:
        return json.dumps({"error": "Not enough data for backtest."})

    returns = [r["return_pct"] for r in results]
    return json.dumps({
        "period": f"{start_date} to {end_date}",
        "total_trades": len(results),
        "avg_return_pct": round(sum(returns) / len(returns), 2),
        "win_rate_pct": round(sum(1 for r in returns if r > 0) / len(returns) * 100, 1),
        "best_trade_pct": round(max(returns), 2),
        "worst_trade_pct": round(min(returns), 2),
        "caveat": "Past performance does not guarantee future results.",
    })


def run_server():
    """Start the MCP server."""
    init_db()
    mcp.run()


if __name__ == "__main__":
    run_server()
