"""
cli.py — Command-line interface for the India Stock Research Agent.

Usage:
    india-stock rank [--top N] [--sector SECTOR]
    india-stock update                              # Download data + compute features
    india-stock train                               # Train/retrain the model
    india-stock news SYMBOL                         # Get recent news for a stock
    india-stock pipeline                            # Run full pipeline end-to-end
"""

import argparse
import sys
from datetime import datetime
from tabulate import tabulate


def cmd_update(args):
    """Download latest data and compute features."""
    from src.core.database import init_db
    from src.data.fetcher import fetch_and_store_all
    from src.data.features import compute_and_store_all

    init_db()
    print(f"\n🕐 Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    fetch_and_store_all()
    compute_and_store_all()
    print("✅ Data update complete.\n")


def cmd_train(args):
    """Train the ML model."""
    from src.models.trainer import run_training
    run_training()


def cmd_news(args):
    """Fetch and display news for a stock."""
    from src.news.fetcher import fetch_news
    from src.news.sentiment import analyze_sentiment_batch, get_aggregate_sentiment
    from src.data.universe import get_symbol_name

    symbol = args.symbol.upper()
    if not symbol.endswith(".NS"):
        symbol += ".NS"

    name = get_symbol_name(symbol)
    print(f"\n📰 News for {name} ({symbol})\n")

    articles = fetch_news(name, max_results=10)
    if not articles:
        print("  No recent news found.\n")
        return

    sentiments = analyze_sentiment_batch([a["title"] for a in articles])
    agg = get_aggregate_sentiment(sentiments)

    for article, sent in zip(articles, sentiments):
        icon = "🟢" if sent["label"] == "positive" else "🔴" if sent["label"] == "negative" else "⚪"
        print(f"  {icon} [{sent['compound']:+.3f}] {article['title']}")
        print(f"     📅 {article.get('published_date', 'N/A')} | 📰 {article.get('source', 'N/A')}")
        print()

    print(f"  ─── Aggregate Sentiment ───")
    print(f"  Overall: {agg['overall_label'].upper()} (avg: {agg['avg_compound']:+.4f})")
    print(f"  🟢 Positive: {agg['positive_pct']}% | 🔴 Negative: {agg['negative_pct']}% | ⚪ Neutral: {agg['neutral_pct']}%\n")


def cmd_rank(args):
    """Score and rank stocks."""
    from src.models.scorer import score_all_stocks

    print(f"\n📈 Stock Rankings — {datetime.now().strftime('%Y-%m-%d')}\n")

    try:
        df = score_all_stocks()
    except FileNotFoundError:
        print("  ✗ Model not trained yet. Run: india-stock train\n")
        return

    if df.empty:
        print("  ✗ No data available. Run: india-stock update\n")
        return

    if args.sector:
        df = df[df["sector"].str.lower() == args.sector.lower()]

    top = df.head(args.top)
    if top.empty:
        print(f"  No stocks found for sector '{args.sector}'.\n")
        return

    # Format table
    table_data = []
    for i, (_, row) in enumerate(top.iterrows(), 1):
        table_data.append([
            i,
            row["name"],
            row["symbol"],
            row["sector"],
            f"{row['final_score']:.4f}",
            f"{row['model_score']:.3f}",
            f"{row['sentiment']:+.3f}",
            f"{row['momentum_20d']:+.1f}%",
            f"{row['rsi']:.0f}",
        ])

    headers = ["#", "Stock", "Symbol", "Sector", "Score", "Model", "Sentiment", "Mom(20d)", "RSI"]
    print(tabulate(table_data, headers=headers, tablefmt="rounded_grid"))

    print(f"\n  ⚠️  DISCLAIMER: These are research signals only, NOT investment advice.")
    print(f"  Always do your own research and consult a financial advisor.\n")


def cmd_pipeline(args):
    """Run the full pipeline: update → train → score → display."""
    print("=" * 60)
    print("🚀 FULL PIPELINE — India Stock Research Agent")
    print("=" * 60)

    # Step 1: Update data
    cmd_update(args)

    # Step 2: Fetch news
    print("📰 Fetching news...\n")
    from src.core.database import init_db
    from src.news.fetcher import fetch_all_news
    init_db()
    fetch_all_news()

    # Step 3: Train model
    cmd_train(args)

    # Step 4: Rank & display
    args.top = getattr(args, "top", 10)
    args.sector = getattr(args, "sector", "")
    cmd_rank(args)

    print("=" * 60)
    print("✅ Pipeline complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        prog="india-stock",
        description="🇮🇳 India Stock Market Research Agent — AI-powered stock analysis (NOT trading advice)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # update
    sub_update = subparsers.add_parser("update", help="Download data and compute features")

    # train
    sub_train = subparsers.add_parser("train", help="Train the ML model")

    # news
    sub_news = subparsers.add_parser("news", help="Get recent news for a stock")
    sub_news.add_argument("symbol", type=str, help="Stock symbol (e.g., RELIANCE or RELIANCE.NS)")

    # rank
    sub_rank = subparsers.add_parser("rank", help="Rank stocks by combined score")
    sub_rank.add_argument("--top", type=int, default=10, help="Number of top stocks (default: 10)")
    sub_rank.add_argument("--sector", type=str, default="", help="Filter by sector")

    # pipeline
    sub_pipeline = subparsers.add_parser("pipeline", help="Run full pipeline end-to-end")

    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()

    commands = {
        "update": cmd_update,
        "train": cmd_train,
        "news": cmd_news,
        "rank": cmd_rank,
        "pipeline": cmd_pipeline,
    }

    func = commands.get(args.command)
    if func:
        func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
