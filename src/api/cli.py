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
import pandas as pd


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
        pub_date = article.get("published_date", "Recent")[:16] # Truncate to YYYY-MM-DD HH:MM
        print(f"  {icon} [{sent['compound']:+.2f}] {pub_date} | {article['title']}")
        print(f"     📅 {article.get('published_date', 'N/A')} | 📰 {article.get('source', 'N/A')}")
        print()

    print("  ─── Aggregate Sentiment ───")
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
        ci = f"[{row.get('pred_return_low', 0)*100:+.1f}%, {row.get('pred_return_high', 0)*100:+.1f}%]"
        table_data.append([
            i,
            row["name"],
            row["sector"],
            f"{row['final_score']:.4f}",
            f"{row.get('predicted_return', 0)*100:+.1f}%",
            ci,
            f"{row['model_score']:.3f}",
            f"{row['sentiment']:+.3f}",
            f"{row['momentum_20d']:+.1f}%",
        ])

    # Regime banner
    regime = top.iloc[0].get("regime", "normal") if not top.empty else "normal"
    if regime == "crash":
        print("  🚨🚨🚨 MARKET REGIME: CRASH — All bullish forecasts are heavily suppressed! 🚨🚨🚨")
    elif regime == "stressed":
        print("  ⚠️  MARKET REGIME: STRESSED — Bullish signals dampened for safety.")
    else:
        print("  ✅ MARKET REGIME: NORMAL")
    print()

    headers = ["#", "Stock", "Sector", "Score", "Return", "80% CI", "Model", "Sent", "Mom"]
    print(tabulate(table_data, headers=headers, tablefmt="rounded_grid"))

    print("\n  ⚠️  DISCLAIMER: Research signals only, NOT investment advice.")
    print("  Always do your own research and consult a financial advisor.\n")


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


def cmd_analyze(args):
    """Analyze a single stock in depth."""
    from src.models.scorer import score_all_stocks
    from src.news.fetcher import fetch_news
    from src.news.sentiment import analyze_sentiment_batch, get_aggregate_sentiment
    from src.data.universe import get_symbol_name
    from src.core.database import get_db
    import yfinance as yf
    import math

    symbol = args.symbol.upper()
    if not symbol.endswith(".NS"):
        symbol += ".NS"

    name = get_symbol_name(symbol)
    if not name:
        print(f"✗ Symbol {symbol} not in universe.")
        return

    print(f"\n🔍 Analyzing: {name} ({symbol})")
    print("━" * 60)

    # 1. Get Score & Target Price
    df = score_all_stocks()
    stock_data = df[df["symbol"] == symbol]

    if stock_data.empty:
        print("  ✗ No model data available for this stock yet. Run 'india-stock update' and 'train'.\n")
        return

    # 1.1 Market Regime Check
    median_20d_momentum = df["momentum_20d"].median()
    in_crash_regime = median_20d_momentum < -4.0
    
    if in_crash_regime:
        print(f"\n  ⚠️  MARKET CRASH REGIME DETECTED (Universe Median 20-Day: {median_20d_momentum:.1f}%)")
        print("  ⚠️  Model trained on normal conditions; treat forecasts as low-confidence, use only as baseline.\n")

    row = stock_data.iloc[0]
    
    # 2. Extract Data
    last_price = row.get("last_price", 0)
    target_price = row.get("target_price", 0)
    pred_return = row.get("predicted_return", 0) * 100
    model_score = row.get("model_score", 0)
    momentum = row.get("momentum_20d", 0)
    rsi = row.get("rsi", 0)
    final_score = row.get("final_score", 0)

    # 3. Categorize Signal
    if final_score > 0.65 and model_score > 0.60:
        signal_tag = "🟩 STRONG BULLISH"
    elif final_score > 0.55 and model_score > 0.50:
        signal_tag = "🟨 MILD BULLISH"
    elif final_score < 0.40 and model_score < 0.40:
        signal_tag = "🟥 BEARISH"
    else:
        signal_tag = "⬜ NEUTRAL"

    # 4. Volatility Context
    vol = row.get("volatility_20", 0)
    if isinstance(vol, pd.Series): 
        vol = float(vol.iloc[0]) if not vol.empty else 0.0
    daily_move = (vol / math.sqrt(252)) * 100 if vol > 0 else 0.0

    # 5. Print Dashboard
    macro_nifty = row.get("macro_nifty_drawdown", 0)
    macro_vix = row.get("macro_vix_percentile", 0)
    
    print(f"📊 Model Analysis (Date: {row['date']})")
    print(f"  Current Price:    ₹{last_price:.2f}")
    print(f"  Signal Strength:  {signal_tag}")
    print(f"  Target Price:     ₹{target_price:.2f} ({pred_return:+.2f}%)  <-- XGBRegressor 20-Day Forecast")
    print(f"  Upside Prob:      {model_score*100:.1f}%              <-- XGBClassifier Probability")
    print(f"  RSI (14-day):     {rsi:.1f}")
    print(f"  Momentum (20d):   {momentum:+.1f}%")
    print(f"  Volatility (20d): {vol*100:.1f}% Annualized (Typical daily move: ±{daily_move:.1f}%)")
    print(f"  Macro Context:    Nifty Drawdown: {macro_nifty:.1f}% | VIX Percentile: {macro_vix:.1f}%")
    print(f"  Ranker Score:     {final_score:.4f}/1.000")

    # 5.5 Fundamentals Layer
    with get_db() as conn:
        fund = conn.execute("SELECT * FROM fundamentals WHERE symbol = ?", (symbol,)).fetchone()
    
    if fund:
        pe = fund["pe_ratio"] or 0
        fpe = fund["forward_pe"] or 0
        roe = (fund["roe"] or 0) * 100
        debt = fund["debt_to_equity"] or 0
        print("\n💼 Core Fundamentals")
        print(f"  - Trailing P/E:   {pe:.1f}x")
        print(f"  - Forward P/E:    {fpe:.1f}x")
        print(f"  - Return on Eq:   {roe:.1f}%")
        print(f"  - Debt/Equity:    {debt:.2f}")

    # 5.6 SHAP AI Explainability
    shap_bull = row.get("shap_bullish", [])
    shap_bear = row.get("shap_bearish", [])
    
    if shap_bull or shap_bear:
        print("\n🧠 AI Explainability (Top SHAP Drivers)")
        for f in shap_bull:
            print(f"  🟢 {f['feature']:<20} (+{f['impact']:.3f} impact)")
        for f in shap_bear:
            print(f"  🔴 {f['feature']:<20} ({f['impact']:.3f} impact)")

    # 6. External Analyst Targets
    print("\n🎯 External Price Targets (Street Consensus)")
    ticker = yf.Ticker(symbol)
    info = ticker.info
    mean_tgt = info.get("targetMeanPrice")
    high_tgt = info.get("targetHighPrice")
    low_tgt = info.get("targetLowPrice")
    analysts = info.get("numberOfAnalystOpinions", 0)
    
    if mean_tgt and last_price > 0:
        upside = ((mean_tgt - last_price) / last_price) * 100
        print(f"  - Consensus Mean: ₹{mean_tgt:.2f} ({upside:+.1f}% vs Current)")
        print(f"  - Target Range:   ₹{low_tgt:.2f} (Low) - ₹{high_tgt:.2f} (High)")
        print(f"  - Analyst Count:  {analysts} Brokers")
    else:
        print("  - No consensus targets available for this stock.")

    print("\n📰 Recent News Sentiment")
    
    articles = fetch_news(name, max_results=5)
    if not articles:
        print("  No recent news found.")
    else:
        sentiments = analyze_sentiment_batch([a["title"] for a in articles])
        agg = get_aggregate_sentiment(sentiments)
        avg_score = agg["avg_compound"]
        ov_label = agg["overall_label"].upper()
        main_icon = "🟢" if "POS" in ov_label else "🔴" if "NEG" in ov_label else "⚪"
        print(f"  Overall: {ov_label} {main_icon} (avg: {avg_score:+.3f})")
        for article, sent in zip(articles, sentiments):
            icon = "🟢" if sent["label"] == "positive" else "🔴" if sent["label"] == "negative" else "⚪"
            pub_date = article.get("published_date", "Recent")[:16]
            print(f"    {icon} [{sent['compound']:+.2f}] {pub_date} | {article['title']}")
            
    print("\n" + "=" * 60 + "\n")


def cmd_portfolio(args):
    """Analyze a predefined list of portfolio stocks."""
    import os
    import json
    from src.models.scorer import score_all_stocks
    
    # Check for local private portfolio config
    if os.path.exists("my_portfolio.json"):
        with open("my_portfolio.json", "r") as f:
            PORTFOLIO = json.load(f)
    else:
        # Generic public placeholder for Github
        PORTFOLIO = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"
        ]

    print("\n💼 PORTFOLIO TRACKER — India Stock Research Agent")
    print("=" * 70)

    try:
        df = score_all_stocks()
    except Exception:
        print("  ✗ Error loading model data. Run 'india-stock train' first.")
        return

    if df.empty:
        print("  ✗ No data available. Run: india-stock update\n")
        return
        
    median_20d_momentum = df["momentum_20d"].median()
    in_crash_regime = median_20d_momentum < -4.0
    
    if in_crash_regime:
        print(f"  ⚠️  MARKET CRASH REGIME DETECTED (Universe Median 20-Day: {median_20d_momentum:.1f}%)")
        print("  ⚠️  Bullish predictions are severely down-weighted for safety.\n")
        df['predicted_return'] = df['predicted_return'].apply(lambda x: x * 0.4 if x > 0 else x * 1.2)
        df['target_price'] = df['last_price'] * (1 + df['predicted_return'])

    portfolio_df = df[df["symbol"].isin(PORTFOLIO)]
    
    # Sector Exposure Calculation
    from src.data.universe import get_symbol_sector
    sector_weights = {}
    for sym in PORTFOLIO:
        sec = get_symbol_sector(sym)
        sector_weights[sec] = sector_weights.get(sec, 0) + 1
        
    print("  📊 Portfolio Sector Exposure (Equal-Weight Assumption):")
    total_assets = len(PORTFOLIO)
    for sec, count in sorted(sector_weights.items(), key=lambda x: x[1], reverse=True):
        weight_pct = (count / total_assets) * 100
        bar = "█" * int(weight_pct / 5)
        print(f"    {sec:<16} {bar} {weight_pct:.1f}%")
    print()
    
    fund_dict = {}
    from src.core.database import get_db
    with get_db() as conn:
        for fund_row in conn.execute("SELECT symbol, pe_ratio, roe FROM fundamentals"):
            fund_dict[fund_row["symbol"]] = {"pe": fund_row["pe_ratio"], "roe": fund_row["roe"]}
    
    table_data = []
    for i, (_, row) in enumerate(portfolio_df.iterrows(), 1):
        target_pct = row['predicted_return'] * 100
        signal_prob = row['model_score'] * 100
        
        # Volatility warning
        vol_warn = "🚨" if row['volatility_20'] > 0.40 else ""
        
        sym = row["symbol"]
        pe = fund_dict.get(sym, {}).get("pe")
        roe = fund_dict.get(sym, {}).get("roe")
        pe_str = f"{pe:.1f}" if pe else "-"
        roe_str = f"{roe*100:.1f}%" if roe else "-"
        
        table_data.append([
            sym.replace('.NS', ''),
            f"₹{row['last_price']:.2f}",
            f"₹{row['target_price']:.2f} ({target_pct:+.1f}%)",
            f"{signal_prob:.1f}%",
            f"{row['momentum_20d']:+.1f}%",
            f"{row['rsi']:.0f} {vol_warn}",
            pe_str,
            roe_str
        ])

    headers = ["Symbol", "CMP", "AI Target (20d)", "Bull Prob", "Mom(20d)", "RSI", "P/E", "ROE"]
    from tabulate import tabulate
    print(tabulate(table_data, headers=headers, tablefmt="rounded_grid"))
    print("=" * 70 + "\n")
    
    print("📰 LATEST PORTFOLIO NEWS")
    print("━" * 70)
    
    from src.news.fetcher import fetch_news
    from src.news.sentiment import analyze_sentiment_batch
    from src.data.universe import get_symbol_name
    
    for symbol in PORTFOLIO:
        name = get_symbol_name(symbol)
        articles = fetch_news(name, max_results=10)
        if articles:
            print(f"🔹 {symbol.replace('.NS', '')}")
            sentiments = analyze_sentiment_batch([a["title"] for a in articles])
            for article, sent in zip(articles, sentiments):
                icon = "🟢" if sent["label"] == "positive" else "🔴" if sent["label"] == "negative" else "⚪"
                pub_date = article.get("published_date", "Recent")[:10]
                print(f"    {icon} [{pub_date}] {article['title']}")
            print()
    
    print("=" * 70 + "\n")





def main():
    parser = argparse.ArgumentParser(
        prog="india-stock",
        description="🇮🇳 India Stock Market Research Agent — AI-powered stock analysis (NOT trading advice)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # update
    subparsers.add_parser("update", help="Download data and compute features")

    # train
    subparsers.add_parser("train", help="Train the ML model")

    # news
    sub_news = subparsers.add_parser("news", help="Get recent news for a stock")
    sub_news.add_argument("symbol", type=str, help="Stock symbol (e.g., RELIANCE or RELIANCE.NS)")

    # rank
    sub_rank = subparsers.add_parser("rank", help="Rank stocks by combined score")
    sub_rank.add_argument("--top", type=int, default=10, help="Number of top stocks (default: 10)")
    sub_rank.add_argument("--sector", type=str, default="", help="Filter by sector")

    # portfolio
    subparsers.add_parser("portfolio", help="Track and predict your custom 12-stock holding portfolio")

    # analyze
    sub_analyze = subparsers.add_parser("analyze", help="Detailed single-stock dashboard")
    sub_analyze.add_argument("symbol", type=str, help="Stock symbol (e.g., ITC or ITC.NS)")

    # pipeline
    subparsers.add_parser("pipeline", help="Run full pipeline end-to-end")

    # discover
    subparsers.add_parser("discover", help="Discover new stocks from news")

    # monitor
    subparsers.add_parser("monitor", help="Run data quality & model health checks")

    # schedule
    subparsers.add_parser("schedule", help="Start automated daily/weekly scheduler")

    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()

    commands = {
        "update": cmd_update,
        "train": cmd_train,
        "news": cmd_news,
        "rank": cmd_rank,
        "portfolio": cmd_portfolio,
        "analyze": cmd_analyze,
        "pipeline": cmd_pipeline,
        "discover": cmd_discover,
        "monitor": cmd_monitor,
        "schedule": cmd_schedule,
    }

    func = commands.get(args.command)
    if func:
        func(args)
    else:
        parser.print_help()


def cmd_discover(args):
    """Run news-driven stock discovery."""
    from src.discovery.discovery import run_discovery
    run_discovery()


def cmd_monitor(args):
    """Run data quality and model health checks."""
    from src.core.monitor import run_monitor
    run_monitor()


def cmd_schedule(args):
    """Start the automated scheduler."""
    from src.core.scheduler import start_scheduler
    start_scheduler()


if __name__ == "__main__":
    main()
