# 🇮🇳 India Stock Market Research Agent

[![CI](https://github.com/Arvinth16/stock-market-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/Arvinth16/stock-market-analysis/actions/workflows/ci.yml)

> **⚠️ DISCLAIMER**: This project is for **research and educational purposes only**.
> It does **NOT** constitute financial advice. Never make investment decisions based
> solely on automated signals. Always consult a qualified financial advisor.

## What is this?

An AI-powered research system that analyzes Indian stock market trends and suggests
candidate stocks for **human review**. It combines:

- **Technical analysis** — SMA, RSI, MACD, Bollinger Bands, ATR, volume ratios
- **Machine learning** — XGBoost classifier with walk-forward cross-validation
- **News sentiment** — Automated news fetching + VADER sentiment scoring
- **MCP integration** — Tools for Claude to query stock data, scores, and news
- **CLI** — Ranked recommendations with rationale and risk warnings

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/Arvinth16/stock-market-analysis.git
cd stock-market-analysis
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# 2. (macOS only) Install OpenMP for XGBoost
brew install libomp

# 3. Copy environment config
cp .env.example .env

# 4. Run the full pipeline
india-stock pipeline
```

## CLI Commands

| Command | Description |
|---|---|
| `india-stock update` | Download OHLCV data + compute technical features |
| `india-stock train` | Train XGBoost model with walk-forward CV |
| `india-stock rank --top 10` | Rank stocks by combined score |
| `india-stock rank --sector IT` | Filter by sector |
| `india-stock news RELIANCE` | Fetch and analyze news for a stock |
| `india-stock pipeline` | Run full pipeline end-to-end |

## Project Structure

```
src/
├── core/         # Config, database, utilities
│   ├── config.py       # Central configuration (.env, paths, model params)
│   └── database.py     # SQLite setup with WAL mode
├── data/         # OHLCV fetcher, universe, features
│   ├── fetcher.py      # Download from Yahoo Finance (NSE tickers)
│   ├── universe.py     # NIFTY 50 constituent list
│   └── features.py     # 19 technical indicators + label generation
├── models/       # ML training, evaluation, scoring
│   ├── trainer.py      # Walk-forward cross-validation
│   └── scorer.py       # Score all stocks, combine signals
├── news/         # News fetching & sentiment
│   ├── fetcher.py      # Google News via gnews
│   └── sentiment.py    # VADER sentiment analysis
├── mcp_server/   # MCP tools for Claude
│   └── server.py       # 5 tools: OHLCV, fundamentals, news, signals, backtest
└── api/          # CLI interface
    └── cli.py          # 5 subcommands
tests/            # Unit and integration tests
```

## Tech Stack

| Layer | Choice | Rationale |
|---|---|---|
| Stock data | `yfinance` (`.NS` tickers) | Free, reliable, supports NSE |
| News | `gnews` (Google News) | Free, no API key needed |
| Sentiment | `vaderSentiment` | Lightweight, no GPU needed |
| Database | SQLite + WAL | Zero config, file-based |
| ML model | XGBoost | Best-in-class for tabular data |
| MCP server | `mcp` Python SDK | Official MCP specification |

## Features & Model

See [FEATURES.md](FEATURES.md) for the complete feature documentation.

**19 technical indicators** across 5 categories:
- **Trend**: SMA(20/50/200), price vs SMA
- **Momentum**: RSI(14), MACD, 5/10/20-day returns
- **Volatility**: Bollinger Bands, ATR(14), 20-day vol
- **Volume**: Volume ratio vs 20-day average
- **Price action**: Distance from 52-week high/low

**Model**: XGBoost binary classifier predicting 20-day forward return > 2%.

**Scoring**: 55% model probability + 30% news sentiment + 15% momentum.

## MCP Server

Start the server for Claude integration:

```bash
python -m src.mcp_server.server
```

Available tools:
- `get_ohlcv(symbol, start_date, end_date)` — Historical price data
- `get_fundamentals(symbol)` — Basic info + price stats
- `get_recent_news(symbol, days)` — News with sentiment analysis
- `get_signals(top_n, sector)` — AI model-scored rankings
- `run_backtest(start_date, end_date, top_n)` — Simple backtest

## Development

```bash
# Run tests
pytest tests/ -v

# Lint
ruff check src/ tests/

# Initialize database
python -m src.core.database
```

## License

MIT
