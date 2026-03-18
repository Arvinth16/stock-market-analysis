# 🇮🇳 India Stock Market Research Agent

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

## Quick Start

```bash
# Create venv and install
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env

# Download data and train
python -m src.data.fetcher
python -m src.models.trainer

# Get recommendations
india-stock rank --top 10
```

## Project Structure

```
src/
├── core/         # Config, database, utilities
├── data/         # OHLCV fetcher, universe, features
├── models/       # ML training, evaluation, scoring
├── news/         # News fetching & sentiment
├── mcp_server/   # MCP tools for Claude
└── api/          # CLI interface
tests/            # Unit and integration tests
```

## License

MIT
