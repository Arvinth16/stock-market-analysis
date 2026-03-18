# Feature Set Documentation

> This document describes the features and labels used by the ML model.

## Technical Indicators (computed from OHLCV)

| Feature | Description | Window | Category |
|---|---|---|---|
| `close_zscore_20` | Z-score of Close Price | 20 days | Trend |
| `close_zscore_50` | Z-score of Close Price | 50 days | Trend |
| `price_vs_sma20` | Price distance from SMA(20) as % | 20 days | Trend |
| `price_vs_sma50` | Price distance from SMA(50) as % | 50 days | Trend |
| `price_vs_sma200` | Price distance from SMA(200) as % | 200 days | Trend |
| `rsi_14` | Relative Strength Index | 14 days | Momentum |
| `macd_norm` | MACD normalized by price (%) | 12/26 days | Momentum |
| `macd_signal_norm` | MACD signal normalized (%) | 9 days | Momentum |
| `macd_hist_norm` | MACD histogram normalized (%) | — | Momentum |
| `bb_width` | Bollinger Band width (% of SMA) | 20 days | Volatility |
| `atr_percent` | Average True Range as % of price | 14 days | Volatility |
| `volatility_20` | Annualized 20-day rolling volatility | 20 days | Volatility |
| `volume_ratio` | Current volume ÷ 20-day avg volume | 20 days | Volume |
| `dist_52w_high` | Distance from 52-week high (%) | 252 days | Price Action |
| `dist_52w_low` | Distance from 52-week low (%) | 252 days | Price Action |
| `return_5d` | 5-day return (backward-looking) | 5 days | Momentum |
| `return_10d` | 10-day return (backward-looking) | 10 days | Momentum |
| `return_20d` | 20-day return (backward-looking) | 20 days | Momentum |

## Label

- **Target**: Binary classification
- **Positive (1)**: 20-trading-day forward return > +2%
- **Negative (0)**: 20-trading-day forward return ≤ +2%
- **Horizon**: ~1 calendar month (20 trading days)

## Leakage Prevention

- All features use **only** data available at or before the prediction date
- Labels use **strictly future** data (forward returns)
- Walk-forward cross-validation with expanding training window

## Scoring Weights

The final stock score combines three signals:

| Signal | Weight | Source |
|---|---|---|
| Model probability | 55% | XGBoost predicted probability of positive class |
| News sentiment | 30% | Average VADER compound score from recent news |
| Price momentum | 15% | 20-day backward return (normalized) |
