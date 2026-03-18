"""Tests for news sentiment."""
from src.news.sentiment import analyze_sentiment, analyze_sentiment_batch, get_aggregate_sentiment

def test_positive():
    r = analyze_sentiment("Company reports record profit and strong growth")
    assert r["label"] == "positive" and r["compound"] > 0

def test_negative():
    r = analyze_sentiment("Stock crashes amid fraud allegations and losses")
    assert r["label"] == "negative" and r["compound"] < 0

def test_batch():
    assert len(analyze_sentiment_batch(["great earnings", "terrible loss"])) == 2

def test_aggregate():
    agg = get_aggregate_sentiment(analyze_sentiment_batch(["great profit", "bad loss"]))
    assert "avg_compound" in agg and "overall_label" in agg
