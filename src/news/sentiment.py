"""
sentiment.py — Sentiment analysis for news headlines using VADER.

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lightweight,
rule-based sentiment analyzer well-suited for financial news headlines.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = None


def _get_analyzer() -> SentimentIntensityAnalyzer:
    """Lazy-load the VADER analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def analyze_sentiment(text: str) -> dict:
    """
    Analyze sentiment of a single text string.

    Returns:
        dict with keys: compound (-1 to 1), pos, neg, neu, label (positive/negative/neutral)
    """
    analyzer = _get_analyzer()
    scores = analyzer.polarity_scores(text)

    # Classify
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return {
        "compound": compound,
        "pos": scores["pos"],
        "neg": scores["neg"],
        "neu": scores["neu"],
        "label": label,
    }


def analyze_sentiment_batch(texts: list[str]) -> list[dict]:
    """Analyze sentiment for a batch of text strings."""
    return [analyze_sentiment(text) for text in texts]


def get_aggregate_sentiment(sentiments: list[dict]) -> dict:
    """
    Compute aggregate sentiment from a list of individual scores.

    Returns:
        dict with avg_compound, positive_pct, negative_pct, neutral_pct, overall_label
    """
    if not sentiments:
        return {
            "avg_compound": 0.0,
            "positive_pct": 0.0,
            "negative_pct": 0.0,
            "neutral_pct": 0.0,
            "overall_label": "neutral",
        }

    compounds = [s["compound"] for s in sentiments]
    labels = [s["label"] for s in sentiments]
    n = len(sentiments)

    avg = sum(compounds) / n
    pos_pct = labels.count("positive") / n * 100
    neg_pct = labels.count("negative") / n * 100
    neu_pct = labels.count("neutral") / n * 100

    if avg >= 0.05:
        overall = "positive"
    elif avg <= -0.05:
        overall = "negative"
    else:
        overall = "neutral"

    return {
        "avg_compound": round(avg, 4),
        "positive_pct": round(pos_pct, 1),
        "negative_pct": round(neg_pct, 1),
        "neutral_pct": round(neu_pct, 1),
        "overall_label": overall,
    }


if __name__ == "__main__":
    # Quick demo
    headlines = [
        "Reliance Industries reports record quarterly profit",
        "TCS shares fall on weak guidance",
        "HDFC Bank maintains stable growth in Q3",
        "Infosys faces regulatory scrutiny over whistleblower complaints",
        "ITC stock surges after demerger announcement",
    ]
    print("─── Sentiment Analysis Demo ───\n")
    results = analyze_sentiment_batch(headlines)
    for headline, result in zip(headlines, results):
        print(f"  [{result['label']:>8s}] ({result['compound']:+.4f})  {headline}")

    agg = get_aggregate_sentiment(results)
    print(f"\n  Overall: {agg['overall_label']} (avg: {agg['avg_compound']:+.4f})")
    print(f"  Positive: {agg['positive_pct']}% | Negative: {agg['negative_pct']}% | Neutral: {agg['neutral_pct']}%")
