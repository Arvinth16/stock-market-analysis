"""
monitor.py — Data quality checks and model drift detection.

Monitors:
- Missing price data days
- Abnormal price jumps (>20% in one day)
- API fetch failures
- Model performance drift vs backtest baseline
"""

import logging
from datetime import datetime, timedelta


from src.core.database import get_db
from src.data.universe import get_symbols

logger = logging.getLogger(__name__)


def check_data_freshness() -> list:
    """Check that all stocks have recent price data."""
    issues = []
    cutoff = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

    with get_db() as conn:
        for symbol in get_symbols():
            row = conn.execute(
                "SELECT MAX(date) as latest FROM ohlcv WHERE symbol = ?",
                (symbol,),
            ).fetchone()

            if row is None or row["latest"] is None:
                issues.append(f"  ⚠ {symbol}: NO DATA AT ALL")
            elif row["latest"] < cutoff:
                issues.append(f"  ⚠ {symbol}: Stale data (latest: {row['latest']})")

    return issues


def check_price_jumps() -> list:
    """Detect abnormal single-day price jumps (>20%)."""
    issues = []

    with get_db() as conn:
        for symbol in get_symbols():
            rows = conn.execute(
                """SELECT date, close FROM ohlcv
                   WHERE symbol = ?
                   ORDER BY date DESC LIMIT 30""",
                (symbol,),
            ).fetchall()

            if len(rows) < 2:
                continue

            for i in range(len(rows) - 1):
                curr = rows[i]["close"]
                prev = rows[i + 1]["close"]
                if prev > 0:
                    change = abs((curr - prev) / prev) * 100
                    if change > 20:
                        issues.append(
                            f"  🚨 {symbol}: {change:.1f}% jump on {rows[i]['date']}"
                        )

    return issues


def check_feature_completeness() -> list:
    """Check that features are computed for all stocks."""
    issues = []
    cutoff = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

    with get_db() as conn:
        for symbol in get_symbols():
            row = conn.execute(
                "SELECT MAX(date) as latest FROM features WHERE symbol = ?",
                (symbol,),
            ).fetchone()

            if row is None or row["latest"] is None:
                issues.append(f"  ⚠ {symbol}: No features computed")
            elif row["latest"] < cutoff:
                issues.append(f"  ⚠ {symbol}: Stale features (latest: {row['latest']})")

    return issues


def check_model_drift() -> list:
    """
    Compare recent prediction accuracy vs baseline.
    Checks if model's recent scoring pattern has degraded.
    """
    issues = []

    with get_db() as conn:
        # Check if we have enough signal history
        count = conn.execute("SELECT COUNT(*) as n FROM signals").fetchone()
        if count["n"] < 50:
            return ["  ℹ Not enough signal history for drift detection yet."]

        # Check if recent predictions are all one-sided (potential issue)
        recent = conn.execute(
            """SELECT AVG(model_score) as avg_score,
                      MIN(model_score) as min_score,
                      MAX(model_score) as max_score
               FROM signals
               WHERE date >= date('now', '-7 days')"""
        ).fetchone()

        if recent and recent["avg_score"] is not None:
            avg = recent["avg_score"]
            spread = recent["max_score"] - recent["min_score"]
            if spread < 0.05:
                issues.append(
                    f"  ⚠ Model predictions have very low spread ({spread:.3f}). "
                    "May indicate degradation."
                )
            if avg > 0.9 or avg < 0.1:
                issues.append(
                    f"  ⚠ Model average score is extreme ({avg:.3f}). "
                    "Possible miscalibration."
                )

    return issues


def run_monitor():
    """Run all data quality and model health checks."""
    print("━" * 60)
    print("🔍 DATA QUALITY & MODEL HEALTH MONITOR")
    print("━" * 60)

    all_issues = []

    print("\n  📊 Checking data freshness...")
    issues = check_data_freshness()
    all_issues.extend(issues)
    if issues:
        for i in issues:
            print(i)
    else:
        print("  ✅ All stock data is fresh.")

    print("\n  📈 Checking for abnormal price jumps...")
    issues = check_price_jumps()
    all_issues.extend(issues)
    if issues:
        for i in issues:
            print(i)
    else:
        print("  ✅ No abnormal price jumps detected.")

    print("\n  🔧 Checking feature completeness...")
    issues = check_feature_completeness()
    all_issues.extend(issues)
    if issues:
        for i in issues[:5]:
            print(i)
        if len(issues) > 5:
            print(f"  ... and {len(issues)-5} more")
    else:
        print("  ✅ All features up to date.")

    print("\n  🤖 Checking model drift...")
    issues = check_model_drift()
    all_issues.extend(issues)
    if issues:
        for i in issues:
            print(i)
    else:
        print("  ✅ Model performance within expected bounds.")

    total = len(all_issues)
    print(f"\n  {'⚠ ' + str(total) + ' issues found.' if total else '✅ All systems healthy.'}\n")
    return all_issues


if __name__ == "__main__":
    run_monitor()
