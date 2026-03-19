"""
scheduler.py — Automated daily/weekly scheduling for the stock research pipeline.

Daily (evening): fetch prices → fetch news → recompute features → score stocks
Weekly (Sunday): refresh fundamentals → retrain models → run backtests
"""

import logging
import schedule
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def daily_pipeline():
    """Run the daily update pipeline."""
    logger.info("=" * 60)
    logger.info("🕐 DAILY PIPELINE STARTED")
    logger.info("=" * 60)

    try:
        # 1. Fetch latest prices
        logger.info("Step 1: Fetching latest OHLCV data...")
        from src.core.database import init_db
        from src.data.fetcher import fetch_and_store_all
        init_db()
        fetch_and_store_all()

        # 2. Fetch macro data
        logger.info("Step 2: Fetching macro data...")
        from src.data.fetcher import fetch_and_store_macro
        fetch_and_store_macro()

        # 3. Recompute features
        logger.info("Step 3: Computing features (V2: 34 features)...")
        from src.data.features import compute_and_store_all
        compute_and_store_all()

        # 4. Score all stocks
        logger.info("Step 4: Scoring all stocks with ensemble...")
        from src.models.scorer import run_scoring
        run_scoring()

        # 5. Run discovery
        logger.info("Step 5: Running news-driven discovery...")
        from src.discovery.discovery import run_discovery
        run_discovery()

        logger.info("✅ DAILY PIPELINE COMPLETED SUCCESSFULLY")

    except Exception as e:
        logger.error(f"❌ DAILY PIPELINE FAILED: {e}")
        raise


def weekly_pipeline():
    """Run the weekly maintenance pipeline."""
    logger.info("=" * 60)
    logger.info("📅 WEEKLY PIPELINE STARTED")
    logger.info("=" * 60)

    try:
        # 1. Refresh fundamentals
        logger.info("Step 1: Refreshing fundamental data...")
        from src.data.fundamentals import fetch_and_store_all_fundamentals
        fetch_and_store_all_fundamentals()

        # 2. Retrain models
        logger.info("Step 2: Retraining ensemble models...")
        from src.models.trainer import run_training
        run_training()

        # 3. Run data quality checks
        logger.info("Step 3: Running data quality monitor...")
        from src.core.monitor import run_monitor
        run_monitor()

        logger.info("✅ WEEKLY PIPELINE COMPLETED SUCCESSFULLY")

    except Exception as e:
        logger.error(f"❌ WEEKLY PIPELINE FAILED: {e}")
        raise


def start_scheduler():
    """Start the automated scheduler."""
    logger.info("🚀 Starting automated scheduler...")

    # Daily at 6 PM IST (12:30 UTC)
    schedule.every().day.at("18:00").do(daily_pipeline)

    # Weekly on Sunday at 10 AM IST
    schedule.every().sunday.at("10:00").do(weekly_pipeline)

    logger.info("  📋 Daily pipeline scheduled at 18:00 IST")
    logger.info("  📋 Weekly pipeline scheduled for Sunday 10:00 IST")

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "daily":
        daily_pipeline()
    elif len(sys.argv) > 1 and sys.argv[1] == "weekly":
        weekly_pipeline()
    else:
        start_scheduler()
