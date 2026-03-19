import pandas as pd
import numpy as np
import joblib
import sqlite3
import os
import sys
from datetime import datetime, timedelta
from tabulate import tabulate

# Add project root to sys path
sys.path.append("/Users/arvinth/Desktop/Ideas/stock-market-analysis")
from src.core.config import DB_PATH
import xgboost as xgb

MODEL_PATH = "/Users/arvinth/Desktop/Ideas/stock-market-analysis/models/xgb_regressor.joblib"
FEATURE_COLS = [
    'close_zscore_20', 'close_zscore_50', 'price_vs_sma20', 'price_vs_sma50', 'price_vs_sma200',
    'rsi_14', 'macd_norm', 'macd_signal_norm', 'macd_hist_norm', 'bb_width',
    'atr_percent', 'volatility_20', 'volume_ratio', 'dist_52w_high', 'dist_52w_low',
    'return_5d', 'return_10d', 'return_20d'
]

def run_eval():
    if not os.path.exists(MODEL_PATH):
        print("Model not found.")
        return
        
    model = joblib.load(MODEL_PATH)
    conn = sqlite3.connect(DB_PATH)
    
    # Get the latest date
    query = "SELECT MAX(date) FROM features"
    latest_date_str = pd.read_sql(query, conn).iloc[0, 0]
    
    # Get the 20th most recent trading date globally
    query_dates = "SELECT DISTINCT date FROM features ORDER BY date DESC LIMIT 25"
    recent_dates = pd.read_sql(query_dates, conn)['date'].tolist()
    if len(recent_dates) < 21:
        print("Not enough history.")
        return
        
    target_past_date = recent_dates[20] # exactly 20 trading days ago
    
    # We want to compare what the model would predict on target_past_date for the latest date
    
    print(f"Comparing predictions made on {target_past_date} to outcomes on {latest_date_str}")
    print("=" * 70)
    
    # Fetch features from 20 days ago
    query_features = f"SELECT * FROM features WHERE date = '{target_past_date}'"
    past_df = pd.read_sql(query_features, conn)
        
    query_today = f"SELECT symbol, close as actual_today FROM ohlcv WHERE date = '{latest_date_str}'"
    today_df = pd.read_sql(query_today, conn)
    
    # Fetch prices from 20 days ago
    query_past_price = f"""
    SELECT symbol, close as past_price 
    FROM ohlcv 
    WHERE date = '{target_past_date}'
    """
    past_price_df = pd.read_sql(query_past_price, conn)
    
    # Merge
    merged = pd.merge(past_df, past_price_df, on='symbol', how='inner')
    merged = pd.merge(merged, today_df, on='symbol', how='inner')
    
    if merged.empty:
        print("No matching data found.")
        return
        
    X = merged[FEATURE_COLS]
    preds = model.predict(X)
    merged['raw_predicted_return'] = preds
    
    # ---------------------------------------------------------
    # NEW MODEL LOGIC: Regime-Aware Crash Overlay
    # Calculate the median 20-day return of the market 20 days ago
    median_20d = past_df['return_20d'].median() * 100
    
    # If the market was already crashing, heavily down-weight the Regressor's optimism!
    if median_20d < -4.0:
        print(f"** ACTIVE OVERLAY: Backtest was in CRASH REGIME (Median Return: {median_20d:.2f}%) **")
        print("** Down-weighting all bullish single-stock signals by 60% **\n")
        # Suppress positive predictions heavily, let negative predictions stay or amplify
        adjusted_preds = np.where(preds > 0, preds * 0.4, preds * 1.2)
    else:
        adjusted_preds = preds
        
    merged['predicted_return'] = adjusted_preds
    # ---------------------------------------------------------
    
    merged['predicted_today'] = merged['past_price'] * (1 + merged['predicted_return'])
    
    # Calculate errors
    merged['actual_return'] = (merged['actual_today'] - merged['past_price']) / merged['past_price']
    merged['error_pct'] = ((merged['predicted_today'] - merged['actual_today']) / merged['actual_today']) * 100
    merged['direction_correct'] = np.sign(merged['predicted_return']) == np.sign(merged['actual_return'])
    
    # Formatting
    merged['Symbol'] = merged['symbol']
    merged['Past Price'] = merged['past_price'].round(2)
    merged['Actual Today'] = merged['actual_today'].round(2)
    merged['Predicted Today'] = merged['predicted_today'].round(2)
    merged['Actual Return'] = (merged['actual_return'] * 100).map("{:+.2f}%".format)
    merged['Predicted Return'] = (merged['predicted_return'] * 100).map("{:+.2f}%".format)
    merged['Error'] = merged['error_pct'].map("{:+.2f}%".format)
    
    # Sort by error magnitude
    merged['abs_error'] = merged['error_pct'].abs()
    merged = merged.sort_values('abs_error')
    
    display_cols = ['Symbol', 'Past Price', 'Actual Today', 'Predicted Today', 'Actual Return', 'Predicted Return', 'Error']
    final_table = merged[display_cols]
    
    print(tabulate(final_table, headers='keys', tablefmt='pipe', showindex=False))
    
    mae = merged['abs_error'].mean()
    direction_acc = merged['direction_correct'].mean() * 100
    print("\n" + "=" * 70)
    print(f"Average Target Miss Error: {mae:.2f}%")
    print(f"Directional Accuracy (Hit Rate): {direction_acc:.1f}%")

if __name__ == "__main__":
    try:
        run_eval()
    except Exception as e:
        import traceback
        traceback.print_exc()

