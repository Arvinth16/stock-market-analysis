import pandas as pd
from src.models.scorer import score_all_stocks
from src.core.database import get_db

def dump_table():
    df = score_all_stocks()
    if df.empty:
        print("No predictions found.")
        return
        
    # Get median momentum
    median_20d = df['momentum_20d'].median()
    in_crash_regime = median_20d < -4.0
    
    # We apply the same regime overlay if they want to see the "new model" predictions
    if in_crash_regime:
        print(f"⚠️ MARKET CRASH REGIME DETECTED (Universe Median 20-Day: {median_20d:.1f}%)")
        print("⚠️ Down-weighting all bullish single-stock signals by 60%\n")
        # Ensure predicted_return is down-weighted
        df['predicted_return'] = df['predicted_return'].apply(
            lambda x: x * 0.4 if x > 0 else x * 1.2
        )
        # Recalculate target price
        df['target_price'] = df['last_price'] * (1 + df['predicted_return'])

    # Format the table
    df_out = pd.DataFrame()
    df_out['Symbol'] = df['symbol']
    df_out['Current Price'] = df['last_price'].map("₹{:.2f}".format)
    df_out['Target (20d)'] = df['target_price'].map("₹{:.2f}".format)
    df_out['Target %'] = (df['predicted_return'] * 100).map("{:+.2f}%".format)
    df_out['Signal (Class Prob)'] = (df['model_score'] * 100).map("{:.1f}%".format)

    # Sort by absolute Target % so the biggest movers are at the top/bottom
    df_out['abs_target'] = df['predicted_return'].abs()
    df_out = df_out.sort_values(by='abs_target', ascending=False).drop(columns=['abs_target'])
    
    print(df_out.to_markdown(index=False))

if __name__ == "__main__":
    dump_table()
