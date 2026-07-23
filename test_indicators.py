import sqlite3
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from backend.indicators.core import pattern
from backend.indicators.core import volatility
from backend.indicators.core import support_resistance

db_path = "/data/data/com.termux/files/home/Green-Bull-Data-Engine/database/market.db"
conn = sqlite3.connect(db_path)
df = pd.read_sql_query("SELECT date, open, high, low, close, volume FROM historical_data WHERE symbol='20MICRONS' ORDER BY date DESC LIMIT 300", conn)
conn.close()

df["date"] = pd.to_datetime(df["date"])
df.sort_values("date", inplace=True)
df.set_index("date", inplace=True)

print("\n==================================================")
print("1. TESTING PATTERN MODULE")
print("==================================================")
pat_df = pattern.calculate_patterns(df)
print(f"Total Rows: {len(pat_df)}")
if "channel_detected" in pat_df.columns:
    print(f"Nulls in CHANNEL_DETECTED: {pat_df['channel_detected'].isnull().sum()}")
    print(f"Unique values in CHANNEL_DETECTED: {pat_df['channel_detected'].dropna().unique()}")

print("\n==================================================")
print("2. TESTING VOLATILITY MODULE (ATR PERCENTILE)")
print("==================================================")
atr_pct = volatility.atr_percentile(df)
print(f"Total Rows: {len(atr_pct)}")
print(f"Nulls in ATR_PERCENTILE: {atr_pct.isnull().sum()}")

print("\n==================================================")
print("3. TESTING SR MODULE (YEARLY LEVELS)")
print("==================================================")
yearly = support_resistance.yearly_levels(df)
print(f"Total Rows: {len(yearly)}")
print(f"Nulls in YEARLY_HIGH: {yearly['upper'].isnull().sum()}")
print("==================================================\n")
