import sqlite3
import pandas as pd
from backend.analyzers.analyzer_engine import AnalyzerEngine

# ইঞ্জিন ইনিশিয়ালাইজ করা
engine = AnalyzerEngine()
conn = engine._get_db_connection()
symbol = "RELIANCE"

# ১. ডেটা লোড করা
df = engine._fetch_feature_history(conn, symbol)
ctx_dicts = [
    engine._fetch_single_row(conn, "fundamental_data", symbol),
    engine._fetch_single_row(conn, "financial_data", symbol)
]

# ২. কলামগুলো মার্জ করা
for ctx in ctx_dicts:
    for k, v in ctx.items():
        if k not in ['symbol', 'date', 'updated_at']:
            df[k] = v

conn.close()

# ৩. ডিটেকটিভ কাজ: চেক করা টার্গেট কি গুলো DF-এ আছে কিনা
print(f"\n🔥 DEBUGGING MAPPING FOR {symbol} 🔥")
print(f"Total Columns in DataFrame: {len(df.columns)}")
print("--------------------------------------------------")

target_keys = ['sales', 'net_income', 'vwap', 'rsi'] # এক্সাম্পল কি

for tk in target_keys:
    if tk in df.columns:
        val = df[tk].iloc[-1]
        print(f"Target '{tk}': FOUND in DF! Value: {val}")
    else:
        # যদি ফাউন্ড না হয়, তবে দেখাব DF-এ ঠিক কী কী কলাম আছে
        print(f"Target '{tk}': NOT FOUND. Searching for Aliases...")
        aliases = engine.ALIAS_MAP.get(tk, [])
        found_alias = None
        for a in aliases:
            if a in df.columns:
                found_alias = a
                break
        
        if found_alias:
            print(f"  -> Found Alias '{found_alias}' in DataFrame! But mapping failed.")
        else:
            print(f"  -> NONE of these aliases found in DF: {aliases}")
            print(f"  -> DF Columns list (Sample): {df.columns.tolist()[:20]}...")

