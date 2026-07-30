import json
import pandas as pd
from backend.repository.stock_repository import repository
from backend.db.connection import db  # 🔴 FIX: db সরাসরি ইম্পোর্ট করা হলো
from backend.analyzers.analyzer_engine import analyzer_engine

symbol = "AAKASH"
print(f"\n{'='*70}\n 🕵️ EXPOSING ENGINE DATA FOR: {symbol}\n{'='*70}\n")

# 1. Fetch DB Data using the correct db connection
rows = db.fetchall("SELECT * FROM feature_history WHERE symbol=? ORDER BY date ASC", (symbol,))
if not rows:
    rows = db.fetchall("SELECT * FROM historical_data WHERE symbol=? ORDER BY date ASC", (symbol,))

df = pd.DataFrame([dict(r) for r in rows])
original_cols = set(df.columns)

# 2. Normalize and check missing columns
normalized_df = analyzer_engine._normalize_columns(df)
normalized_cols = set(normalized_df.columns)

added_cols = normalized_cols - original_cols
print(f"✅ ENGINE RECEIVED: {len(original_cols)} actual features directly from your Database.")
print(f"⚠️ ENGINE PADDED (0.0): {len(added_cols)} missing features were auto-filled to prevent crash.")
if added_cols:
    print(f"   -> Some padded columns: {list(added_cols)[:10]} ...\n")

# 3. Check Institutional (Shareholding) Data Source
shareholding = repository.get_shareholding(symbol)
print(f"📊 SHAREHOLDING DATA FETCHED FROM DB:")
if shareholding:
    for s in shareholding:
        print(f"   -> {s}")
else:
    print("   -> ❌ No Shareholding Data Found (Sanitizer made it 0.0 inside engine)")
print("\n")

# 4. Execute Engine
results = analyzer_engine.run(symbol, df)

print(f"{'='*70}\n 🕯️ CANDLE ENGINE FINAL OUTPUT (Was Crashing Before)\n{'='*70}")
print(json.dumps(results.get("candle", {}), indent=4))

print(f"\n{'='*70}\n 🏦 INSTITUTIONAL ENGINE FINAL OUTPUT (Was Crashing Before)\n{'='*70}")
print(json.dumps(results.get("institutional", {}), indent=4))
print(f"\n{'='*70}\n")

