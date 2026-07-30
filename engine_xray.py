import pandas as pd
from backend.repository.stock_repository import repository
from backend.db.connection import db
from backend.analyzers.analyzer_engine import analyzer_engine

symbol = "AAKASH"

# 1. Fetch exact DB columns
rows = db.fetchall("SELECT * FROM feature_history WHERE symbol=? ORDER BY date ASC LIMIT 1", (symbol,))
if not rows:
    rows = db.fetchall("SELECT * FROM historical_data WHERE symbol=? ORDER BY date ASC LIMIT 1", (symbol,))

if not rows:
    print(f"❌ No data found in DB for {symbol}")
    exit()

db_columns = {str(c).lower().strip() for c in rows[0].keys()}

# Simulate the Aliases mapping from Analyzer Engine
mapped_aliases = ['rsi', 'atr', 'macd_line', 'macd_signal', 'macd_histogram', 'adx', 'roc', 'momentum', 'linreg_slope', 'linreg_r2', 'vwap', 'supertrend', 'bos', 'choch']
db_columns.update(mapped_aliases)

print(f"\n{'='*75}\n 🕵️ X-RAY: WHAT THE 12 ANALYZERS ARE GETTING VS MISSING\n{'='*75}")

for name, engine in analyzer_engine.engines.items():
    print(f"\n⚙️ [ENGINE]: {name.upper()}")
    
    # Check Technical DataFrame Features
    if hasattr(engine, 'EXPECTED_SCHEMA'):
        expected = {str(f).lower().strip() for f in engine.EXPECTED_SCHEMA}
        found = expected.intersection(db_columns)
        missing = expected - db_columns
        
        print(f"   📊 TECHNICAL INDICATORS:")
        print(f"      ✅ GOT FROM DB ({len(found)}): {', '.join(sorted(list(found))) if found else 'None'}")
        if missing:
            print(f"      ❌ MISSING (Auto-padded to 0.0) ({len(missing)}): {', '.join(sorted(list(missing)))}")
        else:
            print(f"      ❌ MISSING: None (Engine got 100% of required technical data!)")
    else:
        print(f"   📊 TECHNICAL INDICATORS: No strict EXPECTED_SCHEMA. It digests all DB columns dynamically.")

    # Check External Data (Fundamental, Institutional, IPO)
    if name == "fundamental":
        print(f"   💼 EXTERNAL DATA (API/DB):")
        f_data = repository.get_fundamental(symbol) or {}
        if f_data:
            valid_keys = [k for k, v in f_data.items() if v is not None]
            null_keys = [k for k, v in f_data.items() if v is None]
            print(f"      ✅ Valid Fundamentals: {len(valid_keys)} keys found.")
            if null_keys: print(f"      ❌ Missing/Null Fundamentals (Sanitized to 0.0): {len(null_keys)} keys.")
        else:
            print("      ❌ Missing: Entire fundamental dictionary is empty.")
            
    elif name == "institutional":
        print(f"   🏦 EXTERNAL DATA (Shareholding DB):")
        s_data = repository.get_shareholding(symbol) or []
        if s_data:
            valid_keys = [k for k, v in s_data[0].items() if v is not None]
            null_keys = [k for k, v in s_data[0].items() if v is None]
            print(f"      ✅ Valid Holdings: {', '.join(valid_keys)}")
            print(f"      ❌ Missing Holdings (Sanitized to 0.0): {', '.join(null_keys)}")
        else:
            print("      ❌ Missing: No shareholding records found.")

    elif name == "ipo":
        print(f"   🚀 EXTERNAL DATA (Corporate Actions DB):")
        c_data = repository.get_corporate_actions(symbol) or []
        print(f"      ✅ Found {len(c_data)} corporate actions events (Splits/Dividends).")

print(f"\n{'='*75}\n")
