import json
import pandas as pd
import numpy as np
import warnings
from backend.db.connection import db
from backend.analyzers.analyzer_engine import analyzer_engine

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('future.no_silent_downcasting', True)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super(NumpyEncoder, self).default(obj)

symbol = "AAKASH"
print(f"\n{'='*75}\n 🚀 FLAWLESS DB SCAN & ENGINE EXECUTION\n{'='*75}\n")

try:
    rows = db.fetchall("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [dict(r).get('name', list(dict(r).values())[0]) for r in rows]
except:
    rows = db.fetchall("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
    tables = [dict(r).get('table_name', list(dict(r).values())[0]) for r in rows]

tables = [t for t in tables if not t.startswith('sqlite_')]

master_df = pd.DataFrame()
static_data = {} # For tables without dates (IPO, Fundamental)

print("🔍 EXTRACTING & NORMALIZING DATA FROM 27 TABLES...")
for table in tables:
    try:
        col_info = db.fetchall(f"PRAGMA table_info({table});")
        cols = [dict(c)['name'].lower() for c in col_info]
        
        if 'symbol' in cols and 'date' in cols:
            rows = db.fetchall(f"SELECT * FROM {table} WHERE symbol='{symbol}' ORDER BY date ASC")
        elif 'date' in cols and table == 'macro_environment':
            rows = db.fetchall(f"SELECT * FROM {table} ORDER BY date ASC")
        elif 'symbol' in cols and 'date' not in cols:
            rows = db.fetchall(f"SELECT * FROM {table} WHERE symbol='{symbol}' LIMIT 1")
        else:
            continue
            
        if rows:
            temp_df = pd.DataFrame([dict(r) for r in rows])
            # 🟢 FIX 1: LOWERCASE ALL COLUMNS IMMEDIATELY
            temp_df.columns = [str(c).lower().strip() for c in temp_df.columns]
            
            if 'date' in temp_df.columns:
                temp_df.set_index('date', inplace=True)
                if master_df.empty:
                    master_df = temp_df
                else:
                    if 'date' not in master_df.index.names:
                        master_df.set_index('date', inplace=True)
                    cols_to_add = temp_df.columns.difference(master_df.columns)
                    master_df = master_df.join(temp_df[cols_to_add], how='outer')
            else:
                # Store non-timeseries data to broadcast later
                for col in temp_df.columns:
                    if col not in static_data:
                        static_data[col] = temp_df[col].iloc[0]
    except Exception as e:
        pass

if master_df.empty:
    print("❌ Critical Error: No data found!")
    exit()

master_df.reset_index(inplace=True)

# 🟢 FIX 2: BROADCAST STATIC DATA (IPO, Fundamental) TO MASTER_DF
for col, val in static_data.items():
    if col not in master_df.columns:
        master_df[col] = val

# Pass through analyzer normalization (handles aliases like BOS, CHOCH)
master_df = analyzer_engine._normalize_columns(master_df)

print(f"✅ SUCCESSFULLY ASSEMBLED {len(master_df.columns)} LOWERCASE COLUMNS!\n")

print(f"📊 ACCURATE DATA INTEGRITY CHECK (X-RAY):")
for name, engine in analyzer_engine.engines.items():
    if hasattr(engine, 'EXPECTED_SCHEMA'):
        expected = {str(f).lower().strip() for f in engine.EXPECTED_SCHEMA}
        found = expected.intersection(set(master_df.columns))
        missing = expected - set(master_df.columns)
        print(f"   ⚙️ {name.upper()}: Found {len(found)} columns. Missing {len(missing)}")

print(f"\n⚡ FIRING ALL 12 ENGINES...")
results = analyzer_engine.run(symbol, master_df)

print(f"\n{'='*75}\n 🧠 12 ANALYZERS RAW JSON OUTPUT\n{'='*75}\n")
print(json.dumps(results, indent=4, cls=NumpyEncoder))

