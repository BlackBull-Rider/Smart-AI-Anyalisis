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
print(f"\n{'='*75}\n 🚀 ULTIMATE DB SCAN + ENGINE EXECUTION (NO EXCUSES)\n{'='*75}\n")

# 1. FETCH DYNAMICALLY FROM ALL TABLES
try:
    rows = db.fetchall("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [dict(r).get('name', list(dict(r).values())[0]) for r in rows]
except:
    rows = db.fetchall("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
    tables = [dict(r).get('table_name', list(dict(r).values())[0]) for r in rows]

tables = [t for t in tables if not t.startswith('sqlite_')]
master_df = pd.DataFrame()

print("🔍 EXTRACTING DATA FROM THE ENTIRE DATABASE...")
for table in tables:
    try:
        col_info = db.fetchall(f"PRAGMA table_info({table});")
        cols = [dict(c)['name'].lower() for c in col_info]
        
        if 'symbol' in cols and 'date' in cols:
            rows = db.fetchall(f"SELECT * FROM {table} WHERE symbol='{symbol}' ORDER BY date ASC")
        elif 'date' in cols and table == 'macro_environment':
            rows = db.fetchall(f"SELECT * FROM {table} ORDER BY date ASC")
        else:
            continue
            
        if rows:
            temp_df = pd.DataFrame([dict(r) for r in rows])
            temp_df.set_index('date', inplace=True)
            
            if master_df.empty:
                master_df = temp_df
            else:
                if 'date' not in master_df.index.names:
                    master_df.set_index('date', inplace=True)
                
                # Safely merge without creating duplicate columns (_dup)
                cols_to_add = temp_df.columns.difference(master_df.columns)
                master_df = master_df.join(temp_df[cols_to_add], how='outer')
    except Exception as e:
        pass

if master_df.empty:
    print("❌ Critical Error: No data found in ANY table!")
    exit()

# Reset index to make 'date' a regular column again
master_df.reset_index(inplace=True)
master_df = master_df.loc[:, ~master_df.columns.duplicated()]

print(f"✅ SUCCESSFULLY ASSEMBLED {len(master_df.columns)} COLUMNS FROM ACROSS ALL TABLES!\n")

# 2. X-RAY (What engines see vs what they want)
print(f"📊 DATA INTEGRITY CHECK (X-RAY):")
for name, engine in analyzer_engine.engines.items():
    if hasattr(engine, 'EXPECTED_SCHEMA'):
        expected = {str(f).lower().strip() for f in engine.EXPECTED_SCHEMA}
        found = expected.intersection(set(master_df.columns))
        missing = expected - set(master_df.columns)
        print(f"   ⚙️ {name.upper()}: Found {len(found)} columns. Missing {len(missing)}")
    else:
        print(f"   ⚙️ {name.upper()}: Dynamic schema (consumes all {len(master_df.columns)} columns)")

# 3. RUN ENGINES
print(f"\n⚡ FIRING ALL 12 ENGINES...")
results = analyzer_engine.run(symbol, master_df)

# 4. OUTPUT
print(f"\n{'='*75}\n 🧠 12 ANALYZERS RAW JSON OUTPUT\n{'='*75}\n")
print(json.dumps(results, indent=4, cls=NumpyEncoder))
print(f"\n{'='*75}\n✅ ALL ENGINES EXECUTED SUCCESSFULLY!\n{'='*75}\n")

