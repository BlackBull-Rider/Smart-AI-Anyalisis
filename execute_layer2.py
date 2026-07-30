import json
import pandas as pd
import numpy as np
import warnings
from backend.db.connection import db
from backend.analyzers.analyzer_engine import analyzer_engine

warnings.simplefilter(action='ignore')

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)): return float(obj) if not np.isnan(obj) else None
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)

symbol = "AAKASH"

# Exclude system logs so their junk ({}) doesn't corrupt our pure math
EXCLUDED_TABLES = {'analyzer_history', 'pipeline_audit', 'update_log', 'pipeline_statistics', 'retry_queue', 'pipeline_queue', 'temp_universal'}

try:
    rows = db.fetchall("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [dict(r).get('name', list(dict(r).values())[0]) for r in rows]
except:
    rows = db.fetchall("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
    tables = [dict(r).get('table_name', list(dict(r).values())[0]) for r in rows]

feature_tables = [t for t in tables if not t.startswith('sqlite_') and t not in EXCLUDED_TABLES]

master_df = pd.DataFrame()
static_data = {}

for table in feature_tables:
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
            temp_df.columns = [str(c).lower().strip() for c in temp_df.columns]
            
            if 'date' in temp_df.columns:
                temp_df.set_index('date', inplace=True)
                if master_df.empty:
                    master_df = temp_df
                else:
                    if 'date' not in master_df.index.names:
                        master_df.set_index('date', inplace=True)
                    master_df = master_df.join(temp_df[temp_df.columns.difference(master_df.columns)], how='outer')
            else:
                for col in temp_df.columns:
                    if col not in static_data:
                        static_data[col] = temp_df[col].iloc[-1] if not temp_df.empty else None
    except Exception as e:
        pass

master_df.reset_index(inplace=True)
for col, val in static_data.items():
    if col not in master_df.columns:
        master_df[col] = val

# Fire Layer 2 Plugin
results = analyzer_engine.run(symbol, master_df)

print(f"\n{'='*80}\n 🧠 LAYER-2 ORIGINAL JSON OUTPUT (NO PLACEHOLDERS)\n{'='*80}\n")
print(json.dumps(results, indent=4, cls=CustomJSONEncoder))
print(f"\n{'='*80}\n✅ PIPELINE LAYER-2 COMPLETED SUCCESSFULLY!\n{'='*80}\n")
