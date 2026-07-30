import json
import pandas as pd
import numpy as np
from backend.db.connection import db
from backend.repository.stock_repository import repository
from backend.analyzers.analyzer_engine import analyzer_engine
import warnings
warnings.simplefilter(action='ignore')

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if pd.isna(obj): return None
        if isinstance(obj, (np.integer, np.floating)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)

symbol = "AAKASH" # তোর টেস্টিং সিম্বল

# =========================================================================
# 1. PULL REAL DATA FROM ENTIRE DATABASE (NO PLACEHOLDERS)
# =========================================================================
EXCLUDED_TABLES = {'analyzer_history', 'pipeline_audit', 'update_log', 'pipeline_statistics', 'retry_queue'}
try:
    rows = db.fetchall("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [dict(r).get('name', list(dict(r).values())[0]) for r in rows]
except:
    rows = db.fetchall("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
    tables = [dict(r).get('table_name', list(dict(r).values())[0]) for r in rows]

target_tables = [t for t in tables if not t.startswith('sqlite_') and t not in EXCLUDED_TABLES]
master_df = pd.DataFrame()

for table in target_tables:
    try:
        col_info = db.fetchall(f"PRAGMA table_info({table});")
        cols = [dict(c)['name'].lower() for c in col_info]
        if 'symbol' in cols and 'date' in cols:
            rows = db.fetchall(f"SELECT * FROM {table} WHERE symbol='{symbol}' ORDER BY date ASC")
        elif 'date' in cols and table == 'macro_environment':
            rows = db.fetchall(f"SELECT * FROM {table} ORDER BY date ASC")
        else: continue
        
        if rows:
            temp_df = pd.DataFrame([dict(r) for r in rows])
            temp_df.columns = [str(c).lower().strip() for c in temp_df.columns]
            temp_df.set_index('date', inplace=True)
            if master_df.empty: master_df = temp_df
            else: master_df = master_df.join(temp_df[temp_df.columns.difference(master_df.columns)], how='outer')
    except: pass

master_df.reset_index(inplace=True)
for col in master_df.columns:
    if col not in ['date', 'symbol', 'timestamp']:
        master_df[col] = pd.to_numeric(master_df[col], errors='coerce')

# Pulling specific dictionary data via repository
fundamental_data = repository.get_fundamental(symbol) or {}
profile_data = repository.get_company_profile(symbol) or {}
financials_data = repository.get_financials(symbol) or {}
shareholding_data = repository.get_shareholding(symbol) or []
earnings_data = repository.get_earnings(symbol) or []
corporate_actions_data = repository.get_corporate_actions(symbol) or []

# =========================================================================
# 2. THE REAL DATA AUDIT DUMP (PRINTING ACTUAL VALUES)
# =========================================================================
print(f"\n{'='*80}")
print(f" 🔍 REAL DATA AUDIT FOR [{symbol}] (VALUES ABOUT TO ENTER ENGINE)")
print(f"{'='*80}")

if not master_df.empty:
    latest_data = master_df.iloc[-1].to_dict()
    print("\n[ DATAFRAME LATEST ROW VALUES ]:")
    for key, value in sorted(latest_data.items()):
        val_str = f"{value:.4f}" if isinstance(value, float) and not pd.isna(value) else str(value)
        print(f"  {key:<25} : {val_str}")
else:
    print("\n❌ DATAFRAME IS EMPTY!")

print("\n[ REPOSITORY DICTIONARY VALUES ]:")
for dict_name, dict_data in [("Fundamental", fundamental_data), ("Profile", profile_data), ("Financials", financials_data)]:
    print(f"\n--- {dict_name} ---")
    if dict_data:
        for key, value in sorted(dict_data.items()):
            print(f"  {key:<25} : {value}")
    else:
        print("  (Empty or None)")

print("\n[ LIST DATA (Shareholding, Earnings, Corp Actions) ]:")
print(f"  Shareholding Records   : {len(shareholding_data)}")
print(f"  Earnings Records       : {len(earnings_data)}")
print(f"  Corp Actions Records   : {len(corporate_actions_data)}")

# =========================================================================
# 3. PUSH REAL DATA INTO THE 12 ENGINES & EXTRACT RAW JSON
# =========================================================================
input_data = {
    'df': master_df,
    'fundamental': fundamental_data,
    'profile': profile_data,
    'financials': financials_data,
    'shareholding': shareholding_data,
    'earnings': earnings_data,
    'corporate_actions': corporate_actions_data
}

print(f"\n{'='*80}")
print(f" ⚙️  FIRING ALL 12 ANALYZERS WITH THIS RAW DATA...")
print(f"{'='*80}")

# ENGINE EXECUTION
results = analyzer_engine.run(symbol, input_data)

# =========================================================================
# 4. FINAL JSON OUTPUT
# =========================================================================
print(f"\n{'='*80}")
print(f" 🧠 FINAL RAW JSON OUTPUT FROM 12 ENGINES")
print(f"{'='*80}\n")
print(json.dumps(results, indent=4, cls=CustomJSONEncoder))
print(f"\n{'='*80}\n")

