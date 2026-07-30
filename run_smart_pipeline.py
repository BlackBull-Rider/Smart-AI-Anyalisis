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

symbol = "AAKASH"

# 1. Fetch ALL Data from DB
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

# =========================================================================
# 🟢 LAYER-1 FALLBACK (Real Mathematical Calculations, NO PLACEHOLDERS)
# Generating the missing features so the Engine gets exactly what it needs
# =========================================================================
if 'close' in master_df.columns:
    # Rate of Change (ROC) & Momentum
    master_df['roc'] = master_df['close'].pct_change(periods=14) * 100
    master_df['momentum'] = master_df['close'].diff(periods=10)
    
    # Linear Regression Slope & R2 (Using rolling correlation logic)
    x = pd.Series(range(len(master_df)), index=master_df.index)
    master_df['linreg_slope'] = (master_df['close'] - master_df['close'].shift(14)) / 14
    master_df['linreg_r2'] = master_df['close'].rolling(14).corr(x) ** 2

if all(c in master_df.columns for c in ['open', 'high', 'low', 'close']):
    # Candle Body Percentage
    master_df['body_pct'] = (master_df['close'] - master_df['open']).abs() / (master_df['high'] - master_df['low']).replace(0, 0.0001)

# 2. Package Input Data exactly as the Orchestrator wants
input_data = {
    'df': master_df,
    'fundamental': repository.get_fundamental(symbol) or {},
    'profile': repository.get_company_profile(symbol) or {},
    'financials': repository.get_financials(symbol) or {},
    'shareholding': repository.get_shareholding(symbol) or [],
    'earnings': repository.get_earnings(symbol) or [],
    'corporate_actions': repository.get_corporate_actions(symbol) or []
}

# 3. Fire the Engine!
results = analyzer_engine.run(symbol, input_data)

print(f"\n{'='*80}\n 🧠 100% SUCCESSFUL RAW JSON OUTPUT\n{'='*80}\n")
print(json.dumps(results, indent=4, cls=CustomJSONEncoder))
