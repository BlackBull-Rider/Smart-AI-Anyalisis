import json
import pandas as pd
import numpy as np
import warnings

# Mute Pandas Warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('future.no_silent_downcasting', True)

from backend.db.connection import db
from backend.repository.stock_repository import repository
from backend.analyzers.analyzer_engine import analyzer_engine

# Numpy to JSON Encoder (To prevent boolean/float errors)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super(NumpyEncoder, self).default(obj)

symbol = "AAKASH"

print(f"\n{'='*75}\n ⚙️ ASSEMBLING MEGA-DATAFRAME & FIRING ENGINES...\n{'='*75}\n")

# 1. Fetch and Merge Time-Series Data (Historical + Features)
rows_hist = db.fetchall("SELECT * FROM historical_data WHERE symbol=? ORDER BY date ASC", (symbol,))
df_hist = pd.DataFrame([dict(r) for r in rows_hist])

rows_feat = db.fetchall("SELECT * FROM feature_history WHERE symbol=? ORDER BY date ASC", (symbol,))
df_feat = pd.DataFrame([dict(r) for r in rows_feat])

if not df_hist.empty and not df_feat.empty:
    df = pd.merge(df_hist, df_feat, on=['date', 'symbol'], how='outer', suffixes=('', '_dup'))
    df = df.loc[:, ~df.columns.str.endswith('_dup')]
elif not df_feat.empty:
    df = df_feat
else:
    df = df_hist

if df.empty:
    print(f"❌ No time-series data found for {symbol}!")
    exit()

print(f"✅ Master DataFrame ready with {len(df.columns)} columns!")

# 2. Run the Analyzer Engine Pipeline
results = analyzer_engine.run(symbol, df)

# 3. Print the RAW JSON Dump
print(f"\n{'='*75}\n 🧠 12 ANALYZERS RAW JSON OUTPUT FOR: {symbol}\n{'='*75}\n")
print(json.dumps(results, indent=4, cls=NumpyEncoder))
print(f"\n{'='*75}\n✅ ALL 12 ENGINES EXECUTED SUCCESSFULLY!\n{'='*75}\n")

