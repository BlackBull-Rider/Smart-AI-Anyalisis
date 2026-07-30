import json
import pandas as pd
import numpy as np
from backend.db.connection import db
from backend.analyzers.analyzer_engine import analyzer_engine

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if pd.isna(obj): return None
        if isinstance(obj, (np.integer, np.floating)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)

symbol = "AAKASH"

# Fetch only the time-series data, Analyzer Engine will fetch the rest!
rows = db.fetchall("SELECT * FROM feature_history WHERE symbol=? ORDER BY date ASC", (symbol,))
df = pd.DataFrame([dict(r) for r in rows])

if df.empty:
    print("❌ No feature history found!")
    exit()

print(f"\n🚀 FIRING LAYER-2 PURE ENGINE FOR {symbol}...")
results = analyzer_engine.run(symbol, df)

print(f"\n{'='*80}\n 🧠 LAYER-2 ORIGINAL JSON OUTPUT (NO PLACEHOLDERS)\n{'='*80}\n")
print(json.dumps(results, indent=4, cls=NumpyEncoder))
