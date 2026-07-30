import json
import pandas as pd
from backend.repository.stock_repository import repository
from backend.db.connection import db  # 🔴 FIX: db সরাসরি ইম্পোর্ট করা হলো

# তোর সোর্স কোডের লোকেশন অনুযায়ী ইম্পোর্ট
try:
    from backend.analyzers.analyzer_engine import analyzer_engine
except ModuleNotFoundError:
    from backend.engine.analyzer_engine import analyzer_engine

def run_12_analyzers(symbol: str):
    print(f"\n🔍 Fetching Data for {symbol}...")

    # 🔴 FIX: repository.db এর বদলে সরাসরি db ব্যবহার করা হলো
    rows = db.fetchall("SELECT * FROM feature_history WHERE symbol=? ORDER BY date ASC", (symbol,))
    if not rows:
        print("⚠️ No feature_history found. Falling back to raw historical_data...")
        rows = db.fetchall("SELECT * FROM historical_data WHERE symbol=? ORDER BY date ASC", (symbol,))

    if not rows:
        print(f"❌ No data found for {symbol}!")
        return

    df = pd.DataFrame([dict(r) for r in rows])
    print(f"📊 Loaded {len(df)} candles with {len(df.columns)} features. Firing up the 12 Analyzers...\n")
    
    # Analyzer Engine রান করা হচ্ছে
    results = analyzer_engine.run(symbol, df)
    
    print(f"{'='*70}\n 🧠 12 ANALYZERS RAW JSON OUTPUT FOR: {symbol}\n{'='*70}\n")
    print(json.dumps(results, indent=4, default=str))
    print(f"\n{'='*70}\n")

if __name__ == '__main__':
    run_12_analyzers("AAKASH")
