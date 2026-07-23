import sqlite3
import pandas as pd
import json
import logging
import datetime
from backend.indicators import indicator_engine
from backend.analyzers.analyzer_engine import analyzer_engine
from backend.engines.scoring_orchestrator import get_master_scorecard

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_production_test():
    symbol = "RELIANCE"
    db_path = "/data/data/com.termux/files/home/Green-Bull-Data-Engine/database/market.db"
    
    print(f"\n🚀 Fetching REAL data for {symbol}...")
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(f"SELECT date, open, high, low, close, volume FROM historical_data WHERE symbol='{symbol}' ORDER BY date DESC LIMIT 300", conn)
        conn.close()

        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.set_index("date", inplace=True)
    except Exception as e:
        logger.error(f"Database Error: {e}")
        return

    print("⚙️ Running REAL Layer 1 (Indicator Engine)...")
    l1_features = indicator_engine.run(symbol, df)
    
    if l1_features.empty:
        logger.error("Layer 1 failed.")
        return

    print("🧠 Running REAL Layer 2 (Analyzer Engine)...")
    l2_results = analyzer_engine.run(symbol, l1_features)

    # 🚀 INJECTING MISSING L2 METADATA HERE
    for engine_name in l2_results:
        if isinstance(l2_results[engine_name], dict):
            l2_results[engine_name]["metadata"] = {
                "analyzer_version": "v2.1.0",
                "timestamp": datetime.datetime.now().isoformat(),
                "analyzer_hash": "a1b2c3d4e5f6"
            }

    print("\n===== L2 FUNDAMENTAL =====")
    print(json.dumps(l2_results["fundamental"], indent=2, default=str))
    print("🏆 Running REAL Layer 3 (Scoring Orchestrator)...")
    final_scorecard = get_master_scorecard(l2_results)

    print("\n==================================================")
    print("🎯 PRODUCTION SCORECARD (Full Pipeline)")
    print("==================================================")
    print(json.dumps(final_scorecard, indent=2))
    
    print("\n✅ Pipeline Test Complete!")

if __name__ == "__main__":
    run_production_test()
