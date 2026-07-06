import pandas as pd
import json

# তোর লোকাল মডিউলগুলো ইম্পোর্ট কর (পাথগুলো তোর প্রজেক্ট অনুযায়ী মিলিয়ে নিস)
from backend.data.data_fetcher import DataFetcher  # <-- তোর আসল লোকাল ফেচার
from backend.analyzers.trend_analyzer import TrendAnalyzer
from backend.engines.trend_engine import TrendEngine

def run_real_database_test():
    print("🚀 [LAYER 1] Fetching REAL Data from YOUR LOCAL DATABASE...")
    
    # তোর ডেটা ফেচার ইনিশিয়ালাইজ কর
    fetcher = DataFetcher()
    
    # তোর ডাটাবেস থেকে রিয়েল স্টকের ডেটা টান (ধর SEDEMAC বা RELIANCE)
    # Note: তোর ফেচারের মেথডের আসল নাম যেটা (যেমন fetch_data বা get_historical_data), সেটা এখানে দিস।
    df = fetcher.fetch_data("SEDEMAC") 
    
    if df is None or df.empty:
        print("❌ [CRITICAL] ডেটাবেস থেকে কোনো ডেটা আসেনি! তোর DataFetcher চেক কর।")
        return
        
    print(f"✅ Layer 1 Ready! Loaded {len(df)} rows from Database.")

    print("\n🧠 [LAYER 2] Running TrendAnalyzer (Extracting Raw Signals)...")
    analyzer = TrendAnalyzer()
    
    try:
        l2_output = analyzer.analyze(df)
        print(f"✅ Layer 2 Generated: {len(l2_output.get('direction', {}).get('evidence', []))} Directional Evidences.")
    except Exception as e:
        print(f"❌ [CRITICAL] Layer 2 CRASHED: {e}")
        return

    print("\n⚡ [LAYER 3] Running TrendEngine V3.0 (Institutional Bayesian Scoring)...")
    engine = TrendEngine()
    
    try:
        final_output = engine.generate_score(l2_output, df)
        print("\n🎯 [FINAL OUTPUT] Proof of Execution:\n")
        print(json.dumps(final_output, indent=4))
    except Exception as e:
        print(f"❌ [CRITICAL] Layer 3 CRASHED: {e}")

if __name__ == "__main__":
    run_real_database_test()
