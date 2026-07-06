import json
import traceback
import pandas as pd
import numpy as np  # <-- এটা ইমপোর্ট করা না থাকলে করে নিস

# ----------------- CUSTOM JSON ENCODER -----------------
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# ----------------- LAYER 1 IMPORTS -----------------
from backend.data.data_fetcher import fetch_ohlcv
from backend.registry.feature_engine import build_features

# ----------------- TREND PIPELINE IMPORTS -----------------
from backend.analyzers.trend_analyzer import TrendAnalyzer
from backend.engines.trend_engine import TrendEngine

# ----------------- MOMENTUM PIPELINE IMPORTS -----------------
from backend.analyzers.momentum_analyzer import MomentumAnalyzer
from backend.engines.momentum_engine import MomentumEngine

def run_combined_institutional_test(symbol="FEDERALBNK"):
    print(f"🚀 [LAYER 1A] Fetching REAL Data for {symbol} from SQLite Database...")
    try:
        df_raw = fetch_ohlcv(symbol, limit=500)
        print(f"✅ Layer 1A Ready: Loaded {len(df_raw)} rows of raw OHLCV data.")
    except Exception as e:
        print(f"❌ [CRITICAL] DataFetcher Failed: {e}")
        return

    print("\n⚙️ [LAYER 1B] Pumping Data through Feature Engine (Strictly No Placeholders)...")
    try:
        df_featured = build_features(df_raw)
        print(f"✅ Layer 1B Ready: Features calculated. Total columns: {len(df_featured.columns)}")
    except Exception as e:
        print(f"❌ [CRITICAL] FeatureEngine Failed: {e}")
        traceback.print_exc()
        return

    # ==========================================
    # TREND PIPELINE (LAYER 2 & 3)
    # ==========================================
    print("\n🧠 [TREND PIPELINE] Running Analyzer & Engine...")
    trend_output = {}
    try:
        t_analyzer = TrendAnalyzer()
        trend_l2 = t_analyzer.analyze(df_featured)
        print(f"   ✅ Trend Layer 2: Generated {len(trend_l2.get('direction', {}).get('evidence', []))} Evidences.")
        
        t_engine = TrendEngine()
        trend_output = t_engine.generate_score(trend_l2, df_featured)
        print("   ✅ Trend Layer 3: Scored successfully!")
    except Exception as e:
        print(f"   ❌ [ERROR] Trend Pipeline Failed: {e}")
        traceback.print_exc()

    # ==========================================
    # MOMENTUM PIPELINE (LAYER 2 & 3)
    # ==========================================
    print("\n⚡ [MOMENTUM PIPELINE] Running Analyzer & Engine V6.0...")
    momentum_output = {}
    try:
        m_analyzer = MomentumAnalyzer()
        momentum_l2 = m_analyzer.analyze(df_featured)
        print(f"   ✅ Momentum Layer 2: Generated Base Evidences.")
        
        m_engine = MomentumEngine()
        momentum_output = m_engine.generate_score(momentum_l2, df_featured)
        print("   ✅ Momentum Layer 3: Physics & Markov math executed perfectly!")
    except Exception as e:
        print(f"   ❌ [ERROR] Momentum Pipeline Failed: {e}")
        traceback.print_exc()

    # ==========================================
    # FINAL MASTER OUTPUT
    # ==========================================
    print("\n🎯 [FINAL MASTER OUTPUT] Proof of 100% Real Execution:\n")
    
    master_result = {
        "Symbol": symbol,
        "Trend_Analysis": trend_output,
        "Momentum_Analysis": momentum_output
    }
    
    # সুন্দর করে ফরম্যাট করে প্রিন্ট করা (Custom Encoder ব্যবহার করে)
    print(json.dumps(master_result, indent=4, cls=NpEncoder))

if __name__ == "__main__":
    # তোর পোর্টফোলিও স্টক FEDERALBNK দিয়ে ফুল পাইপলাইন টেস্ট
    run_combined_institutional_test("FEDERALBNK") 
