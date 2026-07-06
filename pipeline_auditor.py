import pandas as pd
from backend.data.data_fetcher import fetch_ohlcv
from backend.registry.feature_engine import build_features
from backend.analyzers.trend_analyzer import TrendAnalyzer
from backend.engines.trend_engine import TrendEngine

def audit_pipeline(symbol="FEDERALBNK"):
    print(f"🔍 AUDITING PIPELINE FOR: {symbol}\n")

    # --- LAYER 1: DATA FETCHING ---
    df_raw = fetch_ohlcv(symbol, limit=100)
    print(f"Layer 1 (Raw) - Schema: {list(df_raw.columns)}")
    print(f"Layer 1 - Nulls: {df_raw.isna().sum().sum()}\n")

    # --- LAYER 1B: FEATURE ENGINE ---
    df_featured = build_features(df_raw)
    print(f"Layer 1B (Featured) - Schema: {list(df_featured.columns)}")
    print(f"Layer 1B - Nulls: {df_featured.isna().sum().sum()}")
    
    # Check for NaN guessing
    if df_featured.isna().sum().sum() > 0:
        print("⚠️ WARNING: Layer 1B generated NaNs! (Potential 'Guessing' risk)")
    print("-" * 50)

    # --- LAYER 2: ANALYZER (The Contract Check) ---
    analyzer = TrendAnalyzer()
    # আমরা এখানে দেখব Analyzer ঠিক কী কী ইনপুট চাইছে
    required_cols = analyzer.req_cols
    print(f"Layer 2 (Analyzer) - Required Columns: {required_cols}")
    
    missing = [c for c in required_cols if c not in df_featured.columns]
    if missing:
        print(f"❌ CRITICAL: Analyzer is missing data! Missing: {missing}")
    else:
        print("✅ Contract Met: All columns found for Analyzer.")
    print("-" * 50)

    # --- LAYER 3: ENGINE ---
    l2_output = analyzer.analyze(df_featured)
    # ইঞ্জিন কী কী এভিডেন্স পাচ্ছে?
    evidence = l2_output.get("direction", {}).get("evidence", [])
    print(f"Layer 3 (Engine) - Evidence Feed count: {len(evidence)}")
    for ev in evidence[:5]: # প্রথম ৫টা স্যাম্পল দেখি
        print(f"  > Input: {ev['type']} | Weight: {ev['weight']}")

audit_pipeline()
