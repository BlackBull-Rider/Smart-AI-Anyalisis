import yfinance as yf
import pandas as pd
from backend.registry.feature_engine import build_features
from backend.analyzers.trend_analyzer import TrendAnalyzer
from backend.analyzers.momentum_analyzer import MomentumAnalyzer

def run_test():
    print("--- টেষ্টিং শুরু হচ্ছে ---")
    
    # ১. ডেটা ডাউনলোড (MultiIndex ইস্যু ফিক্সড)
    df = yf.download("RELIANCE.NS", period="200d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]

    # ২. ইঞ্জিন রান (৭০০+ ইন্ডিকেটর অটোমেটিক প্রসেস হবে)
    print("ফিচার ইঞ্জিন রান করছি...")
    df_processed = build_features(df)
    print(f"ফিচার তৈরি হয়েছে। টোটাল কলাম সংখ্যা: {len(df_processed.columns)}")

    # ৩. TrendAnalyzer টেস্ট
    print("\n--- TrendAnalyzer টেস্ট ---")
    try:
        t_analyzer = TrendAnalyzer()
        t_res = t_analyzer.analyze(df_processed)
        print(f"Status: {t_res['direction']['status']}")
        print(f"Score: {t_res['direction']['score']}")
    except Exception as e:
        print(f"TrendAnalyzer Error: {e}")

    # ৪. MomentumAnalyzer টেস্ট
    print("\n--- MomentumAnalyzer টেস্ট ---")
    try:
        m_analyzer = MomentumAnalyzer()
        m_res = m_analyzer.analyze(df_processed)
        print(f"Momentum Status: {m_res['momentum_strength']['status']}")
        print(f"Swing Readiness: {m_res['swing_readiness']['state']}")
    except Exception as e:
        print(f"MomentumAnalyzer Error: {e}")

if __name__ == "__main__":
    run_test()
