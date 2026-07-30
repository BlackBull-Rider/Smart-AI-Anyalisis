import pandas as pd
from backend.db.connection import db
from backend.analyzers.analyzer_engine import analyzer_engine

import warnings
warnings.simplefilter(action='ignore')

symbol = 'RELIANCE'

print(f"\n{'='*80}")
print(f" 🔬 DEEP X-RAY: LIVE PAYLOADS (NO SCHEMA BULLSHIT) FOR [{symbol}]")
print(f"{'='*80}")

def fetch_table_data(query, is_list=False):
    try:
        res = db.fetchall(query)
        if res: return [dict(r) for r in res] if is_list else dict(res[0])
    except Exception: pass
    return [] if is_list else {}

df_rows = fetch_table_data(f"SELECT * FROM feature_history WHERE symbol='{symbol}' ORDER BY date ASC LIMIT 200", is_list=True)
df = pd.DataFrame(df_rows) if df_rows else pd.DataFrame()
fundamental = fetch_table_data(f"SELECT * FROM fundamental_data WHERE symbol='{symbol}'")
financials = fetch_table_data(f"SELECT * FROM financial_data WHERE symbol='{symbol}' ORDER BY fiscal_year DESC LIMIT 1")
profile = fetch_table_data(f"SELECT * FROM company_profile WHERE symbol='{symbol}'")
ipo = fetch_table_data(f"SELECT * FROM ipo_data WHERE symbol='{symbol}'")
shareholding = fetch_table_data(f"SELECT * FROM shareholding_data WHERE symbol='{symbol}' ORDER BY quarter DESC LIMIT 4", is_list=True)
macro = fetch_table_data("SELECT * FROM macro_environment ORDER BY date DESC LIMIT 1")

input_data = {
    "df": df, "fundamental": fundamental, "financials": financials,
    "profile": profile, "ipo": ipo, "macro": macro, "shareholding": shareholding
}

# 🔴 THE PURE INTERCEPTOR (No EXPECTED_SCHEMA)
original_analyzers = {}

for name, engine in analyzer_engine.engines.items():
    original_analyzers[name] = engine.analyze
    
    def make_wrapper(eng_name, real_analyze_method):
        def wrapper(analyzer_df, **kwargs):
            print(f"\n{'='*60}")
            print(f" 📥 REAL PAYLOAD DELIVERED TO: {eng_name.upper()} ENGINE")
            print(f"{'='*60}")
            
            last_row = analyzer_df.iloc[-1].to_dict() if not analyzer_df.empty else {}
            
            print(f"📊 DataFrame Injected: {len(analyzer_df.columns)} Total Columns")
            print(f"🔍 Top 15 Exact Values Passed (From Latest Date):")
            
            # ডাইরেক্ট ডেটাফ্রেমের লেটেস্ট রোর ভ্যালুগুলো প্রিন্ট করছি (স্কিমা ছাড়া)
            count = 0
            for key, val in last_row.items():
                if count < 15: # শুধু প্রথম ১৫টা দেখাচ্ছি যাতে স্ক্রিন ভরে না যায়
                    if isinstance(val, float):
                        print(f"   ├── {key:<20}: {val:.4f}")
                    else:
                        print(f"   ├── {key:<20}: {val}")
                    count += 1
            print(f"   └── ... (and {len(analyzer_df.columns) - 15} more columns)")

            # এক্সট্রা ডিকশনারি কী কী যাচ্ছে
            if kwargs:
                print(f"\n📁 Extra Raw Dictionaries Passed Directly:")
                for kwarg_name, kwarg_val in kwargs.items():
                    data_size = len(kwarg_val) if isinstance(kwarg_val, (dict, list)) else 'Empty'
                    print(f"   └── {kwarg_name.upper():<15}: Contains {data_size} keys")
                    
            return real_analyze_method(analyzer_df, **kwargs)
        return wrapper
        
    engine.analyze = make_wrapper(name, engine.analyze)

print("\n🚀 Running Orchestrator with Pure Interceptors...\n")

results = analyzer_engine.run(symbol, input_data)

# রিভোক ইন্টারসেপ্টর
for name, engine in analyzer_engine.engines.items():
    engine.analyze = original_analyzers[name]

print("\n✅ LIVE X-RAY COMPLETE!")
