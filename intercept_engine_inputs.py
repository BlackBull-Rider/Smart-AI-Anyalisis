import pandas as pd
from backend.db.connection import db
from backend.analyzers.analyzer_engine import analyzer_engine
import warnings
warnings.simplefilter(action='ignore')

symbol = 'RELIANCE'

print(f"\n{'='*90}")
print(f" 🚨 STRICT AUDIT: INTERCEPTING RAW INPUT DATA PASSED TO 12 ANALYZERS")
print(f"{'='*90}")

# ১. নরমাল ডেটা ফেচিং (এটাই ইঞ্জিনে যাবে)
df = pd.DataFrame([dict(r) for r in db.fetchall(f"SELECT * FROM feature_history WHERE symbol='{symbol}' ORDER BY date ASC LIMIT 200")] or [])
fundamental = dict(db.fetchone(f"SELECT * FROM fundamental_data WHERE symbol='{symbol}'") or {})
financials = dict(db.fetchone(f"SELECT * FROM financial_data WHERE symbol='{symbol}' ORDER BY fiscal_year DESC LIMIT 1") or {})
profile = dict(db.fetchone(f"SELECT * FROM company_profile WHERE symbol='{symbol}'") or {})
ipo = dict(db.fetchone(f"SELECT * FROM ipo_data WHERE symbol='{symbol}'") or {})
shareholding = [dict(r) for r in db.fetchall(f"SELECT * FROM shareholding_data WHERE symbol='{symbol}' ORDER BY quarter DESC LIMIT 1")] or []
macro = dict(db.fetchone(f"SELECT * FROM macro_environment ORDER BY date DESC LIMIT 1") or {})

input_data = {
    "df": df, "fundamental": fundamental, "financials": financials,
    "profile": profile, "ipo": ipo, "macro": macro, "shareholding": shareholding
}

original_analyzers = {}

# ২. The Pure Interceptor (No placeholders, no expected schema, just pure input print)
for name, engine in analyzer_engine.engines.items():
    original_analyzers[name] = engine.analyze
    
    def make_interceptor(engine_name, original_method):
        def interceptor(input_df, **input_kwargs):
            print(f"\n\n{'='*80}")
            print(f" 📥 INTERCEPTED EXACT INPUT PASSED TO: [ {engine_name.upper()} ENGINE ]")
            print(f"{'='*80}")
            
            # [A] DataFrame Input Print
            print(f"\n📊 1. DATAFRAME INJECTED (Shape: {len(input_df)} Rows x {len(input_df.columns)} Columns)")
            last_row = input_df.iloc[-1].to_dict() if not input_df.empty else {}
            
            # ডেটাফ্রেমের যে কলামগুলোতে ভ্যালু আছে, সেগুলো সব প্রিন্ট করছি (৫টা করে এক লাইনে)
            valid_df_data = {k: v for k, v in last_row.items() if pd.notna(v) and str(v).strip() != ''}
            cols_str = []
            for k, v in valid_df_data.items():
                if isinstance(v, float):
                    cols_str.append(f"{k}: {v:.2f}")
                else:
                    cols_str.append(f"{k}: {str(v)[:15]}")
            
            print(f"-> Latest DataFrame Values (Showing all mapped & raw columns):")
            for i in range(0, min(len(cols_str), 50), 5): # স্ক্রিন পরিষ্কার রাখতে প্রথম ৫০টা দেখাচ্ছি, তুই চাইলে লুপের লিমিট তুলে দিতে পারিস
                print(" | ".join(cols_str[i:i+5]))
            if len(cols_str) > 50:
                print(f" ... and {len(cols_str) - 50} more valid columns inside the DataFrame.")

            # [B] Kwargs (Extra Dicts) Input Print
            print(f"\n📁 2. EXTRA KWARGS (Dictionaries) INJECTED:")
            if not input_kwargs:
                print("   └── NONE")
            else:
                for kw_key, kw_val in input_kwargs.items():
                    if isinstance(kw_val, dict):
                        print(f"\n   └── {kw_key.upper()} (Dict Size: {len(kw_val)}) -> EXACT KEYS PASSED:")
                        kw_str = [f"{k}: {v}" for k, v in kw_val.items() if pd.notna(v) and str(v).strip() != '']
                        for i in range(0, min(len(kw_str), 12), 4):
                            print("       " + " | ".join(kw_str[i:i+4]))
                        if len(kw_str) > 12:
                            print(f"       ... and {len(kw_str) - 12} more keys.")
                    elif isinstance(kw_val, list) and kw_val:
                        print(f"\n   └── {kw_key.upper()} (List) -> EXACT KEYS PASSED:")
                        if isinstance(kw_val[0], dict):
                            kw_str = [f"{k}: {v}" for k, v in kw_val[0].items() if pd.notna(v) and str(v).strip() != '']
                            for i in range(0, min(len(kw_str), 12), 4):
                                print("       " + " | ".join(kw_str[i:i+4]))
                    else:
                        print(f"   └── {kw_key.upper()}: {kw_val}")
            
            # আসল ইঞ্জিনে ডেটা পাস করে দেওয়া হচ্ছে
            return original_method(input_df, **input_kwargs)
        return interceptor
        
    engine.analyze = make_interceptor(name, original_analyzers[name])

# ইঞ্জিন ফায়ার করা হলো
analyzer_engine.run(symbol, input_data)

# রিভোক ইন্টারসেপ্টর
for name, engine in analyzer_engine.engines.items():
    engine.analyze = original_analyzers[name]

print("\n✅ PURE INPUT INTERCEPTION COMPLETE!")
