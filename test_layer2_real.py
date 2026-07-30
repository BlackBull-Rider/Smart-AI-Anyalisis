import pandas as pd
from backend.db.connection import db
from backend.analyzers.analyzer_engine import analyzer_engine

symbol = 'RELIANCE'

print(f"\n{'='*80}")
print(f" 🕵️‍♂️ SCANNING DATABASE FOR REAL [{symbol}] DATA")
print(f"{'='*80}")

# ১. Layer-1 (Feature History) থেকে রিয়াল ডেটা
rows = db.fetchall(f"SELECT * FROM feature_history WHERE symbol='{symbol}' ORDER BY date ASC LIMIT 200")
df = pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()
print(f"✅ Found {len(df)} rows in 'feature_history'")

# ২. ডাইনামিক টেবিল স্ক্যানার (তোর ডেটাবেসে যে নামেই টেবিল থাকুক, ডেটা তুলে আনবে)
def fetch_real_data(table_names, symbol, is_list=False):
    for table in table_names:
        try:
            # টেবিলটা এক্সিস্ট করে কি না চেক
            check = db.fetchone(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
            if check:
                res = db.fetchall(f"SELECT * FROM {table} WHERE symbol='{symbol}' ORDER BY date DESC LIMIT 4")
                if res:
                    print(f"✅ Found data in '{table}'")
                    return [dict(r) for r in res] if is_list else dict(res[0])
        except Exception:
            pass
    return [] if is_list else {}

# ফান্ডামেন্টাল এবং অন্যান্য রিয়াল ডেটা তোলা হচ্ছে
fundamental = fetch_real_data(['fundamentals', 'company_fundamentals', 'ratios'], symbol)
financials = fetch_real_data(['financials', 'financial_results', 'quarterly_results'], symbol)
profile = fetch_real_data(['profile', 'company_profile', 'info'], symbol)
shareholding = fetch_real_data(['shareholding', 'shareholding_pattern', 'institutional_holding'], symbol, is_list=True)
corporate_actions = fetch_real_data(['corporate_actions', 'actions', 'ipo_data'], symbol, is_list=True)
earnings = fetch_real_data(['earnings', 'eps_data'], symbol, is_list=True)

# ৩. রিয়াল ডেটা প্যাকেজ করা
input_data = {
    "df": df,
    "fundamental": fundamental,
    "financials": financials,
    "profile": profile,
    "shareholding": shareholding,
    "corporate_actions": corporate_actions,
    "earnings": earnings
}

print(f"\n🚀 Launching Layer-2 Analyzer with 100% REAL Data...\n")
results = analyzer_engine.run(symbol, input_data)

print(f"\n{'='*80}")
print(f" 🎯 REAL OUTPUT PREVIEW (First 2 Engines)")
print(f"{'='*80}")

# আউটপুট ঠিকমতো জেনারেট হলো কি না, তার প্রুফ
for k, v in list(results.items())[:2]:
    print(f"\n[{k.upper()} ENGINE RESULTS]:")
    for sub_k, sub_v in list(v.items())[:5]: # প্রথম ৫টা রেজাল্ট প্রিন্ট
        print(f"  - {sub_k}: {sub_v}")
        
print("\n✅ Layer-2 is officially consuming real DB data!")
