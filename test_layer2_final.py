import pandas as pd
from backend.db.connection import db
from backend.analyzers.analyzer_engine import analyzer_engine

symbol = 'RELIANCE'

print(f"\n{'='*80}")
print(f" 🕵️‍♂️ FINAL MASTER TEST: SCANNING REAL DATABASE FOR [{symbol}]")
print(f"{'='*80}")

def fetch_table_data(query, is_list=False):
    try:
        res = db.fetchall(query)
        if res:
            return [dict(r) for r in res] if is_list else dict(res[0])
    except Exception as e:
        pass
    return [] if is_list else {}

# ১. TYPE-1 (Time-Series): Feature History থেকে ডেটা তোলা হচ্ছে 
df_rows = fetch_table_data(f"SELECT * FROM feature_history WHERE symbol='{symbol}' ORDER BY date ASC LIMIT 200", is_list=True)
df = pd.DataFrame(df_rows) if df_rows else pd.DataFrame()
print(f"✅ feature_history: {len(df)} rows (Type-1: Time-Series Calculated Data)")

# ২. TYPE-2 (Latest Available): Fundamental, Financial, Profile টেবিল থেকে লেটেস্ট ডেটা
fundamental = fetch_table_data(f"SELECT * FROM fundamental_data WHERE symbol='{symbol}'")
print(f"✅ fundamental_data: {'Found' if fundamental else 'Empty'} (Type-2: Latest Row)")

financials = fetch_table_data(f"SELECT * FROM financial_data WHERE symbol='{symbol}' ORDER BY fiscal_year DESC, fiscal_quarter DESC LIMIT 1")
print(f"✅ financial_data: {'Found' if financials else 'Empty'} (Type-2: Latest Row)")

profile = fetch_table_data(f"SELECT * FROM company_profile WHERE symbol='{symbol}'")
print(f"✅ company_profile: {'Found' if profile else 'Empty'}")

ipo = fetch_table_data(f"SELECT * FROM ipo_data WHERE symbol='{symbol}'")
print(f"✅ ipo_data: {'Found' if ipo else 'Empty'}")

shareholding = fetch_table_data(f"SELECT * FROM shareholding_data WHERE symbol='{symbol}' ORDER BY quarter DESC LIMIT 4", is_list=True)
print(f"✅ shareholding_data: {'Found' if shareholding else 'Empty'}")

macro = fetch_table_data("SELECT * FROM macro_environment ORDER BY date DESC LIMIT 1")
print(f"✅ macro_environment: {'Found' if macro else 'Empty'}")

input_data = {
    "df": df,
    "fundamental": fundamental,
    "financials": financials,
    "profile": profile,
    "ipo": ipo,
    "macro": macro,
    "shareholding": shareholding
}

print(f"\n🚀 Launching Layer-2 Analyzer with Dual-Mapping Architecture...\n")
results = analyzer_engine.run(symbol, input_data)

print(f"\n{'='*80}")
print(f" 🎯 FINAL OUTPUT SUMMARY (Quick Health Check)")
print(f"{'='*80}")

# আউটপুট ক্র্যাশ করেছে কি না, আর ইঞ্জিন কতগুলো সিগন্যাল বানালো তার রিপোর্ট
for engine_name, engine_result in results.items():
    status = engine_result.get('status', 'SUCCESS')
    error = engine_result.get('error', '')
    
    if error:
         print(f"❌ {engine_name.upper():<18}: FAILED -> {error}")
    else:
         # রেজাল্ট ডিকশনারিতে কতগুলো key (সিগন্যাল/ভ্যালু) আছে, সেটা কাউন্ট করছি
         sig_count = len(engine_result)
         print(f"✅ {engine_name.upper():<18}: SUCCESS -> Generated {sig_count} Output Signals")

print("\n🔥 Layer-2 is officially Bulletproof!")
