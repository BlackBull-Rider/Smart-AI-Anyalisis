import pandas as pd
from backend.db.connection import db
from backend.analyzers.analyzer_engine import analyzer_engine

# ১. Layer-1 (Feature History) থেকে RELIANCE এর লেটেস্ট ডেটা তোলা হচ্ছে
rows = db.fetchall("SELECT * FROM feature_history WHERE symbol='RELIANCE' ORDER BY date ASC LIMIT 100")

if not rows:
    print("❌ No data found for RELIANCE in feature_history table!")
    exit()

df = pd.DataFrame([dict(r) for r in rows])

# ২. Input Data Dictionary রেডি করা (এখানে আপাতত শুধু টেকনিক্যাল ডেটা দিচ্ছি টেস্টিংয়ের জন্য)
# তোর ডেটাবেসে ফান্ডামেন্টাল টেবিল থাকলে সেগুলোও এখানে অ্যাড করা যাবে
input_data = {
    "df": df,
    "fundamental": {"pe_ratio": 25.5, "roe": 15.2, "market_cap": 1800000}, # ডেমো ডেটা
    "financials": {"revenue_growth": 12.5, "profit_growth": 8.0},
    "profile": {"sector": "Energy", "industry": "Oil & Gas"},
    "shareholding": [{"promoter": 50.5, "fii": 25.2, "dii": 15.1}],
    "corporate_actions": [],
    "earnings": []
}

# ৩. Layer-2 Analyzer Engine ফায়ার করা
print("🚀 Launching Layer-2 Analyzer Test with Real Layer-1 Data...\n")
results = analyzer_engine.run('RELIANCE', input_data)

print("\n🎯 LAYER-2 EXECUTION COMPLETE!")
