import pandas as pd
from backend.db.connection import db

symbol = 'RELIANCE'

print(f"\n{'='*80}")
print(f" 🔍 X-RAY VISION: DATA PAYLOAD GOING INTO LAYER-2 ENGINES FOR [{symbol}]")
print(f"{'='*80}")

def fetch_table_data(query, is_list=False):
    try:
        res = db.fetchall(query)
        if res:
            return [dict(r) for r in res] if is_list else dict(res[0])
    except Exception:
        pass
    return [] if is_list else {}

def display_data(title, data_dict, max_display=20):
    print(f"\n📦 {title.upper()} (Total Fields: {len(data_dict)})")
    if not data_dict:
        print("   └── ⚠️ No Data Found")
        return
        
    # শুধু যেগুলোতে ভ্যালু আছে (None/Null নয়) সেগুলো ফিল্টার করছি
    valid_data = {k: v for k, v in data_dict.items() if v is not None and str(v).strip() != ''}
    
    count = 0
    for k, v in valid_data.items():
        if count < max_display:
            # ভ্যালুগুলো সুন্দর করে ফরম্যাট করা হচ্ছে
            if isinstance(v, float):
                formatted_v = f"{v:.4f}"
            else:
                formatted_v = str(v)[:50] # বেশি বড় টেক্সট হলে কেটে দেবে
            
            print(f"   ├── {k:<25}: {formatted_v}")
            count += 1
            
    if len(valid_data) > max_display:
        print(f"   └── ... and {len(valid_data) - max_display} more fields injected!")

# 🟢 TYPE-1: Technical Data (Last Row)
df_rows = fetch_table_data(f"SELECT * FROM feature_history WHERE symbol='{symbol}' ORDER BY date ASC LIMIT 200", is_list=True)
if df_rows:
    df = pd.DataFrame(df_rows)
    last_row = df.iloc[-1].to_dict()
    display_data("Type-1: Technical Indicators (feature_history latest date)", last_row, max_display=25)

# 🔵 TYPE-2: Fundamental & Financial Data
fundamentals = fetch_table_data(f"SELECT * FROM fundamental_data WHERE symbol='{symbol}'")
display_data("Type-2: Fundamentals (fundamental_data)", fundamentals, max_display=20)

financials = fetch_table_data(f"SELECT * FROM financial_data WHERE symbol='{symbol}' ORDER BY fiscal_year DESC LIMIT 1")
display_data("Type-2: Financial Results (financial_data latest row)", financials, max_display=20)

shareholding = fetch_table_data(f"SELECT * FROM shareholding_data WHERE symbol='{symbol}' ORDER BY quarter DESC LIMIT 1")
display_data("Type-2: Institutional Shareholding", shareholding, max_display=10)

ipo = fetch_table_data(f"SELECT * FROM ipo_data WHERE symbol='{symbol}'")
display_data("Type-2: IPO Metrics", ipo, max_display=10)

print(f"\n{'='*80}")
print(" ✅ All these real DB values are merged dynamically and passed to the 12 Engines!")
print(f"{'='*80}\n")
