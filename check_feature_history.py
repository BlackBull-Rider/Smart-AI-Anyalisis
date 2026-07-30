import pandas as pd
from backend.db.connection import db

symbol = "AAKASH"

print(f"\n{'='*80}")
print(f" 🔍 DIRECT DATABASE AUDIT: 'feature_history' TABLE FOR [{symbol}]")
print(f"{'='*80}\n")

try:
    # Fetch all data for the symbol directly from feature_history
    query = f"SELECT * FROM feature_history WHERE symbol='{symbol}' ORDER BY date ASC"
    rows = db.fetchall(query)
    
    if not rows:
        print(f"❌ No data found in 'feature_history' for {symbol}! The table might be empty for this stock.")
    else:
        df = pd.DataFrame([dict(r) for r in rows])
        df.columns = [str(c).lower().strip() for c in df.columns]
        
        print(f"📊 Total Rows found for {symbol} in feature_history: {len(df)}")
        print("-" * 80)
        
        # Check the absolute latest row
        latest_row = df.iloc[-1]
        
        valid_data = {}
        empty_data = []
        
        for col in df.columns:
            val = latest_row[col]
            # Verify if the value is genuinely empty/NaN
            if pd.isna(val) or val is None or str(val).strip().lower() in ['nan', 'none', 'null', '']:
                empty_data.append(col)
            else:
                valid_data[col] = val
                
        print(f"\n✅ COLUMNS WITH ACTUAL VALUES (Latest Row): {len(valid_data)}")
        for k, v in sorted(valid_data.items()):
            # Format float for readability
            if isinstance(v, float):
                print(f"  -> {k:<25} : {v:.4f}")
            else:
                print(f"  -> {k:<25} : {v}")
                
        print(f"\n❌ COLUMNS COMPLETELY EMPTY / NaN (Latest Row): {len(empty_data)}")
        # Print empty columns in a grid to save space
        empty_data = sorted(empty_data)
        for i in range(0, len(empty_data), 3):
            row = empty_data[i:i+3]
            print(" | ".join([f"{item:<25}" for item in row]))
            
except Exception as e:
    print(f"🚨 Error querying 'feature_history' table: {e}")

print(f"\n{'='*80}\n")
