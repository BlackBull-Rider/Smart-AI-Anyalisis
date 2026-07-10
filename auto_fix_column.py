import sqlite3
import re

db_path = "backend/database/universe.db"
try:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(market_data);")
    columns = [row[1] for row in cur.fetchall()]
    
    if not columns:
        print("⚠️ Table 'market_data' does not exist in universe.db!")
    else:
        print(f"📌 Columns actually present in your DB: {columns}")
        
        # Date column খোঁজার চেষ্টা
        possible_names = ["date", "Date", "trade_date", "timestamp", "datetime", "TradeDate", "time"]
        actual_col = next((col for col in possible_names if col in columns), None)
        
        if not actual_col:
            print("⚠️ Date column খুঁজে পাওয়া যায়নি। দয়া করে উপরের কলাম লিস্টটি আমাকে দিন।")
        else:
            print(f"✅ Found the correct date column: '{actual_col}'")
            
            filepath = "backend/pipeline/market_pipeline.py"
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            
            # পাইপলাইনে কলামের নাম আপডেট করা
            content = re.sub(r'ts_col="[^"]+"', f'ts_col="{actual_col}"', content)
            content = re.sub(r'ts = rec\.get\("[^"]+"\)', f'ts = rec.get("{actual_col}")', content)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
                
            print("✅ Successfully patched Market Pipeline with the EXACT column name!")
except Exception as e:
    print(f"Error: {e}")
