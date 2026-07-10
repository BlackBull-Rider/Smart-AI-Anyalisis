import re
import os

filepath = "backend/database/schema.py"

# নতুন সঠিক স্টক মাস্টার টেবিল ডেফিনিশন
new_stock_master_ddl = '''    """CREATE TABLE IF NOT EXISTS stock_master (
        symbol TEXT PRIMARY KEY,
        company_name TEXT NOT NULL CHECK(length(company_name) > 0),
        exchange TEXT NOT NULL CHECK(exchange IN ('NSE', 'BSE', 'MCX', 'NYSE', 'NASDAQ')),
        sector TEXT,
        industry TEXT,
        market_cap_category TEXT CHECK(market_cap_category IN ('LARGE', 'MID', 'SMALL', 'MICRO')),
        is_active BOOLEAN NOT NULL DEFAULT 1,
        is_fno BOOLEAN DEFAULT 0,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );""",'''

try:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # স্টক মাস্টার টেবিল ব্লকটি খুঁজে বের করা এবং রিপ্লেস করা
    pattern = r'"""CREATE TABLE IF NOT EXISTS stock_master \((.*?)\);""",\n'
    
    # re.DOTALL দিয়ে মাল্টিলাইন ম্যাচ করছি
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, new_stock_master_ddl + '\n', content, flags=re.DOTALL)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print("✅ schema.py successfully overwritten with correct stock_master DDL.")
    else:
        print("❌ Could not find stock_master block in schema.py! Please check the file.")

except Exception as e:
    print(f"❌ Error: {e}")
