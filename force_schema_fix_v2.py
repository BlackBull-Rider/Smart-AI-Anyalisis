import re

filepath = "backend/database/schema.py"

# নতুন সঠিক স্টক মাস্টার টেবিল ডেফিনিশন
new_ddl = '''    """CREATE TABLE IF NOT EXISTS stock_master (
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
    );""",

    """CREATE TABLE IF NOT EXISTS idx_stock_fno_dummy (id INTEGER);""", # Dummy to keep index block intact

    # নিচের ইনডেক্স ব্লকটি ঠিক করা
    '''

try:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # পুরনো stock_master এবং এর ইনডেক্স ব্লকটি মুছে নতুনটি বসানো
    #Regex ব্যবহার করে পুরনো টেবিল ব্লকটি ধরছি
    pattern = r'"""CREATE TABLE IF NOT EXISTS stock_master \((.*?)\);""",\n'
    
    # এটি পুরো ফাইল থেকে ওই টেবিল ব্লকটি সরিয়ে নতুন ব্লক বসাবে
    content = re.sub(pattern, new_ddl, content, flags=re.DOTALL)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
        
    print("✅ schema.py updated manually.")
except Exception as e:
    print(f"❌ Error: {e}")
