import sqlite3
import re

db_path = "/data/data/com.termux/files/home/Smart-AI-Anyalisis/backend/database/universe.db"

# ১. schema.py ফিক্স করা
filepath = "backend/database/schema.py"
try:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # is_fno যোগ করা
    target = 'is_active BOOLEAN NOT NULL DEFAULT 1,'
    if target in content and 'is_fno' not in content:
        content = content.replace(target, 'is_active BOOLEAN NOT NULL DEFAULT 1,\n        is_fno BOOLEAN DEFAULT 0,')
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print("✅ schema.py updated.")
except Exception as e:
    print(f"❌ Patching Error: {e}")

# ২. ডাটাবেসে কলাম চেক ও অ্যাড করা
try:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # কলাম আছে কি না চেক
    cur.execute("PRAGMA table_info(stock_master)")
    cols = [col[1] for col in cur.fetchall()]
    
    if 'is_fno' not in cols:
        print("🔧 Adding missing column 'is_fno'...")
        cur.execute("ALTER TABLE stock_master ADD COLUMN is_fno BOOLEAN DEFAULT 0;")
        conn.commit()
    else:
        print("✅ Column 'is_fno' already exists.")
    conn.close()
except Exception as e:
    print(f"❌ DB Error: {e}")
