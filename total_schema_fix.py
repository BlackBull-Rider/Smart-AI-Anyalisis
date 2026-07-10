import sqlite3
import os

db_path = "/data/data/com.termux/files/home/Smart-AI-Anyalisis/backend/database/universe.db"
schema_file = "backend/database/schema.py"

print("Step 1: Patching schema.py...")
with open(schema_file, "r", encoding="utf-8") as f:
    content = f.read()

# is_fno যোগ করা (যদি না থাকে)
if 'is_fno BOOLEAN DEFAULT 0,' not in content:
    content = content.replace('is_active BOOLEAN NOT NULL DEFAULT 1,', 
                              'is_active BOOLEAN NOT NULL DEFAULT 1,\n        is_fno BOOLEAN DEFAULT 0,')
    with open(schema_file, "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ schema.py patched.")

print("Step 2: Dropping existing 'stock_master' table...")
try:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = OFF;")
    cur.execute("DROP TABLE IF EXISTS stock_master;")
    cur.execute("PRAGMA foreign_keys = ON;")
    conn.commit()
    conn.close()
    print("✅ Table 'stock_master' dropped successfully.")
except Exception as e:
    print(f"❌ Error dropping table: {e}")

print("Step 3: Running init_db logic...")
from backend.database.schema import schema_engine
try:
    schema_engine.create_schema()
    print("🎉 SUCCESS: Everything is clean and synchronized!")
except Exception as e:
    print(f"❌ FINAL ERROR: {e}")
