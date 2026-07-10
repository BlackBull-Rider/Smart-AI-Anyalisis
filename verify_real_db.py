import sqlite3
import os

db_path = "backend/database/universe.db"

if not os.path.exists(db_path):
    print(f"❌ ERROR: File not found at {db_path}")
    exit()

try:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cur.fetchall()]
    print(f"✅ Connection successful. Tables in {db_path}:")
    print(tables)
    
    if 'equity_master' in tables:
        count = cur.execute("SELECT count(*) FROM equity_master").fetchone()[0]
        print(f"✅ equity_master found with {count} rows.")
    else:
        print("❌ ERROR: 'equity_master' table does not exist in this file!")
    
    conn.close()
except Exception as e:
    print(f"❌ DB Access Error: {e}")
