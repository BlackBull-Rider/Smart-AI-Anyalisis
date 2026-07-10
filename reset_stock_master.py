import sqlite3
db_path = "/data/data/com.termux/files/home/Smart-AI-Anyalisis/backend/database/universe.db"

try:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    print("⚠️ Dropping old 'stock_master' table...")
    # ফরেন কি চেক বন্ধ রাখা যাতে এরর না আসে
    cur.execute("PRAGMA foreign_keys = OFF;")
    cur.execute("DROP TABLE IF EXISTS stock_master;")
    cur.execute("PRAGMA foreign_keys = ON;")
    conn.commit()
    conn.close()
    print("✅ Successfully dropped 'stock_master'. Now run init_db.py again.")
except Exception as e:
    print(f"❌ Error: {e}")
