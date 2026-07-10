import sqlite3
db_path = "/data/data/com.termux/files/home/Smart-AI-Anyalisis/backend/database/universe.db"
try:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("ALTER TABLE stock_master ADD COLUMN is_fno INTEGER DEFAULT 0;")
    conn.commit()
    print("✅ Successfully added 'is_fno' column to existing stock_master table.")
    conn.close()
except Exception as e:
    print(f"⚠️ Column might already exist or error occurred: {e}")
