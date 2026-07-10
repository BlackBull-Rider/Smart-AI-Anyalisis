import sqlite3
import os

db_path = "/data/data/com.termux/files/home/Smart-AI-Anyalisis/backend/database/universe.db"

try:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # আপনার stock_master টেবিল ফোর্সফুলি তৈরি করা হচ্ছে
    cur.execute("""
    CREATE TABLE IF NOT EXISTS stock_master (
        symbol TEXT PRIMARY KEY,
        company_name TEXT,
        sector TEXT,
        industry TEXT,
        exchange TEXT,
        series TEXT,
        isin TEXT,
        face_value NUMERIC,
        is_active INTEGER DEFAULT 1
    );
    """)
    conn.commit()
    
    # টেবিল তৈরি হয়েছে কি না চেক করা
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stock_master';")
    exists = cur.fetchone()
    
    if exists:
        print("✅ SUCCESS: 'stock_master' table created/verified successfully in universe.db!")
    else:
        print("❌ ERROR: Failed to create table.")
        
    conn.close()
except Exception as e:
    print(f"❌ Critical Error: {e}")
