import sqlite3
import os

db_path = "/data/data/com.termux/files/home/Smart-AI-Anyalisis/backend/database/universe.db"

print("1. Creating missing tables in universe.db...")
try:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # market_data টেবিল তৈরি (schema.py এর স্ট্রাকচার অনুযায়ী)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS market_data (
        market_data_id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        trade_date DATE NOT NULL,
        price_open NUMERIC(18,6) NOT NULL CHECK(price_open >= 0),
        price_high NUMERIC(18,6) NOT NULL,
        price_low NUMERIC(18,6) NOT NULL CHECK(price_low >= 0),
        price_close NUMERIC(18,6) NOT NULL CHECK(price_close >= 0),
        volume INTEGER NOT NULL CHECK(volume >= 0),
        delivery_volume INTEGER,
        vwap NUMERIC(18,6),
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(symbol, trade_date)
    );
    """)
    conn.commit()
    print("✅ 'market_data' table created successfully!")
except Exception as e:
    print(f"❌ Failed to create table: {e}")

print("\n2. Patching pipeline to use 'trade_date'...")
filepath = "backend/pipeline/market_pipeline.py"
try:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # পাইপলাইনে timestamp এর জায়গায় trade_date বসানো
    content = content.replace('ts_col="timestamp"', 'ts_col="trade_date"')
    content = content.replace('ts = rec.get("timestamp")', 'ts = rec.get("trade_date")')

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ Pipeline successfully patched!")
except Exception as e:
    print(f"❌ Failed to patch pipeline: {e}")
