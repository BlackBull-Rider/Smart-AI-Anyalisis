import sqlite3

db_path = "/data/data/com.termux/files/home/Smart-AI-Anyalisis/backend/database/universe.db"
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# ১. আগের ভুল টেবিলটা পুরোপুরি মুছে ফেলা
cur.execute("DROP TABLE IF EXISTS market_data;")

# ২. পাইপলাইনের জন্য একদম সঠিক কলামসহ টেবিল তৈরি করা
cur.execute("""
CREATE TABLE market_data (
    symbol TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    PRIMARY KEY (symbol, trade_date)
);
""")

conn.commit()
conn.close()
print("✅ SUCCESS: 'market_data' টেবিল trade_date সহ পারফেক্টভাবে তৈরি হয়েছে!")
