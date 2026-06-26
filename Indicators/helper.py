import sqlite3
import pandas as pd
import numpy as np
import os

# ডেটা ইঞ্জিনের লাইভ ডাটাবেস পাথ
DB_PATH = "../Green-Bull-Data-Engine/database/market.db"

def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    ইনস্টিটিউশনাল গ্রেড ডেটা ভ্যালিডেশন।
    মিসিং কলাম বা করাপ্টেড ডেটা থাকলে ইঞ্জিনকে ক্র্যাশ করা থেকে বাঁচাবে।
    """
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"[CRITICAL] Missing required column '{col}' in database.")
            
    # মেমোরি এবং ম্যাথ অপ্টিমাইজেশনের জন্য float64 এ টাইপকাস্ট করা
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].astype(np.float64)
    
    # করাপ্টেড বা মিসিং (NaN) ডেটা ড্রপ করা যাতে ভেক্টর ক্যালকুলেশনে কোনো এরর না আসে
    df.dropna(subset=numeric_cols, inplace=True)
    return df

def fetch_real_ohlcv(symbol: str, limit: int = 1500) -> pd.DataFrame:
    """
    historical_data টেবিল থেকে নির্দিষ্ট সিম্বলের র (Raw) OHLCV ডেটা ফেচ করে।
    লজিক: লেটেস্ট ডেটা আগে এনে তারপর টাইম-সিরিজ অনুযায়ী সোজা করা।
    """
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"[CRITICAL] Database not found at: {DB_PATH}. Check directory structure.")
        
    try:
        conn = sqlite3.connect(DB_PATH)
        # ORDER BY date DESC দিয়ে লেটেস্ট ডেটা আগে ফেচ করা
        query = """
            SELECT date, open, high, low, close, volume 
            FROM historical_data 
            WHERE symbol = ? 
            ORDER BY date DESC 
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(symbol, limit))
        conn.close()
        
        if df.empty:
            print(f"[WARNING] No historical data found for symbol: {symbol}")
            return pd.DataFrame()
            
        # ভেক্টর ক্যালকুলেশনের জন্য ডেটাকে পুরনো থেকে নতুন অর্ডারে সাজানো
        df = df.sort_values(by="date").reset_index(drop=True)
        
        # ডেটা ক্লিন এবং ভ্যালিডেট করে রিটার্ন করা
        return validate_ohlcv(df)
        
    except sqlite3.Error as e:
        print(f"[DATABASE ERROR]: {e}")
        return pd.DataFrame()

def get_existing_symbol() -> str:
    """
    টেস্টিং এবং ইনিশিয়ালাইজেশনের জন্য ডাটাবেস থেকে যেকোনো একটি ভ্যালিড সিম্বল স্ক্যান করে।
    """
    if not os.path.exists(DB_PATH):
        return None
        
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT symbol FROM historical_data LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
    except sqlite3.Error as e:
        print(f"[DATABASE ERROR]: {e}")
        return None
