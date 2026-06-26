import sqlite3
import pandas as pd
import logging
from typing import Dict, Any
import sys
import os

# config.py ইমপোর্ট করার জন্য রুট ডিরেক্টরি পাথ অ্যাড করা হলো
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import DB_PATH, DEFAULT_LOOKBACK

logger = logging.getLogger("DatabaseHandler")

class DatabaseHandler:
    def __init__(self, db_path: str = str(DB_PATH)):
        """ডাটাবেস কানেকশন ইনিশিয়ালাইজ ও ভ্যালিডেট করবে।"""
        self.db_path = db_path
        if not os.path.exists(self.db_path):
            logger.critical(f"Database not found at {self.db_path}")
            raise FileNotFoundError(f"Missing DB: {self.db_path}")

    def _get_connection(self):
        """সিকিউর SQLite কানেকশন রিটার্ন করবে।"""
        try:
            return sqlite3.connect(self.db_path)
        except sqlite3.Error as e:
            logger.error(f"Database Connection Failed: {e}")
            raise

    def get_historical_buffer(self, symbol: str, lookback: int = DEFAULT_LOOKBACK) -> pd.DataFrame:
        """Layer 1 & 2 এর জন্য historical_data টেবিল থেকে OHLCV ডেটা আনবে।"""
        query = "SELECT * FROM historical_data WHERE symbol = ? ORDER BY date ASC"
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(symbol,))
        
        if df.empty:
            return pd.DataFrame()
        
        df['date'] = pd.to_datetime(df['date'])
        if len(df) > lookback:
            return df.tail(lookback).reset_index(drop=True)
        return df

    def get_latest_price_data(self, symbol: str) -> Dict[str, Any]:
        """ডিসিশন ইঞ্জিনের জন্য শুধুমাত্র লেটেস্ট Date এর ডেটা ফেচ করবে (তোর রুল অনুযায়ী)।"""
        query = "SELECT * FROM historical_data WHERE symbol = ? ORDER BY date DESC LIMIT 1"
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(symbol,))
        return df.iloc[0].to_dict() if not df.empty else {}

    def get_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """Long Term Engine এর জন্য fundamental_data আনবে।"""
        query = "SELECT * FROM fundamental_data WHERE symbol = ?"
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(symbol,))
        return df.iloc[0].to_dict() if not df.empty else {}

    def get_ipo_data(self, symbol: str) -> Dict[str, Any]:
        """IPO ইঞ্জিনের জন্য ipo_data টেবিল থেকে ডেটা ফেচ করবে।"""
        query = "SELECT * FROM ipo_data WHERE symbol = ?"
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(symbol,))
        return df.iloc[0].to_dict() if not df.empty else {}
