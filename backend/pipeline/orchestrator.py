import sqlite3
import yfinance as yf
import pandas as pd
import time
import gc
from backend.data.market_sync import MarketSync

class MemorySafeHistoricalEngine:
    def __init__(self):
        self.sync_engine = MarketSync()
        self.db_path = "/data/data/com.termux/files/home/Smart-AI-Anyalisis/database/universe.db"

    def run(self):
        # ১. ইউনিভার্স সিঙ্ক
        self.sync_engine.sync_universe()
        
        # ২. অরিজিনাল সিম্বল লিস্ট ফেচ করা (DISTINCT দিয়ে ডুপ্লিকেট বাদ দেওয়া হলো)
        conn = sqlite3.connect(self.db_path)
        all_symbols = [row[0] for row in conn.execute("SELECT DISTINCT symbol FROM equity_master").fetchall()]
        
        # ৩. স্মার্ট রিজিউম: কোন স্টকগুলো অলরেডি সেভ হয়েছে তা বের করা
        try:
            done_symbols = [row[0] for row in conn.execute("SELECT DISTINCT symbol FROM market_data").fetchall()]
        except Exception:
            done_symbols = []
        conn.close()
        
        # ৪. শুধুমাত্র বাকি থাকা স্টকগুলোর লিস্ট বানানো
        pending_symbols = [sym for sym in all_symbols if sym not in done_symbols]
        
        print(f"📊 Total Symbols: {len(all_symbols)} | Already Done: {len(done_symbols)} | Pending: {len(pending_symbols)}")
        print("🚀 Starting Memory-Safe Fetching Engine...")
        
        # ৫. ডিরেক্ট ফেচ এবং রাইট লুপ (RAM ম্যানেজমেন্ট সহ)
        for sym in pending_symbols:
            try:
                ticker = f"{sym}.NS"
                df = yf.download(ticker, start="2000-01-01", progress=False)
                
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    
                    df = df.reset_index()
                    df.rename(columns={
                        'Date': 'trade_date', 'Open': 'open', 'High': 'high', 
                        'Low': 'low', 'Close': 'close', 'Volume': 'volume'
                    }, inplace=True)
                    
                    df['trade_date'] = df['trade_date'].dt.strftime('%Y-%m-%d')
                    df['symbol'] = sym
                    df = df[['symbol', 'trade_date', 'open', 'high', 'low', 'close', 'volume']]
                    
                    rows_saved = self.sync_engine.upsert("market_data", df)
                    print(f"✅ {sym} -> {rows_saved} rows saved.")
                else:
                    print(f"⚠️ {sym} -> No data found.")
                
            except Exception as e:
                print(f"❌ Error {sym}: {e}")
            
            finally:
                # 🧹 মেমোরি লিক ঠেকানোর ব্রহ্মাস্ত্র (Garbage Collection)
                if 'df' in locals():
                    del df
                gc.collect()
                # সার্ভার যেন ব্লক না করে তাই ছোট্ট বিরতি
                time.sleep(0.5)

if __name__ == "__main__":
    MemorySafeHistoricalEngine().run()
