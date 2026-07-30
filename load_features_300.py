import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from backend.db.connection import db
from backend.repository.stock_repository import repository
from backend.indicators import indicator_engine

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(message)s")

def calculate_and_save(symbol):
    try:
        # ১. Historical Data থেকে কাঁচা ডেটা (OHLCV) আনা হচ্ছে
        raw_df = repository.get_history(symbol)
        
        if raw_df is None or raw_df.empty:
            return symbol, False, 0, "No historical data found"
            
        # ইনডেক্স ঠিক করা
        if 'date' in raw_df.columns:
            raw_df['date'] = pd.to_datetime(raw_df['date'])
            raw_df.set_index('date', inplace=True)
            
        raw_df.sort_index(inplace=True)
            
        # ২. ঠিক লাস্ট 300 দিনের রও (Raw) ডেটা কেটে নেওয়া
        if len(raw_df) > 300:
            raw_df = raw_df.tail(300)
            
        if raw_df.empty:
            return symbol, False, 0, "Empty dataframe after processing"
            
        # ৩. লাস্ট 300 দিনের Historical Data ইঞ্জিনে ফিড করা হলো
        feat_df = indicator_engine.run(symbol, raw_df)
        
        if feat_df is not None and not feat_df.empty:
            # ৪. feature_history-তে এন্ট্রি
            inserted = repository.save_features(symbol, feat_df)
            return symbol, True, inserted, None
        else:
            return symbol, False, 0, "Engine returned empty dataframe"
            
    except Exception as e:
        return symbol, False, 0, str(e)

if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(" 🎯 LAYER-1B: stock_master (Symbols) -> Historical Data (300 Days) -> ENGINE")
    print(f"{'='*80}\n")
    
    try:
        # 🔴 ফিক্স: কোনো is_active ফিল্টার ছাড়াই ডাইরেক্ট সিম্বল তোলা হচ্ছে
        rows = db.fetchall("SELECT DISTINCT symbol FROM stock_master")
        symbols = [dict(r)['symbol'] for r in rows if 'symbol' in dict(r)]
    except Exception as e:
        print(f"❌ Database error: {e}")
        exit()
        
    if not symbols:
        print("❌ No symbols found in stock_master table!")
        exit()
        
    total = len(symbols)
    print(f"✅ Found {total} symbols. Fetching 300 days of historical data for each...\n")
    
    success_count = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(calculate_and_save, sym): sym for sym in symbols}
        
        for i, future in enumerate(as_completed(futures), 1):
            sym, success, rows_saved, err = future.result()
            if success:
                success_count += 1
                print(f"✅ [{i}/{total}] {sym:<15} -> {rows_saved} feature rows saved")
            else:
                print(f"❌ [{i}/{total}] {sym:<15} -> ERROR: {err}")
                
    print(f"\n{'='*80}")
    print(f" 🎉 300-DAY FEATURE CALCULATION COMPLETE! Total Processed: {success_count}/{total}")
    print(f"{'='*80}\n")
