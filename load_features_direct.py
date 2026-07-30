import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from backend.db.connection import db
from backend.repository.stock_repository import repository
from backend.indicators import indicator_engine

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(message)s")

def calculate_and_save(symbol):
    try:
        # একদম ডাইরেক্ট stock_master টেবিল থেকে OHLCV তোলা হচ্ছে
        query = f"SELECT * FROM stock_master WHERE symbol='{symbol}' ORDER BY date ASC"
        raw_data = db.fetchall(query)
        
        if not raw_data:
            return symbol, False, 0, "No OHLCV data found in stock_master"
            
        raw_df = pd.DataFrame([dict(r) for r in raw_data])
        raw_df.columns = [str(c).lower().strip() for c in raw_df.columns]
        
        if 'date' in raw_df.columns:
            raw_df['date'] = pd.to_datetime(raw_df['date'])
            raw_df.set_index('date', inplace=True)
            
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in raw_df.columns:
                raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')
                
        if 'close' in raw_df.columns:
            raw_df.dropna(subset=['close'], inplace=True)
        
        if raw_df.empty:
            return symbol, False, 0, "Empty dataframe after cleaning OHLCV"
            
        # ক্যালকুলেশনের জন্য ইঞ্জিনকে stock_master-এর ডেটা পাস করা হচ্ছে
        feat_df = indicator_engine.run(symbol, raw_df)
        
        if feat_df is not None and not feat_df.empty:
            # feature_history-তে এন্ট্রি
            inserted = repository.save_features(symbol, feat_df)
            return symbol, True, inserted, None
        else:
            return symbol, False, 0, "Engine returned empty"
            
    except Exception as e:
        return symbol, False, 0, str(e)

if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(" 🎯 FETCHING OHLCV DIRECTLY FROM [stock_master] & LOADING FEATURES")
    print(f"{'='*80}\n")
    
    try:
        # stock_master থেকে ইউনিক সিম্বলগুলো তোলা হচ্ছে
        rows = db.fetchall("SELECT DISTINCT symbol FROM stock_master")
        symbols = [dict(r)['symbol'] for r in rows if 'symbol' in dict(r)]
    except Exception as e:
        print(f"❌ Database error: {e}")
        exit()
        
    if not symbols:
        print("❌ No symbols found in stock_master table!")
        exit()
        
    total = len(symbols)
    print(f"✅ Found {total} symbols in 'stock_master'. Igniting indicator engine...\n")
    
    success_count = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(calculate_and_save, sym): sym for sym in symbols}
        
        for i, future in enumerate(as_completed(futures), 1):
            sym, success, rows_saved, err = future.result()
            if success:
                success_count += 1
                print(f"✅ [{i}/{total}] {sym:<15} -> {rows_saved} rows loaded into feature_history")
            else:
                print(f"❌ [{i}/{total}] {sym:<15} -> ERROR: {err}")
                
    print(f"\n{'='*80}")
    print(f" 🎉 CALCULATION COMPLETE! Total Processed: {success_count}/{total}")
    print(f"{'='*80}\n")
