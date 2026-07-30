import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from backend.db.connection import db
from backend.repository.stock_repository import repository
from backend.indicators import indicator_engine

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(message)s")

def calculate_and_save(symbol):
    try:
        # ১. Repository ব্যবহার করে OHLCV টানছি (কোনো ফালতু SQL গেসওয়ার্ক নেই)
        raw_df = repository.get_history(symbol)
        
        if raw_df is None or raw_df.empty:
            return symbol, False, 0, "No OHLCV data returned by repository"
            
        # ২. ইঞ্জিন রান (OHLCV ডাটাফ্রেম পাস করে)
        feat_df = indicator_engine.run(symbol, raw_df)
        
        if feat_df is not None and not feat_df.empty:
            # ৩. feature_history-তে এন্ট্রি
            inserted = repository.save_features(symbol, feat_df)
            return symbol, True, inserted, None
        else:
            return symbol, False, 0, "Engine returned empty dataframe"
            
    except Exception as e:
        return symbol, False, 0, str(e)

if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(" 🎯 SYMBOLS FROM [stock_master] -> OHLCV VIA [repository] -> INDICATOR ENGINE")
    print(f"{'='*80}\n")
    
    try:
        # stock_master থেকে শুধু এক্টিভ সিম্বলগুলো তোলা হচ্ছে
        rows = db.fetchall("SELECT symbol FROM stock_master WHERE is_active=1")
        if not rows:
            rows = db.fetchall("SELECT symbol FROM stock_master")
        symbols = [dict(r)['symbol'] for r in rows if 'symbol' in dict(r)]
    except Exception as e:
        print(f"❌ Database error: {e}")
        exit()
        
    if not symbols:
        print("❌ No symbols found in stock_master table!")
        exit()
        
    total = len(symbols)
    print(f"✅ Picked {total} active symbols from 'stock_master'. Igniting indicator engine...\n")
    
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
    print(f" 🎉 CALCULATION COMPLETE! Total Processed: {success_count}/{total}")
    print(f"{'='*80}\n")
