import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from backend.db.connection import db
from backend.repository.stock_repository import repository
from backend.indicators import indicator_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def process_symbol(symbol):
    try:
        # ১. ডাইরেক্ট 'historical_data' টেবিল থেকে ডেটা তুলছি (যেহেতু repository-তে get_history নেই)
        rows = db.fetchall(
            "SELECT * FROM historical_data WHERE symbol=? ORDER BY date ASC", 
            (symbol,)
        )
        
        if not rows:
            return symbol, False, 0, "No history found in historical_data table"
            
        raw_df = pd.DataFrame([dict(r) for r in rows])
        
        # ২. ঠিক লাস্ট ৩০০ দিনের ডেটা নিচ্ছি
        if len(raw_df) > 300:
            raw_df = raw_df.tail(300)
            
        # ৩. ইঞ্জিন রান করছি
        feat_df = indicator_engine.run(symbol, raw_df)
        
        if feat_df is not None and not feat_df.empty:
            # ৪. feature_history তে সেভ করছি
            inserted = repository.save_features(symbol, feat_df)
            return symbol, True, inserted, None
        else:
            return symbol, False, 0, "Engine returned empty"
            
    except Exception as e:
        return symbol, False, 0, str(e)

if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(" 🚀 STARTING FEATURE INGESTION (DIRECT DB ACCESS)")
    print(f"{'='*80}\n")
    
    # Stock Master থেকে সব একটিভ সিম্বল তুলছি
    symbols_data = repository.get_active_symbols()
    symbols = [s['symbol'] for s in symbols_data]
    
    total = len(symbols)
    print(f"✅ Found {total} active symbols. Processing...\n")
    
    success_count = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_symbol, sym): sym for sym in symbols}
        
        for future in as_completed(futures):
            sym, success, rows, err = future.result()
            if success:
                success_count += 1
                print(f"✅ {sym:<15} -> Saved {rows} feature rows")
            else:
                print(f"❌ {sym:<15} -> FAILED: {err}")
                
    print(f"\n{'='*80}")
    print(f" 🎉 DONE! Processed {success_count}/{total} symbols.")
    print(f"{'='*80}\n")
