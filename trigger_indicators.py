import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from backend.repository.stock_repository import repository
from backend.indicators import indicator_engine

# লগিং সেটআপ
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(message)s")

def calculate_and_save(symbol):
    try:
        # 1. Run the massive 601-feature engine
        df = indicator_engine.run(symbol)
        
        if df is not None and not df.empty:
            # 2. Save directly to feature_history table
            inserted_rows = repository.save_features(symbol, df)
            return symbol, True, inserted_rows, None
        else:
            return symbol, False, 0, "Engine returned empty dataframe"
    except Exception as e:
        return symbol, False, 0, str(e)

if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(" 🧠 FIRING LAYER-1B (INDICATOR ENGINE) TO LOAD 601 FEATURES")
    print(f"{'='*80}\n")
    
    # Get all active symbols from DB
    symbols_data = repository.get_active_symbols()
    symbols = [s['symbol'] for s in symbols_data if 'symbol' in s]
    
    total = len(symbols)
    success_count = 0
    
    # Running with 4 Threads (Fast & Stable)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(calculate_and_save, sym): sym for sym in symbols}
        
        for i, future in enumerate(as_completed(futures), 1):
            sym, success, rows, err = future.result()
            if success:
                success_count += 1
                print(f"✅ [{i}/{total}] {sym:<15} -> Generated & Saved {rows} feature rows")
            else:
                print(f"❌ [{i}/{total}] {sym:<15} -> FAILED: {err}")
                
    print(f"\n{'='*80}")
    print(f" 🎉 INDICATOR ENGINE COMPLETE! Successfully processed {success_count}/{total} symbols.")
    print(f"{'='*80}\n")
