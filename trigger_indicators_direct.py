import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from backend.db.connection import db
from backend.repository.stock_repository import repository
from backend.indicators import indicator_engine

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(message)s")

def calculate_and_save(symbol):
    try:
        # 1. Direct SQL fetch to bypass repository layer issues
        # Adjust 'history' table name if it's different in your DB (e.g., historical_data, daily_data)
        query = f"SELECT * FROM history WHERE symbol='{symbol}' ORDER BY date ASC"
        rows = db.fetchall(query)
        
        if not rows:
            return symbol, False, 0, "No raw history data found in 'history' table"
            
        # Create DataFrame
        raw_df = pd.DataFrame([dict(r) for r in rows])
        # Format columns exactly as indicator engine expects
        raw_df.columns = [str(c).lower().strip() for c in raw_df.columns]
        raw_df['date'] = pd.to_datetime(raw_df['date'])
        raw_df.set_index('date', inplace=True)
        
        # Convert numeric columns safely
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in raw_df.columns:
                raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')
        
        raw_df.dropna(subset=['close'], inplace=True)
        
        if raw_df.empty:
            return symbol, False, 0, "DataFrame empty after cleaning"
            
        # 2. Feed raw_df into indicator engine
        df = indicator_engine.run(symbol, raw_df)
        
        if df is not None and not df.empty:
            # 3. Save calculated features back to DB
            inserted_rows = repository.save_features(symbol, df)
            return symbol, True, inserted_rows, None
        else:
            return symbol, False, 0, "Engine returned empty dataframe"
    except Exception as e:
        return symbol, False, 0, str(e)

if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(" 🧠 FIRING LAYER-1B (DIRECT SQL -> INDICATOR ENGINE)")
    print(f"{'='*80}\n")
    
    symbols_data = repository.get_active_symbols()
    symbols = [s['symbol'] for s in symbols_data if 'symbol' in s]
    
    total = len(symbols)
    success_count = 0
    
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
