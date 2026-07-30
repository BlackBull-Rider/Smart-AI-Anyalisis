import logging
import pandas as pd
import multiprocessing
import time
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from backend.db.connection import db
from backend.repository.stock_repository import repository
from backend.indicators import indicator_engine
import warnings

warnings.filterwarnings('ignore')
logging.getLogger('backend.indicators').setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(message)s")

def calculate_engine_task(symbol, raw_df):
    start_time = time.time()
    try:
        feat_df = indicator_engine.run(symbol, raw_df)
        elapsed = time.time() - start_time
        return symbol, True, feat_df, None, elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        return symbol, False, None, str(e), elapsed

if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(" 🚀 STARTING TURBO ENGINE (BUG FIXED: INDEX ALIGNED)")
    print(f"{'='*80}\n")
    
    symbols_data = repository.get_active_symbols()
    symbols = [s['symbol'] for s in symbols_data]
    
    print("📦 Phase 1: Scanning Database for missing data... (Please wait)")
    
    tasks = []
    skipped = 0
    for sym in symbols:
        rows = db.fetchall(
            "SELECT * FROM historical_data WHERE symbol=? ORDER BY date ASC", 
            (sym,)
        )
        if not rows:
            continue
            
        raw_df = pd.DataFrame([dict(r) for r in rows])
            
        if 'date' in raw_df.columns:
            raw_df['date'] = pd.to_datetime(raw_df['date'])
            last_raw_date_str = str(raw_df['date'].max())[:10]
            
            # 🔴 ফিক্স: এই লাইনটাই আমি আগে ভুলে গেছিলাম!
            raw_df.set_index('date', inplace=True)
        else:
            last_raw_date_str = str(raw_df.index.max())[:10]
            
        last_feat_date = repository.get_last_feature_date(sym)
        
        if last_feat_date and str(last_feat_date)[:10] >= last_raw_date_str:
            skipped += 1
            continue
            
        raw_df.sort_index(inplace=True)
        if len(raw_df) > 300:
            raw_df = raw_df.tail(300)
            
        tasks.append((sym, raw_df))
        
    total_tasks = len(tasks)
    print(f"⏭️ Skipped {skipped} up-to-date stocks.")
    
    if total_tasks == 0:
        print("🎉 All stocks are already up-to-date! No calculation needed.")
        exit()
        
    cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"🔥 Phase 2: Igniting Engine with {cores} parallel CPU Cores for {total_tasks} stocks...\n")
    
    completed_count = 0
    success_count = 0
    global_start = time.time()
    
    with ProcessPoolExecutor(max_workers=cores) as executor:
        futures = {executor.submit(calculate_engine_task, sym, df): sym for sym, df in tasks}
        
        for future in as_completed(futures):
            sym = futures[future]
            completed_count += 1
            try:
                sym, success, feat_df, err, task_time = future.result()
                
                elapsed_total = time.time() - global_start
                avg_time = elapsed_total / completed_count
                rem_tasks = total_tasks - completed_count
                eta_secs = int(avg_time * rem_tasks)
                eta_str = str(datetime.timedelta(seconds=eta_secs))
                
                if success and feat_df is not None and not feat_df.empty:
                    inserted = repository.save_features(sym, feat_df)
                    success_count += 1
                    print(f"✅ {sym:<12} -> Saved {inserted} rows | ⏱️ {task_time:.2f}s | 📊 {completed_count}/{total_tasks} | ⏳ ETA: {eta_str}")
                else:
                    print(f"❌ {sym:<12} -> FAILED: {err or 'Empty'} | ⏱️ {task_time:.2f}s | 📊 {completed_count}/{total_tasks} | ⏳ ETA: {eta_str}")
            except Exception as e:
                print(f"❌ {sym:<12} -> CRASHED: {str(e)}")
                
    total_time_str = str(datetime.timedelta(seconds=int(time.time() - global_start)))
    print(f"\n{'='*80}")
    print(f" 🎉 DONE! Processed {success_count} stocks. Total Time Taken: {total_time_str}")
    print(f"{'='*80}\n")
