import time
import logging
from backend.pipeline.ai_pipeline import FeatureLayerManager, DedicatedDBWriter, enqueue_for_ai, pipeline_queues
from backend.repository.stock_repository import repository
from backend.db.connection import db

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

def test_pipeline_all():
    print("--- Starting AI Pipeline Bulk DB Write Test ---")
    
    # ১. তোর অফিসিয়াল Repository ব্যবহার করে সিম্বল ফেচ করা
    try:
        active_stocks = repository.get_active_symbols()
        symbols = [stock["symbol"] for stock in active_stocks]
    except Exception as e:
        print(f"Failed to fetch symbols via repository: {e}")
        return
        
    if not symbols:
        print("No active symbols found in database!")
        return
        
    print(f"Found {len(symbols)} active symbols. Enqueuing all...")

    # ২. Layer 1-এর ওয়ার্কার এবং ডেডিকেটেড রাইটার স্টার্ট করা
    manager = FeatureLayerManager()
    manager.start()
    
    writer = DedicatedDBWriter()
    writer.start()
    
    # ৩. সব সিম্বল কিউতে দেওয়া (Queue Full থাকলে ওয়েট করবে)
    for sym in symbols:
        enqueued = False
        while not enqueued:
            enqueued = enqueue_for_ai(sym)
            if not enqueued:
                print(f"Queue is full. Pausing injector for {sym}...")
                time.sleep(3)
        
    # ৪. প্রসেস শেষ হওয়া পর্যন্ত ওয়েট করা
    print(f"Waiting for {len(symbols)} symbols to be processed by 8 workers and 1 DB writer...")
    
    # কিউ খালি না হওয়া পর্যন্ত লুপ চলবে
    while not pipeline_queues.symbol_queue.empty() or not pipeline_queues.db_write_queue.empty():
        time.sleep(2)
        
    # শেষ ডেটাটুকু ডাটাবেসে সেভ হওয়ার জন্য ছোট একটা বাফার টাইম
    time.sleep(3)
    
    # ৫. ডাটাবেস ভেরিফিকেশন (অফিসিয়াল DB কানেকশন দিয়ে)
    print("\n--- Verifying Database ---")
    try:
        row = db.fetchone("SELECT count(DISTINCT symbol) as sym_count, count(*) as row_count FROM feature_history")
        sym_count = row["sym_count"]
        row_count = row["row_count"]
        
        print(f"Total unique symbols saved in feature_history: {sym_count} / {len(symbols)}")
        print(f"Total rows (with 571 features) written across all symbols: {row_count}")
        
        if sym_count > 0:
            print("🚀 SUCCESS: Bulk Data was written to the database seamlessly!")
        else:
            print("❌ FAILED: No data found in database. Check logs.")
    except Exception as e:
        print(f"Database verification error: {e}")

    # ৬. শাটডাউন
    print("\n--- Shutting Down ---")
    manager.stop()
    writer.stop()
    pipeline_queues.db_write_queue.put(None)
    
    manager.join()
    writer.join()
    print("Test Complete!")

if __name__ == "__main__":
    test_pipeline_all()
