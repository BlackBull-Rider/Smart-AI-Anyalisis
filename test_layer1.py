import time
import logging
from backend.pipeline.ai_pipeline import FeatureLayerManager, enqueue_for_ai, pipeline_queues

# লগিং সেটআপ যাতে টার্মিনালে সব দেখা যায়
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

def test_layer1():
    print("--- Starting Layer 1 Test ---")
    
    # ম্যানেজার স্টার্ট করছি
    manager = FeatureLayerManager()
    manager.start()
    
    # টেস্টের জন্য একটা সিম্বল কিউ-তে দিচ্ছি (তোর ডাটাবেসে যে সিম্বল আছে সেটা দে, যেমন 'RELIANCE.NS')
    test_symbol = "RELIANCE" 
    enqueue_for_ai(test_symbol)
    
    print(f"Waiting for {test_symbol} to process...")
    time.sleep(5) # ৫ সেকেন্ড ওয়েট করছি প্রসেস হওয়ার জন্য
    
    # কিউ থেকে আউটপুট চেক করছি
    if not pipeline_queues.feature_queue.empty():
        payload = pipeline_queues.feature_queue.get()
        print("\n=== SUCCESS ===")
        print(f"Symbol: {payload.symbol}")
        print(f"Timestamp: {payload.timestamp}")
        print(f"DataFrame Shape: {payload.df.shape} (Rows, Columns)")
        print(f"Some Features Calculated: {list(payload.df.columns[-5:])}")
        print("===============\n")
    else:
        print("\n=== FAILED ===")
        print("No data in feature_queue. Check logs for errors.")
        print("===============\n")
        
    # শাটডাউন
    manager.stop()
    manager.join()
    print("--- Test Complete ---")

if __name__ == "__main__":
    test_layer1()
