import time
import logging
from backend.pipeline.ai_pipeline import MasterPipelineManager, enqueue_for_ai

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def run_test():
    print("\n" + "="*50)
    print("🚀 STARTING ENTERPRISE PIPELINE TEST")
    print("="*50 + "\n")
    
    manager = MasterPipelineManager()
    manager.start()

    # ২টো স্টক স্ট্রিমিং কিউ-তে ফেলা হলো
    test_symbols = ["SUNPHARMA"]
    for sym in test_symbols:
        enqueue_for_ai(sym)

    print("\n⏳ Waiting 45 seconds for the Streaming Pipeline to process all layers...\n")
    time.sleep(45)

    print("\n🛑 Initiating Graceful Shutdown for Streaming Workers...")
    manager.stop()
    manager.join()
    
    # 💥 স্ট্রিমিং শেষ, এবার চলবে ম্যাক্রো ইঞ্জিন!
    print("\n🌍 Running End-of-Day Macro Engines (Ranking, Portfolio, Dashboard)...")
    manager.generate_macro_reports()
    
    print("\n" + "="*50)
    print("✅ TEST COMPLETED SUCCESSFULLY!")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_test()
