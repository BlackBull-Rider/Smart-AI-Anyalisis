import pandas as pd
from backend.database.db_handler import DatabaseHandler
from backend.ai.master_engine import generate_final_master_report

def run_live_test():
    print("--- CONNECTING TO DATABASE: RELIANCE ---")
    
    # 1. Initialize DB Handler
    # এখানে তোমার DB_PATH সঠিকভাবে আছে কি না নিশ্চিত হয়ে নাও
    db = DatabaseHandler()
    
    try:
        # 2. Fetch Real Data (Using your specific method name)
        print("Fetching historical buffer for RELIANCE...")
        df = db.get_historical_buffer("RELIANCE", lookback=100)
        
        if df.empty:
            print("ERROR: No data found in database for RELIANCE.")
            return

        print(f"Data received: {len(df)} rows.")

        # 3. Call Master AI
        account_balance = 500000 
        print("Running Master AI Analysis...")
        report = generate_final_master_report(df, account_balance)
        
        # 4. Final Output
        print("\n" + "="*50)
        print("FINAL RELIANCE LIVE REPORT")
        print("="*50)
        
        for category, details in report.items():
            print(f"\n[{category}]")
            print(details)
            
    except Exception as e:
        print(f"SYSTEM FAILURE DURING LIVE TEST: {e}")

if __name__ == "__main__":
    run_live_test()
