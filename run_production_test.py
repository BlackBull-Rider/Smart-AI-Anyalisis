import logging
from backend.ai.master_engine import GreenBullOrchestrator, SystemContext
from backend.database.db_handler import DatabaseHandler

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def run_live_test():
    print("\n--- INITIALIZING GREEN BULL RIDER: PRODUCTION MODE ---")
    
    # 1. Database Connection & Data Fetch
    db = DatabaseHandler()
    print("Fetching real historical data from DB...")
    df = db.get_historical_buffer("RELIANCE", lookback=200) # EMA200 এর জন্য ২০০ ক্যান্ডেল
    
    if df.empty:
        print("CRITICAL: No data found for RELIANCE. Check Database.")
        return

    # 2. Prepare System Context (The Single Source of Truth)
    ctx = SystemContext(
        df=df,
        symbol="RELIANCE",
        balance=500000.0,
        timeframe="5m"
    )

    # 3. Run Orchestrator
    orchestrator = GreenBullOrchestrator(db_path="audit_trail.db")
    
    print("Executing Institutional-grade Analysis...")
    report = orchestrator.generate_final_master_report(ctx)

    # 4. Final Output Display
    print("\n" + "="*60)
    print(f"REPORT GENERATED: {report['metadata']['timestamp']}")
    print(f"MASTER SCORE: {report['master_score']:.2f}")
    print(f"RECOMMENDATION: {report['recommendation']['action']}")
    print("="*60)
    
    # বিস্তারিত রিপোর্ট প্রিন্ট করো
    print("\n--- EXECUTION PLAN ---")
    print(report['execution_plan'])
    
    print("\n--- AUDIT STATUS ---")
    for engine, status in report['audit_trail'].items():
        print(f"{engine}: {status}")

if __name__ == "__main__":
    run_live_test()
