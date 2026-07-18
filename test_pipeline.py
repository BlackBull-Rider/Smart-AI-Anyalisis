import logging
from backend.pipeline.orchestrator import InstitutionalPipeline

# লগিং সেটআপ যাতে টার্মিনালে আউটপুট দেখতে পান
logging.basicConfig(level=logging.INFO)

# ডামি ডেটা তৈরি
dummy_l4_stock = {
    "symbol": "RELIANCE",
    "market_permission": "ALLOWED",
    "conviction_score": 85.0,
    "confidence_score": 90.0,
    "overall_risk": 30.0,
    "reward_quality": 80.0,
    "entry_action": "EXECUTE",
    "entry_quality": 88.0,
    "holding_type": "COMPOUND",
    "trend_score": 82.0,
    "institutional_score": 88.0,
    "fundamental_score": 92.0,
    "regime_score": 75.0,
    "recommendation": "STRONG_BUY"
}

dummy_portfolio = [] # আপাতত ফাঁকা পোর্টফোলিও
macro_data = {"regime_score": 75.0, "market_state": "BULL", "systemic_risk": 20.0}

# পাইপলাইন রান করানো
if __name__ == "__main__":
    print("Starting Pipeline Test...\n" + "-"*40)
    orchestrator = InstitutionalPipeline()
    result = orchestrator.execute_market_scan(
        universe_l4_data=[dummy_l4_stock], 
        current_portfolio=dummy_portfolio, 
        available_cash_pct=100.0,
        market_macro_data=macro_data
    )
    
    print("\n--- Final Dashboard Payload ---")
    import json
    print(json.dumps(result["dashboard"], indent=2))
