import pandas as pd

def calculate_conviction_score(entry_metrics: dict, risk_metrics: dict, reward_metrics: dict) -> dict:
    """
    The Final Judge: Aggregates entry, risk, and reward metrics to generate 
    a final trading conviction score.
    
    entry_metrics: Output from entry_engine
    risk_metrics: Output from risk_engine
    reward_metrics: Output from reward_engine
    """
    
    # 1. Component Weights
    # এন্ট্রি ভালো হলে হবে না, রিস্ক ম্যানেজমেন্ট এবং রিওয়ার্ড স্ট্রাকচার সমান গুরুত্বপূর্ণ
    w_entry = 0.30
    w_risk = 0.40  # Risk management is the top priority
    w_reward = 0.30
    
    # 2. Score Normalization
    # Risk Score (0-100), Reward Score (0-100), Entry Confidence (Assumed 0-100)
    e_score = 100 if entry_metrics['action'] == "EXECUTE_ENTRY" else 0
    r_score = risk_metrics['risk_score'] # This is percentage, we want to inverse risk-to-score
    # Logic: Risk percentage should be low for high conviction
    r_conviction = max(0, 100 - (r_score * 5)) 
    rew_score = reward_metrics['reward_score']
    
    # 3. Aggregation
    conviction_score = (e_score * w_entry) + (r_conviction * w_risk) + (rew_score * w_reward)
    conviction_score = max(0, min(100, round(conviction_score)))
    
    # 4. Final Verdict Logic
    if conviction_score >= 85: rating = "STRONG_BUY_EXECUTE"
    elif conviction_score >= 65: rating = "MODERATE_BUY"
    elif conviction_score > 40: rating = "WAIT_FOR_CONFIRMATION"
    else: rating = "AVOID_TRADE"
    
    return {
        "conviction_score": conviction_score,
        "rating": rating,
        "multi_factor_confirmed": (e_score > 0 and r_conviction > 50 and rew_score > 50),
        "decision_strength": "HIGH" if conviction_score >= 80 else "LOW"
    }
