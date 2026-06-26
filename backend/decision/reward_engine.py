import pandas as pd
from backend.decision.target_engine import calculate_targets

def calculate_reward_metrics(entry_price: float, stop_loss: float, holding_period_days: int) -> dict:
    """
    The Profitability Gauge: Calculates Expected Returns, Reward Score, and CAGR.
    """
    # 1. Get Targets from Target Engine
    targets = calculate_targets(entry_price, stop_loss, position_side="LONG")
    target_price = targets['targets']['t2_momentum'] # We use T2 as standard expectation
    
    # 2. Risk-Reward Ratio (RRR) Calculation
    risk = abs(entry_price - stop_loss)
    reward = abs(target_price - entry_price)
    rrr = reward / risk if risk != 0 else 0
    
    # 3. Expected Return (Percentage)
    expected_return_pct = (reward / entry_price) * 100
    
    # 4. CAGR Estimate (Annualized Return)
    # Formula: (1 + Return)^(365/Days) - 1
    if holding_period_days > 0:
        cagr = ((1 + (expected_return_pct / 100)) ** (365 / holding_period_days) - 1) * 100
    else:
        cagr = expected_return_pct
        
    # 5. Reward Score (0-100)
    # RRR যদি ৩ এর বেশি হয় তবেই স্কোর হাই হবে
    reward_score = min(rrr * 20, 100)
    
    return {
        "expected_return_pct": round(expected_return_pct, 2),
        "reward_score": round(reward_score, 2),
        "risk_reward_ratio": round(rrr, 2),
        "cagr_estimate": round(cagr, 2),
        "rating": "HIGH_PROFITABILITY" if rrr >= 3 else "MODERATE" if rrr >= 2 else "LOW_PROFITABILITY",
        "meta": {
            "target_used": target_price,
            "days_projection": holding_period_days
        }
    }
