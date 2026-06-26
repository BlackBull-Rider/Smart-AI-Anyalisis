import pandas as pd

def calculate_confidence_score(scores: dict) -> dict:
    """
    The Validation Layer: Aggregates scores and applies conflict penalties.
    scores: A dictionary containing scores from various engines.
            e.g., {'trend': 80, 'momentum': 40, 'volume': 60, 'smart_money': 90}
    """
    
    # 1. Weights: Trend and Smart Money are highest priority
    weights = {
        'trend': 0.30,
        'momentum': 0.20,
        'volume': 0.15,
        'smart_money': 0.35
    }
    
    # 2. Weighted Aggregation
    weighted_score = sum(scores.get(k, 50) * weights.get(k, 0.25) for k in weights)
    
    # 3. Conflict Detection (The "A-Grade" logic)
    # যদি Trend এবং Momentum এর মধ্যে গ্যাপ খুব বেশি হয়, তবে পেনাল্টি
    score_gap = abs(scores.get('trend', 50) - scores.get('momentum', 50))
    conflict_penalty = 15 if score_gap > 40 else 0
    
    # 4. Final Confidence Score
    confidence_score = max(0, min(100, round(weighted_score - conflict_penalty)))
    
    # 5. Signal Reliability
    if confidence_score >= 80: reliability = "HIGH_CONVICTION"
    elif confidence_score >= 60: reliability = "CONFIRMED"
    elif confidence_score > 40: reliability = "SPECULATIVE"
    else: reliability = "REJECT_SIGNAL"
    
    return {
        "confidence_score": confidence_score,
        "signal_reliability": reliability,
        "probability_score": f"{confidence_score}%",
        "meta": {
            "conflict_penalty": conflict_penalty,
            "weighted_score": round(weighted_score, 2)
        }
    }
