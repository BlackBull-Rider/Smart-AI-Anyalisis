import pandas as pd
from backend.analyzers.institutional_analyzer import analyze_institutional_flow

def calculate_institutional_score(inst_df: pd.DataFrame) -> dict:
    """
    Converts Institutional Intelligence into Accumulation/Distribution Scores.
    """
    analysis = analyze_institutional_flow(inst_df)
    
    # 1. Base Score (Neutral = 50)
    score = 50
    acc_score = 0
    dist_score = 0
    
    # 2. Institutional Bias Scoring (FII + DII)
    if analysis['institutional_bias'] == "BULLISH_INSTITUTIONAL":
        score += 20
        acc_score += 20
    else:
        score -= 20
        dist_score += 20
        
    # 3. Delivery Analysis Scoring
    # হাই ডেলিভারি মানে কনভিকশন (Accumulation)
    if analysis['delivery_signal'] == "ACCUMULATION":
        score += 20
        acc_score += 20
    elif analysis['delivery_signal'] == "DISTRIBUTION":
        score -= 20
        dist_score += 20
        
    # 4. Block/Bulk Deal Impact
    if analysis['block_deal_alert']:
        score += 10 # স্মার্ট মানির বড় এন্ট্রি
        acc_score += 10
        
    # 5. Promoter Risk (Penalty)
    if analysis['promoter_status'] == "WARNING_PROMOTER_SELLING":
        score -= 20
        dist_score += 20
        
    # Final Clamping
    score = max(0, min(100, round(score)))
    
    # Rating Logic
    if score >= 75: rating = "STRONG_ACCUMULATION"
    elif score >= 60: rating = "INSTITUTIONAL_BULLISH"
    elif score > 40: rating = "NEUTRAL_INSTITUTIONAL"
    elif score > 25: rating = "INSTITUTIONAL_BEARISH"
    else: rating = "STRONG_DISTRIBUTION"
    
    return {
        "institutional_score": score,
        "accumulation_score": acc_score,
        "distribution_score": dist_score,
        "institutional_rating": rating,
        "meta": analysis
    }
