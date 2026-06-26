import pandas as pd
from backend.analyzers.volatility_analyzer import analyze_volatility

def calculate_volatility_score(df: pd.DataFrame) -> dict:
    """
    Converts Volatility Intelligence into a quantifiable Score (0-100).
    """
    analysis = analyze_volatility(df)
    
    # 1. Base Score (Neutral = 50)
    score = 50
    
    # 2. Breakout Score (High Priority)
    breakout_score = 0
    if analysis['squeeze_detected']:
        breakout_score += 30  # Squeeze মানেই বড় মুভমেন্টের সম্ভাবনা
    
    if analysis['breakout_status'] != "NONE":
        breakout_score += 20
        
    # 3. Risk Volatility Analysis
    # হাই ভোল্যাটিলিটি মার্কেট ঝুঁকিপূর্ণ, তাই স্কোর অ্যাডজাস্টমেন্ট
    if analysis['regime'] == "HIGH_VOLATILITY":
        risk_penalty = 15
    else:
        risk_penalty = 0
        
    # Final Score Calculation
    score = score + breakout_score - risk_penalty
    score = max(0, min(100, round(score)))
    
    # 4. Rating Logic
    if score >= 80: rating = "EXPLOSIVE_BREAKOUT"
    elif score >= 60: rating = "VOLATILITY_EXPANSION"
    elif score > 40: rating = "STABLE"
    elif score > 20: rating = "CONTRACTION"
    else: rating = "LOW_VOLATILITY_DORMANT"
    
    # 5. Confidence Logic
    # Squeeze + Breakout থাকলে কনফিডেন্স হাই
    confidence = "HIGH" if (analysis['squeeze_detected'] and analysis['breakout_status'] != "NONE") else "LOW"
    
    return {
        "volatility_score": score,
        "breakout_score": breakout_score,
        "volatility_rating": rating,
        "confidence": confidence,
        "meta": analysis
    }
