import pandas as pd
from backend.analyzers.momentum_analyzer import analyze_momentum

def calculate_momentum_score(df: pd.DataFrame) -> dict:
    """
    Converts Momentum Intelligence into a quantifiable Score (0-100).
    """
    analysis = analyze_momentum(df)
    
    # 1. Base Score (Neutral = 50)
    score = 50
    
    # 2. Score Weighting
    # Divergence is a strong reversal signal (High Weightage)
    if analysis['divergence'] == "BULLISH_DIVERGENCE":
        score += 25
    elif analysis['divergence'] == "BEARISH_DIVERGENCE":
        score -= 25
        
    # Acceleration vs Deceleration
    if analysis['acceleration'] == "ACCELERATING":
        score += 15
    else:
        score -= 15
        
    # MACD Status
    if analysis['macd_status'] == "BULLISH_CROSS":
        score += 10
    else:
        score -= 10
        
    # 3. Clamping Score (0 to 100)
    score = max(0, min(100, round(score)))
    
    # 4. Rating Logic
    if score >= 85: rating = "VERY_BULLISH"
    elif score >= 65: rating = "BULLISH"
    elif score > 35: rating = "NEUTRAL"
    elif score > 15: rating = "BEARISH"
    else: rating = "VERY_BEARISH"
    
    # 5. Confidence Logic
    # যদি মার্কেট ট্রেন্ডিং হয় এবং মোমেন্টাম কনফার্ম করে, তবেই কনফিডেন্স হাই
    confidence = "HIGH" if (analysis['market_regime'] == "TRENDING" and abs(score - 50) > 20) else "LOW"
    
    return {
        "momentum_score": score,
        "momentum_rating": rating,
        "confidence": confidence,
        "meta": analysis
    }
