import pandas as pd
from backend.analyzers.trend_analyzer import analyze_trend_intelligence

def calculate_trend_score(df: pd.DataFrame) -> dict:
    """
    Converts Trend Intelligence into a quantifiable Score (0-100).
    """
    analysis = analyze_trend_intelligence(df)
    
    # 1. Base Score (Neutral = 50)
    score = 50
    
    # 2. Score Weighting
    # Directional Bias
    if analysis['direction'] == "BULLISH":
        score += 20
    else:
        score -= 20
        
    # Trend Strength (ADX Factor)
    if analysis['strength'] == "STRONG":
        score += 15 if analysis['direction'] == "BULLISH" else -15
        
    # Trend Quality (Alignment Factor)
    if analysis['quality'] == "ALIGNED":
        score += 10 if analysis['direction'] == "BULLISH" else -10
        
    # Trend Exhaustion (Penalty)
    # যদি ট্রেন্ড ক্লান্ত থাকে, স্কোর কমে যাবে কারণ ব্রেকআউট বা রিভার্সাল হতে পারে
    if analysis['exhaustion'] == "POTENTIAL_EXHAUSTION":
        score = score * 0.85 
        
    # 3. Clamping Score (0 to 100)
    score = max(0, min(100, round(score)))
    
    # 4. Rating Logic
    if score >= 85: rating = "STRONG_BULL"
    elif score >= 65: rating = "WEAK_BULL"
    elif score > 35: rating = "NEUTRAL"
    elif score > 15: rating = "WEAK_BEAR"
    else: rating = "STRONG_BEAR"
    
    # 5. Confidence Logic
    # যদি Strength এবং Alignment দুটোই হাই থাকে তবেই Confidence High
    confidence = "HIGH" if (analysis['strength'] == "STRONG" and analysis['quality'] == "ALIGNED") else "LOW"
    
    return {
        "trend_score": score,
        "trend_rating": rating,
        "confidence": confidence,
        "meta": analysis
    }
