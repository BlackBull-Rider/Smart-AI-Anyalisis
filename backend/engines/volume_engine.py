import pandas as pd
from backend.analyzers.volume_analyzer import analyze_volume

def calculate_volume_score(df: pd.DataFrame) -> dict:
    """
    Converts Volume Intelligence into a quantifiable Score (0-100).
    """
    analysis = analyze_volume(df)
    
    # 1. Base Score (Neutral = 50)
    score = 50
    
    # 2. Score Weighting
    # Volume Explosion (RVOL) থাকলে কনভিকশন হাই
    if analysis['is_explosion']:
        score += 20
        
    # Trend Confirmation
    if analysis['confirmation'] == "CONFIRMED_TREND":
        score += 15
    elif analysis['divergence'] == "BEARISH_VOLUME_DIVERGENCE":
        score -= 20
        
    # Smart Money Trend (OBV)
    if analysis['smart_money_trend'] == "ACCUMULATION":
        score += 10
    else:
        score -= 10
        
    # VWAP Sentiment
    if analysis['sentiment'] == "BULLISH_SENTIMENT":
        score += 5
    else:
        score -= 5
        
    # 3. Clamping Score (0 to 100)
    score = max(0, min(100, round(score)))
    
    # 4. Rating Logic
    if score >= 80: rating = "STRONG_ACCUMULATION"
    elif score >= 60: rating = "BULLISH_VOLUME"
    elif score > 40: rating = "NEUTRAL"
    elif score > 20: rating = "BEARISH_VOLUME"
    else: rating = "STRONG_DISTRIBUTION"
    
    # 5. Confidence Logic
    # যদি ভলিউম এক্সপ্লোশন এবং কনফার্মেশন একই সাথে হয়, তবেই হাই কনফিডেন্স
    confidence = "HIGH" if (analysis['is_explosion'] and analysis['confirmation'] == "CONFIRMED_TREND") else "LOW"
    
    return {
        "volume_score": score,
        "volume_rating": rating,
        "confidence": confidence,
        "meta": analysis
    }
