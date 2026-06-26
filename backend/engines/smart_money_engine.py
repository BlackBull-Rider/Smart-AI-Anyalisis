import pandas as pd
from backend.analyzers.smart_money_analyzer import analyze_smart_money

def calculate_smart_money_score(df: pd.DataFrame) -> dict:
    """
    Converts Smart Money Intelligence into a quantifiable Score (0-100).
    """
    analysis = analyze_smart_money(df)
    
    # 1. Base Score (Neutral = 50)
    score = 50
    
    # 2. Market Structure Scoring (Highest Priority)
    if analysis['structure']['bos']:
        score += 20  # Trend confirmation
    
    if analysis['structure']['choch']:
        score += 15  # Potential reversal setup
        
    # 3. Order Block & FVG (Quality Scoring)
    # যদি ব্লকটি FRESH হয় তবেই ট্রেড করার কনভিকশন বেশি
    if analysis['order_block']['validity'] == "FRESH":
        score += 20
    elif analysis['order_block']['validity'] == "MITIGATED":
        score += 5
        
    if analysis['fvg']['needs_mitigation']:
        score += 10 # Imbalance means price likely to return
        
    # 4. Liquidity Score (Risk/Opp Factor)
    # লিকুইডিটি থ্রেট থাকলে সেটাকে ট্রেড করার আগে সাবধান হতে হবে (Penalty)
    if analysis['liquidity'] != "CLEAR":
        score -= 15
        
    # 3. Clamping Score (0 to 100)
    score = max(0, min(100, round(score)))
    
    # 4. Rating Logic
    if score >= 80: rating = "INSTITUTIONAL_BUY_ZONE"
    elif score >= 60: rating = "STRUCTURE_BULLISH"
    elif score > 40: rating = "NEUTRAL_SMC"
    elif score > 20: rating = "STRUCTURE_BEARISH"
    else: rating = "INSTITUTIONAL_SELL_ZONE"
    
    # 5. Confidence Logic
    # Structure + Fresh OB থাকলে কনফিডেন্স হাই
    confidence = "HIGH" if (analysis['structure']['bos'] and analysis['order_block']['validity'] == "FRESH") else "LOW"
    
    return {
        "smart_money_score": score,
        "smart_money_rating": rating,
        "confidence": confidence,
        "meta": analysis
    }
