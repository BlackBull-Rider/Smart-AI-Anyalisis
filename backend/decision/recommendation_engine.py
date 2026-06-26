import pandas as pd

def get_recommendation(master_score: float, conviction_score: float) -> dict:
    """
    The Translator: Converts quantitative scores into actionable qualitative recommendations.
    
    Args:
        master_score: 0-100 (Based on trend, volume, smart money, etc.)
        conviction_score: 0-100 (Confidence metric)
    """
    
    # Logic: Score এবং Conviction এর কম্বিনেশনে রেকমেন্ডেশন তৈরি
    # হাই স্কোর + লো কনভিকশন = হোল্ড (অনিশ্চয়তা)
    # হাই স্কোর + হাই কনভিকশন = স্ট্রং বাই
    
    avg_score = (master_score + conviction_score) / 2
    
    if master_score >= 90 and conviction_score >= 80:
        recommendation = "STRONG_BUY"
    elif master_score >= 75 and conviction_score >= 60:
        recommendation = "BUY"
    elif master_score >= 50 and conviction_score >= 40:
        recommendation = "HOLD"
    elif master_score >= 30:
        recommendation = "REDUCE"
    elif master_score >= 15:
        recommendation = "SELL"
    else:
        recommendation = "AVOID"
        
    # Generating Action Plan
    actions = {
        "STRONG_BUY": "EXECUTE_FULL_POSITION_IMMEDIATELY",
        "BUY": "EXECUTE_POSITION_WITH_TIGHT_SL",
        "HOLD": "DO_NOTHING_MONITOR_LEVELS",
        "REDUCE": "BOOK_PARTIAL_PROFIT_OR_TIGHTEN_SL",
        "SELL": "EXIT_ALL_POSITIONS_IMMEDIATELY",
        "AVOID": "DO_NOT_TOUCH_HIGH_RISK"
    }
    
    return {
        "recommendation": recommendation,
        "action_plan": actions.get(recommendation, "WAIT"),
        "confidence_grade": "A+" if conviction_score > 85 else "B" if conviction_score > 50 else "C",
        "is_actionable": recommendation in ["STRONG_BUY", "BUY", "SELL"]
    }
