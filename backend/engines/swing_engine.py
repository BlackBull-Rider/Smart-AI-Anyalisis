import pandas as pd
from backend.analyzers.pattern_analyzer import analyze_patterns

def calculate_swing_score(df: pd.DataFrame) -> dict:
    """
    Converts Pattern Intelligence into quantifiable Swing Scores (0-100).
    """
    analysis = analyze_patterns(df)
    
    # 1. Base Score (Neutral = 50)
    score = 50
    
    # 2. Swing Probability Scoring (Structure Focus)
    # যদি সুইং হাই এবং লো এর কাউন্ট বেশি থাকে, তবে স্ট্রাকচার শক্তিশালী
    swing_count = analysis['metrics']['swing_highs_count'] + analysis['metrics']['swing_lows_count']
    if swing_count >= 4:
        score += 20 # স্ট্রাকচার প্রতিষ্ঠিত
    elif swing_count < 2:
        score -= 20 # স্ট্রাকচার নেই (চপি মার্কেট)
        
    # 3. Swing Quality Analysis
    # Reversal Patterns (Head & Shoulders) থাকলে সুইং এর কোয়ালিটি রিস্কি
    if analysis['reversal_patterns']['head_and_shoulders']:
        score -= 15
        
    # Continuation Patterns (Triangle/Channel) থাকলে সুইং এর কোয়ালিটি হাই
    if analysis['continuation_patterns']['triangle_wedge'] or analysis['continuation_patterns']['channel']:
        score += 25
        
    # 4. Final Clamping
    score = max(0, min(100, round(score)))
    
    # 5. Rating Logic
    if score >= 80: rating = "STRONG_SWING_SETUP"
    elif score >= 60: rating = "VALID_SWING"
    elif score > 40: rating = "CONSOLIDATION"
    else: rating = "NO_CLEAR_SWING"
    
    # 6. Confidence Logic
    # যদি প্যাটার্ন স্পষ্ট হয়, তবে কনফিডেন্স হাই
    confidence = "HIGH" if (analysis['continuation_patterns']['triangle_wedge'] and swing_count >= 3) else "LOW"
    
    return {
        "swing_score": score,
        "swing_probability": score, # স্কোরকেই প্রোবাবিলিটি হিসেবে ধরা হয়েছে
        "swing_quality": rating,
        "confidence": confidence,
        "meta": analysis
    }
