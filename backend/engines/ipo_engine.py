import pandas as pd
from backend.analyzers.ipo_analyzer import analyze_ipo

def calculate_ipo_score(subscription_data: dict, gmp: float, anchor_quality: str) -> dict:
    """
    Converts IPO Intelligence into quantifiable Listing and Opportunity Scores.
    """
    analysis = analyze_ipo(subscription_data, gmp, anchor_quality)
    
    # 1. IPO Score (Subscription Focus)
    # QIB (50%) + HNI (30%) + Retail (20%)
    ipo_score = analysis['listing_probability_score']
    
    # 2. Listing Score (GMP Focus)
    # GMP যদি হাই হয়, লিস্টিং গেইন স্কোর অটোমেটিক হাই হবে
    listing_score = min(gmp * 2, 100) # GMP 50% = 100 Score
    
    # 3. Opportunity Score (Weighted Average)
    # IPO Score (60%) + Listing Score (40%)
    opp_score = (ipo_score * 0.6) + (listing_score * 0.4)
    opp_score = max(0, min(100, round(opp_score)))
    
    # 4. Rating Logic
    if opp_score >= 80: rating = "SUPER_IPO"
    elif opp_score >= 60: rating = "STRONG_LISTING"
    elif opp_score > 40: rating = "AVERAGE"
    else: rating = "AVOID_IPO"
    
    # 5. Confidence Logic
    # QIB যদি খুব বেশি সাবস্ক্রাইব করে (High demand) তবেই কনফিডেন্স হাই
    confidence = "HIGH" if (subscription_data.get('qib', 0) > 20 and opp_score > 60) else "LOW"
    
    return {
        "ipo_score": ipo_score,
        "listing_score": listing_score,
        "opportunity_score": opp_score,
        "rating": rating,
        "confidence": confidence,
        "meta": analysis
    }
