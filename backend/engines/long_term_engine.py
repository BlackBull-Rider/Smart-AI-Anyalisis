import pandas as pd
from backend.analyzers.fundamental_analyzer import analyze_fundamentals

def calculate_long_term_score(fund_df: pd.DataFrame) -> dict:
    """
    Evaluates Long-Term Potential based on CAGR, Quality, and Valuation.
    """
    analysis = analyze_fundamentals(fund_df)
    metrics = analysis['metrics']
    
    # 1. CAGR Score (Proxy for Growth)
    # Revenue Growth কে CAGR এর প্রক্সি হিসেবে ধরা হয়েছে
    cagr_score = min(metrics.get('revenue_growth', 0) * 3, 100) 
    
    # 2. Quality Score (Consistency & Safety)
    # ROE (60%) + Debt (40%)
    quality_score = 0
    quality_score += (min(metrics['roe'], 30) / 30) * 60
    quality_score += (1 - min(metrics['debt_equity'], 1)) * 40
    
    # 3. Investment Score (Valuation Adjusted)
    # PE Ratio কম হলে ভালো, বেশি হলে পেনাল্টি
    pe = metrics['pe_ratio']
    if pe < 15: inv_score = 100
    elif pe < 30: inv_score = 60
    else: inv_score = 20
    
    # 4. Long-Term Composite Score (Weighted)
    # Quality (40%) + Growth (30%) + Valuation (30%)
    lt_score = (quality_score * 0.4) + (cagr_score * 0.3) + (inv_score * 0.3)
    lt_score = max(0, min(100, round(lt_score)))
    
    # 5. Rating Logic
    if lt_score >= 80: rating = "COMPOUNDING_GEM"
    elif lt_score >= 60: rating = "LONG_TERM_BET"
    elif lt_score > 40: rating = "HOLD_NEUTRAL"
    else: rating = "WEAK_LONG_TERM"
    
    # 6. Confidence Logic
    # হাই কোয়ালিটি এবং হাই গ্রোথ থাকলে কনফিডেন্স হাই
    confidence = "HIGH" if (quality_score > 60 and cagr_score > 50) else "LOW"
    
    return {
        "long_term_score": lt_score,
        "investment_score": inv_score,
        "cagr_score": round(cagr_score, 2),
        "quality_rating": rating,
        "confidence": confidence,
        "meta": analysis
    }
