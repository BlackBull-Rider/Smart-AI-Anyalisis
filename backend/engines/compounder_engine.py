import pandas as pd
# Assuming fundamental data is used for compounding analysis
from backend.analyzers.fundamental_analyzer import analyze_fundamentals

def calculate_compounder_score(fund_df: pd.DataFrame) -> dict:
    """
    Evaluates the 'Compounder' potential of a stock.
    Focuses on: Consistency, Moat, and Capital Efficiency.
    """
    analysis = analyze_fundamentals(fund_df)
    metrics = analysis['metrics']
    
    # 1. Base Score (Neutral = 50)
    score = 50
    
    # 2. Wealth Creation Score (ROE & Margin Stability)
    # ROE > 20% is the hallmark of a compounder
    if metrics['roe'] > 25: score += 20
    elif metrics['roe'] > 15: score += 10
    else: score -= 20
    
    # 3. Business Quality Score (Debt & Growth)
    # Low Debt = Sustainable compounding
    if metrics['debt_equity'] < 0.3: score += 15
    elif metrics['debt_equity'] < 0.7: score += 5
    else: score -= 15
    
    # Growth quality
    if analysis['growth'] == "STRONG_GROWTH": score += 15
    
    # 4. Valuation Penalty (Don't overpay for quality)
    # High PE can kill compounding returns (Mean reversion risk)
    if metrics['pe'] > 50: score -= 15
    elif metrics['pe'] < 25: score += 10
    
    # Final Clamping
    score = max(0, min(100, round(score)))
    
    # Rating Logic
    if score >= 80: rating = "COMPOUNDING_MACHINE"
    elif score >= 60: rating = "QUALITY_GROWTH"
    elif score > 40: rating = "AVERAGE"
    else: rating = "WEAK_BUSINESS"
    
    # 5. Confidence Logic
    # High ROE + Low Debt = High Confidence
    confidence = "HIGH" if (metrics['roe'] > 20 and metrics['debt_equity'] < 0.5) else "LOW"
    
    return {
        "compounder_score": score,
        "wealth_creation_score": (metrics['roe'] + metrics['revenue_growth_proxy'] if 'revenue_growth_proxy' in metrics else metrics['roe']),
        "business_quality_score": score,
        "rating": rating,
        "confidence": confidence,
        "meta": analysis
    }
