import pandas as pd
from backend.analyzers.fundamental_analyzer import analyze_fundamentals

def calculate_fundamental_score(fund_df: pd.DataFrame) -> dict:
    """
    Converts Fundamental Intelligence into Business and Financial Scores (0-100).
    """
    analysis = analyze_fundamentals(fund_df)
    metrics = analysis['metrics']
    
    # 1. Business Score (Quality Focus)
    # ROE (60%) + Debt_to_Equity (40%)
    bus_score = 0
    if metrics['roe'] > 20: bus_score += 60
    elif metrics['roe'] > 15: bus_score += 40
    else: bus_score += 10
    
    if metrics['debt_equity'] < 0.5: bus_score += 40
    elif metrics['debt_equity'] < 1.0: bus_score += 20
    else: bus_score += 5
    
    # 2. Financial Score (Growth & Valuation Focus)
    # Revenue Growth (50%) + Valuation (PE) (50%)
    fin_score = 0
    if metrics['pe'] < 15: fin_score += 50
    elif metrics['pe'] < 25: fin_score += 30
    else: fin_score += 10
    
    # Growth factor
    if analysis['growth'] == "STRONG_GROWTH": fin_score += 50
    else: fin_score += 20
    
    # 3. Composite Fundamental Score
    fundamental_score = (bus_score + fin_score) / 2
    fundamental_score = max(0, min(100, round(fundamental_score)))
    
    # 4. Rating Logic
    if fundamental_score >= 80: rating = "BLUE_CHIP_QUALITY"
    elif fundamental_score >= 60: rating = "GROWTH_VALUE"
    elif fundamental_score > 40: rating = "AVERAGE_PERFORMER"
    else: rating = "FUNDAMENTAL_WEAKNESS"
    
    # 5. Confidence
    # যদি বিজনেস এবং ফিন্যান্সিয়াল দুটো স্কোরই হাই হয়
    confidence = "HIGH" if (bus_score > 60 and fin_score > 60) else "LOW"
    
    return {
        "fundamental_score": fundamental_score,
        "business_score": bus_score,
        "financial_score": fin_score,
        "fundamental_rating": rating,
        "confidence": confidence,
        "meta": analysis
    }
