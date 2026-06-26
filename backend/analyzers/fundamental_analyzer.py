import pandas as pd
import numpy as np

def analyze_fundamentals(fund_df: pd.DataFrame) -> dict:
    """
    Fundamental Intelligence: Business Quality, Valuation, and Growth.
    fund_df তে থাকতে হবে: [roe, pe_ratio, debt_to_equity, revenue_growth, profit_margin]
    """
    # শেষ কোয়ার্টারের ডেটা
    last = fund_df.iloc[-1]
    
    # 1. Business Quality (ROE & Debt)
    # ROE > 15% ভালো, Debt/Equity < 1.0 নিরাপদ
    quality_score = 0
    if last['roe'] > 15: quality_score += 1
    if last['debt_to_equity'] < 1.0: quality_score += 1
    business_quality = "HIGH_QUALITY" if quality_score == 2 else "AVERAGE" if quality_score == 1 else "RISKY"
    
    # 2. Valuation Analysis
    # PE Ratio Sector এর তুলনায় কম কি না (এখানে আমরা গ্লোবাল স্ট্যান্ডার্ড ধরেছি)
    valuation = "UNDERVALUED" if last['pe_ratio'] < 20 else "OVERVALUED"
    
    # 3. Growth Analysis
    growth = "STRONG_GROWTH" if last['revenue_growth'] > 10 else "STAGNANT"
    
    # 4. Profitability
    # Profit Margin > 10% হেলদি
    profitability = "HIGH_MARGIN" if last['profit_margin'] > 10 else "LOW_MARGIN"
    
    # 5. Overall Health
    # সব প্যারামিটার মিলিয়ে একটি কনক্লুশন
    health_check = "BUY_THESIS_VALID" if (quality_score >= 1 and valuation == "UNDERVALUED") else "WATCHLIST_ONLY"
    
    return {
        "business_quality": business_quality,
        "valuation": valuation,
        "growth": growth,
        "profitability": profitability,
        "overall_health": health_check,
        "metrics": {
            "roe": last['roe'],
            "pe": last['pe_ratio'],
            "debt_equity": last['debt_to_equity']
        }
    }
