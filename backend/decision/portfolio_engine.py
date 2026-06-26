import pandas as pd

def analyze_portfolio(holdings: list, total_capital: float) -> dict:
    """
    The CIO: Analyzes portfolio health, diversification, and drift.
    holdings: list of {'ticker': str, 'value': float, 'sector': str, 'entry_score': float}
    """
    
    # 1. Sector Concentration Analysis
    sector_exposure = {}
    for h in holdings:
        sector_exposure[h['sector']] = sector_exposure.get(h['sector'], 0) + h['value']
    
    # 2. Diversification Score (0-100)
    # যদি একটি সেক্টরে ৫০% এর বেশি থাকে, তবে ডাইভারসিফিকেশন স্কোর কমবে
    max_sector_pct = max(sector_exposure.values()) / total_capital
    diversification_score = max(0, 100 - (max_sector_pct * 100))
    
    # 3. Portfolio Drift & Rebalancing Advice
    rebalance_suggestions = []
    for h in holdings:
        current_weight = h['value'] / total_capital
        # যদি কোনো পজিশন ১০% এর বেশি গেইন করে, তবে প্রফিট বুকিং এর পরামর্শ
        if current_weight > 0.20: # 20% cap
            rebalance_suggestions.append({
                "ticker": h['ticker'],
                "action": "REDUCE_POSITION",
                "reason": "EXCESS_WEIGHT_THRESHOLD"
            })
            
    return {
        "portfolio_health_score": round(diversification_score, 2),
        "sector_exposure": sector_exposure,
        "rebalance_suggestions": rebalance_suggestions,
        "is_balanced": diversification_score > 60
    }

def suggest_portfolio_adjustment(portfolio_metrics: dict) -> str:
    """
    Final Portfolio Intelligence: Suggests overall action.
    """
    if portfolio_metrics['portfolio_health_score'] < 40:
        return "CRITICAL_OVER_CONCENTRATION_REBALANCE_IMMEDIATELY"
    elif portfolio_metrics['portfolio_health_score'] < 70:
        return "OPTIMIZE_DIVERSIFICATION"
    else:
        return "PORTFOLIO_BALANCED"
