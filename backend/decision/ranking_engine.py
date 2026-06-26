import pandas as pd

def rank_stocks(stock_data_list: list) -> list:
    """
    The Opportunity Finder: Ranks a universe of stocks based on weighted scores.
    
    stock_data_list: List of dictionaries, each containing scores from various engines.
    """
    
    # 1. Weights: Define how much each metric impacts the rank
    # Fundamental & Trend & Smart Money are prioritized
    weights = {
        'trend_score': 0.25,
        'smart_money_score': 0.25,
        'fundamental_score': 0.20,
        'momentum_score': 0.15,
        'volume_score': 0.10,
        'volatility_score': 0.05
    }
    
    ranked_list = []
    
    for stock in stock_data_list:
        # Calculate Weighted Composite Score
        composite_score = sum(stock.get(key, 50) * weights.get(key, 0.1) for key in weights)
        
        # Apply Safety Filter: If fundamental or smart_money is too low, penalize heavily
        if stock.get('smart_money_score', 0) < 30:
            composite_score *= 0.7  # 30% Penalty
            
        ranked_list.append({
            "ticker": stock['ticker'],
            "rank_score": round(composite_score, 2),
            "signal_reliability": stock.get('signal_reliability', 'NEUTRAL'),
            "is_tradeable": stock.get('is_safe', True)
        })
        
    # 2. Sort by rank_score (Descending)
    ranked_list = sorted(ranked_list, key=lambda x: x['rank_score'], reverse=True)
    
    return ranked_list
