import pandas as pd
from backend.decision import market_regime_engine, portfolio_engine, ranking_engine, watchlist_engine

def get_dashboard_summary(df: pd.DataFrame, portfolio_holdings: list, universe_data: list) -> dict:
    """
    The Dashboard Aggregator: Creates a high-level summary of the entire trading system.
    
    Args:
        df: Current market data
        portfolio_holdings: Current position data
        universe_data: List of scores for all tracked stocks
    """
    
    # 1. Market Health
    regime = market_regime_engine.decide_market_permission(df)
    
    # 2. Portfolio Snapshot
    p_health = portfolio_engine.analyze_portfolio(portfolio_holdings, 1000000) # Example total capital
    
    # 3. Intelligence Feed (Rankings & Watchlist)
    ranked = ranking_engine.rank_stocks(universe_data)
    watchlist = watchlist_engine.generate_smart_watchlists(universe_data)
    
    # 4. Aggregating for Dashboard View
    return {
        "market_summary": {
            "status": regime['trade_permission'],
            "bias": regime['market_bias'],
            "risk_level": regime['reasoning']
        },
        "portfolio_snapshot": {
            "health_score": p_health['portfolio_health_score'],
            "is_balanced": p_health['is_balanced'],
            "active_suggestions": len(p_health['rebalance_suggestions'])
        },
        "top_opportunities": ranked[:5], # Show Top 5
        "watchlist_summary": {
            "high_conviction_count": len(watchlist['high_conviction_list']),
            "opportunity_count": len(watchlist['opportunity_list'])
        },
        "system_status": "OPERATIONAL_READY"
    }
