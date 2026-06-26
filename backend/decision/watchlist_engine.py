import pandas as pd

def generate_smart_watchlists(universe_data: list) -> dict:
    """
    The Intelligence Filter: Categorizes the market universe into actionable lists.
    
    universe_data: A list of dicts containing scores for each stock.
    """
    
    smart_watchlist = []     # Overall high quality
    high_conviction = []     # Conviction > 80 (High confidence)
    opportunity_list = []    # Entry strategy active
    auto_watchlist = []      # Volume/Momentum filtered
    
    for stock in universe_data:
        # 1. High Conviction Logic
        if stock.get('conviction_score', 0) >= 80:
            high_conviction.append(stock['ticker'])
            
        # 2. Opportunity Logic (Ready for Entry)
        if stock.get('entry_signal') == "EXECUTE_ENTRY" and stock.get('master_score', 0) > 60:
            opportunity_list.append(stock['ticker'])
            
        # 3. Smart Watchlist (Fundamental + Technical)
        if stock.get('fundamental_score', 0) > 60 and stock.get('trend_score', 0) > 60:
            smart_watchlist.append(stock['ticker'])
            
        # 4. Auto Watchlist (Momentum/Volume filters)
        if stock.get('momentum_score', 0) > 70 and stock.get('volume_score', 0) > 60:
            auto_watchlist.append(stock['ticker'])
            
    return {
        "smart_watchlist": smart_watchlist,
        "high_conviction_list": high_conviction,
        "opportunity_list": opportunity_list,
        "auto_watchlist": auto_watchlist,
        "meta": {
            "total_universe_scanned": len(universe_data),
            "timestamp": "CURRENT_SESSION"
        }
    }
