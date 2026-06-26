import pandas as pd
from backend.engines import trend_engine, momentum_engine, volume_engine, volatility_engine, smart_money_engine
from backend.decision.market_regime_engine import decide_market_permission

def get_entry_strategy(df: pd.DataFrame) -> dict:
    """
    The Sniper: Calculates the optimal entry strategy based on confluence.
    Returns: Strategy, Price, and Zone.
    """
    # 1. Permission Check (Layer 4 Gatekeeper)
    gatekeeper = decide_market_permission(df)
    if gatekeeper['trade_permission'] == "DENIED":
        return {"action": "WAIT", "reason": gatekeeper['reason']}
        
    # 2. Gather Confluence Scores
    t_score = trend_engine.calculate_trend_score(df)
    v_score = volatility_engine.calculate_volatility_score(df)
    sm_score = smart_money_engine.calculate_smart_money_score(df)
    
    # 3. Dynamic Strategy Selection
    # Case A: Breakout Entry
    if v_score['volatility_score'] > 75 and t_score['trend_score'] > 70:
        strategy = "BREAKOUT_ENTRY"
        entry_price = df['high'].iloc[-1] + (df['close'].iloc[-1] * 0.001) # Breakout level + buffer
        entry_zone = "RESISTANCE_LEVEL"
        
    # Case B: Pullback Entry (Smart Money Confluence)
    elif sm_score['smart_money_score'] > 70 and t_score['trend_score'] > 60:
        strategy = "PULLBACK_ENTRY"
        entry_price = df['low'].rolling(5).min().iloc[-1] # Near support
        entry_zone = "ORDER_BLOCK_OR_SUPPORT"
        
    else:
        return {"action": "WAIT", "reason": "NO_CLEAR_SETUP"}
        
    # 4. Final Verification
    # যদি স্মার্ট মানি স্কোর কম থাকে, তবে এন্ট্রি রিজেক্ট করা (Liquidity Trap avoidance)
    if sm_score['smart_money_score'] < 40:
        return {"action": "WAIT", "reason": "HIGH_LIQUIDITY_RISK"}
        
    return {
        "action": "EXECUTE_ENTRY",
        "strategy": strategy,
        "entry_price": round(entry_price, 2),
        "entry_zone": entry_zone,
        "risk_multiplier": gatekeeper['risk_multiplier'],
        "meta": {
            "trend_strength": t_score['trend_score'],
            "volatility_context": v_score['volatility_rating']
        }
    }
