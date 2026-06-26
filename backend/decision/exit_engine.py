import pandas as pd
from backend.engines import trend_engine, volatility_engine, market_regime_engine

def calculate_exit_strategy(df: pd.DataFrame, entry_price: float, position_side: str = "LONG") -> dict:
    """
    The Protector: Determines when to exit based on Volatility, Trend, and Regime.
    Returns: Action (HOLD, EXIT, PARTIAL_EXIT), Reason, and StopLevel.
    """
    # 1. Gather Intelligence
    t_score = trend_engine.calculate_trend_score(df)
    v_score = volatility_engine.calculate_volatility_score(df)
    current_price = df['close'].iloc[-1]
    
    # 2. Emergency Exit (Layer 4 Gatekeeper)
    regime = market_regime_engine.decide_market_permission(df)
    if regime['trade_permission'] == "DENIED":
        return {"action": "EMERGENCY_EXIT", "reason": "MARKET_RISK_SPIKE"}
    
    # 3. Trailing Stop Loss (Volatility-Based)
    # ATR ব্যবহার করে স্টপ লস ট্রেইল করা
    atr = v_score['meta']['metrics']['atr']
    trailing_sl = current_price - (atr * 2) if position_side == "LONG" else current_price + (atr * 2)
    
    # 4. Profit Taking Logic
    # যদি ট্রেন্ড স্কোর নাটকীয়ভাবে কমে যায় (Trend Exhaustion)
    is_trend_weakening = t_score['trend_score'] < 40
    is_profit_target_met = (current_price > entry_price * 1.03) # 3% Target
    
    # 5. Decision Matrix
    # Stop Loss Hit
    if (position_side == "LONG" and current_price <= trailing_sl) or \
       (position_side == "SHORT" and current_price >= trailing_sl):
        return {"action": "STOP_LOSS_EXIT", "reason": "VOLATILITY_BREAKDOWN", "stop_level": round(trailing_sl, 2)}
        
    # Trend Reversal or Profit Taking
    if is_trend_weakening or is_profit_target_met:
        return {
            "action": "PARTIAL_EXIT" if is_profit_target_met else "FULL_EXIT", 
            "reason": "PROFIT_BOOKING" if is_profit_target_met else "TREND_EXHAUSTION",
            "stop_level": round(trailing_sl, 2)
        }
        
    return {"action": "HOLD", "reason": "TREND_INTACT", "stop_level": round(trailing_sl, 2)}
