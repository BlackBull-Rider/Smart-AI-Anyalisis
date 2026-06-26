import pandas as pd
from datetime import datetime, timedelta
from backend.engines import trend_engine, market_regime_engine

def calculate_holding_strategy(df: pd.DataFrame, entry_type: str = "SWING") -> dict:
    """
    The Time Intelligence: Estimates holding duration and confidence based on 
    Trend Maturity and Market Regime.
    """
    # 1. Gather Intelligence
    t_score = trend_engine.calculate_trend_score(df)
    regime = market_regime_engine.decide_market_permission(df)
    
    # 2. Logic: Estimate Expected Duration (Days)
    if entry_type == "INTRADAY":
        duration = 1
        review_date = datetime.now() + timedelta(hours=6)
    elif entry_type == "SWING":
        # ট্রেন্ড যদি স্ট্রং হয় তবে বেশিদিন হোল্ড করা যায়
        duration = 10 if t_score['trend_score'] > 70 else 5
        review_date = datetime.now() + timedelta(days=duration)
    else: # LONG_TERM
        duration = 90
        review_date = datetime.now() + timedelta(days=duration)
        
    # 3. Holding Confidence (Decay Logic)
    # যদি ট্রেন্ড ম্যাচিউরড (পুরানো) হয়ে যায়, তবে কনফিডেন্স কমে যাবে
    trend_strength = t_score['trend_score']
    if trend_strength < 40:
        confidence = "LOW"
        reason = "TREND_WEAKENING_EXIT_SOON"
    elif regime['market_bias'] == "NEUTRAL":
        confidence = "MEDIUM"
        reason = "MARKET_CHOPPY_HOLD_WITH_CAUTION"
    else:
        confidence = "HIGH"
        reason = "TREND_ALIGNED_HOLD"
        
    return {
        "expected_duration_days": duration,
        "holding_confidence": confidence,
        "review_timeline": review_date.strftime('%Y-%m-%d'),
        "reasoning": reason,
        "meta": {
            "trend_score": trend_strength,
            "bias": regime['market_bias']
        }
    }
