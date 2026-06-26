import pandas as pd
from backend.analyzers.market_regime_analyzer import analyze_market_regime
from backend.engines.trend_engine import calculate_trend_score

def decide_market_permission(df: pd.DataFrame) -> dict:
    """
    The Gatekeeper: Determines if market conditions allow for trading.
    Generates Trade Permission, Bias, and Dynamic Risk Multipliers.
    """
    # 1. Gather Intelligence
    regime_data = analyze_market_regime(df)
    trend_data = calculate_trend_score(df)
    
    # 2. Logic: Trade Permission Gatekeeper
    # Default: Permission Denied
    permission = "DENIED"
    reason = "UNSTABLE_MARKET_CONDITIONS"
    risk_multiplier = 0.0 # পজিশন সাইজ ডিফল্ট জিরো
    
    # 3. Decision Logic based on Regime + Trend
    if regime_data['risk_regime'] == "LOW_RISK_ACCUMULATION":
        permission = "ALLOWED"
        reason = "OPTIMAL_MARKET_ENVIRONMENT"
        risk_multiplier = 1.5 # ফুল কনফিডেন্স
        
    elif regime_data['risk_regime'] == "MODERATE_RISK":
        permission = "ALLOWED"
        reason = "STABLE_MARKET"
        risk_multiplier = 1.0 # স্ট্যান্ডার্ড রিস্ক
        
    elif regime_data['risk_regime'] == "EXTREME_RISK":
        permission = "DENIED"
        reason = "VOLATILITY_TOO_HIGH_FOR_TRADING"
        risk_multiplier = 0.0
        
    # 4. Market Bias Determination
    market_bias = "BULLISH" if regime_data['market_regime'] == "BULL_MARKET" else \
                  "BEARISH" if regime_data['market_regime'] == "BEAR_MARKET" else "NEUTRAL"
                  
    # 5. Trend Confirmation
    # যদি মার্কেট বুলিশ হয় কিন্তু ট্রেন্ড স্কোর খারাপ হয়, তবে কনফিডেন্স কমানো
    if market_bias == "BULLISH" and trend_data['trend_score'] < 50:
        market_bias = "BULLISH_WEAK"
        risk_multiplier *= 0.5
        
    return {
        "trade_permission": permission,
        "market_bias": market_bias,
        "risk_multiplier": risk_multiplier,
        "reasoning": reason,
        "meta": {
            "regime": regime_data,
            "trend": trend_data
        }
    }
