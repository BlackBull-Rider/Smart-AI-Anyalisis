import pandas as pd
import numpy as np
from backend.indicators.moving_average import calculate_sma
from backend.indicators.volatility import calculate_bollinger_bands

def analyze_market_regime(df: pd.DataFrame) -> dict:
    """
    Market Regime Intelligence: Bull/Bear/Sideways, Volatility & Risk Regime.
    """
    close = df['close']
    ma200 = calculate_sma(close, 200).iloc[-1]
    ma50 = calculate_sma(close, 50).iloc[-1]
    bb = calculate_bollinger_bands(close, period=20)
    
    # 1. Bull/Bear/Sideways Analysis
    if close.iloc[-1] > ma200 and ma50 > ma200:
        regime = "BULL_MARKET"
    elif close.iloc[-1] < ma200 and ma50 < ma200:
        regime = "BEAR_MARKET"
    else:
        regime = "SIDEWAYS_OR_TRANSITION"
        
    # 2. Volatility Regime
    bb_width = (bb['bb_upper'] - bb['bb_lower']) / bb['bb_middle']
    volatility_regime = "HIGH_VOLATILITY" if bb_width.iloc[-1] > bb_width.rolling(20).mean().iloc[-1] else "LOW_VOLATILITY"
    
    # 3. Risk Regime (The "Brain" Logic)
    # High Volatility + Bear Market = High Risk (Avoid)
    # Low Volatility + Bull Market = Low Risk (Optimal)
    if regime == "BEAR_MARKET" and volatility_regime == "HIGH_VOLATILITY":
        risk_regime = "EXTREME_RISK"
    elif regime == "BULL_MARKET" and volatility_regime == "LOW_VOLATILITY":
        risk_regime = "LOW_RISK_ACCUMULATION"
    else:
        risk_regime = "MODERATE_RISK"
        
    return {
        "market_regime": regime,
        "volatility_regime": volatility_regime,
        "risk_regime": risk_regime,
        "metrics": {
            "distance_from_ma200": (close.iloc[-1] - ma200) / ma200,
            "bb_width": bb_width.iloc[-1]
        }
    }
