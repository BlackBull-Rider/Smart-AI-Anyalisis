import pandas as pd
import numpy as np
from backend.indicators.momentum import calculate_rsi, calculate_macd, calculate_roc

def analyze_momentum(df: pd.DataFrame) -> dict:
    """
    Momentum Intelligence: Strength, Acceleration, and Divergence.
    """
    rsi = calculate_rsi(df['close'])
    macd_df = calculate_macd(df['close'])
    roc = calculate_roc(df['close'], period=10)
    
    # 1. Momentum Strength (Using RSI)
    rsi_val = rsi.iloc[-1]
    strength = "OVERBOUGHT" if rsi_val > 70 else "OVERSOLD" if rsi_val < 30 else "NEUTRAL"
    
    # 2. Momentum Acceleration (Rate of Change of ROC)
    # ROC বাড়ছে মানে মোমেন্টাম বাড়ছে (Acceleration)
    roc_current = roc.iloc[-1]
    roc_prev = roc.iloc[-2]
    acceleration = "ACCELERATING" if roc_current > roc_prev else "DECELERATING"
    
    # 3. MACD Analysis
    macd_hist = macd_df['macd_hist'].iloc[-1]
    macd_signal = "BULLISH_CROSS" if macd_hist > 0 else "BEARISH_CROSS"
    
    # 4. Divergence Detection (Simple Logic)
    # Price Low lower, but RSI Low higher (Bullish Divergence)
    price_low_lower = df['close'].iloc[-1] < df['close'].iloc[-2]
    rsi_low_higher = rsi.iloc[-1] > rsi.iloc[-2]
    divergence = "BULLISH_DIVERGENCE" if (price_low_lower and rsi_low_higher) else "NONE"
    
    # 5. Momentum Regime (Efficiency Ratio - Bonus)
    # মার্কেট কি ট্রেন্ডিং না রেঞ্জবাউন্ড?
    price_change = abs(df['close'].iloc[-1] - df['close'].iloc[-10])
    volatility = df['close'].diff().abs().rolling(10).sum().iloc[-1]
    efficiency_ratio = price_change / volatility if volatility != 0 else 0
    regime = "TRENDING" if efficiency_ratio > 0.3 else "CHOPPY"
    
    return {
        "strength": strength,
        "acceleration": acceleration,
        "macd_status": macd_signal,
        "divergence": divergence,
        "market_regime": regime,
        "metrics": {
            "rsi": rsi_val,
            "roc": roc_current
        }
    }
