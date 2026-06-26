import pandas as pd
import numpy as np
from backend.indicators.moving_average import calculate_sma, calculate_supertrend
from backend.indicators.momentum import calculate_adx_dmi, calculate_rsi
from backend.indicators.volatility import calculate_bollinger_bands

def analyze_trend_intelligence(df: pd.DataFrame) -> dict:
    """
    Indicators ব্যবহার করে ট্রেন্ডের গভীর বিশ্লেষণ (Market Intelligence).
    """
    
    # ইন্ডিকেটর ক্যালকুলেশন
    st = calculate_supertrend(df)
    ma_short = calculate_sma(df['close'], 20)
    ma_long = calculate_sma(df['close'], 50)
    adx_data = calculate_adx_dmi(df)
    rsi = calculate_rsi(df['close'])
    bb = calculate_bollinger_bands(df['close'])
    
    # 1. Trend Direction
    # সুপারট্রেন্ড এবং MA এর রিলেশন অনুযায়ী ডিরেকশন
    direction = "BULLISH" if st['trend_direction'].iloc[-1] == 1 and df['close'].iloc[-1] > ma_short.iloc[-1] else "BEARISH"
    
    # 2. Trend Strength
    # ADX > 25 হলে স্ট্রং ট্রেন্ড
    adx_val = adx_data['adx'].iloc[-1]
    strength = "STRONG" if adx_val > 25 else "WEAK"
    
    # 3. Trend Quality
    # শর্ট টার্ম MA লং টার্ম MA এর উপরে থাকলে 'Aligned'
    quality = "ALIGNED" if ma_short.iloc[-1] > ma_long.iloc[-1] else "DISORDERED"
    
    # 4. Trend Exhaustion
    # RSI > 70 বা BB Upper এর কাছাকাছি থাকলে এক্সহশন
    is_exhausted = (rsi.iloc[-1] > 70) or (df['close'].iloc[-1] > bb['bb_upper'].iloc[-1])
    exhaustion = "POTENTIAL_EXHAUSTION" if is_exhausted else "HEALTHY"
    
    # 5. Trend Continuation
    # পুলব্যাক চেক (Price যদি MA এর কাছাকাছি থাকে)
    is_pullback = (df['close'].iloc[-1] > ma_short.iloc[-1]) and (df['low'].iloc[-1] < ma_short.iloc[-1] * 1.01)
    continuation = "CONTINUATION_SETUP" if is_pullback else "NO_IMMEDIATE_SETUP"
    
    return {
        "direction": direction,
        "strength": strength,
        "quality": quality,
        "exhaustion": exhaustion,
        "continuation": continuation,
        "metrics": {
            "adx": adx_val,
            "rsi": rsi.iloc[-1]
        }
    }
