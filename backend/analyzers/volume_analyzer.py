import pandas as pd
import numpy as np
from backend.indicators.volume import calculate_obv, calculate_vwap

def analyze_volume(df: pd.DataFrame) -> dict:
    """
    Volume Intelligence: Confirmation, Divergence, and RVOL analysis.
    """
    vol = df['volume']
    price = df['close']
    ma_vol = vol.rolling(window=20).mean()
    
    # 1. Volume Explosion (RVOL calculation)
    # বর্তমান ভলিউম যদি ২০ দিনের গড়ের ১.৫ গুণের বেশি হয়
    rvol = vol.iloc[-1] / ma_vol.iloc[-1]
    is_explosion = rvol > 1.5
    
    # 2. Volume Confirmation
    # প্রাইস মুভমেন্ট এবং ভলিউম একই দিকে কি না
    price_change = price.iloc[-1] > price.iloc[-2]
    vol_increase = vol.iloc[-1] > vol.iloc[-2]
    confirmation = "CONFIRMED_TREND" if (price_change == vol_increase) else "DIVERGENCE_WARNING"
    
    # 3. Smart Volume (OBV Trend)
    obv = calculate_obv(df)
    obv_trend = "ACCUMULATION" if obv.iloc[-1] > obv.rolling(10).mean().iloc[-1] else "DISTRIBUTION"
    
    # 4. Volume Divergence Detection
    # প্রাইস হাই বানাচ্ছে কিন্তু ভলিউম কমছে
    price_high = price.iloc[-1] > price.iloc[-5:].max()
    vol_low = vol.iloc[-1] < vol.iloc[-5:].min()
    divergence = "BEARISH_VOLUME_DIVERGENCE" if (price_high and vol_low) else "NONE"
    
    # 5. VWAP Relationship (Market Sentiment)
    vwap = calculate_vwap(df)
    sentiment = "BULLISH_SENTIMENT" if price.iloc[-1] > vwap.iloc[-1] else "BEARISH_SENTIMENT"
    
    return {
        "rvol": round(rvol, 2),
        "is_explosion": is_explosion,
        "confirmation": confirmation,
        "smart_money_trend": obv_trend,
        "divergence": divergence,
        "sentiment": sentiment,
        "metrics": {
            "volume_delta": vol.iloc[-1] - vol.iloc[-2],
            "vwap_distance": price.iloc[-1] - vwap.iloc[-1]
        }
    }
