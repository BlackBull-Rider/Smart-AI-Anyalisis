import pandas as pd
import numpy as np
from backend.indicators.smart_money import calculate_market_structure

def analyze_market_structure(df: pd.DataFrame) -> dict:
    """
    মার্কেট স্ট্রাকচারের গভীর বিশ্লেষণ (BOS, CHOCH, Liquidity).
    এটি শুধু র ডেটা নয়, মার্কেট এখন কোন স্টেজে আছে তা বলবে।
    """
    structure = calculate_market_structure(df)
    
    # লাস্ট ২টা সুইং হাই/লো এর ডেটা
    last_swing_high = structure['swing_high'].tail(10).sum()
    last_swing_low = structure['swing_low'].tail(10).sum()
    
    # মার্কেট বায়াস (Bias) ডিটারমাইন করা
    if last_swing_high > 0 and last_swing_low > 0:
        market_bias = "CONSOLIDATION"
    elif last_swing_high > last_swing_low:
        market_bias = "BULLISH_STRUCTURE"
    else:
        market_bias = "BEARISH_STRUCTURE"
        
    # CHOCH Detection Logic
    # যদি আগের সুইং লো ব্রেক করে প্রাইস নিচে নামে, তবে সেটা CHOCH
    is_choch = (df['close'].iloc[-1] < df['low'].rolling(window=10).min().iloc[-1]) and (market_bias == "BULLISH_STRUCTURE")
    
    return {
        "market_bias": market_bias,
        "is_choch_pending": is_choch,
        "structure_strength": "HIGH" if (last_swing_high + last_swing_low) > 2 else "LOW"
    }
