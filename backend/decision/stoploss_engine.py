import pandas as pd
from backend.analyzers.volatility_analyzer import analyze_volatility

def calculate_stoploss_levels(df: pd.DataFrame, entry_price: float, position_side: str = "LONG") -> dict:
    """
    The Safety Net: Calculates Initial, Dynamic, and Trailing SL levels.
    """
    # 1. Gather Volatility Data
    v_data = analyze_volatility(df)
    atr = v_data['metrics']['atr']
    
    # 2. Initial Stop Loss (Using 2x ATR for buffer)
    # মার্কেট নয়েজ থেকে বাঁচার জন্য ২x ATR স্ট্যান্ডার্ড
    initial_sl_offset = atr * 2
    
    if position_side == "LONG":
        initial_sl = entry_price - initial_sl_offset
    else:
        initial_sl = entry_price + initial_sl_offset
        
    # 3. Dynamic Stop Loss (Regime Adjusted)
    # হাই ভোল্যাটিলিটি মার্কেট হলে স্টপ লস সামান্য বাড়িয়ে দেওয়া (Safety Buffer)
    if v_data['regime'] == "HIGH_VOLATILITY":
        dynamic_sl_offset = atr * 2.5
    else:
        dynamic_sl_offset = atr * 1.5
        
    if position_side == "LONG":
        dynamic_sl = entry_price - dynamic_sl_offset
    else:
        dynamic_sl = entry_price + dynamic_sl_offset
        
    # 4. Trailing Stop Loss Logic
    # ট্রেলিং স্টপ লস কারেন্ট প্রাইস থেকে ১.৫ ATR দূরে থাকবে
    current_price = df['close'].iloc[-1]
    trailing_sl_offset = atr * 1.5
    
    if position_side == "LONG":
        # ট্রেলিং স্টপ লস সবসময় আগের হাই এর দিকে মুভ করবে
        trailing_sl = max(initial_sl, current_price - trailing_sl_offset)
    else:
        trailing_sl = min(initial_sl, current_price + trailing_sl_offset)
        
    return {
        "initial_sl": round(initial_sl, 2),
        "dynamic_sl": round(dynamic_sl, 2),
        "trailing_sl": round(trailing_sl, 2),
        "atr_value": round(atr, 2),
        "meta": {
            "regime": v_data['regime'],
            "buffer_used": "2x_ATR"
        }
    }
