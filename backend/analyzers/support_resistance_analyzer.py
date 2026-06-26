import pandas as pd
import numpy as np
from backend.indicators.support_resistance import calculate_pivots, get_dynamic_levels

def analyze_support_resistance(df: pd.DataFrame) -> dict:
    """
    S/R Intelligence: Strength, Retests, and Breakout signals.
    """
    levels = calculate_pivots(df)
    dynamic = get_dynamic_levels(df)
    
    current_price = df['close'].iloc[-1]
    res_level = dynamic['resistance'].iloc[-1]
    supp_level = dynamic['support'].iloc[-1]
    
    # 1. Level Strength Analysis
    # যদি প্রাইস লেভেলের খুব কাছাকাছি থাকে তবে লেভেলটি টেস্ট হচ্ছে
    res_strength = "STRONG" if abs(current_price - res_level) < (res_level * 0.002) else "NEUTRAL"
    supp_strength = "STRONG" if abs(current_price - supp_level) < (supp_level * 0.002) else "NEUTRAL"
    
    # 2. Breakout/Breakdown Detection
    # প্রাইস যদি রেজিস্ট্যান্সের উপরে ক্লোজ দেয়
    is_breakout = current_price > res_level and df['close'].iloc[-2] <= res_level
    is_breakdown = current_price < supp_level and df['close'].iloc[-2] >= supp_level
    
    # 3. Retest Detection
    # যদি আগের ক্যান্ডেলে ব্রেকআউট হয় আর বর্তমান ক্যান্ডেলে প্রাইস আবার লেভেলে ফিরে আসে
    was_breakout = df['close'].iloc[-2] > res_level and df['close'].iloc[-3] <= res_level
    is_retest = was_breakout and (current_price <= res_level * 1.001) and (current_price >= res_level * 0.999)
    
    return {
        "resistance": {
            "level": round(res_level, 2),
            "strength": res_strength,
            "is_breakout": is_breakout
        },
        "support": {
            "level": round(supp_level, 2),
            "strength": supp_strength,
            "is_breakdown": is_breakdown
        },
        "retest_status": "VALID_RETEST" if is_retest else "NONE",
        "metrics": {
            "distance_to_res": res_level - current_price,
            "distance_to_supp": current_price - supp_level
        }
    }
