import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

def find_swings(data: pd.Series, window: int = 5) -> tuple:
    """স্বিং হাই এবং সুইং লো খোঁজার জন্য সুইং ডিটেক্টর"""
    ilocs_max = argrelextrema(data.values, np.greater_equal, order=window)[0]
    ilocs_min = argrelextrema(data.values, np.less_equal, order=window)[0]
    return ilocs_max, ilocs_min

def analyze_patterns(df: pd.DataFrame, window: int = 5) -> dict:
    """
    Pattern Intelligence: Reversal & Continuation.
    """
    highs, lows = find_swings(df['high'], window), find_swings(df['low'], window)
    
    # লাস্ট ৩টি সুইং হাই এবং লো (প্যাটার্ন ডিটেকশনের জন্য)
    last_highs = df['high'].iloc[highs[0][-3:]].values if len(highs[0]) >= 3 else []
    last_lows = df['low'].iloc[lows[0][-3:]].values if len(lows[0]) >= 3 else []
    
    # 1. Head & Shoulders Detection
    is_hns = False
    if len(last_highs) >= 3:
        # লেফট শোল্ডার < হেড > রাইট শোল্ডার
        if last_highs[1] > last_highs[0] and last_highs[1] > last_highs[2]:
            is_hns = True
            
    # 2. Triangle/Wedge Detection
    # সাপোর্ট এবং রেজিস্ট্যান্স এর ডিস্টেন্স কমছে কি না (Convergence)
    is_triangle = False
    if len(last_highs) >= 2 and len(last_lows) >= 2:
        high_slope = last_highs[1] - last_highs[0]
        low_slope = last_lows[1] - last_lows[0]
        # যদি রেজিস্ট্যান্স নিচের দিকে নামে আর সাপোর্ট ওপরের দিকে ওঠে
        if high_slope < 0 and low_slope > 0:
            is_triangle = True
            
    # 3. Cup & Handle (Simple Approximation)
    # প্রাইস এর variance কম এবং একটা dip এর পর ব্রেকআউট
    is_cup = False
    if df['close'].rolling(20).std().iloc[-1] < df['close'].mean() * 0.05:
        is_cup = True

    return {
        "reversal_patterns": {
            "head_and_shoulders": is_hns,
            "cup_and_handle": is_cup
        },
        "continuation_patterns": {
            "triangle_wedge": is_triangle,
            "channel": not is_triangle if len(last_highs) >= 2 else False
        },
        "metrics": {
            "swing_highs_count": len(last_highs),
            "swing_lows_count": len(last_lows)
        }
    }
