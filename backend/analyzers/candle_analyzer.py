import pandas as pd
import numpy as np
from backend.indicators.candle import calculate_candle_metrics

def analyze_candle(df: pd.DataFrame) -> dict:
    """
    Candle Psychology Intelligence: Bull/Bear power, Reversals, and Gap analysis.
    """
    metrics = calculate_candle_metrics(df)
    last = metrics.iloc[-1]
    prev = metrics.iloc[-2]
    
    # 1. Candle Psychology (Body vs Wick Ratio)
    # যদি বডি খুব ছোট আর উইক বড় হয় -> Indecision
    if last['body_size'] < (last['candle_range'] * 0.2):
        sentiment = "INDECISION"
    elif last['body_size'] > (last['candle_range'] * 0.7):
        sentiment = "DECISIVE_MOMENTUM"
    else:
        sentiment = "BALANCED"
        
    # 2. Bull/Bear Power Analysis
    power_balance = "BULLISH_DOMINANCE" if last['bull_power'] > last['bear_power'] else "BEARISH_DOMINANCE"
    
    # 3. Reversal Detection (Pinbar & Engulfing)
    # Pinbar: যদি একটা উইক বডির অন্তত ২ গুণ বড় হয়
    is_pinbar = (last['upper_wick'] > (last['body_size'] * 2)) or (last['lower_wick'] > (last['body_size'] * 2))
    
    # Engulfing: কারেন্ট ক্যান্ডেল প্রিভিয়াস ক্যান্ডেলকে গিলে ফেলেছে (বডি সাইজ অনুযায়ী)
    is_engulfing = (last['body_size'] > prev['body_size']) and \
                   ((df['close'].iloc[-1] > df['open'].iloc[-2]) if last['bull_power'] > last['bear_power'] else \
                    (df['close'].iloc[-1] < df['open'].iloc[-2]))

    # 4. Continuation Detection
    # যদি বডি স্ট্রং হয় এবং গ্যাপের দিকে যায়
    is_continuation = (sentiment == "DECISIVE_MOMENTUM") and (last['gap_up'] > 0 or last['gap_down'] > 0)
    
    # 5. Gap Analysis
    gap_status = "GAP_UP" if last['gap_up'] > 0 else "GAP_DOWN" if last['gap_down'] > 0 else "NO_GAP"
    
    return {
        "market_sentiment": sentiment,
        "power_balance": power_balance,
        "reversal_signal": "PINBAR_OR_ENGULFING" if (is_pinbar or is_engulfing) else "NONE",
        "continuation_signal": "STRONG_SETUP" if is_continuation else "WAIT",
        "gap_analysis": gap_status,
        "metrics": {
            "bull_power_ratio": last['bull_power'] / (last['bull_power'] + last['bear_power'] + 0.001),
            "volatility_context": last['candle_range']
        }
    }
