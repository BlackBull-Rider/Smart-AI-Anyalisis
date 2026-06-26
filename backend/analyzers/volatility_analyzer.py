import pandas as pd
import numpy as np
from backend.indicators.volatility import calculate_atr, calculate_bollinger_bands, calculate_keltner_channel

def analyze_volatility(df: pd.DataFrame) -> dict:
    """
    Volatility Intelligence: Squeeze, Expansion, and Breakout signals.
    """
    # ক্যালকুলেশন
    atr = calculate_atr(df, period=14).iloc[-1]
    bb = calculate_bollinger_bands(df['close'], period=20)
    kc = calculate_keltner_channel(df, period=20, atr_period=14)
    
    # 1. Volatility Expansion/Contraction
    # Bollinger Width এর পরিবর্তন দেখে
    bb_width = (bb['bb_upper'] - bb['bb_lower']) / bb['bb_middle']
    is_expanding = bb_width.iloc[-1] > bb_width.rolling(20).mean().iloc[-1]
    volatility_state = "EXPANDING" if is_expanding else "CONTRACTING"
    
    # 2. Squeeze Detection (The Holy Grail)
    # যখন Bollinger Bands Keltner Channel এর ভেতরে ঢুকে যায় (Compression)
    squeeze_trigger = (bb['bb_upper'].iloc[-1] < kc['kc_upper'].iloc[-1]) and \
                      (bb['bb_lower'].iloc[-1] > kc['kc_lower'].iloc[-1])
    
    # 3. Breakout Detection
    price = df['close'].iloc[-1]
    breakout_signal = "UPPER_BREAKOUT" if price > bb['bb_upper'].iloc[-1] else \
                      "LOWER_BREAKOUT" if price < bb['bb_lower'].iloc[-1] else "NONE"
    
    # 4. Volatility Regime (Relative Volatility)
    # বর্তমান ATR যদি ২০ পিরিয়ডের মুভিং এভারেজের চেয়ে বেশি হয়
    atr_ma = calculate_atr(df, 14).rolling(20).mean().iloc[-1]
    regime = "HIGH_VOLATILITY" if atr > atr_ma else "LOW_VOLATILITY"
    
    return {
        "state": volatility_state,
        "squeeze_detected": squeeze_trigger,
        "breakout_status": breakout_signal,
        "regime": regime,
        "metrics": {
            "atr": atr,
            "bb_width_percent": bb_width.iloc[-1] * 100
        }
    }
