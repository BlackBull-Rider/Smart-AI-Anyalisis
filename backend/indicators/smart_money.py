import pandas as pd
import numpy as np

# ==========================================
# 1. MARKET STRUCTURE (BOS/CHOCH)
# ==========================================

def calculate_market_structure(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Detects Swing Highs/Lows for BOS/CHOCH"""
    highs = df['high'].rolling(window=2*window+1, center=True).max()
    lows = df['low'].rolling(window=2*window+1, center=True).min()
    
    swing_high = (df['high'] == highs)
    swing_low = (df['low'] == lows)
    
    return pd.DataFrame({'swing_high': swing_high.astype(int), 'swing_low': swing_low.astype(int)})

# ==========================================
# 2. FAIR VALUE GAPS (FVG)
# ==========================================

def calculate_fvg(df: pd.DataFrame) -> pd.DataFrame:
    """Fair Value Gap detection (Bullish & Bearish)"""
    # Bullish FVG: High[i-2] < Low[i]
    bullish_fvg = (df['high'].shift(2) < df['low'])
    # Bearish FVG: Low[i-2] > High[i]
    bearish_fvg = (df['low'].shift(2) > df['high'])
    
    return pd.DataFrame({
        'bullish_fvg': bullish_fvg.astype(int),
        'bearish_fvg': bearish_fvg.astype(int)
    })

# ==========================================
# 3. LIQUIDITY ANALYSIS (Equal Highs/Lows)
# ==========================================

def calculate_liquidity_zones(df: pd.DataFrame, tolerance: float = 0.001) -> pd.DataFrame:
    """Equal Highs and Equal Lows Detection"""
    equal_high = (df['high'] - df['high'].shift(1)).abs() < (df['high'] * tolerance)
    equal_low = (df['low'] - df['low'].shift(1)).abs() < (df['low'] * tolerance)
    
    return pd.DataFrame({
        'equal_high': equal_high.astype(int),
        'equal_low': equal_low.astype(int)
    })

# ==========================================
# 4. PREMIUM / DISCOUNT ZONES
# ==========================================

def calculate_premium_discount(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Premium/Discount Zones based on range"""
    highest = df['high'].rolling(window=period).max()
    lowest = df['low'].rolling(window=period).min()
    
    mid_range = (highest + lowest) / 2
    
    premium = (df['close'] > mid_range).astype(int)
    discount = (df['close'] < mid_range).astype(int)
    
    return pd.DataFrame({'premium_zone': premium, 'discount_zone': discount})

# ==========================================
# 5. ORDER & MITIGATION BLOCKS (Logic)
# ==========================================

def calculate_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """Identify Order Blocks based on impulsive momentum"""
    # Simplified Logic: Huge body candle with low wick relative to range
    candle_range = df['high'] - df['low']
    body = (df['open'] - df['close']).abs()
    
    is_impulsive = (body > (candle_range * 0.7))
    order_block = is_impulsive.astype(int)
    
    return pd.DataFrame({'order_block': order_block})
