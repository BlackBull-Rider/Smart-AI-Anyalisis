import pandas as pd
import numpy as np

# ==========================================
# 1. PIVOT POINTS
# ==========================================

def calculate_pivots(df: pd.DataFrame) -> pd.DataFrame:
    """Classic Pivot Points calculation"""
    pp = (df['high'] + df['low'] + df['close']) / 3
    r1 = 2 * pp - df['low']
    s1 = 2 * pp - df['high']
    r2 = pp + (df['high'] - df['low'])
    s2 = pp - (df['high'] - df['low'])
    
    return pd.DataFrame({
        'pivot': pp, 'r1': r1, 's1': s1, 'r2': r2, 's2': s2
    })

# ==========================================
# 2. SWING & FRACTAL LEVELS
# ==========================================

def calculate_fractals(df: pd.DataFrame, window: int = 2) -> pd.DataFrame:
    """Williams Fractals (High > 2 bars left/right, Low < 2 bars left/right)"""
    high = df['high']
    low = df['low']
    
    fractal_high = high[(high > high.shift(1)) & (high > high.shift(2)) & 
                        (high > high.shift(-1)) & (high > high.shift(-2))]
    fractal_low = low[(low < low.shift(1)) & (low < low.shift(2)) & 
                      (low < low.shift(-1)) & (low < low.shift(-2))]
                      
    return pd.DataFrame({'fractal_high': fractal_high, 'fractal_low': fractal_low})

def calculate_swings(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Identify Swing Highs and Swing Lows"""
    swing_high = df['high'][(df['high'] == df['high'].rolling(window=2*window+1, center=True).max())]
    swing_low = df['low'][(df['low'] == df['low'].rolling(window=2*window+1, center=True).min())]
    
    return pd.DataFrame({'swing_high': swing_high, 'swing_low': swing_low})

# ==========================================
# 3. DYNAMIC SUPPORT/RESISTANCE & BREAKOUTS
# ==========================================

def get_dynamic_levels(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Simple Support/Resistance based on recent highs/lows"""
    support = df['low'].rolling(window=window).min()
    resistance = df['high'].rolling(window=window).max()
    
    return pd.DataFrame({'support': support, 'resistance': resistance})

def calculate_breakouts(df: pd.DataFrame, resistance: pd.Series, support: pd.Series) -> pd.DataFrame:
    """Identify Breakout and Breakdown levels"""
    breakout = (df['close'] > resistance) & (df['close'].shift(1) <= resistance)
    breakdown = (df['close'] < support) & (df['close'].shift(1) >= support)
    
    return pd.DataFrame({
        'breakout': breakout.astype(int),
        'breakdown': breakdown.astype(int)
    })
