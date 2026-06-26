import pandas as pd
import numpy as np

# ==========================================
# 1. TRUE RANGE & ATR FAMILY
# ==========================================

def calculate_true_range(df: pd.DataFrame) -> pd.Series:
    """1. True Range (TR)"""
    high = df['high']
    low = df['low']
    prev_close = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """2. Average True Range (ATR) - Wilder's Smoothing method"""
    tr = calculate_true_range(df)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def calculate_natr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """3. Normalized Average True Range (NATR)"""
    atr = calculate_atr(df, period)
    return (atr / df['close']) * 100

# ==========================================
# 2. BOLLINGER BANDS FAMILY
# ==========================================

def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """4. Bollinger Bands (Upper, Middle, Lower)"""
    middle_band = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return pd.DataFrame({
        'bb_upper': upper_band,
        'bb_middle': middle_band,
        'bb_lower': lower_band
    })

def calculate_bollinger_width(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """5. Bollinger Band Width"""
    bb = calculate_bollinger_bands(series, period, std_dev)
    return (bb['bb_upper'] - bb['bb_lower']) / bb['bb_middle']

def calculate_bollinger_pb(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """6. Bollinger %B"""
    bb = calculate_bollinger_bands(series, period, std_dev)
    return (series - bb['bb_lower']) / (bb['bb_upper'] - bb['bb_lower'])

# ==========================================
# 3. PRICE CHANNELS
# ==========================================

def calculate_donchian_channel(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """7. Donchian Channel"""
    upper_band = df['high'].rolling(window=period).max()
    lower_band = df['low'].rolling(window=period).min()
    middle_band = (upper_band + lower_band) / 2
    
    return pd.DataFrame({
        'dc_upper': upper_band,
        'dc_middle': middle_band,
        'dc_lower': lower_band
    })

def calculate_keltner_channel(df: pd.DataFrame, period: int = 20, atr_period: int = 10, multiplier: float = 2.0) -> pd.DataFrame:
    """8. Keltner Channel"""
    middle_band = df['close'].ewm(span=period, adjust=False).mean()
    atr = calculate_atr(df, atr_period)
    
    upper_band = middle_band + (multiplier * atr)
    lower_band = middle_band - (multiplier * atr)
    
    return pd.DataFrame({
        'kc_upper': upper_band,
        'kc_middle': middle_band,
        'kc_lower': lower_band
    })

# ==========================================
# 4. VOLATILITY INDICES & EXPANSION
# ==========================================

def calculate_historical_volatility(series: pd.Series, period: int = 20, trading_days: int = 252) -> pd.Series:
    """9. Historical Volatility (Annualized)"""


