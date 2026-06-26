import pandas as pd
import numpy as np

# ==========================================
# 1. CUMULATIVE & WEIGHTED VOLUME
# ==========================================

def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """1. On-Balance Volume (OBV)"""
    close = df['close']
    volume = df['volume']
    
    direction = np.where(close > close.shift(1), volume, np.where(close < close.shift(1), -volume, 0))
    return pd.Series(direction, index=df.index).cumsum()

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """2. Volume Weighted Average Price (VWAP)"""
    q = df['close'] * df['volume']
    return q.cumsum() / df['volume'].cumsum()

def calculate_vwma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """3. Volume Weighted Moving Average (VWMA)"""
    price_volume = df['close'] * df['volume']
    return price_volume.rolling(window=period, min_periods=1).sum() / df['volume'].rolling(window=period, min_periods=1).sum()

# ==========================================
# 2. MONEY FLOW & BREADTH
# ==========================================

def calculate_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """4. Money Flow Index (MFI)"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    
    # Positive / Negative Money Flow
    diff = typical_price.diff()
    pos_mf = np.where(diff > 0, money_flow, 0)
    neg_mf = np.where(diff < 0, money_flow, 0)
    
    pos_sum = pd.Series(pos_mf).rolling(window=period).sum()
    neg_sum = pd.Series(neg_mf).rolling(window=period).sum()
    
    mfi_val = 100 - (100 / (1 + (pos_sum / neg_sum)))
    return pd.Series(mfi_val, index=df.index)

def calculate_cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """5. Chaikin Money Flow (CMF)"""
    mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mf_volume = mf_multiplier * df['volume']
    return mf_volume.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()

def calculate_ad_line(df: pd.DataFrame) -> pd.Series:
    """6. Accumulation/Distribution Line (A/D Line)"""
    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    clv = clv.fillna(0)
    return (clv * df['volume']).cumsum()

# ==========================================
# 3. PRESSURE & VOLUME OSCILLATORS
# ==========================================

def calculate_force_index(df: pd.DataFrame, period: int = 13) -> pd.Series:
    """7. Elder-Ray Force Index"""
    fi = df['close'].diff() * df['volume']
    return fi.ewm(span=period, adjust=False).mean()

def calculate_ease_of_movement(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """8. Ease of Movement (EoM)"""
    distance = ((df['high'] + df['low']) / 2) - ((df['high'].shift(1) + df['low'].shift(1)) / 2)
    box_ratio = (df['volume'] / 100000000) / (df['high'] - df['low'])
    eom = distance / box_ratio
    return eom.rolling(window=period).mean()

def calculate_volume_oscillator(series: pd.Series, short: int = 5, long: int = 10) -> pd.Series:
    """9. Volume Oscillator"""
    short_ema = series.ewm(span=short, adjust=False).mean()
    long_ema = series.ewm(span=long, adjust=False).mean()
    return ((short_ema - long_ema) / long_ema) * 100

def calculate_klinger_oscillator(df: pd.DataFrame, fast: int = 34, slow: int = 55, signal: int = 13) -> pd.DataFrame:
    """10. Klinger Oscillator"""
    hlc = (df['high'] + df['low'] + df['close']) / 3
    trend = np.where(hlc > hlc.shift(1), 1, -1)
    dm = df['high'] - df['low']
    
    cm = dm.cumsum()
    vf = df['volume'] * np.abs(2 * (dm / dm.ewm(span=2).mean()) - 1) * trend * 100
    
    ko_line = vf.ewm(span=fast, adjust=False).mean() - vf.ewm(span=slow, adjust=False).mean()
    signal_line = ko_line.ewm(span=signal, adjust=False).mean()
    
    return pd.DataFrame({'ko_line': ko_line, 'ko_signal': signal_line})

# ==========================================
# 4. NEGATIVE & POSITIVE VOLUME INDICES
# ==========================================

def calculate_nvi(df: pd.DataFrame) -> pd.Series:
    """11. Negative Volume Index (NVI)"""
    close = df['close']
    volume = df['volume']
    
    nvi = [1000.0] 
    for i in range(1, len(df)):
        prev_nvi = nvi[-1]
        if volume.iloc[i] < volume.iloc[i-1]:
            change = ((close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1]) * prev_nvi
            nvi.append(prev_nvi + change)
        else:
            nvi.append(prev_nvi)
            
    return pd.Series(nvi, index=df.index)

def calculate_pvi(df: pd.DataFrame) -> pd.Series:
    """12. Positive Volume Index (PVI)"""
    close = df['close']
    volume = df['volume']
    
    pvi = [1000.0]
    for i in range(1, len(df)):
        prev_pvi = pvi[-1]
        if volume.iloc[i] > volume.iloc[i-1]:
            change = ((close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1]) * prev_pvi
            pvi.append(prev_pvi + change)
        else:
            pvi.append(prev_pvi)
            
    return pd.Series(pvi, index=df.index)
