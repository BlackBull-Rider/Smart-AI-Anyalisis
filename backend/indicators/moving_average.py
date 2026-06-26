import pandas as pd
import numpy as np

# ==========================================
# 1. MOVING AVERAGE LIBRARY (As per Blueprint)
# ==========================================

def calculate_sma(series: pd.Series, period: int = 20) -> pd.Series:
    """1. Simple Moving Average (SMA)"""
    return series.rolling(window=period, min_periods=1).mean()

def calculate_ema(series: pd.Series, period: int = 20) -> pd.Series:
    """2. Exponential Moving Average (EMA)"""
    return series.ewm(span=period, adjust=False).mean()

def calculate_wma(series: pd.Series, period: int = 20) -> pd.Series:
    """3. Weighted Moving Average (WMA)"""
    weights = np.arange(1, period + 1)
    return series.rolling(window=period).apply(
        lambda prices: np.dot(prices, weights) / weights.sum(), raw=True
    )

def calculate_vwma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """4. Volume Weighted Moving Average (VWMA)"""
    price_volume = df['close'] * df['volume']
    return price_volume.rolling(window=period, min_periods=1).sum() / df['volume'].rolling(window=period, min_periods=1).sum()

def calculate_dema(series: pd.Series, period: int = 20) -> pd.Series:
    """5. Double Exponential Moving Average (DEMA)"""
    ema1 = calculate_ema(series, period)
    ema2 = calculate_ema(ema1, period)
    return 2 * ema1 - ema2

def calculate_tema(series: pd.Series, period: int = 20) -> pd.Series:
    """6. Triple Exponential Moving Average (TEMA)"""
    ema1 = calculate_ema(series, period)
    ema2 = calculate_ema(ema1, period)
    ema3 = calculate_ema(ema2, period)
    return (3 * ema1) - (3 * ema2) + ema3

def calculate_trima(series: pd.Series, period: int = 20) -> pd.Series:
    """7. Triangular Moving Average (TRIMA)"""
    window1 = int(np.ceil((period + 1) / 2))
    window2 = int(np.floor((period + 1) / 2))
    return series.rolling(window1).mean().rolling(window2).mean()

def calculate_hma(series: pd.Series, period: int = 20) -> pd.Series:
    """8. Hull Moving Average (HMA)"""
    half_length = int(period / 2)
    sqrt_length = int(np.sqrt(period))
    
    wma_half = calculate_wma(series, half_length)
    wma_full = calculate_wma(series, period)
    
    raw_hma = (2 * wma_half) - wma_full
    return calculate_wma(raw_hma, sqrt_length)

def calculate_zlema(series: pd.Series, period: int = 20) -> pd.Series:
    """9. Zero Lag Exponential Moving Average (ZLEMA)"""
    lag = int((period - 1) / 2)
    ema_data = series + (series - series.shift(lag))
    return calculate_ema(ema_data, period)

def calculate_kama(series: pd.Series, period: int = 10, fast_ema: int = 2, slow_ema: int = 30) -> pd.Series:
    """10. Kaufman's Adaptive Moving Average (KAMA)"""
    change = series.diff(period).abs()
    volatility = series.diff().abs().rolling(window=period).sum()
    
    # Handle division by zero
    er = np.where(volatility == 0, 0, change / volatility)
    
    fast_sc = 2 / (fast_ema + 1)
    slow_sc = 2 / (slow_ema + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama = np.zeros(len(series))
    kama[:] = np.nan
    
    series_vals = series.values
    
    # KAMA needs a loop for recursive calculation, optimized with numpy arrays
    start_idx = period
    if start_idx < len(series):
        kama[start_idx-1] = series_vals[start_idx-1]
        for i in range(start_idx, len(series)):
            if np.isnan(kama[i-1]):
                kama[i] = series_vals[i]
            else:
                kama[i] = kama[i-1] + sc[i] * (series_vals[i] - kama[i-1])
                
    return pd.Series(kama, index=series.index)

def calculate_alma(series: pd.Series, period: int = 9, offset: float = 0.85, sigma: float = 6.0) -> pd.Series:
    """11. Arnaud Legoux Moving Average (ALMA)"""
    m = int(offset * (period - 1))
    s = period / sigma
    k = np.arange(period)
    weights = np.exp(-((k - m) ** 2) / (2 * s ** 2))
    weights /= weights.sum()
    
    return series.rolling(period).apply(lambda x: np.dot(x, weights), raw=True)

def calculate_t3(series: pd.Series, period: int = 5, v_factor: float = 0.7) -> pd.Series:
    """12. Tillson T3 Moving Average"""
    e1 = calculate_ema(series, period)
    e2 = calculate_ema(e1, period)
    e3 = calculate_ema(e2, period)
    e4 = calculate_ema(e3, period)
    e5 = calculate_ema(e4, period)
    e6 = calculate_ema(e5, period)
    
    c1 = - (v_factor ** 3)
    c2 = 3 * (v_factor ** 2) + 3 * (v_factor ** 3)
    c3 = - 6 * (v_factor ** 2) - 3 * v_factor - 3 * (v_factor ** 3)
    c4 = 1 + 3 * v_factor + (v_factor ** 3) + 3 * (v_factor ** 2)
    
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


# ==========================================
# 2. TREND FOLLOWING & TRAILING STOPLOSS
# ==========================================

def calculate_atr(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """Average True Range for Supertrend"""
    high = df['high'].values
    low = df['low'].values
    prev_close = df['close'].shift(1).values
    
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    return pd.Series(tr, index=df.index).ewm(alpha=1/period, adjust=False).mean()

def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Supertrend Implementation for Simultaneous Trend Following and Exit Trailing"""
    atr = calculate_atr(df, period)
    hl2 = (df['high'] + df['low']) / 2
    
    basic_ub = hl2 + (multiplier * atr)
    basic_lb = hl2 - (multiplier * atr)
    close = df['close'].values
    
    arr_ub = basic_ub.values
    arr_lb = basic_lb.values
    
    length = len(df)
    final_ub = np.zeros(length)
    final_lb = np.zeros(length)
    supertrend = np.zeros(length)
    direction = np.zeros(length)
    
    if length > 0:
        final_ub[0] = arr_ub[0]
        final_lb[0] = arr_lb[0]
        direction[0] = 1
        
        for i in range(1, length):
            if arr_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]:
                final_ub[i] = arr_ub[i]
            else:
                final_ub[i] = final_ub[i-1]
                
            if arr_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]:
                final_


