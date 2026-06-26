import pandas as pd
import numpy as np

# ==========================================
# 1. RSI & MOMENTUM OSCILLATORS
# ==========================================

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """1. Relative Strength Index (RSI)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_rsi_slope(rsi_series: pd.Series, period: int = 5) -> pd.Series:
    """2. RSI Slope (For Divergence & Momentum Shift)"""
    # Slope calculated as simple difference over the period
    return rsi_series.diff(period) / period

def calculate_roc(series: pd.Series, period: int = 9) -> pd.Series:
    """9. Rate of Change (ROC)"""
    return ((series - series.shift(period)) / series.shift(period)) * 100

def calculate_momentum(series: pd.Series, period: int = 10) -> pd.Series:
    """11. Momentum"""
    return series - series.shift(period)

def calculate_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """17. Williams %R"""
    highest_high = df['high'].rolling(window=period).max()
    lowest_low = df['low'].rolling(window=period).min()
    return ((highest_high - df['close']) / (highest_high - lowest_low)) * -100

# ==========================================
# 2. MACD & PRICE OSCILLATORS
# ==========================================

def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """3, 4, 5. MACD, Signal, and Histogram"""
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'macd': macd_line,
        'macd_signal': signal_line,
        'macd_hist': histogram
    })

def calculate_ppo(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """13. Percentage Price Oscillator (PPO)"""
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    ppo_line = ((fast_ema - slow_ema) / slow_ema) * 100
    ppo_signal = ppo_line.ewm(span=signal, adjust=False).mean()
    
    return pd.DataFrame({'ppo': ppo_line, 'ppo_signal': ppo_signal})

def calculate_dpo(series: pd.Series, period: int = 20) -> pd.Series:
    """14. Detrended Price Oscillator (DPO)"""
    shifted_sma = series.rolling(window=period).mean().shift(int((period / 2) + 1))
    return series - shifted_sma

def calculate_trix(series: pd.Series, period: int = 15) -> pd.Series:
    """12. TRIX"""
    ema1 = series.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    return ema3.pct_change() * 100

# ==========================================
# 3. TREND STRENGTH (ADX, DMI) & CCI
# ==========================================

def calculate_adx_dmi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """6, 7, 8. ADX, DI+, DI-"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.abs().ewm(alpha=1/period, adjust=False).mean() / atr)
    
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).abs() * 100
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return pd.DataFrame({'adx': adx, 'di_plus': plus_di, 'di_minus': minus_di})

def calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """10. Commodity Channel Index (CCI)"""
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: pd.Series(x).mad(), raw=True)
    return (tp - sma_tp) / (0.015 * mad)

# ==========================================
# 4. STOCHASTICS & ULTIMATE OSCILLATOR
# ==========================================

def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """15. Stochastic Oscillator"""
    highest_high = df['high'].rolling(window=k_period).max()
    lowest_low = df['low'].rolling(window=k_period).min()
    
    stoch_k = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
    stoch_d = stoch_k.rolling(window=d_period).mean()
    
    return pd.DataFrame({'stoch_k': stoch_k, 'stoch_d': stoch_d})

def calculate_stochastic_rsi(series: pd.Series, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
    """16. Stochastic RSI"""
    rsi = calculate_rsi(series, period)
    rsi_min = rsi.rolling(window=period).min()
    rsi_max = rsi.rolling(window=period).max()
    
    stoch_rsi_k = 100 * ((rsi - rsi_min) / (rsi_max - rsi_min))
    stoch_rsi_k = stoch_rsi_k.rolling(window=smooth_k).mean()
    stoch_rsi_

