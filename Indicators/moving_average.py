import numpy as np
from Indicators.math_utils import enforce_writeable_float_array_fast, replace_nan_with_zero

# ==========================================
# GRACEFUL DEGRADATION FOR NUMBA (Termux Fix)
# ==========================================
try:
    from numba import njit
except ImportError:
    print("[WARNING] Numba is not installed. Running in CPU-Fallback mode for testing.")
    # Numba না পেলে একটা ডামি ডেকোরেটর বানিয়ে নেওয়া, যাতে কোড ক্র্যাশ না করে
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# ==========================================
# 1. THE CLASSICS & VOLUME WEIGHTED
# ==========================================

def calculate_sma(data: np.ndarray, period: int) -> np.ndarray:
    arr = enforce_writeable_float_array_fast(data)
    if len(arr) < period:
        return np.zeros_like(arr)
    weights = np.ones(period, dtype=np.float64) / period
    sma_valid = np.convolve(arr, weights, mode='valid')
    pad = np.zeros(period - 1, dtype=np.float64)
    return replace_nan_with_zero(np.concatenate((pad, sma_valid)))

@njit(cache=True, fastmath=True)
def _ema_numba_core(arr: np.ndarray, period: int, alpha: float) -> np.ndarray:
    ema = np.zeros_like(arr)
    initial_sma = 0.0
    for i in range(period):
        initial_sma += arr[i]
    ema[period - 1] = initial_sma / period
    for i in range(period, len(arr)):
        ema[i] = (arr[i] - ema[i - 1]) * alpha + ema[i - 1]
    return ema

def calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
    arr = enforce_writeable_float_array_fast(data)
    if len(arr) < period:
        return np.zeros_like(arr)
    alpha = 2.0 / (period + 1.0)
    return replace_nan_with_zero(_ema_numba_core(arr, period, alpha))

def calculate_wma(data: np.ndarray, period: int) -> np.ndarray:
    arr = enforce_writeable_float_array_fast(data)
    if len(arr) < period:
        return np.zeros_like(arr)
    weights = np.arange(1, period + 1, dtype=np.float64)
    weights /= weights.sum()
    wma_valid = np.convolve(arr, weights[::-1], mode='valid')
    pad = np.zeros(period - 1, dtype=np.float64)
    return replace_nan_with_zero(np.concatenate((pad, wma_valid)))

def calculate_vwma(close: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
    c = enforce_writeable_float_array_fast(close)
    v = enforce_writeable_float_array_fast(volume)
    cv = c * v
    sma_cv = calculate_sma(cv, period)
    sma_v = calculate_sma(v, period)
    vwma = np.divide(sma_cv, sma_v, out=np.zeros_like(sma_cv), where=sma_v != 0)
    return replace_nan_with_zero(vwma)

# ==========================================
# 2. ZERO-LAG & SMOOTHERS
# ==========================================

def calculate_dema(data: np.ndarray, period: int) -> np.ndarray:
    ema1 = calculate_ema(data, period)
    ema2 = calculate_ema(ema1, period)
    return replace_nan_with_zero(2 * ema1 - ema2)

def calculate_tema(data: np.ndarray, period: int) -> np.ndarray:
    ema1 = calculate_ema(data, period)
    ema2 = calculate_ema(ema1, period)
    ema3 = calculate_ema(ema2, period)
    return replace_nan_with_zero(3 * ema1 - 3 * ema2 + ema3)

def calculate_hma(data: np.ndarray, period: int) -> np.ndarray:
    half_length = int(period / 2)
    sqrt_length = int(np.sqrt(period))
    wma1 = calculate_wma(data, half_length)
    wma2 = calculate_wma(data, period)
    raw_hma = 2 * wma1 - wma2
    return calculate_wma(raw_hma, sqrt_length)

def calculate_zlema(data: np.ndarray, period: int) -> np.ndarray:
    arr = enforce_writeable_float_array_fast(data)
    lag = int((period - 1) / 2)
    if len(arr) <= lag:
        return np.zeros_like(arr)
    lagged_data = np.zeros_like(arr)
    lagged_data[lag:] = arr[:-lag]
    adjusted_data = arr + (arr - lagged_data)
    return calculate_ema(adjusted_data, period)

# ==========================================
# 3. ADAPTIVE & INSTITUTIONAL QUANT
# ==========================================

@njit(cache=True, fastmath=True)
def _kama_numba_core(arr: np.ndarray, period: int, fast_ema: float, slow_ema: float) -> np.ndarray:
    kama = np.zeros_like(arr)
    # KAMA ইনিশিয়ালাইজেশন
    kama[period-1] = arr[period-1] 
    for i in range(period, len(arr)):
        change = abs(arr[i] - arr[i - period])
        volatility = 0.0
        for j in range(period):
            volatility += abs(arr[i-j] - arr[i-j-1])
            
        er = change / volatility if volatility != 0 else 0.0
        sc = (er * (fast_ema - slow_ema) + slow_ema) ** 2
        kama[i] = kama[i - 1] + sc * (arr[i] - kama[i - 1])
    return kama

def calculate_kama(data: np.ndarray, period: int = 10, fast_len: int = 2, slow_len: int = 30) -> np.ndarray:
    arr = enforce_writeable_float_array_fast(data)
    if len(arr) < period:
        return np.zeros_like(arr)
    fast_ema = 2.0 / (fast_len + 1.0)
    slow_ema = 2.0 / (slow_len + 1.0)
    kama = _kama_numba_core(arr, period, fast_ema, slow_ema)
    return replace_nan_with_zero(kama)

def calculate_alma(data: np.ndarray, period: int, offset: float = 0.85, sigma: float = 6.0) -> np.ndarray:
    arr = enforce_writeable_float_array_fast(data)
    if len(arr) < period:
        return np.zeros_like(arr)
    m = offset * (period - 1)
    s = period / sigma
    weights = np.exp(-((np.arange(period) - m) ** 2) / (2 * s * s))
    weights /= weights.sum()
    alma_valid = np.convolve(arr, weights[::-1], mode='valid')
    pad = np.zeros(period - 1, dtype=np.float64)
    return replace_nan_with_zero(np.concatenate((pad, alma_valid)))

def calculate_t3(data: np.ndarray, period: int, v_factor: float = 0.7) -> np.ndarray:
    a = v_factor
    c1 = -a**3
    c2 = 3*a**2 + 3*a**3
    c3 = -6*a**2 - 3*a - 3*a**3
    c4 = 1 + 3*a + a**3 + 3*a**2
    
    e1 = calculate_ema(data, period)
    e2 = calculate_ema(e1, period)
    e3 = calculate_ema(e2, period)
    e4 = calculate_ema(e3, period)
    e5 = calculate_ema(e4, period)
    e6 = calculate_ema(e5, period)
    
    t3 = c1*e6 + c2*e5 + c3*e4 + c4*e3
    return replace_nan_with_zero(t3)

# ==========================================
# 4. THE MASTER ENGINE MATRIX
# ==========================================

def moving_average_matrix(close: np.ndarray, volume: np.ndarray, periods: list) -> dict:
    """
    সবগুলো MA-এর ক্রসওভার এবং ট্রেন্ড ডিরেকশন একবারে ক্যালকুলেট করার ফাস্ট ব্লক।
    এটি সরাসরি Layer 2 (Analyzers) এবং সুপারট্রেন্ডের ট্রেইলিং লজিকের সাথে ফিড হবে।
    """
    matrix = {}
    for p in periods:
        matrix[f'SMA_{p}'] = calculate_sma(close, p)
        matrix[f'EMA_{p}'] = calculate_ema(close, p)
        matrix[f'WMA_{p}'] = calculate_wma(close, p)
        matrix[f'VWMA_{p}'] = calculate_vwma(close, volume, p)
        matrix[f'HMA_{p}'] = calculate_hma(close, p)
        matrix[f'ALMA_{p}'] = calculate_alma(close, p)
    
    # অ্যাডভান্সড ইন্ডিকেটরগুলোর জন্য ডিফল্ট পিরিয়ড
    matrix['DEMA_20'] = calculate_dema(close, 20)
    matrix['TEMA_20'] = calculate_tema(close, 20)
    matrix['ZLEMA_20'] = calculate_zlema(close, 20)
    matrix['KAMA_10'] = calculate_kama(close, 10)
    matrix['T3_10'] = calculate_t3(close, 10)
    
    return matrix
