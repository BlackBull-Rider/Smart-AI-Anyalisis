import numpy as np
from Indicators.math_utils import enforce_writeable_float_array_fast, replace_nan_with_zero

# ==========================================
# GRACEFUL DEGRADATION FOR NUMBA (Termux Fix)
# ==========================================
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# ==========================================
# VOLATILITY ENGINE (ATR & BOLLINGER BANDS)
# ==========================================

def calculate_true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    True Range (TR) ক্যালকুলেশন - ভেক্টরাইজড।
    """
    h = enforce_writeable_float_array_fast(high)
    l = enforce_writeable_float_array_fast(low)
    c = enforce_writeable_float_array_fast(close)
    
    # আগের দিনের ক্লোজ প্রাইস শিফট করা
    prev_close = np.roll(c, 1)
    prev_close[0] = l[0]  # প্রথম দিনের জন্য প্রিভিয়াস ক্লোজ হিসেবে লো ধরা
    
    tr1 = h - l
    tr2 = np.abs(h - prev_close)
    tr3 = np.abs(l - prev_close)
    
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    return replace_nan_with_zero(tr)

@njit(cache=True, fastmath=True)
def _wilder_ma_core(tr: np.ndarray, period: int, alpha: float) -> np.ndarray:
    """
    Wilder's Smoothing (EMA-style) Numba JIT ইঞ্জিন।
    """
    atr = np.zeros_like(tr)
    # প্রথম ATR ভ্যালু হবে সাধারণ SMA
    sum_tr = 0.0
    for i in range(period):
        sum_tr += tr[i]
    atr[period - 1] = sum_tr / period
    
    # ওয়াইল্ডার স্মুথিং লজিক
    for i in range(period, len(tr)):
        atr[i] = atr[i - 1] + alpha * (tr[i] - atr[i - 1])
        
    return atr

def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Average True Range (ATR) - ওয়াইল্ডার মুভিং এভারেজ বেসড।
    """
    tr = calculate_true_range(high, low, close)
    if len(tr) < period:
        return np.zeros_like(tr)
        
    alpha = 1.0 / period
    atr = _wilder_ma_core(tr, period, alpha)
    return replace_nan_with_zero(atr)

def calculate_bollinger_bands(close: np.ndarray, period: int = 20, num_std: float = 2.0) -> tuple:
    """
    Bollinger Bands (Middle, Upper, Lower) - মেমোরি অপ্টিমাইজড।
    """
    c = enforce_writeable_float_array_fast(close)
    if len(c) < period:
        return np.zeros_like(c), np.zeros_like(c), np.zeros_like(c)
        
    # সিম্পল মুভিং এভারেজ (মিডল ব্যান্ড)
    weights = np.ones(period, dtype=np.float64) / period
    sma_valid = np.convolve(c, weights, mode='valid')
    pad = np.zeros(period - 1, dtype=np.float64)
    middle_band = np.concatenate((pad, sma_valid))
    
    # রোলিং স্ট্যান্ডার্ড ডেভিয়েশন
    # (স্ট্রাইড ট্রিকস ব্যবহার করে ফাস্ট উইন্ডো ক্যালকুলেশন)
    shape = (c.size - period + 1, period)
    strides = (c.strides[0], c.strides[0])
    rolling_window = np.lib.stride_tricks.as_strided(c, shape=shape, strides=strides)
    
    std_valid = np.std(rolling_window, axis=1)
    std_dev = np.concatenate((pad, std_valid))
    
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    
    return replace_nan_with_zero(middle_band), replace_nan_with_zero(upper_band), replace_nan_with_zero(lower_band)
