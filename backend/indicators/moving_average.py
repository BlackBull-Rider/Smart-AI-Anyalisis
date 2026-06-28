import logging
import numpy as np
import pandas as pd
from typing import Union, Optional, Any

from backend.indicators.helper import (
    ensure_series,
    standardize_column_names,
    hl2,
    hlc3,
    ohlc4,
    weighted_price
)
from backend.indicators.math_utils import (
    safe_divide, 
    rolling_mean, 
    EPSILON
)

# ==============================================================================
# LOGGING & CONSTANTS
# ==============================================================================

logger = logging.getLogger(__name__)

DEFAULT_MA_LENGTH = 20
DEFAULT_FAST_LENGTH = 2
DEFAULT_SLOW_LENGTH = 30
DEFAULT_ALMA_OFFSET = 0.85
DEFAULT_ALMA_SIGMA = 6.0
DEFAULT_T3_VFACTOR = 0.7
DEFAULT_MCGINLEY_K = 0.6

# Computed source requirements mapping
COMPUTED_SOURCES = {
    'hl2': (hl2, ['high', 'low']),
    'median_price': (hl2, ['high', 'low']),
    'hlc3': (hlc3, ['high', 'low', 'close']),
    'typical_price': (hlc3, ['high', 'low', 'close']),
    'ohlc4': (ohlc4, ['open', 'high', 'low', 'close']),
    'weighted_price': (weighted_price, ['high', 'low', 'close'])
}

# ==============================================================================
# EXCEPTIONS
# ==============================================================================

class MovingAverageError(Exception):
    """Custom exception for errors in moving average calculations."""
    pass

# ==============================================================================
# COMMON UTILITIES
# ==============================================================================

def validate_length(length: int) -> None:
    if not isinstance(length, int) or length <= 0:
        logger.error(f"Invalid moving average length: {length}")
        raise MovingAverageError(f"Length must be a positive integer, got {length}")

def get_price_source(data: Union[pd.DataFrame, pd.Series], source: str = 'close') -> pd.Series:
    """
    Extracts the specified price source. Uses dictionary mapping for clean maintainability.
    Avoids strict DatetimeIndex validation to support basic RangeIndex DataFrames.
    """
    if isinstance(data, pd.Series):
        return data
        
    df = standardize_column_names(data)
    src = source.lower().strip()
    
    if src in df.columns:
        return df[src]
        
    if src in COMPUTED_SOURCES:
        func, req_cols = COMPUTED_SOURCES[src]
        
        missing = [c for c in req_cols if c not in df.columns]
        if missing:
            logger.error(f"Missing required columns for computed source '{src}': {missing}")
            raise MovingAverageError(f"DataFrame is missing columns needed for '{src}': {missing}")
            
        args = [df[c] for c in req_cols]
        return func(*args)
            
    logger.error(f"Invalid source '{source}'.")
    raise MovingAverageError(f"Invalid source '{source}' or column not found in DataFrame.")

def validate_ma_input(data: Union[pd.DataFrame, pd.Series], length: int) -> None:
    validate_length(length)
    if data is None or len(data) == 0:
        raise MovingAverageError("Input data cannot be empty.")

def prepare_series(data: Union[pd.DataFrame, pd.Series], source: str, length: int) -> pd.Series:
    validate_ma_input(data, length)
    series = get_price_source(data, source)
    return ensure_series(series)

def _finalize_series(series: pd.Series, offset: int, fillna: Any, name: str) -> pd.Series:
    res = series.shift(offset) if offset != 0 else series
    if fillna is not None:
        res = res.fillna(fillna)
    res.name = name
    return res

def _nan_safe_convolve(arr: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Performs 1D convolution safely ignoring NaNs by forward/backward filling temporarily.
    Restores the original NaN mask to the output array.
    """
    length = len(weights)
    mask = np.isnan(arr)
    
    s_filled = pd.Series(arr).ffill().bfill().to_numpy(dtype=np.float64)
    conv = np.convolve(s_filled, weights, mode='valid')
    
    out = np.full(len(arr), np.nan)
    out[length - 1:] = conv
    out[mask] = np.nan
    return out

# ==============================================================================
# STANDARD INDICATOR IMPLEMENTATIONS
# ==============================================================================

def sma(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_MA_LENGTH, source: str = 'close',
        offset: int = 0, fillna: Any = None, min_periods: Optional[int] = None) -> pd.Series:
    """Simple Moving Average (SMA)."""
    s = prepare_series(data, source, length)
    min_p = min_periods if min_periods is not None else length
    
    res = rolling_mean(s, window=length)
    if min_p != length:
        res = s.rolling(window=length, min_periods=min_p).mean()
        
    return _finalize_series(res, offset, fillna, f"SMA_{length}")

def ema(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_MA_LENGTH, source: str = 'close',
        offset: int = 0, fillna: Any = None, min_periods: Optional[int] = None) -> pd.Series:
    """Exponential Moving Average (EMA)."""
    s = prepare_series(data, source, length)
    min_p = min_periods if min_periods is not None else length
    res = s.ewm(span=length, min_periods=min_p, adjust=False).mean()
    return _finalize_series(res, offset, fillna, f"EMA_{length}")

def smma(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_MA_LENGTH, source: str = 'close',
         offset: int = 0, fillna: Any = None, min_periods: Optional[int] = None) -> pd.Series:
    """Smoothed Moving Average (RMA)."""
    s = prepare_series(data, source, length)
    min_p = min_periods if min_periods is not None else length
    res = s.ewm(alpha=1.0/length, min_periods=min_p, adjust=False).mean()
    return _finalize_series(res, offset, fillna, f"SMMA_{length}")

def wma(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_MA_LENGTH, source: str = 'close',
        offset: int = 0, fillna: Any = None, min_periods: Optional[int] = None) -> pd.Series:
    """Weighted Moving Average (WMA)."""
    s = prepare_series(data, source, length)
    arr = s.to_numpy(dtype=np.float64)
    
    weights = np.arange(1, length + 1, dtype=np.float64)
    weights /= weights.sum()
    
    out = _nan_safe_convolve(arr, weights[::-1])
    res = pd.Series(out, index=s.index)
    
    if min_periods is not None:
        valid_count = s.notna().rolling(window=length, min_periods=0).sum()
        res.loc[valid_count < min_periods] = np.nan
        
    return _finalize_series(res, offset, fillna, f"WMA_{length}")

def vwma(data: Union[pd.DataFrame, pd.Series], volume: Optional[pd.Series] = None,
         length: int = DEFAULT_MA_LENGTH, source: str = 'close', offset: int = 0, fillna: Any = None,
         min_periods: Optional[int] = None) -> pd.Series:
    """Volume Weighted Moving Average (VWMA)."""
    validate_ma_input(data, length)
    s = get_price_source(data, source)
    
    if volume is None:
        if isinstance(data, pd.DataFrame):
            df = standardize_column_names(data)
            if 'volume' not in df.columns:
                raise MovingAverageError("VWMA requires a 'volume' column.")
            vol = df['volume']
        else:
            raise MovingAverageError("VWMA requires volume data.")
    else:
        vol = ensure_series(volume)
        
    if not s.index.equals(vol.index):
        raise MovingAverageError("Price series and Volume series indices do not match.")
        
    min_p = min_periods if min_periods is not None else length
    
    price_vol = s * vol
    sma_price_vol = price_vol.rolling(window=length, min_periods=min_p).sum()
    sma_vol = vol.rolling(window=length, min_periods=min_p).sum()
    
    res = safe_divide(sma_price_vol.to_numpy(dtype=np.float64), sma_vol.to_numpy(dtype=np.float64), default=np.nan)
    return _finalize_series(pd.Series(res, index=s.index), offset, fillna, f"VWMA_{length}")

def lsma(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_MA_LENGTH, source: str = 'close',
         offset: int = 0, fillna: Any = None) -> pd.Series:
    """Least Squares Moving Average (Linear Regression)."""
    s = prepare_series(data, source, length)
    arr = s.to_numpy(dtype=np.float64)
    
    weights = np.arange(1, length + 1, dtype=np.float64)
    weights = (6 * weights - 2 * (length + 1)) / (length * (length + 1))
    
    out = _nan_safe_convolve(arr, weights[::-1])
    return _finalize_series(pd.Series(out, index=s.index), offset, fillna, f"LSMA_{length}")

# ==============================================================================
# ADVANCED INDICATOR IMPLEMENTATIONS
# ==============================================================================

def dema(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_MA_LENGTH, source: str = 'close',
         offset: int = 0, fillna: Any = None, min_periods: Optional[int] = None) -> pd.Series:
    s = prepare_series(data, source, length)
    e1 = ema(s, length=length, min_periods=min_periods)
    e2 = ema(e1, length=length, min_periods=min_periods)
    return _finalize_series((2 * e1) - e2, offset, fillna, f"DEMA_{length}")

def tema(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_MA_LENGTH, source: str = 'close',
         offset: int = 0, fillna: Any = None, min_periods: Optional[int] = None) -> pd.Series:
    s = prepare_series(data, source, length)
    e1 = ema(s, length=length, min_periods=min_periods)
    e2 = ema(e1, length=length, min_periods=min_periods)
    e3 = ema(e2, length=length, min_periods=min_periods)
    return _finalize_series((3 * e1) - (3 * e2) + e3, offset, fillna, f"TEMA_{length}")

def trima(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_MA_LENGTH, source: str = 'close',
          offset: int = 0, fillna: Any = None, min_periods: Optional[int] = None) -> pd.Series:
    s = prepare_series(data, source, length)
    half_length = (length + 1) // 2
    sma1 = sma(s, length=half_length, min_periods=min_periods)
    smooth_length = half_length if length % 2 != 0 else half_length + 1
    res = sma(sma1, length=smooth_length, min_periods=min_periods)
    return _finalize_series(res, offset, fillna, f"TRIMA_{length}")

def hma(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_MA_LENGTH, source: str = 'close',
        offset: int = 0, fillna: Any = None, min_periods: Optional[int] = None) -> pd.Series:
    s = prepare_series(data, source, length)
    half_length = max(1, int(length / 2))
    sqrt_length = max(1, int(round(np.sqrt(length))))
    
    wma_half = wma(s, length=half_length, min_periods=min_periods)
    wma_full = wma(s, length=length, min_periods=min_periods)
    
    raw_hma = (2 * wma_half) - wma_full
    res = wma(raw_hma, length=sqrt_length, min_periods=min_periods)
    return _finalize_series(res, offset, fillna, f"HMA_{length}")

def zlema(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_MA_LENGTH, source: str = 'close',
          offset: int = 0, fillna: Any = None, min_periods: Optional[int] = None) -> pd.Series:
    s = prepare_series(data, source, length)
    lag = int((length - 1) / 2)
    s_lagged = s.shift(lag)
    res = ema(s + (s - s_lagged), length=length, min_periods=min_periods)
    return _finalize_series(res, offset, fillna, f"ZLEMA_{length}")

def kama(data: Union[pd.DataFrame, pd.Series], length: int = 10, fast_len: int = DEFAULT_FAST_LENGTH, 
         slow_len: int = DEFAULT_SLOW_LENGTH, source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    """
    Kaufman Adaptive Moving Average (KAMA).
    FUTURE_OPTIMIZATION: Candidate for Numba @jit compilation.
    """
    s = prepare_series(data, source, length)
    arr = s.to_numpy(dtype=np.float64)
    
    if len(arr) < length:
        return _finalize_series(pd.Series(np.full(len(arr), np.nan), index=s.index), offset, fillna, f"KAMA_{length}")
        
    change = np.abs(s.diff(length).to_numpy(dtype=np.float64))
    volatility = s.diff(1).abs().rolling(window=length).sum().to_numpy(dtype=np.float64)
    
    er = safe_divide(change, volatility, default=0.0)
    fast_sc = 2.0 / (fast_len + 1.0)
    slow_sc = 2.0 / (slow_len + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama_arr = np.full_like(arr, np.nan)
    first_valid = length
    
    if first_valid < len(arr):
        kama_arr[first_valid - 1] = arr[first_valid - 1]
        for i in range(first_valid, len(arr)):
            if np.isnan(sc[i]) or np.isnan(arr[i]):
                kama_arr[i] = kama_arr[i-1] 
            else:
                kama_arr[i] = kama_arr[i-1] + sc[i] * (arr[i] - kama_arr[i-1])
                
    return _finalize_series(pd.Series(kama_arr, index=s.index), offset, fillna, f"KAMA_{length}")

def alma(data: Union[pd.DataFrame, pd.Series], length: int = 9, offset_alma: float = DEFAULT_ALMA_OFFSET, 
         sigma: float = DEFAULT_ALMA_SIGMA, source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    s = prepare_series(data, source, length)
    arr = s.to_numpy(dtype=np.float64)
    
    m = np.floor(offset_alma * (length - 1))
    s_val = length / sigma
    
    weights = np.exp(-((np.arange(length) - m) ** 2) / (2 * s_val * s_val))
    weights /= weights.sum()
    
    out = _nan_safe_convolve(arr, weights[::-1])
    return _finalize_series(pd.Series(out, index=s.index), offset, fillna, f"ALMA_{length}")

def t3(data: Union[pd.DataFrame, pd.Series], length: int = 5, vfactor: float = DEFAULT_T3_VFACTOR, 
       source: str = 'close', offset: int = 0, fillna: Any = None, 
       min_periods: Optional[int] = None) -> pd.Series:
    s = prepare_series(data, source, length)
    a = vfactor
    c1, c2 = -(a ** 3), 3 * (a ** 2) + 3 * (a ** 3)
    c3, c4 = -6 * (a ** 2) - 3 * a - 3 * (a ** 3), 1 + 3 * a + (a ** 3) + 3 * (a ** 2)
    
    e1 = ema(s, length=length, min_periods=min_periods)
    e2 = ema(e1, length=length, min_periods=min_periods)
    e3 = ema(e2, length=length, min_periods=min_periods)
    e4 = ema(e3, length=length, min_periods=min_periods)
    e5 = ema(e4, length=length, min_periods=min_periods)
    e6 = ema(e5, length=length, min_periods=min_periods)
    
    res = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
    return _finalize_series(res, offset, fillna, f"T3_{length}")

def mcginley_dynamic(data: Union[pd.DataFrame, pd.Series], length: int = 14, k: float = DEFAULT_MCGINLEY_K,
                     source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    """
    McGinley Dynamic Indicator.
    FUTURE_OPTIMIZATION: Candidate for Numba @jit compilation.
    """
    s = prepare_series(data, source, length)
    arr = s.to_numpy(dtype=np.float64)
    md_arr = np.full_like(arr, np.nan)
    
    first_valid = ~np.isnan(arr)
    if not first_valid.any():
        return _finalize_series(pd.Series(md_arr, index=s.index), offset, fillna, f"McGinley_{length}")
        
    start_idx = first_valid.argmax()
    md_arr[start_idx] = arr[start_idx]
    
    for i in range(start_idx + 1, len(arr)):
        if np.isnan(arr[i]):
            md_arr[i] = md_arr[i-1]
        else:
            prev_md = md_arr[i-1]
            if prev_md <= EPSILON:
                md_arr[i] = arr[i]
            else:
                denominator = max(k * length * ((arr[i] / prev_md) ** 4), EPSILON)
                md_arr[i] = prev_md + (arr[i] - prev_md) / denominator
                
    return _finalize_series(pd.Series(md_arr, index=s.index), offset, fillna, f"McGinley_{length}")

def vidya(data: Union[pd.DataFrame, pd.Series], length: int = 9, source: str = 'close',
          offset: int = 0, fillna: Any = None) -> pd.Series:
    """
    Variable Index Dynamic Average (VIDYA).
    FUTURE_OPTIMIZATION: Candidate for Numba @jit compilation.
    """
    s = prepare_series(data, source, length)
    arr = s.to_numpy(dtype=np.float64)
    mask = np.isnan(arr)
    
    if len(arr) < length:
        return _finalize_series(pd.Series(np.full(len(arr), np.nan), index=s.index), offset, fillna, f"VIDYA_{length}")

    s_filled = pd.Series(arr).ffill().bfill().to_numpy(dtype=np.float64)
    diff = np.diff(s_filled, prepend=0.0)
    
    pos_move = np.where(diff > 0, diff, 0.0)
    neg_move = np.where(diff < 0, np.abs(diff), 0.0)
    
    pos_sum = pd.Series(pos_move).rolling(window=length).sum().to_numpy(dtype=np.float64)
    neg_sum = pd.Series(neg_move).rolling(window=length).sum().to_numpy(dtype=np.float64)
    
    cmo = safe_divide(np.abs(pos_sum - neg_sum), (pos_sum + neg_sum), default=0.0)
    alpha = 2.0 / (length + 1.0)
    
    vidya_arr = np.full_like(arr, np.nan)
    first_valid = length
    
    if first_valid < len(arr):
        vidya_arr[first_valid - 1] = arr[first_valid - 1]
        for i in range(first_valid, len(arr)):
            if np.isnan(s_filled[i]) or np.isnan(cmo[i]):
                vidya_arr[i] = vidya_arr[i-1]
            else:
                weight = alpha * cmo[i]
                vidya_arr[i] = weight * s_filled[i] + (1 - weight) * vidya_arr[i-1]
                
    vidya_arr[mask] = np.nan
    return _finalize_series(pd.Series(vidya_arr, index=s.index), offset, fillna, f"VIDYA_{length}")

# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "MovingAverageError",
    "validate_length",
    "get_price_source",
    "validate_ma_input",
    "prepare_series",
    "sma",
    "ema",
    "smma",
    "wma",
    "vwma",
    "lsma",
    "dema",
    "tema",
    "trima",
    "hma",
    "zlema",
    "kama",
    "alma",
    "t3",
    "mcginley_dynamic",
    "vidya"
]
