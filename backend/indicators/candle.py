import logging
import numpy as np
import pandas as pd
from typing import Union, Optional, Any, Tuple

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper

from backend.indicators.helper import (
    standardize_column_names,
    validate_ohlc,
    hl2,
    hlc3,
    ohlc4,
    weighted_price
)
from backend.indicators.math_utils import (
    safe_divide,
    rolling_mean,
    rolling_std,
    EPSILON
)

# ==============================================================================
# LOGGING & CONSTANTS
# ==============================================================================

logger = logging.getLogger(__name__)

DEFAULT_LENGTH = 20
DOJI_THRESHOLD = 0.05
SMALL_BODY_THRESHOLD = 0.25
LARGE_BODY_THRESHOLD = 0.60
LONG_WICK_THRESHOLD = 0.50

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

class CandleIndicatorError(Exception):
    """Custom exception for errors in candlestick metric calculations."""
    pass

# ==============================================================================
# COMMON UTILITIES
# ==============================================================================

def validate_length(length: int, name: str = "length") -> None:
    if not isinstance(length, int) or length <= 0:
        logger.error(f"Invalid candle window length for {name}: {length}")
        raise CandleIndicatorError(f"Length for {name} must be a positive integer, got {length}")

def get_price_source(data: Union[pd.DataFrame, pd.Series], source: str = 'close') -> pd.Series:
    """Extracts the specific price source requested."""
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
            raise CandleIndicatorError(f"DataFrame is missing columns needed for '{src}': {missing}")
        return func(*[df[c] for c in req_cols])
            
    logger.error(f"Invalid source '{source}'.")
    raise CandleIndicatorError(f"Invalid source '{source}' or column not found.")

def _finalize_output(output: Union[pd.Series, pd.DataFrame], offset: int, fillna: Any) -> Union[pd.Series, pd.DataFrame]:
    res = output.shift(offset) if offset != 0 else output
    if fillna is not None:
        res = res.fillna(fillna)
    return res

def _extract_ohlc_arrays(data: Union[pd.DataFrame, pd.Series]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    """Helper for ultra-fast NumPy array extraction ensuring pure vectorized operations."""
    if isinstance(data, pd.Series):
        raise CandleIndicatorError("OHLC DataFrame required for candle metrics, received Series.")
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    o = df['open'].to_numpy(dtype=np.float64)
    h = df['high'].to_numpy(dtype=np.float64)
    l = df['low'].to_numpy(dtype=np.float64)
    c = df['close'].to_numpy(dtype=np.float64)
    return o, h, l, c, df.index

def _shift_array(arr: np.ndarray, periods: int = 1, fill_val: float = np.nan) -> np.ndarray:
    """Safe NumPy array shift."""
    shifted = np.empty_like(arr)
    if periods > 0:
        shifted[:periods] = fill_val
        shifted[periods:] = arr[:-periods]
    elif periods < 0:
        shifted[periods:] = fill_val
        shifted[:periods] = arr[-periods:]
    else:
        shifted[:] = arr
    return shifted

# ==============================================================================
# BODY METRICS
# ==============================================================================

def body_size(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    res = np.abs(c - o)
    return _finalize_output(pd.Series(res, index=idx, name="BodySize"), offset, fillna)

def real_body(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    res = c - o
    return _finalize_output(pd.Series(res, index=idx, name="RealBody"), offset, fillna)

def body_percent(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    body = np.abs(c - o)
    rng = h - l
    res = 100.0 * safe_divide(body, rng, default=0.0)
    return _finalize_output(pd.Series(res, index=idx, name="BodyPercent"), offset, fillna)

def body_midpoint(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    res = (o + c) / 2.0
    return _finalize_output(pd.Series(res, index=idx, name="BodyMidpoint"), offset, fillna)

def body_ratio(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    body = np.abs(c - o)
    rng = h - l
    res = safe_divide(body, rng, default=0.0)
    return _finalize_output(pd.Series(res, index=idx, name="BodyRatio"), offset, fillna)

def body_strength(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    real = c - o
    rng = h - l
    res = safe_divide(real, rng, default=0.0)
    return _finalize_output(pd.Series(res, index=idx, name="BodyStrength"), offset, fillna)

def body_position(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    mid = (o + c) / 2.0
    rng = h - l
    res = safe_divide(mid - l, rng, default=0.5)
    return _finalize_output(pd.Series(res, index=idx, name="BodyPosition"), offset, fillna)

# ==============================================================================
# BODY CLASSIFICATION
# ==============================================================================

def bullish_body(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    res = c > o
    return _finalize_output(pd.Series(res, index=idx, name="BullishBody"), offset, fillna)

def bearish_body(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    res = c < o
    return _finalize_output(pd.Series(res, index=idx, name="BearishBody"), offset, fillna)

def small_body(data: pd.DataFrame, threshold: float = SMALL_BODY_THRESHOLD, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    body = np.abs(c - o)
    rng = h - l
    res = safe_divide(body, rng, default=0.0) <= threshold
    return _finalize_output(pd.Series(res, index=idx, name="SmallBody"), offset, fillna)

def large_body(data: pd.DataFrame, threshold: float = LARGE_BODY_THRESHOLD, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    body = np.abs(c - o)
    rng = h - l
    res = safe_divide(body, rng, default=0.0) >= threshold
    return _finalize_output(pd.Series(res, index=idx, name="LargeBody"), offset, fillna)

# ==============================================================================
# BODY MOMENTUM
# ==============================================================================

def body_change(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    body = np.abs(c - o)
    prev_body = _shift_array(body, 1)
    res = body - prev_body
    return _finalize_output(pd.Series(res, index=idx, name="BodyChange"), offset, fillna)

def body_average(data: pd.DataFrame, length: int = DEFAULT_LENGTH, offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Body Average")
    b_size = body_size(data)
    res = rolling_mean(b_size, length)
    res.name = f"BodyAvg_{length}"
    return _finalize_output(res, offset, fillna)

def body_expansion(data: pd.DataFrame, length: int = DEFAULT_LENGTH, multiplier: float = 1.5, offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Body Expansion")
    b_size = body_size(data).to_numpy()
    b_avg = rolling_mean(pd.Series(b_size, index=data.index), length).to_numpy()
    res = b_size > (b_avg * multiplier)
    return _finalize_output(pd.Series(res, index=data.index, name=f"BodyExp_{length}"), offset, fillna)

def body_contraction(data: pd.DataFrame, length: int = DEFAULT_LENGTH, multiplier: float = 0.75, offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Body Contraction")
    b_size = body_size(data).to_numpy()
    b_avg = rolling_mean(pd.Series(b_size, index=data.index), length).to_numpy()
    res = b_size < (b_avg * multiplier)
    return _finalize_output(pd.Series(res, index=data.index, name=f"BodyCont_{length}"), offset, fillna)

# ==============================================================================
# WICK METRICS
# ==============================================================================

def upper_wick(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    res = h - np.maximum(o, c)
    return _finalize_output(pd.Series(res, index=idx, name="UpperWick"), offset, fillna)

def lower_wick(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    res = np.minimum(o, c) - l
    return _finalize_output(pd.Series(res, index=idx, name="LowerWick"), offset, fillna)

def upper_shadow(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    return upper_wick(data, offset=offset, fillna=fillna)

def lower_shadow(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    return lower_wick(data, offset=offset, fillna=fillna)

def wick_size(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    uw = h - np.maximum(o, c)
    lw = np.minimum(o, c) - l
    res = uw + lw
    return _finalize_output(pd.Series(res, index=idx, name="WickSize"), offset, fillna)

def wick_ratio(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    ws = (h - np.maximum(o, c)) + (np.minimum(o, c) - l)
    rng = h - l
    res = safe_divide(ws, rng, default=0.0)
    return _finalize_output(pd.Series(res, index=idx, name="WickRatio"), offset, fillna)

def wick_balance(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    uw = h - np.maximum(o, c)
    lw = np.minimum(o, c) - l
    res = uw - lw
    return _finalize_output(pd.Series(res, index=idx, name="WickBalance"), offset, fillna)

def wick_strength(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    uw = h - np.maximum(o, c)
    lw = np.minimum(o, c) - l
    rng = h - l
    res = safe_divide(lw - uw, rng, default=0.0)
    return _finalize_output(pd.Series(res, index=idx, name="WickStrength"), offset, fillna)

def wick_percent(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    wr = wick_ratio(data) * 100.0
    wr.name = "WickPercent"
    return _finalize_output(wr, offset, fillna)

def long_upper_wick(data: pd.DataFrame, threshold: float = LONG_WICK_THRESHOLD, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    uw = h - np.maximum(o, c)
    rng = h - l
    res = uw > (rng * threshold)
    return _finalize_output(pd.Series(res, index=idx, name="LongUpperWick"), offset, fillna)

def long_lower_wick(data: pd.DataFrame, threshold: float = LONG_WICK_THRESHOLD, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    lw = np.minimum(o, c) - l
    rng = h - l
    res = lw > (rng * threshold)
    return _finalize_output(pd.Series(res, index=idx, name="LongLowerWick"), offset, fillna)

def small_wick(data: pd.DataFrame, threshold: float = 0.20, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    ws = (h - np.maximum(o, c)) + (np.minimum(o, c) - l)
    rng = h - l
    res = ws < (rng * threshold)
    return _finalize_output(pd.Series(res, index=idx, name="SmallWick"), offset, fillna)

# ==============================================================================
# RANGE METRICS
# ==============================================================================

def candle_range(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    res = h - l
    return _finalize_output(pd.Series(res, index=idx, name="CandleRange"), offset, fillna)

def true_range(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    prev_c = _shift_array(c, 1)
    
    tr1 = h - l
    tr2 = np.abs(h - prev_c)
    tr3 = np.abs(l - prev_c)
    
    res = np.maximum(tr1, np.maximum(tr2, tr3))
    return _finalize_output(pd.Series(res, index=idx, name="TrueRange"), offset, fillna)

def body_to_range(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    return body_ratio(data, offset=offset, fillna=fillna)

def range_expansion(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    rng = h - l
    prev_rng = _shift_array(rng, 1)
    res = rng > prev_rng
    return _finalize_output(pd.Series(res, index=idx, name="RangeExp"), offset, fillna)

def range_contraction(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    rng = h - l
    prev_rng = _shift_array(rng, 1)
    res = rng < prev_rng
    return _finalize_output(pd.Series(res, index=idx, name="RangeCont"), offset, fillna)

def average_range(data: pd.DataFrame, length: int = DEFAULT_LENGTH, offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Average Range")
    rng = candle_range(data)
    res = rolling_mean(rng, length)
    res.name = f"AvgRange_{length}"
    return _finalize_output(res, offset, fillna)

def rolling_range(data: pd.DataFrame, length: int = DEFAULT_LENGTH, offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Rolling Range")
    df = standardize_column_names(data)
    roll_max = df['high'].rolling(window=length).max()
    roll_min = df['low'].rolling(window=length).min()
    res = roll_max - roll_min
    res.name = f"RollRange_{length}"
    return _finalize_output(res, offset, fillna)

def range_percentile(data: pd.DataFrame, length: int = 252, offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Range Percentile")
    rng = candle_range(data)
    res = rng.rolling(window=length).rank(pct=True) * 100.0
    res.name = f"RangePct_{length}"
    return _finalize_output(res, offset, fillna)

# ==============================================================================
# GAP ANALYSIS
# ==============================================================================

def gap_up(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    """Traditional Gap Up: Open > Previous Close."""
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    prev_c = _shift_array(c, 1)
    res = o > prev_c
    return _finalize_output(pd.Series(res, index=idx, name="GapUp"), offset, fillna)

def gap_down(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    """Traditional Gap Down: Open < Previous Close."""
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    prev_c = _shift_array(c, 1)
    res = o < prev_c
    return _finalize_output(pd.Series(res, index=idx, name="GapDown"), offset, fillna)

def breakaway_gap_up(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    """Breakaway Gap Up: Open > Previous High."""
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    prev_h = _shift_array(h, 1)
    res = o > prev_h
    return _finalize_output(pd.Series(res, index=idx, name="BreakawayGapUp"), offset, fillna)

def breakaway_gap_down(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    """Breakaway Gap Down: Open < Previous Low."""
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    prev_l = _shift_array(l, 1)
    res = o < prev_l
    return _finalize_output(pd.Series(res, index=idx, name="BreakawayGapDown"), offset, fillna)

def gap_size(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    prev_c = _shift_array(c, 1)
    res = o - prev_c
    return _finalize_output(pd.Series(res, index=idx, name="GapSize"), offset, fillna)

def gap_percent(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    prev_c = _shift_array(c, 1)
    res = 100.0 * safe_divide(o - prev_c, prev_c, default=0.0)
    return _finalize_output(pd.Series(res, index=idx, name="GapPercent"), offset, fillna)

def opening_gap(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    return gap_size(data, offset=offset, fillna=fillna)

def closing_gap(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    prev_c = _shift_array(c, 1)
    res = c - prev_c
    return _finalize_output(pd.Series(res, index=idx, name="CloseGap"), offset, fillna)

def gap_fill(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    """True if today's price range overlaps and fills the gap created at open."""
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    prev_c = _shift_array(c, 1)
    
    g_up = o > prev_c
    g_up_filled = g_up & (l <= prev_c)
    
    g_dn = o < prev_c
    g_dn_filled = g_dn & (h >= prev_c)
    
    res = g_up_filled | g_dn_filled
    return _finalize_output(pd.Series(res, index=idx, name="GapFill"), offset, fillna)

def inside_gap(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    """Gap occurred at open but remains inside the previous bar's H/L range."""
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    prev_h = _shift_array(h, 1)
    prev_l = _shift_array(l, 1)
    prev_c = _shift_array(c, 1)
    
    has_gap = np.abs(o - prev_c) > EPSILON
    is_inside = (o <= prev_h) & (o >= prev_l)
    res = has_gap & is_inside
    return _finalize_output(pd.Series(res, index=idx, name="InsideGap"), offset, fillna)

# ==============================================================================
# PATTERN COMPONENTS
# ==============================================================================

def doji(data: pd.DataFrame, threshold: float = DOJI_THRESHOLD, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    body = np.abs(c - o)
    rng = h - l
    res = safe_divide(body, rng, default=0.0) <= threshold
    return _finalize_output(pd.Series(res, index=idx, name="Doji"), offset, fillna)

def dragonfly_doji(data: pd.DataFrame, threshold: float = DOJI_THRESHOLD, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    is_doji = doji(data, threshold).to_numpy()
    uw = h - np.maximum(o, c)
    lw = np.minimum(o, c) - l
    res = is_doji & (lw > (uw * 3.0)) & (uw < ((h - l) * 0.1))
    return _finalize_output(pd.Series(res, index=idx, name="DragonflyDoji"), offset, fillna)

def gravestone_doji(data: pd.DataFrame, threshold: float = DOJI_THRESHOLD, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    is_doji = doji(data, threshold).to_numpy()
    uw = h - np.maximum(o, c)
    lw = np.minimum(o, c) - l
    res = is_doji & (uw > (lw * 3.0)) & (lw < ((h - l) * 0.1))
    return _finalize_output(pd.Series(res, index=idx, name="GravestoneDoji"), offset, fillna)

def long_legged_doji(data: pd.DataFrame, threshold: float = DOJI_THRESHOLD, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    is_doji = doji(data, threshold).to_numpy()
    uw = h - np.maximum(o, c)
    lw = np.minimum(o, c) - l
    rng = h - l
    res = is_doji & (uw > (rng * 0.3)) & (lw > (rng * 0.3))
    return _finalize_output(pd.Series(res, index=idx, name="LongLeggedDoji"), offset, fillna)

def marubozu(data: pd.DataFrame, threshold: float = 0.95, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    body = np.abs(c - o)
    rng = h - l
    res = safe_divide(body, rng, default=0.0) >= threshold
    return _finalize_output(pd.Series(res, index=idx, name="Marubozu"), offset, fillna)

def bullish_marubozu(data: pd.DataFrame, threshold: float = 0.95, offset: int = 0, fillna: Any = None) -> pd.Series:
    is_mb = marubozu(data, threshold).to_numpy()
    is_bull = bullish_body(data).to_numpy()
    res = is_mb & is_bull
    return _finalize_output(pd.Series(res, index=data.index, name="BullMarubozu"), offset, fillna)

def bearish_marubozu(data: pd.DataFrame, threshold: float = 0.95, offset: int = 0, fillna: Any = None) -> pd.Series:
    is_mb = marubozu(data, threshold).to_numpy()
    is_bear = bearish_body(data).to_numpy()
    res = is_mb & is_bear
    return _finalize_output(pd.Series(res, index=data.index, name="BearMarubozu"), offset, fillna)

def spinning_top(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    body = np.abs(c - o)
    uw = h - np.maximum(o, c)
    lw = np.minimum(o, c) - l
    
    is_small_body = small_body(data).to_numpy()
    res = is_small_body & (uw > body) & (lw > body)
    return _finalize_output(pd.Series(res, index=idx, name="SpinningTop"), offset, fillna)

def high_wave(data: pd.DataFrame, length: int = DEFAULT_LENGTH, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    rng = h - l
    uw = h - np.maximum(o, c)
    lw = np.minimum(o, c) - l
    
    is_small_body = small_body(data).to_numpy()
    avg_rng = average_range(data, length).to_numpy()
    
    res = is_small_body & (uw > rng * 0.3) & (lw > rng * 0.3) & (rng > avg_rng)
    return _finalize_output(pd.Series(res, index=idx, name="HighWave"), offset, fillna)

def hammer_shape(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    """
    Strict Hammer definition: long lower wick, small upper wick, real body present.
    NOTE: Detects purely the structural shape. Trend context and confirmation belong to the Strategy Layer.
    """
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    body = np.abs(c - o)
    uw = h - np.maximum(o, c)
    lw = np.minimum(o, c) - l
    rng = h - l
    
    res = (lw > (body * 2.0)) & (uw < (rng * 0.1)) & (body > EPSILON)
    return _finalize_output(pd.Series(res, index=idx, name="HammerShape"), offset, fillna)

def hanging_man_shape(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    """
    Structurally identical to Hammer Shape. 
    NOTE: Detects purely the structural shape. Trend context and confirmation belong to the Strategy Layer.
    """
    res = hammer_shape(data).to_numpy()
    return _finalize_output(pd.Series(res, index=data.index, name="HangingManShape"), offset, fillna)

def inverted_hammer_shape(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    """
    Strict Inverted Hammer definition: long upper wick, small lower wick, real body present.
    NOTE: Detects purely the structural shape. Trend context and confirmation belong to the Strategy Layer.
    """
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    body = np.abs(c - o)
    uw = h - np.maximum(o, c)
    lw = np.minimum(o, c) - l
    rng = h - l
    
    res = (uw > (body * 2.0)) & (lw < (rng * 0.1)) & (body > EPSILON)
    return _finalize_output(pd.Series(res, index=idx, name="InvHammerShape"), offset, fillna)

def shooting_star_shape(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    """
    Structurally identical to Inverted Hammer Shape.
    NOTE: Detects purely the structural shape. Trend context and confirmation belong to the Strategy Layer.
    """
    res = inverted_hammer_shape(data).to_numpy()
    return _finalize_output(pd.Series(res, index=data.index, name="ShootingStarShape"), offset, fillna)

def belt_hold(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    """Bullish: Opens at Low, closes near High. Bearish: Opens at High, closes near Low."""
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    is_bull = bullish_body(data).to_numpy()
    is_bear = bearish_body(data).to_numpy()
    
    bull_hold = is_bull & (np.abs(o - l) < EPSILON) & (c > l + (h - l) * 0.8)
    bear_hold = is_bear & (np.abs(o - h) < EPSILON) & (c < h - (h - l) * 0.8)
    
    res = bull_hold | bear_hold
    return _finalize_output(pd.Series(res, index=idx, name="BeltHold"), offset, fillna)

def shaven_head(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    uw = h - np.maximum(o, c)
    res = uw < EPSILON
    return _finalize_output(pd.Series(res, index=idx, name="ShavenHead"), offset, fillna)

def shaven_bottom(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    lw = np.minimum(o, c) - l
    res = lw < EPSILON
    return _finalize_output(pd.Series(res, index=idx, name="ShavenBottom"), offset, fillna)

# ==============================================================================
# STRENGTH METRICS
# ==============================================================================

def bull_power(data: pd.DataFrame, length: int = 13, offset: int = 0, fillna: Any = None) -> pd.Series:
    """High - EMA(Close). Utilizes native Pandas ewm for speed and moving_average independence."""
    validate_length(length, "Bull Power")
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    ema = df['close'].ewm(span=length, adjust=False).mean()
    res = df['high'] - ema
    return _finalize_output(pd.Series(res, index=df.index, name=f"BullPower_{length}"), offset, fillna)

def bear_power(data: pd.DataFrame, length: int = 13, offset: int = 0, fillna: Any = None) -> pd.Series:
    """Low - EMA(Close)."""
    validate_length(length, "Bear Power")
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    ema = df['close'].ewm(span=length, adjust=False).mean()
    res = df['low'] - ema
    return _finalize_output(pd.Series(res, index=df.index, name=f"BearPower_{length}"), offset, fillna)

def buying_pressure(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    res = c - l
    return _finalize_output(pd.Series(res, index=idx, name="BuyingPressure"), offset, fillna)

def selling_pressure(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    res = h - c
    return _finalize_output(pd.Series(res, index=idx, name="SellingPressure"), offset, fillna)

def candle_strength(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    """(Close - Low) / (High - Low)"""
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    res = safe_divide(c - l, h - l, default=0.5)
    return _finalize_output(pd.Series(res, index=idx, name="CandleStrength"), offset, fillna)

def direction_strength(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    """(Close - Open) / (High - Low)"""
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    res = safe_divide(c - o, h - l, default=0.0)
    return _finalize_output(pd.Series(res, index=idx, name="DirStrength"), offset, fillna)

def dominance_score(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    """(Buying Pressure - Selling Pressure) / Candle Range"""
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    bp = c - l
    sp = h - c
    res = safe_divide(bp - sp, h - l, default=0.0)
    return _finalize_output(pd.Series(res, index=idx, name="DominanceScore"), offset, fillna)

def pressure_score(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    """Buying Pressure / (Buying Pressure + Selling Pressure)"""
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    bp = c - l
    sp = h - c
    res = safe_divide(bp, bp + sp, default=0.5)
    return _finalize_output(pd.Series(res, index=idx, name="PressureScore"), offset, fillna)

def balance_score(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    """1 - abs(BP - SP) / Candle Range"""
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    bp = c - l
    sp = h - c
    diff = np.abs(bp - sp)
    res = 1.0 - safe_divide(diff, h - l, default=1.0)
    return _finalize_output(pd.Series(res, index=idx, name="BalanceScore"), offset, fillna)

# ==============================================================================
# CLOSE / OPEN POSITION METRICS
# ==============================================================================

def close_location_value(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    """((C - L) - (H - C)) / (H - L)"""
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    res = safe_divide((c - l) - (h - c), h - l, default=0.0)
    return _finalize_output(pd.Series(res, index=idx, name="CLV"), offset, fillna)

def close_percent(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    """(C - L) / (H - L) * 100"""
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    res = 100.0 * safe_divide(c - l, h - l, default=0.5)
    return _finalize_output(pd.Series(res, index=idx, name="ClosePercent"), offset, fillna)

def close_to_high(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    res = h - c
    return _finalize_output(pd.Series(res, index=idx, name="CloseToHigh"), offset, fillna)

def close_to_low(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    res = c - l
    return _finalize_output(pd.Series(res, index=idx, name="CloseToLow"), offset, fillna)

def close_position(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    res = safe_divide(c - l, h - l, default=0.5)
    return _finalize_output(pd.Series(res, index=idx, name="ClosePos"), offset, fillna)

def open_location(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    res = safe_divide((o - l) - (h - o), h - l, default=0.0)
    return _finalize_output(pd.Series(res, index=idx, name="OpenLoc"), offset, fillna)

def open_position(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    res = safe_divide(o - l, h - l, default=0.5)
    return _finalize_output(pd.Series(res, index=idx, name="OpenPos"), offset, fillna)

# ==============================================================================
# SESSION METRICS
# ==============================================================================

def bullish_candle(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    return bullish_body(data, offset=offset, fillna=fillna)

def bearish_candle(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    return bearish_body(data, offset=offset, fillna=fillna)

def neutral_candle(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    res = np.abs(c - o) < EPSILON
    return _finalize_output(pd.Series(res, index=idx, name="NeutralCandle"), offset, fillna)

def inside_bar(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    prev_h = _shift_array(h, 1)
    prev_l = _shift_array(l, 1)
    res = (h <= prev_h) & (l >= prev_l)
    return _finalize_output(pd.Series(res, index=idx, name="InsideBar"), offset, fillna)

def outside_bar(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    prev_h = _shift_array(h, 1)
    prev_l = _shift_array(l, 1)
    res = (h > prev_h) & (l < prev_l)
    return _finalize_output(pd.Series(res, index=idx, name="OutsideBar"), offset, fillna)

def engulfing_body(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    """
    NOTE: Detects purely the structural shape. Trend context and confirmation belong to the Strategy Layer.
    """
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    prev_o = _shift_array(o, 1)
    prev_c = _shift_array(c, 1)
    
    bull_engulf = (c > o) & (prev_c < prev_o) & (c > prev_o) & (o < prev_c)
    bear_engulf = (c < o) & (prev_c > prev_o) & (c < prev_o) & (o > prev_c)
    
    # +1 for Bullish, -1 for Bearish, 0 otherwise
    res = np.where(bull_engulf, 1, np.where(bear_engulf, -1, 0))
    return _finalize_output(pd.Series(res, index=idx, name="EngulfingBody"), offset, fillna)

def body_overlap(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    top = np.maximum(o, c)
    bot = np.minimum(o, c)
    
    prev_top = _shift_array(top, 1)
    prev_bot = _shift_array(bot, 1)
    
    overlap_top = np.minimum(top, prev_top)
    overlap_bot = np.maximum(bot, prev_bot)
    
    res = np.maximum(0.0, overlap_top - overlap_bot)
    return _finalize_output(pd.Series(res, index=idx, name="BodyOverlap"), offset, fillna)

def range_overlap(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    prev_h = _shift_array(h, 1)
    prev_l = _shift_array(l, 1)
    
    overlap_top = np.minimum(h, prev_h)
    overlap_bot = np.maximum(l, prev_l)
    
    res = np.maximum(0.0, overlap_top - overlap_bot)
    return _finalize_output(pd.Series(res, index=idx, name="RangeOverlap"), offset, fillna)

# ==============================================================================
# VOLATILITY STYLE METRICS
# ==============================================================================

def expansion_candle(data: pd.DataFrame, length: int = DEFAULT_LENGTH, multiplier: float = 1.5, offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Expansion Candle")
    rng = candle_range(data).to_numpy()
    avg_rng = rolling_mean(pd.Series(rng, index=data.index), length).to_numpy()
    res = rng > (avg_rng * multiplier)
    return _finalize_output(pd.Series(res, index=data.index, name=f"ExpCandle_{length}"), offset, fillna)

def compression_candle(data: pd.DataFrame, length: int = DEFAULT_LENGTH, multiplier: float = 0.75, offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Compression Candle")
    rng = candle_range(data).to_numpy()
    avg_rng = rolling_mean(pd.Series(rng, index=data.index), length).to_numpy()
    res = rng < (avg_rng * multiplier)
    return _finalize_output(pd.Series(res, index=data.index, name=f"CompCandle_{length}"), offset, fillna)

def impulse_candle(data: pd.DataFrame, length: int = DEFAULT_LENGTH, offset: int = 0, fillna: Any = None) -> pd.Series:
    is_exp = expansion_candle(data, length=length).to_numpy()
    is_large = large_body(data).to_numpy()
    res = is_exp & is_large
    return _finalize_output(pd.Series(res, index=data.index, name=f"ImpulseCandle_{length}"), offset, fillna)

def indecision_candle(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    is_small = small_body(data).to_numpy()
    ws = wick_size(data).to_numpy()
    real = body_size(data).to_numpy()
    res = is_small & (ws > (real * 2.0))
    return _finalize_output(pd.Series(res, index=data.index, name="IndecisionCandle"), offset, fillna)

# ==============================================================================
# STATISTICAL CANDLE METRICS
# ==============================================================================

def rolling_body_mean(data: pd.DataFrame, length: int = DEFAULT_LENGTH, offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Rolling Body Mean")
    b_size = body_size(data)
    res = rolling_mean(b_size, length)
    res.name = f"RollBodyMean_{length}"
    return _finalize_output(res, offset, fillna)

def rolling_body_std(data: pd.DataFrame, length: int = DEFAULT_LENGTH, offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Rolling Body Std")
    b_size = body_size(data)
    res = rolling_std(b_size, length)
    res.name = f"RollBodyStd_{length}"
    return _finalize_output(res, offset, fillna)

def rolling_body_zscore(data: pd.DataFrame, length: int = DEFAULT_LENGTH, offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Rolling Body Z-Score")
    b_size = body_size(data).to_numpy()
    b_mean = rolling_body_mean(data, length=length).to_numpy()
    b_std = rolling_body_std(data, length=length).to_numpy()
    
    res = safe_divide(b_size - b_mean, b_std, default=np.nan)
    return _finalize_output(pd.Series(res, index=data.index, name=f"RollBodyZ_{length}"), offset, fillna)

def rolling_range_mean(data: pd.DataFrame, length: int = DEFAULT_LENGTH, offset: int = 0, fillna: Any = None) -> pd.Series:
    return average_range(data, length=length, offset=offset, fillna=fillna)

def rolling_range_std(data: pd.DataFrame, length: int = DEFAULT_LENGTH, offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Rolling Range Std")
    rng = candle_range(data)
    res = rolling_std(rng, length)
    res.name = f"RollRangeStd_{length}"
    return _finalize_output(res, offset, fillna)

def rolling_wick_mean(data: pd.DataFrame, length: int = DEFAULT_LENGTH, offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Rolling Wick Mean")
    ws = wick_size(data)
    res = rolling_mean(ws, length)
    res.name = f"RollWickMean_{length}"
    return _finalize_output(res, offset, fillna)

def rolling_wick_std(data: pd.DataFrame, length: int = DEFAULT_LENGTH, offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Rolling Wick Std")
    ws = wick_size(data)
    res = rolling_std(ws, length)
    res.name = f"RollWickStd_{length}"
    return _finalize_output(res, offset, fillna)

def rolling_body_percentile(data: pd.DataFrame, length: int = 252, offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Rolling Body Percentile")
    b_size = body_size(data)
    res = b_size.rolling(window=length).rank(pct=True) * 100.0
    res.name = f"RollBodyPct_{length}"
    return _finalize_output(res, offset, fillna)

def rolling_range_percentile(data: pd.DataFrame, length: int = 252, offset: int = 0, fillna: Any = None) -> pd.Series:
    return range_percentile(data, length=length, offset=offset, fillna=fillna)

def rolling_wick_percentile(data: pd.DataFrame, length: int = 252, offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Rolling Wick Percentile")
    ws = wick_size(data)
    res = ws.rolling(window=length).rank(pct=True) * 100.0
    res.name = f"RollWickPct_{length}"
    return _finalize_output(res, offset, fillna)

# ==============================================================================
# INSTITUTIONAL METRICS
# ==============================================================================

def institutional_body(data: pd.DataFrame, length: int = DEFAULT_LENGTH, z_thresh: float = 2.0, offset: int = 0, fillna: Any = None) -> pd.Series:
    """Identifies anomalously large bodies (Z-Score > Z_Thresh)."""
    validate_length(length, "Institutional Body")
    z_score = rolling_body_zscore(data, length=length).to_numpy()
    res = z_score > z_thresh
    return _finalize_output(pd.Series(res, index=data.index, name=f"InstBody_{length}_{z_thresh}"), offset, fillna)

def institutional_wick(data: pd.DataFrame, length: int = DEFAULT_LENGTH, z_thresh: float = 2.0, offset: int = 0, fillna: Any = None) -> pd.Series:
    """Identifies anomalously large wicks."""
    validate_length(length, "Institutional Wick")
    ws = wick_size(data).to_numpy()
    w_mean = rolling_wick_mean(data, length=length).to_numpy()
    w_std = rolling_wick_std(data, length=length).to_numpy()
    
    threshold = w_mean + (z_thresh * w_std)
    res = ws > threshold
    return _finalize_output(pd.Series(res, index=data.index, name=f"InstWick_{length}"), offset, fillna)

def institutional_imbalance(data: pd.DataFrame, length: int = DEFAULT_LENGTH, offset: int = 0, fillna: Any = None) -> pd.Series:
    """
    Extreme momentum candle: Massive body, large range, and optional volume confirmation.
    Indicates institutional tracking/sponsorship.
    """
    validate_length(length, "Institutional Imbalance")
    is_large = institutional_body(data, length=length, z_thresh=1.5).to_numpy()
    is_exp = expansion_candle(data, length=length, multiplier=2.0).to_numpy()
    
    res = is_large & is_exp
    
    # Optional Volume Integration if available in Data
    df = standardize_column_names(data)
    if 'volume' in df.columns:
        vol = df['volume']
        vol_sma = rolling_mean(vol, length)
        res = res & (vol > vol_sma * 1.5).to_numpy()
        
    return _finalize_output(pd.Series(res, index=data.index, name=f"InstImbalance_{length}"), offset, fillna)

def institutional_pressure(data: pd.DataFrame, length: int = DEFAULT_LENGTH, offset: int = 0, fillna: Any = None) -> pd.Series:
    """Wick strength multiplied by an expansion factor."""
    validate_length(length, "Institutional Pressure")
    w_str = wick_strength(data).to_numpy()
    is_exp = expansion_candle(data, length=length).to_numpy()
    
    res = w_str * is_exp
    return _finalize_output(pd.Series(res, index=data.index, name=f"InstPressure_{length}"), offset, fillna)

def absorption_candle(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    """Identifies long lower wicks with small bodies, typically indicating selling absorption."""
    is_small = small_body(data).to_numpy()
    llw = long_lower_wick(data).to_numpy()
    is_down = bearish_body(data).to_numpy()
    
    res = is_small & llw & is_down
    return _finalize_output(pd.Series(res, index=data.index, name="AbsorptionCandle"), offset, fillna)

def rejection_candle(data: pd.DataFrame, length: int = DEFAULT_LENGTH, offset: int = 0, fillna: Any = None) -> pd.Series:
    """Identifies a candle with a strong wick that reverses from a local extreme."""
    validate_length(length, "Rejection Candle")
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    
    roll_max = pd.Series(h, index=data.index).rolling(length).max().shift(1).to_numpy()
    roll_min = pd.Series(l, index=data.index).rolling(length).min().shift(1).to_numpy()
    
    is_long_up = long_upper_wick(data).to_numpy()
    is_long_dn = long_lower_wick(data).to_numpy()
    
    reject_high = is_long_up & (h >= roll_max)
    reject_low = is_long_dn & (l <= roll_min)
    
    res = reject_high | reject_low
    return _finalize_output(pd.Series(res, index=idx, name=f"RejectionCandle_{length}"), offset, fillna)

def acceptance_candle(data: pd.DataFrame, length: int = DEFAULT_LENGTH, offset: int = 0, fillna: Any = None) -> pd.Series:
    """Identifies a large body candle closing beyond recent extremes (breakout acceptance)."""
    validate_length(length, "Acceptance Candle")
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    
    roll_max = pd.Series(h, index=data.index).rolling(length).max().shift(1).to_numpy()
    roll_min = pd.Series(l, index=data.index).rolling(length).min().shift(1).to_numpy()
    
    is_large = large_body(data).to_numpy()
    
    accept_high = is_large & (c > roll_max)
    accept_low = is_large & (c < roll_min)
    
    res = accept_high | accept_low
    return _finalize_output(pd.Series(res, index=idx, name=f"AcceptanceCandle_{length}"), offset, fillna)

def liquidity_sweep_candle(data: pd.DataFrame, length: int = DEFAULT_LENGTH, offset: int = 0, fillna: Any = None) -> pd.Series:
    """Takes out previous high/low but closes inside."""
    validate_length(length, "Liquidity Sweep")
    o, h, l, c, idx = _extract_ohlc_arrays(data)
    
    roll_max = pd.Series(h, index=data.index).rolling(length).max().shift(1).to_numpy()
    roll_min = pd.Series(l, index=data.index).rolling(length).min().shift(1).to_numpy()
    
    sweep_high = (h > roll_max) & (c < roll_max)
    sweep_low = (l < roll_min) & (c > roll_min)
    
    res = sweep_high | sweep_low
    return _finalize_output(pd.Series(res, index=idx, name=f"LiqSweep_{length}"), offset, fillna)

def smart_money_candle(data: pd.DataFrame, length: int = DEFAULT_LENGTH, offset: int = 0, fillna: Any = None) -> pd.Series:
    """Combination of liquidity sweep + rejection + institutional wick sizing."""
    is_sweep = liquidity_sweep_candle(data, length=length).to_numpy()
    is_reject = rejection_candle(data, length=length).to_numpy()
    is_inst_wick = institutional_wick(data, length=length).to_numpy()
    
    res = is_sweep & is_reject & is_inst_wick
    
    # Optional Volume Confirmation
    df = standardize_column_names(data)
    if 'volume' in df.columns:
        vol = df['volume']
        vol_sma = rolling_mean(vol, length)
        res = res & (vol > vol_sma * 1.2).to_numpy()
        
    return _finalize_output(pd.Series(res, index=data.index, name=f"SmartMoneyCandle_{length}"), offset, fillna)

# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "CandleIndicatorError",
    "validate_length",
    "get_price_source",
    
    # Body Metrics
    "body_size",
    "real_body",
    "body_percent",
    "body_midpoint",
    "body_ratio",
    "body_strength",
    "body_position",
    
    # Body Classification
    "bullish_body",
    "bearish_body",
    "small_body",
    "large_body",
    
    # Body Momentum
    "body_change",
    "body_average",
    "body_expansion",
    "body_contraction",
    
    # Wick Metrics
    "upper_wick",
    "lower_wick",
    "upper_shadow",
    "lower_shadow",
    "wick_size",
    "wick_ratio",
    "wick_balance",
    "wick_strength",
    "wick_percent",
    "long_upper_wick",
    "long_lower_wick",
    "small_wick",
    
    # Range Metrics
    "candle_range",
    "true_range",
    "body_to_range",
    "range_expansion",
    "range_contraction",
    "average_range",
    "rolling_range",
    "range_percentile",
    
    # Gap Analysis
    "gap_up",
    "gap_down",
    "gap_size",
    "gap_percent",
    "opening_gap",
    "closing_gap",
    "gap_fill",
    "inside_gap",
    "breakaway_gap_up",
    "breakaway_gap_down",
    
    # Pattern Components
    "doji",
    "dragonfly_doji",
    "gravestone_doji",
    "long_legged_doji",
    "marubozu",
    "bullish_marubozu",
    "bearish_marubozu",
    "spinning_top",
    "high_wave",
    "hammer_shape",
    "hanging_man_shape",
    "inverted_hammer_shape",
    "shooting_star_shape",
    "belt_hold",
    "shaven_head",
    "shaven_bottom",
    
    # Strength Metrics
    "bull_power",
    "bear_power",
    "buying_pressure",
    "selling_pressure",
    "candle_strength",
    "direction_strength",
    "dominance_score",
    "pressure_score",
    "balance_score",
    
    # Close / Open Position Metrics
    "close_location_value",
    "close_percent",
    "close_to_high",
    "close_to_low",
    "close_position",
    "open_location",
    "open_position",
    
    # Session Metrics
    "bullish_candle",
    "bearish_candle",
    "neutral_candle",
    "inside_bar",
    "outside_bar",
    "engulfing_body",
    "body_overlap",
    "range_overlap",
    
    # Volatility Style Metrics
    "expansion_candle",
    "compression_candle",
    "impulse_candle",
    "indecision_candle",
    
    # Statistical Candle Metrics
    "rolling_body_mean",
    "rolling_body_std",
    "rolling_body_zscore",
    "rolling_range_mean",
    "rolling_range_std",
    "rolling_wick_mean",
    "rolling_wick_std",
    "rolling_body_percentile",
    "rolling_range_percentile",
    "rolling_wick_percentile",
    
    # Institutional Metrics
    "institutional_body",
    "institutional_wick",
    "institutional_imbalance",
    "institutional_pressure",
    "absorption_candle",
    "rejection_candle",
    "acceptance_candle",
    "liquidity_sweep_candle",
    "smart_money_candle"
]
