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
    validate_datetime_index,
    hl2,
    hlc3,
    ohlc4,
    weighted_price
)
from backend.indicators.math_utils import (
    safe_divide,
    rolling_mean,
    rolling_sum,
    rolling_std,
    EPSILON
)

# ==============================================================================
# LOGGING & CONSTANTS
# ==============================================================================

logger = logging.getLogger(__name__)

DEFAULT_SWING_LEN = 5
DEFAULT_ZONE_LEN = 20
DEFAULT_CHANNEL_LEN = 20
DEFAULT_CLUSTER_BINS = 10
DEFAULT_MTF_W = 5
DEFAULT_MTF_M = 21
DEFAULT_MTF_Y = 252

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

class SupportResistanceError(Exception):
    """Custom exception for errors in support/resistance calculations."""
    pass

# ==============================================================================
# NUMBA JIT ACCELERATED CORE FUNCTIONS
# ==============================================================================

@jit(nopython=True, cache=True)
def _get_nanmin_nanmax(arr: np.ndarray) -> Tuple[float, float]:
    """Numba-safe min/max ignoring NaNs."""
    min_val = np.inf
    max_val = -np.inf
    for x in arr:
        if not np.isnan(x):
            if x < min_val: min_val = x
            if x > max_val: max_val = x
    if np.isinf(min_val):
        return np.nan, np.nan
    return min_val, max_val

@jit(nopython=True, cache=True)
def _swing_levels_jit(high: np.ndarray, low: np.ndarray, left: int, right: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(high)
    sh, sl = np.full(n, np.nan), np.full(n, np.nan)
    last_sh, last_sl = np.full(n, np.nan), np.full(n, np.nan)
    curr_sh, curr_sl = np.nan, np.nan
    
    for i in range(left, n - right):
        is_sh = True
        for j in range(1, left + 1):
            if high[i - j] > high[i]:
                is_sh = False; break
        if is_sh:
            for j in range(1, right + 1):
                if high[i + j] >= high[i]:
                    is_sh = False; break
        if is_sh:
            sh[i], curr_sh = high[i], high[i]
            
        is_sl = True
        for j in range(1, left + 1):
            if low[i - j] < low[i]:
                is_sl = False; break
        if is_sl:
            for j in range(1, right + 1):
                if low[i + j] <= low[i]:
                    is_sl = False; break
        if is_sl:
            sl[i], curr_sl = low[i], low[i]
            
        last_sh[i], last_sl[i] = curr_sh, curr_sl

    for i in range(n - right, n):
        last_sh[i], last_sl[i] = curr_sh, curr_sl
    return sh, sl, last_sh, last_sl

@jit(nopython=True, cache=True)
def _price_cluster_jit(price: np.ndarray, window: int, bins: int) -> np.ndarray:
    n = len(price)
    clusters = np.full(n, np.nan)
    if n >= window:
        for i in range(window - 1, n):
            p_win = price[i - window + 1 : i + 1]
            p_min, p_max = _get_nanmin_nanmax(p_win)
            
            if np.isnan(p_min) or np.isnan(p_max) or p_min == p_max:
                clusters[i] = p_win[-1]
                continue
            bin_edges = np.linspace(p_min, p_max, bins + 1)
            hist = np.zeros(bins)
            for j in range(window):
                val = p_win[j]
                if np.isnan(val): continue
                b = 0
                while b < bins and val >= bin_edges[b+1]:
                    b += 1
                if b == bins: b -= 1
                hist[b] += 1
            max_bin = np.argmax(hist)
            clusters[i] = (bin_edges[max_bin] + bin_edges[max_bin + 1]) / 2.0
    return clusters

@jit(nopython=True, cache=True)
def _equal_cluster_jit(levels: np.ndarray, atr: np.ndarray, multiplier: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(levels)
    is_eq = np.zeros(n, dtype=np.bool_)
    c_size = np.zeros(n, dtype=np.float64)
    c_strength = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        curr = levels[i]
        if np.isnan(curr): continue
        count = 1
        eps = atr[i] * multiplier
        for j in range(i-1, max(-1, i-50), -1):
            if not np.isnan(levels[j]):
                if np.abs(curr - levels[j]) <= eps:
                    count += 1
        if count > 1:
            is_eq[i] = True
            c_size[i] = count
            c_strength[i] = count * 10.0
    return is_eq, c_size, c_strength

@jit(nopython=True, cache=True)
def _fvg_mitigation_jit(high: np.ndarray, low: np.ndarray, fvg_top: np.ndarray, fvg_bot: np.ndarray, is_bullish: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(high)
    mitigated = np.zeros(n, dtype=np.bool_)
    unfilled_top = np.copy(fvg_top)
    unfilled_bot = np.copy(fvg_bot)
    
    for i in range(n):
        if np.isnan(fvg_top[i]): continue
        t, b = fvg_top[i], fvg_bot[i]
        mitig = False
        for j in range(i+1, n):
            if is_bullish[i]:
                if low[j] <= b: 
                    mitig = True; break
                elif low[j] < t:
                    t = low[j]
            else:
                if high[j] >= t:
                    mitig = True; break
                elif high[j] > b:
                    b = high[j]
        mitigated[i] = mitig
        unfilled_top[i] = t
        unfilled_bot[i] = b
    return mitigated, unfilled_top, unfilled_bot

@jit(nopython=True, cache=True)
def _ob_lifecycle_jit(ob_high: np.ndarray, ob_low: np.ndarray, high: np.ndarray, low: np.ndarray, is_bull: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(high)
    state = np.zeros(n, dtype=np.int8)
    
    for i in range(n):
        if np.isnan(ob_high[i]): continue
        t, b = ob_high[i], ob_low[i]
        curr_state = 1 
        for j in range(i+1, n):
            if is_bull[i]:
                if low[j] < b:
                    curr_state = 3 
                    break
                elif low[j] <= t:
                    curr_state = 2 
            else:
                if high[j] > t:
                    curr_state = 3 
                    break
                elif high[j] >= b:
                    curr_state = 2 
        state[i] = curr_state
    
    fresh = state == 1
    mitigated = state == 2
    invalidated = state == 3
    return fresh, mitigated, invalidated

@jit(nopython=True, cache=True)
def _adaptive_vol_profile_jit(price: np.ndarray, vol: np.ndarray, length: int, atr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(price)
    poc, vah, val = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    if n >= length:
        for i in range(length - 1, n):
            p_win = price[i - length + 1 : i + 1]
            v_win = vol[i - length + 1 : i + 1]
            p_min, p_max = _get_nanmin_nanmax(p_win)
            
            if np.isnan(p_min) or np.isnan(p_max) or p_min == p_max:
                poc[i] = vah[i] = val[i] = p_win[-1]
                continue
                
            bin_width = max(atr[i] * 0.1, EPSILON)
            bins = max(3, int((p_max - p_min) / bin_width))
            if bins > 50: bins = 50 
            
            bin_edges = np.linspace(p_min, p_max, bins + 1)
            vol_profile = np.zeros(bins)
            for j in range(length):
                b = 0
                while b < bins and p_win[j] >= bin_edges[b+1]: b += 1
                if b == bins: b -= 1
                vol_profile[b] += v_win[j]
                
            poc_idx = np.argmax(vol_profile)
            poc[i] = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2.0
            
            t_vol = np.sum(vol_profile) * 0.70
            c_vol = vol_profile[poc_idx]
            u, d_ = poc_idx + 1, poc_idx - 1
            while c_vol < t_vol and (u < bins or d_ >= 0):
                vu = vol_profile[u] if u < bins else -1.0
                vd = vol_profile[d_] if d_ >= 0 else -1.0
                if vu > vd:
                    c_vol += vu; u += 1
                else:
                    c_vol += vd; d_ -= 1
            vah[i] = bin_edges[min(u, bins)]
            val[i] = bin_edges[max(0, d_)]
    return poc, vah, val

@jit(nopython=True, cache=True)
def _touch_count_jit(price_array: np.ndarray, target_array: np.ndarray, epsilon_arr: np.ndarray, length: int) -> np.ndarray:
    n = len(price_array)
    touches = np.zeros(n, dtype=np.float64)
    for i in range(length, n):
        target = target_array[i]
        if np.isnan(target): continue
        win = price_array[i-length:i]
        eps = epsilon_arr[i]
        
        count = 0
        for j in range(len(win)):
            if not np.isnan(win[j]):
                denom = target if target != 0 else EPSILON
                if np.abs(win[j] - target) / denom <= eps:
                    count += 1
        touches[i] = count
    return touches

@jit(nopython=True, cache=True)
def _cluster_density_jit(price: np.ndarray, clusters: np.ndarray, atr: np.ndarray, length: int) -> np.ndarray:
    n = len(price)
    density = np.zeros(n, dtype=np.float64)
    for i in range(length, n):
        c_target = clusters[i]
        if np.isnan(c_target): continue
        c_win = price[i-length:i]
        c_eps = atr[i] * 0.5
        count = 0
        for j in range(len(c_win)):
            if not np.isnan(c_win[j]) and (c_target - c_eps <= c_win[j] <= c_target + c_eps):
                count += 1
        density[i] = count
    return density

@jit(nopython=True, cache=True)
def _vol_profile_sr_jit(price: np.ndarray, vol: np.ndarray, length: int, bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(price)
    poc, vah, val = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    hvn, lvn = np.full(n, np.nan), np.full(n, np.nan)
    
    if n >= length:
        for i in range(length - 1, n):
            p_win = price[i - length + 1 : i + 1]
            v_win = vol[i - length + 1 : i + 1]
            p_min, p_max = _get_nanmin_nanmax(p_win)
            
            if np.isnan(p_min) or np.isnan(p_max) or p_min == p_max:
                poc[i] = vah[i] = val[i] = hvn[i] = lvn[i] = p_win[-1]
                continue
            
            bin_edges = np.linspace(p_min, p_max, bins + 1)
            vol_profile = np.zeros(bins)
            for j in range(length):
                b = 0
                while b < bins and p_win[j] >= bin_edges[b+1]: b += 1
                if b == bins: b -= 1
                vol_profile[b] += v_win[j]
                
            poc_idx = np.argmax(vol_profile)
            poc[i] = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2.0
            hvn[i] = poc[i]
            
            min_v = np.inf
            lvn_b = -1
            for b in range(bins):
                if vol_profile[b] > 0 and vol_profile[b] < min_v:
                    min_v = vol_profile[b]
                    lvn_b = b
            if lvn_b != -1:
                lvn[i] = (bin_edges[lvn_b] + bin_edges[lvn_b + 1]) / 2.0
            
            t_vol = np.sum(vol_profile) * 0.70
            c_vol = vol_profile[poc_idx]
            u, d_ = poc_idx + 1, poc_idx - 1
            while c_vol < t_vol and (u < bins or d_ >= 0):
                vu = vol_profile[u] if u < bins else -1.0
                vd = vol_profile[d_] if d_ >= 0 else -1.0
                if vu > vd:
                    c_vol += vu; u += 1
                else:
                    c_vol += vd; d_ -= 1
            vah[i] = bin_edges[min(u, bins)]
            val[i] = bin_edges[max(0, d_)]
            
    return poc, vah, val, hvn, lvn

def _shift_array(arr: np.ndarray, periods: int = 1, fill_val: float = np.nan) -> np.ndarray:
    """Safe NumPy array shift avoiding object overhead."""
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
# COMMON UTILITIES (Pure Extractors without Global Cache)
# ==============================================================================

def validate_length(length: int, name: str = "length") -> None:
    if not isinstance(length, int) or length <= 0:
        logger.error(f"Invalid window length for {name}: {length}")
        raise SupportResistanceError(f"Length for {name} must be a positive integer, got {length}")

def _finalize_output(output: Union[pd.Series, pd.DataFrame], offset: int, fillna: Any) -> Union[pd.Series, pd.DataFrame]:
    res = output.shift(offset) if offset != 0 else output
    if fillna is not None:
        res = res.fillna(fillna)
    return res

def _extract_ohlc(data: Union[pd.DataFrame, pd.Series]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    """Direct, pure extraction without unsafe global caching."""
    if isinstance(data, pd.Series):
        raise SupportResistanceError("OHLC DataFrame required.")
    df = standardize_column_names(data)
    validate_ohlc(df)
    return (df['open'].to_numpy(dtype=np.float64), df['high'].to_numpy(dtype=np.float64), 
           df['low'].to_numpy(dtype=np.float64), df['close'].to_numpy(dtype=np.float64), df.index)

def _extract_volume(data: Union[pd.DataFrame, pd.Series]) -> np.ndarray:
    if isinstance(data, pd.Series):
        raise SupportResistanceError("Volume extraction requires DataFrame.")
    df = standardize_column_names(data)
    if 'volume' not in df.columns:
        raise SupportResistanceError("DataFrame must contain a 'volume' column.")
    return df['volume'].to_numpy(dtype=np.float64)

def get_price_source(data: Union[pd.DataFrame, pd.Series], source: str = 'close') -> pd.Series:
    if isinstance(data, pd.Series): return data
    df = standardize_column_names(data)
    src = source.lower().strip()
    if src in df.columns: return df[src]
    if src in COMPUTED_SOURCES:
        func, req_cols = COMPUTED_SOURCES[src]
        return func(*[df[c] for c in req_cols])
    raise SupportResistanceError(f"Invalid source '{source}'.")

def _atr_proxy(data: pd.DataFrame, length: int = 14) -> np.ndarray:
    """Localized dynamic ATR calculation."""
    o, h, l, c, idx = _extract_ohlc(data)
    prev_c = _shift_array(c, 1)
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    return rolling_mean(pd.Series(tr, index=idx), length).ffill().to_numpy(dtype=np.float64)

# ==============================================================================
# PIVOT INDICATOR FUNCTIONS
# ==============================================================================

def classic_pivot_levels(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    o, h, l, c, idx = _extract_ohlc(data)
    prev_h, prev_l, prev_c = _shift_array(h, 1), _shift_array(l, 1), _shift_array(c, 1)
    pp = (prev_h + prev_l + prev_c) / 3.0
    r1, s1 = (2.0 * pp) - prev_l, (2.0 * pp) - prev_h
    r2, s2 = pp + (prev_h - prev_l), pp - (prev_h - prev_l)
    r3, s3 = prev_h + 2.0 * (pp - prev_l), prev_l - 2.0 * (prev_h - pp)
    out = pd.DataFrame({"R3": r3, "R2": r2, "R1": r1, "PP": pp, "S1": s1, "S2": s2, "S3": s3}, index=idx)
    return _finalize_output(out, offset, fillna)

def pivot_point(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return classic_pivot_levels(data, offset, fillna)['PP']
def pivot_support_1(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return classic_pivot_levels(data, offset, fillna)['S1']
def pivot_support_2(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return classic_pivot_levels(data, offset, fillna)['S2']
def pivot_support_3(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return classic_pivot_levels(data, offset, fillna)['S3']
def pivot_resistance_1(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return classic_pivot_levels(data, offset, fillna)['R1']
def pivot_resistance_2(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return classic_pivot_levels(data, offset, fillna)['R2']
def pivot_resistance_3(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return classic_pivot_levels(data, offset, fillna)['R3']

def floor_pivot(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return pivot_point(data, offset, fillna)
def floor_s1(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return pivot_support_1(data, offset, fillna)
def floor_s2(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return pivot_support_2(data, offset, fillna)
def floor_s3(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return pivot_support_3(data, offset, fillna)
def floor_r1(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return pivot_resistance_1(data, offset, fillna)
def floor_r2(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return pivot_resistance_2(data, offset, fillna)
def floor_r3(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return pivot_resistance_3(data, offset, fillna)

def _woodie_levels(data: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c, idx = _extract_ohlc(data)
    prev_h, prev_l, prev_c = _shift_array(h, 1), _shift_array(l, 1), _shift_array(c, 1)
    pp = (prev_h + prev_l + 2.0 * prev_c) / 4.0
    rng = prev_h - prev_l
    r1, s1 = (2.0 * pp) - prev_l, (2.0 * pp) - prev_h
    r2, s2 = pp + rng, pp - rng
    r3, s3 = prev_h + 2.0 * (pp - prev_l), prev_l - 2.0 * (prev_h - pp)
    return pd.DataFrame({"PP": pp, "R1": r1, "R2": r2, "R3": r3, "S1": s1, "S2": s2, "S3": s3}, index=idx)

def woodie_pivot(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return _finalize_output(_woodie_levels(data)['PP'], offset, fillna)
def woodie_r1(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return _finalize_output(_woodie_levels(data)['R1'], offset, fillna)
def woodie_r2(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return _finalize_output(_woodie_levels(data)['R2'], offset, fillna)
def woodie_r3(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return _finalize_output(_woodie_levels(data)['R3'], offset, fillna)
def woodie_s1(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return _finalize_output(_woodie_levels(data)['S1'], offset, fillna)
def woodie_s2(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return _finalize_output(_woodie_levels(data)['S2'], offset, fillna)
def woodie_s3(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return _finalize_output(_woodie_levels(data)['S3'], offset, fillna)

def _camarilla_levels(data: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c, idx = _extract_ohlc(data)
    prev_h, prev_l, prev_c = _shift_array(h, 1), _shift_array(l, 1), _shift_array(c, 1)
    rng = prev_h - prev_l
    pp = (prev_h + prev_l + prev_c) / 3.0
    h4, h3, h2, h1 = prev_c + (rng * 1.1)/2.0, prev_c + (rng * 1.1)/4.0, prev_c + (rng * 1.1)/6.0, prev_c + (rng * 1.1)/12.0
    l1, l2, l3, l4 = prev_c - (rng * 1.1)/12.0, prev_c - (rng * 1.1)/6.0, prev_c - (rng * 1.1)/4.0, prev_c - (rng * 1.1)/2.0
    h5 = h4 + 1.168 * (h4 - h3)
    l5 = prev_c - (h5 - prev_c)
    
    den = np.where(prev_l == 0, EPSILON, prev_l)
    h6 = safe_divide(prev_h, den, default=1.0) * prev_c
    l6 = prev_c - (h6 - prev_c)
    return pd.DataFrame({"PP": pp, "H1": h1, "H2": h2, "H3": h3, "H4": h4, "H5": h5, "H6": h6,
                         "L1": l1, "L2": l2, "L3": l3, "L4": l4, "L5": l5, "L6": l6}, index=idx)

def camarilla_pivot(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return _finalize_output(_camarilla_levels(data)['PP'], offset, fillna)
def camarilla_h1(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return _finalize_output(_camarilla_levels(data)['H1'], offset, fillna)
def camarilla_h2(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return _finalize_output(_camarilla_levels(data)['H2'], offset, fillna)
def camarilla_h3(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return _finalize_output(_camarilla_levels(data)['H3'], offset, fillna)
def camarilla_h4(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return _finalize_output(_camarilla_levels(data)['H4'], offset, fillna)
def camarilla_h5(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return _finalize_output(_camarilla_levels(data)['H5'], offset, fillna)
def camarilla_h6(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return _finalize_output(_camarilla_levels(data)['H6'], offset, fillna)
def camarilla_l1(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return _finalize_output(_camarilla_levels(data)['L1'], offset, fillna)
def camarilla_l2(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return _finalize_output(_camarilla_levels(data)['L2'], offset, fillna)
def camarilla_l3(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return _finalize_output(_camarilla_levels(data)['L3'], offset, fillna)
def camarilla_l4(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return _finalize_output(_camarilla_levels(data)['L4'], offset, fillna)
def camarilla_l5(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return _finalize_output(_camarilla_levels(data)['L5'], offset, fillna)
def camarilla_l6(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return _finalize_output(_camarilla_levels(data)['L6'], offset, fillna)

def _demark_levels(data: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c, idx = _extract_ohlc(data)
    prev_o, prev_h, prev_l, prev_c = _shift_array(o, 1), _shift_array(h, 1), _shift_array(l, 1), _shift_array(c, 1)
    x = np.where(prev_c < prev_o, prev_h + 2.0 * prev_l + prev_c,
                 np.where(prev_c > prev_o, 2.0 * prev_h + prev_l + prev_c, prev_h + prev_l + 2.0 * prev_c))
    return pd.DataFrame({"PP": x / 4.0, "R1": (x / 2.0) - prev_l, "S1": (x / 2.0) - prev_h}, index=idx)

def demark_pivot(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return _finalize_output(_demark_levels(data)['PP'], offset, fillna)
def demark_resistance(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return _finalize_output(_demark_levels(data)['R1'], offset, fillna)
def demark_support(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return _finalize_output(_demark_levels(data)['S1'], offset, fillna)

# ==============================================================================
# SWING & BREAKOUT FUNCTIONS
# ==============================================================================

def _get_swings(data: pd.DataFrame, length: int) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    o, h, l, c, idx = _extract_ohlc(data)
    sh, sl, lsh, lsl = _swing_levels_jit(h, l, length, length)
    return (pd.Series(sh, index=idx), pd.Series(sl, index=idx), pd.Series(lsh, index=idx), pd.Series(lsl, index=idx))

def swing_high(data: pd.DataFrame, length: int = DEFAULT_SWING_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    return _finalize_output(_get_swings(data, length)[0], offset, fillna)

def swing_low(data: pd.DataFrame, length: int = DEFAULT_SWING_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    return _finalize_output(_get_swings(data, length)[1], offset, fillna)

def highest_swing(data: pd.DataFrame, length: int = 20, swing_len: int = DEFAULT_SWING_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = swing_high(data, swing_len).rolling(window=length, min_periods=1).max().ffill()
    return _finalize_output(res, offset, fillna)

def lowest_swing(data: pd.DataFrame, length: int = 20, swing_len: int = DEFAULT_SWING_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = swing_low(data, swing_len).rolling(window=length, min_periods=1).min().ffill()
    return _finalize_output(res, offset, fillna)

def last_swing_high(data: pd.DataFrame, length: int = DEFAULT_SWING_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    return _finalize_output(_get_swings(data, length)[2].ffill(), offset, fillna)

def last_swing_low(data: pd.DataFrame, length: int = DEFAULT_SWING_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    return _finalize_output(_get_swings(data, length)[3].ffill(), offset, fillna)

def confirmed_swing_high(data: pd.DataFrame, length: int = DEFAULT_SWING_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    return _finalize_output(last_swing_high(data, length).shift(length).ffill(), offset, fillna)

def confirmed_swing_low(data: pd.DataFrame, length: int = DEFAULT_SWING_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    return _finalize_output(last_swing_low(data, length).shift(length).ffill(), offset, fillna)

def adaptive_swing(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    o, h, l, c, idx = _extract_ohlc(data)
    atr = _atr_proxy(data)
    volatility_ratio = safe_divide(atr, rolling_mean(pd.Series(atr, index=idx), 50).to_numpy(dtype=np.float64), default=1.0)
    length = pd.Series(np.where(volatility_ratio > 1.5, 20, np.where(volatility_ratio > 1.0, 10, 5)), index=idx)
    out = pd.DataFrame({
        "swing_2": last_swing_high(data, 2), "swing_5": last_swing_high(data, 5),
        "swing_10": last_swing_high(data, 10), "swing_20": last_swing_high(data, 20), "adaptive_len": length
    }, index=idx)
    return _finalize_output(out, offset, fillna)

def highest_high_breakout(data: pd.DataFrame, length: int = DEFAULT_CHANNEL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc(data)
    res = pd.Series(c, index=idx) > rolling_highest_high(data, length).shift(1)
    return _finalize_output(res.rename(f"HHB_{length}"), offset, fillna)

def lowest_low_breakdown(data: pd.DataFrame, length: int = DEFAULT_CHANNEL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc(data)
    res = pd.Series(c, index=idx) < rolling_lowest_low(data, length).shift(1)
    return _finalize_output(res.rename(f"LLB_{length}"), offset, fillna)

def previous_high_breakout(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc(data)
    res = pd.Series(c, index=idx) > pd.Series(h, index=idx).shift(1)
    return _finalize_output(res.rename("PrevHighBO"), offset, fillna)

def previous_low_breakdown(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc(data)
    res = pd.Series(c, index=idx) < pd.Series(l, index=idx).shift(1)
    return _finalize_output(res.rename("PrevLowBD"), offset, fillna)

def donchian_breakout(data: pd.DataFrame, length: int = DEFAULT_CHANNEL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = pd.Series(np.where(highest_high_breakout(data, length), 1, np.where(lowest_low_breakdown(data, length), -1, 0)), index=data.index)
    return _finalize_output(res.rename(f"DonchianBO_{length}"), offset, fillna)

def range_breakout(data: pd.DataFrame, length: int = DEFAULT_ZONE_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    r_high, r_low = rolling_highest_high(data, length).shift(1), rolling_lowest_low(data, length).shift(1)
    o, h, l, c, idx = _extract_ohlc(data)
    is_consolidation = (r_high - r_low) < (rolling_mean(pd.Series(h - l, index=idx), length).shift(1) * 1.5)
    bo = (pd.Series(c, index=idx) > r_high) | (pd.Series(c, index=idx) < r_low)
    return _finalize_output((is_consolidation & bo).rename(f"RangeBO_{length}"), offset, fillna)

# ==============================================================================
# FRACTAL & STRIDE/CHANNEL FUNCTIONS
# ==============================================================================

def bill_williams_fractal_high(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc(data)
    h_s = pd.Series(h, index=idx)
    is_fractal = (h_s > h_s.shift(1)) & (h_s > h_s.shift(2)) & (h_s > h_s.shift(-1)) & (h_s > h_s.shift(-2))
    return _finalize_output(pd.Series(np.where(is_fractal, h_s, np.nan), index=idx, name="FractalHigh"), offset, fillna)

def bill_williams_fractal_low(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc(data)
    l_s = pd.Series(l, index=idx)
    is_fractal = (l_s < l_s.shift(1)) & (l_s < l_s.shift(2)) & (l_s < l_s.shift(-1)) & (l_s < l_s.shift(-2))
    return _finalize_output(pd.Series(np.where(is_fractal, l_s, np.nan), index=idx, name="FractalLow"), offset, fillna)

def fractal_pivot(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    return _finalize_output(((bill_williams_fractal_high(data).ffill() + bill_williams_fractal_low(data).ffill()) / 2.0).rename("FractalPivot"), offset, fillna)

def fractal_support(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    return _finalize_output(bill_williams_fractal_low(data).ffill().rename("FractalSupport"), offset, fillna)

def fractal_resistance(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    return _finalize_output(bill_williams_fractal_high(data).ffill().rename("FractalResistance"), offset, fillna)

def rolling_lowest_low(data: pd.DataFrame, length: int = DEFAULT_ZONE_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc(data)
    return _finalize_output(pd.Series(l, index=idx).rolling(length).min().rename(f"RLL_{length}"), offset, fillna)

def rolling_highest_high(data: pd.DataFrame, length: int = DEFAULT_ZONE_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc(data)
    return _finalize_output(pd.Series(h, index=idx).rolling(length).max().rename(f"RHH_{length}"), offset, fillna)

def dynamic_support(data: pd.DataFrame, length: int = DEFAULT_ZONE_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    return _finalize_output(rolling_lowest_low(data, length).ewm(span=length, adjust=False).mean().rename(f"DynSup_{length}"), offset, fillna)

def dynamic_resistance(data: pd.DataFrame, length: int = DEFAULT_ZONE_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    return _finalize_output(rolling_highest_high(data, length).ewm(span=length, adjust=False).mean().rename(f"DynRes_{length}"), offset, fillna)

def adaptive_support(data: pd.DataFrame, length: int = DEFAULT_ZONE_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    df = standardize_column_names(data)
    res = rolling_sum(df['low'] * df['volume'], length) / rolling_sum(df['volume'], length) if 'volume' in df.columns else rolling_mean(df['low'], length)
    return _finalize_output(res.rename(f"AdpSup_{length}"), offset, fillna)

def adaptive_resistance(data: pd.DataFrame, length: int = DEFAULT_ZONE_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    df = standardize_column_names(data)
    res = rolling_sum(df['high'] * df['volume'], length) / rolling_sum(df['volume'], length) if 'volume' in df.columns else rolling_mean(df['high'], length)
    return _finalize_output(res.rename(f"AdpRes_{length}"), offset, fillna)

def donchian_channel(data: pd.DataFrame, length: int = DEFAULT_CHANNEL_LEN, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    upper, lower = rolling_highest_high(data, length), rolling_lowest_low(data, length)
    return _finalize_output(pd.DataFrame({"upper": upper, "middle": (upper + lower) / 2.0, "lower": lower}, index=data.index), offset, fillna)

def price_channel(data: pd.DataFrame, length: int = DEFAULT_CHANNEL_LEN, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    return donchian_channel(data, length, offset, fillna)

def highest_channel(data: pd.DataFrame, length: int = DEFAULT_CHANNEL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    return donchian_channel(data, length, offset, fillna)['upper']

def lowest_channel(data: pd.DataFrame, length: int = DEFAULT_CHANNEL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    return donchian_channel(data, length, offset, fillna)['lower']

def channel_midline(data: pd.DataFrame, length: int = DEFAULT_CHANNEL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    return donchian_channel(data, length, offset, fillna)['middle']

# ==============================================================================
# ZONE & ORDER FLOW FUNCTIONS
# ==============================================================================

def support_zone(data: pd.DataFrame, length: int = DEFAULT_SWING_LEN, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    lsl = last_swing_low(data, length)
    atr = pd.Series(_atr_proxy(data), index=data.index)
    return _finalize_output(pd.DataFrame({"zone_upper": lsl + (atr * 0.2), "zone_lower": lsl}, index=data.index), offset, fillna)

def resistance_zone(data: pd.DataFrame, length: int = DEFAULT_SWING_LEN, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    lsh = last_swing_high(data, length)
    atr = pd.Series(_atr_proxy(data), index=data.index)
    return _finalize_output(pd.DataFrame({"zone_upper": lsh, "zone_lower": lsh - (atr * 0.2)}, index=data.index), offset, fillna)

def demand_zone(data: pd.DataFrame, length: int = DEFAULT_ZONE_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc(data)
    strong_move = pd.Series((c > o) & ((h - l) > _atr_proxy(data) * 1.5), index=idx)
    return _finalize_output(pd.Series(np.where(strong_move, pd.Series(l, index=idx).shift(1), np.nan), index=idx).ffill().rename(f"DemandZone_{length}"), offset, fillna)

def supply_zone(data: pd.DataFrame, length: int = DEFAULT_ZONE_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc(data)
    strong_move = pd.Series((c < o) & ((h - l) > _atr_proxy(data) * 1.5), index=idx)
    return _finalize_output(pd.Series(np.where(strong_move, pd.Series(h, index=idx).shift(1), np.nan), index=idx).ffill().rename(f"SupplyZone_{length}"), offset, fillna)

def reaction_zone(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    return _finalize_output(pd.DataFrame({"demand": demand_zone(data), "supply": supply_zone(data)}, index=data.index), offset, fillna)

def congestion_zone(data: pd.DataFrame, length: int = DEFAULT_ZONE_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc(data)
    res = (rolling_highest_high(data, length) - rolling_lowest_low(data, length)) < rolling_mean(pd.Series(h - l, index=idx), length * 2).shift(1)
    return _finalize_output(res.rename(f"CongestionZone_{length}"), offset, fillna)

# ==============================================================================
# SMART MONEY CONCEPTS (SMC): FVG, BPR, OB LIFECYCLE
# ==============================================================================

def bullish_fvg(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    o, h, l, c, idx = _extract_ohlc(data)
    h_shift, l_s = _shift_array(h, 2), pd.Series(l, index=idx)
    atr = pd.Series(_atr_proxy(data), index=idx)
    is_fvg = (l_s > h_shift) & ((l_s - h_shift) > (atr * 0.1)) 
    return _finalize_output(pd.DataFrame({"is_fvg": is_fvg, "top": np.where(is_fvg, l_s, np.nan), "bottom": np.where(is_fvg, h_shift, np.nan)}, index=idx), offset, fillna)

def bearish_fvg(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    o, h, l, c, idx = _extract_ohlc(data)
    l_shift, h_s = _shift_array(l, 2), pd.Series(h, index=idx)
    atr = pd.Series(_atr_proxy(data), index=idx)
    is_fvg = (h_s < l_shift) & ((l_shift - h_s) > (atr * 0.1))
    return _finalize_output(pd.DataFrame({"is_fvg": is_fvg, "top": np.where(is_fvg, l_shift, np.nan), "bottom": np.where(is_fvg, h_s, np.nan)}, index=idx), offset, fillna)

def mitigated_fvg(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    o, h, l, c, idx = _extract_ohlc(data)
    b_fvg, br_fvg = bullish_fvg(data), bearish_fvg(data)
    fvg_top = np.where(b_fvg['is_fvg'], b_fvg['top'], np.where(br_fvg['is_fvg'], br_fvg['top'], np.nan))
    fvg_bot = np.where(b_fvg['is_fvg'], b_fvg['bottom'], np.where(br_fvg['is_fvg'], br_fvg['bottom'], np.nan))
    mitig, un_top, un_bot = _fvg_mitigation_jit(h, l, fvg_top, fvg_bot, b_fvg['is_fvg'].to_numpy(dtype=np.bool_))
    out = pd.DataFrame({"is_mitigated": mitig, "unfilled_top": un_top, "unfilled_bottom": un_bot, "is_active": ~pd.Series(mitig, index=idx).fillna(False)}, index=idx)
    return _finalize_output(out, offset, fillna)

def balanced_price_range(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    b_fvg, br_fvg = bullish_fvg(data), bearish_fvg(data)
    bpr = (b_fvg['is_fvg'].shift(1) & br_fvg['is_fvg']) | (br_fvg['is_fvg'].shift(1) & b_fvg['is_fvg'])
    return _finalize_output(bpr.rename("BPR"), offset, fillna)

def order_blocks(data: pd.DataFrame, length: int = DEFAULT_ZONE_LEN, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    """True SMC Order Blocks requiring FVG Displacement + Break of Structure."""
    o, h, l, c, idx = _extract_ohlc(data)
    bos = market_structure_shift(data, length=length) 
    fvg_bull, fvg_bear = bullish_fvg(data)['is_fvg'], bearish_fvg(data)['is_fvg']
    
    strong_bull = bos['mss_bullish'] & fvg_bull
    strong_bear = bos['mss_bearish'] & fvg_bear
    
    ob_h = np.where(strong_bull, _shift_array(h, 1), np.where(strong_bear, _shift_array(h, 1), np.nan))
    ob_l = np.where(strong_bull, _shift_array(l, 1), np.where(strong_bear, _shift_array(l, 1), np.nan))
    
    fresh, mitigated, invalidated = _ob_lifecycle_jit(ob_h, ob_l, h, l, strong_bull.to_numpy(dtype=np.bool_))
    out = pd.DataFrame({"fresh_ob": fresh, "mitigated_ob": mitigated, "invalidated_ob": invalidated, "ob_high": ob_h, "ob_low": ob_l, "is_bullish": strong_bull.to_numpy()}, index=idx)
    return _finalize_output(out, offset, fillna)

def breaker_block(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    return _finalize_output(order_blocks(data)['invalidated_ob'].rename("BreakerBlock"), offset, fillna)

def flip_zone(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    return breaker_block(data, offset=offset, fillna=fillna).rename("FlipZone")

def smc_structure(data: pd.DataFrame, length: int = 5, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    o, h, l, c, idx = _extract_ohlc(data)
    sh, sl, c_s = confirmed_swing_high(data, length).to_numpy(), confirmed_swing_low(data, length).to_numpy(), pd.Series(c, index=idx)
    trend = pd.Series(np.where(c_s > sh, 1, np.where(c_s < sl, -1, np.nan)), index=idx).ffill()
    out = pd.DataFrame({"bos_up": (trend == 1) & (c_s > sh), "bos_dn": (trend == -1) & (c_s < sl),
                         "choch_up": (trend.shift(1) == -1) & (c_s > sh), "choch_dn": (trend.shift(1) == 1) & (c_s < sl), "trend": trend}, index=idx)
    return _finalize_output(out, offset, fillna)

def bos_level(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    smc = smc_structure(data)
    return _finalize_output(pd.Series(np.where(smc['bos_up'], 1, np.where(smc['bos_dn'], -1, 0)), index=data.index, name="BOS"), offset, fillna)

def choch_level(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    smc = smc_structure(data)
    return _finalize_output(pd.Series(np.where(smc['choch_up'], 1, np.where(smc['choch_dn'], -1, 0)), index=data.index, name="CHOCH"), offset, fillna)

def market_structure_shift(data: pd.DataFrame, length: int = 5, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    smc = smc_structure(data, length)
    has_fvg = bullish_fvg(data)['is_fvg'] | bearish_fvg(data)['is_fvg']
    return _finalize_output(pd.DataFrame({"mss_bullish": smc['bos_up'] & has_fvg, "mss_bearish": smc['bos_dn'] & has_fvg}, index=data.index), offset, fillna)

def premium_discount_zone(data: pd.DataFrame, length: int = 20, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    rhh, rll = rolling_highest_high(data, length), rolling_lowest_low(data, length)
    return _finalize_output(pd.DataFrame({"premium_level": rhh, "equilibrium": (rhh + rll) / 2.0, "discount_level": rll}, index=data.index), offset, fillna)

def inducement_level(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    return last_swing_low(data, length=2, offset=offset, fillna=fillna).rename("Inducement")

# ==============================================================================
# LIQUIDITY & ADVANCED SMC PATTERNS
# ==============================================================================

def buy_side_liquidity(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    sh = swing_high(data).ffill()
    atr = _atr_proxy(data)
    eqh, _, _ = _equal_cluster_jit(sh.to_numpy(), atr, 0.1) 
    return _finalize_output(pd.Series(np.where(eqh, sh, np.nan), index=data.index).ffill().rename("BSL"), offset, fillna)

def sell_side_liquidity(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    sl = swing_low(data).ffill()
    atr = _atr_proxy(data)
    eql, _, _ = _equal_cluster_jit(sl.to_numpy(), atr, 0.1)
    return _finalize_output(pd.Series(np.where(eql, sl, np.nan), index=data.index).ffill().rename("SSL"), offset, fillna)

def internal_liquidity(data: pd.DataFrame, length: int = 5, offset: int = 0, fillna: Any = None) -> pd.Series:
    return last_swing_low(data, length=length, offset=offset, fillna=fillna).rename("InternalLiq")

def external_liquidity(data: pd.DataFrame, length: int = 20, offset: int = 0, fillna: Any = None) -> pd.Series:
    return rolling_lowest_low(data, length=length, offset=offset, fillna=fillna).rename("ExternalLiq")

def equal_highs(data: pd.DataFrame, epsilon_mult: float = 0.1, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    sh = swing_high(data, length=5).to_numpy()
    atr = _atr_proxy(data)
    is_eq, c_size, c_str = _equal_cluster_jit(sh, atr, epsilon_mult)
    return _finalize_output(pd.DataFrame({"is_equal": is_eq, "cluster_size": c_size, "cluster_strength": c_str}, index=data.index), offset, fillna)

def equal_lows(data: pd.DataFrame, epsilon_mult: float = 0.1, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    sl = swing_low(data, length=5).to_numpy()
    atr = _atr_proxy(data)
    is_eq, c_size, c_str = _equal_cluster_jit(sl, atr, epsilon_mult)
    return _finalize_output(pd.DataFrame({"is_equal": is_eq, "cluster_size": c_size, "cluster_strength": c_str}, index=data.index), offset, fillna)

def liquidity_high(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return highest_swing(data, length=50, offset=offset, fillna=fillna).rename("LiqHigh")
def liquidity_low(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return lowest_swing(data, length=50, offset=offset, fillna=fillna).rename("LiqLow")
def liquidity_pool(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return _finalize_output((equal_highs(data)['is_equal'] | equal_lows(data)['is_equal']).rename("LiqPool"), offset, fillna)
def institutional_resistance(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return liquidity_high(data, offset=offset, fillna=fillna).rename("InstRes")
def institutional_support(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return liquidity_low(data, offset=offset, fillna=fillna).rename("InstSup")

def liquidity_sweep_level(data: pd.DataFrame, length: int = 20, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc(data)
    rh, rl = rolling_highest_high(data, length).shift(1), rolling_lowest_low(data, length).shift(1)
    res = ((pd.Series(h, index=idx) > rh) & (pd.Series(c, index=idx) < rh)) | ((pd.Series(l, index=idx) < rl) & (pd.Series(c, index=idx) > rl))
    return _finalize_output(res.rename(f"LiqSweep_{length}"), offset, fillna)

def stop_hunt_zone(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return liquidity_sweep_level(data).rename("StopHuntZone")
def smart_money_level(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return liquidity_pool(data, offset=offset, fillna=fillna).rename("SmartMoneyLevel")

def turtle_soup(data: pd.DataFrame, length: int = 20, offset: int = 0, fillna: Any = None) -> pd.Series:
    """Sweep of 20-period High/Low, then immediate reversal (close back inside)."""
    sweep = liquidity_sweep_level(data, length=length)
    o, h, l, c, idx = _extract_ohlc(data)
    rh, rl = rolling_highest_high(data, length).shift(1), rolling_lowest_low(data, length).shift(1)
    
    soup_short = sweep & (pd.Series(h, index=idx) > rh) & (pd.Series(c, index=idx) < pd.Series(o, index=idx))
    soup_long = sweep & (pd.Series(l, index=idx) < rl) & (pd.Series(c, index=idx) > pd.Series(o, index=idx))
    return _finalize_output((soup_short | soup_long).rename("TurtleSoup"), offset, fillna)

def judas_swing(data: pd.DataFrame, length: int = 10, offset: int = 0, fillna: Any = None) -> pd.Series:
    """False initial breakout masking the true institutional direction."""
    return false_breakout(data, length=length, offset=offset, fillna=fillna).rename("JudasSwing")

def kill_zones(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    """Time-based Kill Zones (Asian, London, NY) for Forex/Crypto."""
    df = standardize_column_names(data)
    validate_datetime_index(df)
    times = df.index.strftime('%H:%M')
    asian = (times >= '20:00') | (times <= '00:00')
    london = (times >= '02:00') & (times <= '05:00')
    ny = (times >= '07:00') & (times <= '10:00')
    out = pd.DataFrame({"Asian_KZ": asian, "London_KZ": london, "NY_KZ": ny}, index=df.index)
    return _finalize_output(out, offset, fillna)

def amd_cycle(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    """Accumulation, Manipulation, Distribution cycle proxy using Session profile and Swings."""
    ib = initial_balance(data)
    csh, csl = confirmed_swing_high(data), confirmed_swing_low(data)
    o, h, l, c, idx = _extract_ohlc(data)
    
    accum = (ib['IB_High'] - ib['IB_Low']) < (_atr_proxy(data) * 1.5)
    manip = turtle_soup(data).to_numpy()
    distrib = (pd.Series(c, index=idx) > csh) | (pd.Series(c, index=idx) < csl)
    
    out = pd.DataFrame({"Accumulation": accum, "Manipulation": manip, "Distribution": distrib}, index=idx)
    return _finalize_output(out, offset, fillna)

# ==============================================================================
# TOUCH COUNT & STRENGTH SYSTEMS
# ==============================================================================

def support_touch_count(data: pd.DataFrame, length: int = 50, epsilon_mult: float = 0.1, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc(data)
    eps = _atr_proxy(data) * epsilon_mult
    touches = _touch_count_jit(l, last_swing_low(data, 5).to_numpy(), eps, length)
    return _finalize_output(pd.Series(touches, index=idx, name="SupTouches"), offset, fillna)

def resistance_touch_count(data: pd.DataFrame, length: int = 50, epsilon_mult: float = 0.1, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc(data)
    eps = _atr_proxy(data) * epsilon_mult
    touches = _touch_count_jit(h, last_swing_high(data, 5).to_numpy(), eps, length)
    return _finalize_output(pd.Series(touches, index=idx, name="ResTouches"), offset, fillna)

def level_strength(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    return _finalize_output((support_touch_count(data) + resistance_touch_count(data)).rename("LevelStrength"), offset, fillna)

def level_confidence(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    return _finalize_output(np.clip(level_strength(data) / 5.0, 0.0, 1.0).rename("LevelConfidence"), offset, fillna)

def zone_width(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    sz = support_zone(data)
    return _finalize_output((sz['zone_upper'] - sz['zone_lower']).rename("ZoneWidth"), offset, fillna)

def bounce_count(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    return level_strength(data, offset=offset, fillna=fillna).rename("BounceCount")

# ==============================================================================
# DISTANCE & RETEST VALIDATIONS
# ==============================================================================

def distance_to_support(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc(data)
    return _finalize_output((pd.Series(c, index=idx) - last_swing_low(data, 5)).rename("DistToSup"), offset, fillna)

def distance_to_resistance(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc(data)
    return _finalize_output((last_swing_high(data, 5) - pd.Series(c, index=idx)).rename("DistToRes"), offset, fillna)

def nearest_support(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return last_swing_low(data, length=5, offset=offset, fillna=fillna)
def nearest_resistance(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return last_swing_high(data, length=5, offset=offset, fillna=fillna)
def risk_distance(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return distance_to_support(data, offset=offset, fillna=fillna)
def reward_distance(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return distance_to_resistance(data, offset=offset, fillna=fillna)

def confirmed_breakout(data: pd.DataFrame, length: int = DEFAULT_CHANNEL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc(data)
    body = np.abs(c - o)
    res = highest_high_breakout(data, length) & (pd.Series(body, index=idx) > rolling_mean(pd.Series(body, index=idx), length).shift(1))
    return _finalize_output(res.rename(f"ConfBO_{length}"), offset, fillna)

def confirmed_breakdown(data: pd.DataFrame, length: int = DEFAULT_CHANNEL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc(data)
    body = np.abs(c - o)
    res = lowest_low_breakdown(data, length) & (pd.Series(body, index=idx) > rolling_mean(pd.Series(body, index=idx), length).shift(1))
    return _finalize_output(res.rename(f"ConfBD_{length}"), offset, fillna)

def false_breakout(data: pd.DataFrame, length: int = DEFAULT_CHANNEL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc(data)
    prev = rolling_highest_high(data, length).shift(1)
    return _finalize_output(((pd.Series(h, index=idx) > prev) & (pd.Series(c, index=idx) <= prev)).rename(f"FalseBO_{length}"), offset, fillna)

def false_breakdown(data: pd.DataFrame, length: int = DEFAULT_CHANNEL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc(data)
    prev = rolling_lowest_low(data, length).shift(1)
    return _finalize_output(((pd.Series(l, index=idx) < prev) & (pd.Series(c, index=idx) >= prev)).rename(f"FalseBD_{length}"), offset, fillna)

def retest_level(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc(data)
    csh, csl = confirmed_swing_high(data, 5), confirmed_swing_low(data, 5)
    eps = pd.Series(_atr_proxy(data), index=idx) * 0.1
    t_sh = (pd.Series(l, index=idx) <= csh + eps) & (pd.Series(c, index=idx) >= csh)
    t_sl = (pd.Series(h, index=idx) >= csl - eps) & (pd.Series(c, index=idx) <= csl)
    return _finalize_output((t_sh | t_sl).rename("Retest"), offset, fillna)

def break_strength(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc(data)
    body = np.abs(c - o)
    return _finalize_output(pd.Series(safe_divide(body, rolling_mean(pd.Series(body, index=idx), 20).shift(1).to_numpy(dtype=np.float64), default=1.0), index=idx, name="BreakStrength"), offset, fillna)

# ==============================================================================
# STATISTICAL & VOLATILITY ADJUSTED LEVELS
# ==============================================================================

def rolling_support(data: pd.DataFrame, length: int = DEFAULT_ZONE_LEN, offset: int = 0, fillna: Any = None) -> pd.Series: return rolling_lowest_low(data, length=length, offset=offset, fillna=fillna).rename("RollSup")
def rolling_resistance(data: pd.DataFrame, length: int = DEFAULT_ZONE_LEN, offset: int = 0, fillna: Any = None) -> pd.Series: return rolling_highest_high(data, length=length, offset=offset, fillna=fillna).rename("RollRes")

def support_percentile(data: pd.DataFrame, length: int = 252, offset: int = 0, fillna: Any = None) -> pd.Series:
    return _finalize_output((rolling_support(data, length=20).rolling(window=length).rank(pct=True) * 100.0).rename(f"SupPct_{length}"), offset, fillna)

def resistance_percentile(data: pd.DataFrame, length: int = 252, offset: int = 0, fillna: Any = None) -> pd.Series:
    return _finalize_output((rolling_resistance(data, length=20).rolling(window=length).rank(pct=True) * 100.0).rename(f"ResPct_{length}"), offset, fillna)

def support_zscore(data: pd.DataFrame, length: int = 50, offset: int = 0, fillna: Any = None) -> pd.Series:
    sup = rolling_support(data, length=20)
    z = safe_divide((sup - rolling_mean(sup, length)).to_numpy(dtype=np.float64), rolling_std(sup, length).to_numpy(dtype=np.float64), default=np.nan)
    return _finalize_output(pd.Series(z, index=data.index, name=f"SupZ_{length}"), offset, fillna)

def resistance_zscore(data: pd.DataFrame, length: int = 50, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = rolling_resistance(data, length=20)
    z = safe_divide((res - rolling_mean(res, length)).to_numpy(dtype=np.float64), rolling_std(res, length).to_numpy(dtype=np.float64), default=np.nan)
    return _finalize_output(pd.Series(z, index=data.index, name=f"ResZ_{length}"), offset, fillna)

def atr_support(data: pd.DataFrame, length: int = 14, mult: float = 2.0, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc(data)
    return _finalize_output((pd.Series(c, index=idx) - (pd.Series(_atr_proxy(data), index=idx) * mult)).rename(f"ATRSup_{length}_{mult}"), offset, fillna)

def atr_resistance(data: pd.DataFrame, length: int = 14, mult: float = 2.0, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc(data)
    return _finalize_output((pd.Series(c, index=idx) + (pd.Series(_atr_proxy(data), index=idx) * mult)).rename(f"ATRRes_{length}_{mult}"), offset, fillna)

def volatility_support(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return atr_support(data, offset=offset, fillna=fillna).rename("VolSup")
def volatility_resistance(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return atr_resistance(data, offset=offset, fillna=fillna).rename("VolRes")
def adaptive_breakout(data: pd.DataFrame, length: int = 20, mult: float = 2.0, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    return _finalize_output(pd.DataFrame({"breakout_upper": atr_resistance(data, length, mult), "breakout_lower": atr_support(data, length, mult)}, index=data.index), offset, fillna)

def daily_levels(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.DataFrame: return donchian_channel(data, length=1, offset=offset, fillna=fillna)
def weekly_levels(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.DataFrame: return donchian_channel(data, length=DEFAULT_MTF_W, offset=offset, fillna=fillna)
def monthly_levels(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.DataFrame: return donchian_channel(data, length=DEFAULT_MTF_M, offset=offset, fillna=fillna)
def yearly_levels(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.DataFrame: return donchian_channel(data, length=DEFAULT_MTF_Y, offset=offset, fillna=fillna)

def merged_support(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = np.maximum(daily_levels(data)['lower'].to_numpy(), np.maximum(weekly_levels(data)['lower'].to_numpy(), monthly_levels(data)['lower'].to_numpy()))
    return _finalize_output(pd.Series(res, index=data.index, name="MergedSupport"), offset, fillna)

def merged_resistance(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = np.minimum(daily_levels(data)['upper'].to_numpy(), np.minimum(weekly_levels(data)['upper'].to_numpy(), monthly_levels(data)['upper'].to_numpy()))
    return _finalize_output(pd.Series(res, index=data.index, name="MergedResistance"), offset, fillna)

def price_cluster(data: pd.DataFrame, length: int = 50, bins: int = DEFAULT_CLUSTER_BINS, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc(data)
    return _finalize_output(pd.Series(_price_cluster_jit((h + l + c) / 3.0, length, bins), index=idx, name=f"PriceCluster_{length}"), offset, fillna)

def cluster_support(data: pd.DataFrame, length: int = 50, bins: int = DEFAULT_CLUSTER_BINS, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc(data)
    return _finalize_output((price_cluster(data, length, bins) - (pd.Series(_atr_proxy(data), index=idx) * 0.5)).rename("ClusterSup"), offset, fillna)

def cluster_resistance(data: pd.DataFrame, length: int = 50, bins: int = DEFAULT_CLUSTER_BINS, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc(data)
    return _finalize_output((price_cluster(data, length, bins) + (pd.Series(_atr_proxy(data), index=idx) * 0.5)).rename("ClusterRes"), offset, fillna)

def cluster_density(data: pd.DataFrame, length: int = 50, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, idx = _extract_ohlc(data)
    density = _cluster_density_jit(c, price_cluster(data, length).to_numpy(), _atr_proxy(data), length)
    return _finalize_output(pd.Series(density, index=idx, name="ClusterDensity"), offset, fillna)

def cluster_strength(data: pd.DataFrame, length: int = 50, offset: int = 0, fillna: Any = None) -> pd.Series:
    return _finalize_output((cluster_density(data, length) / length).rename("ClusterStrength"), offset, fillna)

def market_structure_high(data: pd.DataFrame, length: int = 5, offset: int = 0, fillna: Any = None) -> pd.Series: return swing_high(data, length=length, offset=offset, fillna=fillna).rename("MSH")
def market_structure_low(data: pd.DataFrame, length: int = 5, offset: int = 0, fillna: Any = None) -> pd.Series: return swing_low(data, length=length, offset=offset, fillna=fillna).rename("MSL")
def structure_resistance(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return market_structure_high(data, offset=offset, fillna=fillna).ffill().rename("StructRes")
def structure_support(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return market_structure_low(data, offset=offset, fillna=fillna).ffill().rename("StructSup")
def higher_high(data: pd.DataFrame, length: int = 5, offset: int = 0, fillna: Any = None) -> pd.Series: return _finalize_output((confirmed_swing_high(data, length) > confirmed_swing_high(data, length).shift(1)).rename("HH"), offset, fillna)
def lower_low(data: pd.DataFrame, length: int = 5, offset: int = 0, fillna: Any = None) -> pd.Series: return _finalize_output((confirmed_swing_low(data, length) < confirmed_swing_low(data, length).shift(1)).rename("LL"), offset, fillna)
def higher_low(data: pd.DataFrame, length: int = 5, offset: int = 0, fillna: Any = None) -> pd.Series: return _finalize_output((confirmed_swing_low(data, length) > confirmed_swing_low(data, length).shift(1)).rename("HL"), offset, fillna)
def lower_high(data: pd.DataFrame, length: int = 5, offset: int = 0, fillna: Any = None) -> pd.Series: return _finalize_output((confirmed_swing_high(data, length) < confirmed_swing_high(data, length).shift(1)).rename("LH"), offset, fillna)

# ==============================================================================
# ADVANCED MARKET PROFILE & ANCHORED VWAP
# ==============================================================================

def adaptive_volume_profile(data: pd.DataFrame, length: int = 20, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    vol = _extract_volume(data)
    o, h, l, c, idx = _extract_ohlc(data)
    price = (h + l + c) / 3.0
    poc, vah, val = _adaptive_vol_profile_jit(price, vol, length, _atr_proxy(data))
    return _finalize_output(pd.DataFrame({"POC": poc, "VAH": vah, "VAL": val}, index=idx), offset, fillna)

def swing_high_vwap(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    if 'anchored_vwap' not in globals():
        from backend.indicators.volume import anchored_vwap
    csh = confirmed_swing_high(data)
    return _finalize_output(anchored_vwap(data, anchor=(csh != csh.shift(1)) & (~csh.isna())).rename("SH_VWAP"), offset, fillna)

def swing_low_vwap(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    if 'anchored_vwap' not in globals():
        from backend.indicators.volume import anchored_vwap
    csl = confirmed_swing_low(data)
    return _finalize_output(anchored_vwap(data, anchor=(csl != csl.shift(1)) & (~csl.isna())).rename("SL_VWAP"), offset, fillna)

# ==============================================================================
# FIBONACCI, GANN, MURREY MATH & PSYCHOLOGICAL LEVELS
# ==============================================================================

def fibonacci_levels(data: pd.DataFrame, length: int = 20, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    rhh, rll = rolling_highest_high(data, length), rolling_lowest_low(data, length)
    diff = rhh - rll
    return _finalize_output(pd.DataFrame({"Fib_0": rll, "Fib_236": rll + diff * 0.236, "Fib_382": rll + diff * 0.382,
                                          "Fib_500": rll + diff * 0.500, "Fib_618": rll + diff * 0.618, "Fib_786": rll + diff * 0.786,
                                          "Fib_100": rhh, "GoldenZone_Upper": rll + diff * 0.618, "GoldenZone_Lower": rll + diff * 0.500}, index=data.index), offset, fillna)

def gann_levels(data: pd.DataFrame, length: int = 20, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    rhh, rll = rolling_highest_high(data, length), rolling_lowest_low(data, length)
    diff = rhh - rll
    return _finalize_output(pd.DataFrame({f"Gann_{i}_8": rll + diff * (i / 8.0) for i in range(9)}, index=data.index), offset, fillna)

def murrey_math_levels(data: pd.DataFrame, length: int = 64, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    rhh, rll = rolling_highest_high(data, length).to_numpy(), rolling_lowest_low(data, length).to_numpy()
    diff = np.where((rhh - rll) == 0, EPSILON, rhh - rll)
    shift = np.log10(diff)
    sf = 10 ** np.floor(shift)
    m_range = sf * 0.125
    octave = np.floor(rll / m_range) * m_range
    return _finalize_output(pd.DataFrame({f"MM_{i}_8": octave + m_range * i for i in range(9)}, index=data.index), offset, fillna)

def psychological_levels(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    c = get_price_source(data).to_numpy()
    intervals = np.array([10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000], dtype=np.float64)
    atr = _atr_proxy(data)
    res = np.zeros(len(c), dtype=np.float64)
    for i in range(len(c)):
        if np.isnan(c[i]) or np.isnan(atr[i]): 
            res[i] = np.nan
            continue
        target_interval = max(atr[i] * 5.0, EPSILON)
        step = intervals[np.argmin(np.abs(intervals - target_interval))]
        res[i] = np.round(c[i] / step) * step
    return _finalize_output(pd.Series(res, index=data.index, name="PsychLevel"), offset, fillna)

def gap_levels(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    o, h, l, c, idx = _extract_ohlc(data)
    prev_c = pd.Series(c, index=idx).shift(1).to_numpy()
    return _finalize_output(pd.DataFrame({"gap_support": np.where(o > prev_c, prev_c, np.nan), "gap_resistance": np.where(o < prev_c, prev_c, np.nan)}, index=idx).ffill(), offset, fillna)

def previous_session_levels(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    df = standardize_column_names(data)
    validate_datetime_index(df)
    return _finalize_output(pd.DataFrame({"PDH": df['high'].groupby(df.index.date).transform('max').shift(1), "PDL": df['low'].groupby(df.index.date).transform('min').shift(1)}, index=df.index), offset, fillna)

# ==============================================================================
# ADVANCED SMC PATTERNS (Turtle Soup, Judas Swing, Kill Zones, AMD)
# ==============================================================================

def turtle_soup(data: pd.DataFrame, length: int = 20, offset: int = 0, fillna: Any = None) -> pd.Series:
    """Sweep of 20-period High/Low, then immediate reversal (close back inside)."""
    sweep = liquidity_sweep_level(data, length=length)
    o, h, l, c, idx = _extract_ohlc(data)
    rh = rolling_highest_high(data, length).shift(1).to_numpy(dtype=np.float64)
    rl = rolling_lowest_low(data, length).shift(1).to_numpy(dtype=np.float64)
    
    soup_short = sweep.to_numpy() & (h > rh) & (c < o)
    soup_long = sweep.to_numpy() & (l < rl) & (c > o)
    return _finalize_output(pd.Series(soup_short | soup_long, index=idx, name="TurtleSoup"), offset, fillna)

def judas_swing(data: pd.DataFrame, length: int = 10, offset: int = 0, fillna: Any = None) -> pd.Series:
    """False initial breakout masking the true institutional direction."""
    return false_breakout(data, length=length, offset=offset, fillna=fillna).rename("JudasSwing")

def kill_zones(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    """Time-based Kill Zones (Asian, London, NY) for Forex/Crypto/Indices."""
    df = standardize_column_names(data)
    validate_datetime_index(df)
    times = df.index.strftime('%H:%M')
    asian = (times >= '20:00') | (times <= '00:00')
    london = (times >= '02:00') & (times <= '05:00')
    ny = (times >= '07:00') & (times <= '10:00')
    out = pd.DataFrame({"Asian_KZ": asian, "London_KZ": london, "NY_KZ": ny}, index=df.index)
    return _finalize_output(out, offset, fillna)

def amd_cycle(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    """Accumulation, Manipulation, Distribution cycle proxy."""
    ib = initial_balance(data)
    csh, csl = confirmed_swing_high(data).to_numpy(), confirmed_swing_low(data).to_numpy()
    o, h, l, c, idx = _extract_ohlc(data)
    
    accum = (ib['IB_High'] - ib['IB_Low']).to_numpy() < (_atr_proxy(data) * 1.5)
    manip = turtle_soup(data).to_numpy()
    distrib = (c > csh) | (c < csl)
    
    out = pd.DataFrame({"Accumulation": accum, "Manipulation": manip, "Distribution": distrib}, index=idx)
    return _finalize_output(out, offset, fillna)


# ==============================================================================
# 19. THE ULTIMATE CONFLUENCE & RELIABILITY ENGINE
# ==============================================================================

def institutional_confluence_score(data: pd.DataFrame, epsilon_pct: float = 0.005, offset: int = 0, fillna: Any = None) -> pd.Series:
    """
    Evaluates 10 Core Institutional metrics to generate an absolute 0-100 Confluence Score.
    Runs entirely on pre-extracted arrays without re-triggering pandas wrappers overhead.
    """
    o, h, l, c, idx = _extract_ohlc(data)
    
    sh, sl, _, _ = _swing_levels_jit(h, l, 5, 5)
    sh_filled, sl_filled = pd.Series(sh).ffill().to_numpy(), pd.Series(sl).ffill().to_numpy()
    
    prev_h, prev_l, prev_c = _shift_array(h, 1), _shift_array(l, 1), _shift_array(c, 1)
    pp = (prev_h + prev_l + prev_c) / 3.0
    
    rhh, rll = pd.Series(h).rolling(20).max().to_numpy(), pd.Series(l).rolling(20).min().to_numpy()
    fib_618 = rll + (rhh - rll) * 0.618
    
    intervals = np.array([10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000], dtype=np.float64)
    atr = _atr_proxy(data)
    psych = np.zeros(len(c), dtype=np.float64)
    for i in range(len(c)):
        if np.isnan(c[i]) or np.isnan(atr[i]): continue
        step = intervals[np.argmin(np.abs(intervals - max(atr[i] * 5.0, EPSILON)))]
        psych[i] = np.round(c[i] / step) * step
        
    gap_sup = pd.Series(np.where(o > prev_c, prev_c, np.nan)).ffill().to_numpy()
    
    diff = np.where((rhh - rll) == 0, EPSILON, rhh - rll)
    m_range = (10 ** np.floor(np.log10(diff))) * 0.125
    mm_4_8 = np.floor(rll / m_range) * m_range + m_range * 4
    
    df = standardize_column_names(data)
    if 'volume' in df.columns:
        vol = df['volume'].to_numpy(dtype=np.float64)
        poc, _, _ = _adaptive_vol_profile_jit((h+l+c)/3.0, vol, 20, atr)
    else:
        poc = np.full_like(c, np.nan)
        
    levels = [sh_filled, sl_filled, pp, fib_618, psych, gap_sup, mm_4_8, poc]
    
    score = np.zeros(len(c))
    for lvl in levels:
        valid = ~np.isnan(lvl) & (lvl > 0)
        close_match = np.abs(c - lvl) / np.where(lvl==0, EPSILON, lvl) <= epsilon_pct
        score += np.where(valid & close_match, 1.0, 0.0)
        
    final_score = (score / len(levels)) * 100.0
    return _finalize_output(pd.Series(final_score, index=idx, name="InstConfluenceScore"), offset, fillna)

def mtf_confluence_score(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    base = institutional_confluence_score(data)
    d = donchian_channel(data, length=1)['lower'].to_numpy()
    w = donchian_channel(data, length=DEFAULT_MTF_W)['lower'].to_numpy()
    m = donchian_channel(data, length=DEFAULT_MTF_M)['lower'].to_numpy()
    
    c = get_price_source(data).to_numpy()
    bonus = np.where(np.abs(c - d) <= c * 0.005, 10.0, 0.0) + np.where(np.abs(c - w) <= c * 0.005, 15.0, 0.0) + np.where(np.abs(c - m) <= c * 0.005, 25.0, 0.0)
    return _finalize_output(pd.Series(np.clip(base + bonus, 0.0, 100.0), index=data.index, name="MTF_ConfluenceScore"), offset, fillna)

def level_reliability_ranking(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    score = mtf_confluence_score(data).fillna(0).to_numpy()
    ranks = np.select([score >= 90, score >= 75, score >= 50, score >= 30, score >= 15], ['Elite', 'Institutional', 'Strong', 'Moderate', 'Weak'], default='Very Weak')
    return _finalize_output(pd.Series(ranks, index=data.index, name="ReliabilityRank"), offset, fillna)

# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "turtle_soup", "judas_swing", "kill_zones", "amd_cycle",
    "SupportResistanceError", "validate_length", "get_price_source",
    "pivot_point", "pivot_support_1", "pivot_support_2", "pivot_support_3", "pivot_resistance_1", "pivot_resistance_2", "pivot_resistance_3", "classic_pivot_levels",
    "floor_pivot", "floor_s1", "floor_s2", "floor_s3", "floor_r1", "floor_r2", "floor_r3",
    "woodie_pivot", "woodie_r1", "woodie_r2", "woodie_r3", "woodie_s1", "woodie_s2", "woodie_s3",
    "camarilla_pivot", "camarilla_h1", "camarilla_h2", "camarilla_h3", "camarilla_h4", "camarilla_h5", "camarilla_h6", "camarilla_l1", "camarilla_l2", "camarilla_l3", "camarilla_l4", "camarilla_l5", "camarilla_l6",
    "demark_pivot", "demark_resistance", "demark_support",
    "swing_high", "swing_low", "highest_swing", "lowest_swing", "last_swing_high", "last_swing_low", "confirmed_swing_high", "confirmed_swing_low", "adaptive_swing",
    "bill_williams_fractal_high", "bill_williams_fractal_low", "fractal_pivot", "fractal_support", "fractal_resistance",
    "rolling_lowest_low", "rolling_highest_high", "dynamic_support", "dynamic_resistance", "adaptive_support", "adaptive_resistance",
    "highest_high_breakout", "lowest_low_breakdown", "previous_high_breakout", "previous_low_breakdown", "donchian_breakout", "range_breakout",
    "donchian_channel", "price_channel", "highest_channel", "lowest_channel", "channel_midline",
    "support_zone", "resistance_zone", "demand_zone", "supply_zone", "reaction_zone", "congestion_zone",
    "bullish_fvg", "bearish_fvg", "mitigated_fvg", "balanced_price_range", "order_blocks", "breaker_block", "flip_zone",
    "smc_structure", "bos_level", "choch_level", "premium_discount_zone", "inducement_level", "market_structure_shift",
    "buy_side_liquidity", "sell_side_liquidity", "internal_liquidity", "external_liquidity", "equal_highs", "equal_lows",
    "support_touch_count", "resistance_touch_count", "level_strength", "level_confidence", "zone_width", "bounce_count",
    "distance_to_support", "distance_to_resistance", "nearest_support", "nearest_resistance", "risk_distance", "reward_distance",
    "confirmed_breakout", "confirmed_breakdown", "false_breakout", "false_breakdown", "retest_level", "break_strength",
    "liquidity_high", "liquidity_low", "liquidity_pool", "institutional_resistance", "institutional_support", "liquidity_sweep_level", "stop_hunt_zone", "smart_money_level",
    "rolling_support", "rolling_resistance", "support_percentile", "resistance_percentile", "support_zscore", "resistance_zscore",
    "atr_support", "atr_resistance", "volatility_support", "volatility_resistance", "adaptive_breakout",
    "daily_levels", "weekly_levels", "monthly_levels", "yearly_levels", "merged_support", "merged_resistance",
    "price_cluster", "cluster_support", "cluster_resistance", "cluster_density", "cluster_strength",
    "market_structure_high", "market_structure_low", "structure_resistance", "structure_support", "higher_high", "lower_low", "higher_low", "lower_high",
    "adaptive_volume_profile", "swing_high_vwap", "swing_low_vwap",
    "fibonacci_levels", "gann_levels", "murrey_math_levels", "psychological_levels", "gap_levels", "previous_session_levels",
    "institutional_confluence_score", "mtf_confluence_score", "level_reliability_ranking", "turtle_soup", "judas_swing", "kill_zones", "amd_cycle"
]
