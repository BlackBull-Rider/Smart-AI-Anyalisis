import logging
import numpy as np
import pandas as pd
from typing import Union, Optional, Any, Tuple

# Optional Numba compilation for institutional speed
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
    ensure_series,
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

DEFAULT_VOL_LEN = 20
DEFAULT_MFI_LEN = 14
DEFAULT_CMF_LEN = 20
DEFAULT_FORCE_LEN = 13
DEFAULT_EOM_LEN = 14
DEFAULT_PVO_FAST = 12
DEFAULT_PVO_SLOW = 26
DEFAULT_PVO_SIGNAL = 9
DEFAULT_KLINGER_FAST = 34
DEFAULT_KLINGER_SLOW = 55
DEFAULT_KLINGER_SIGNAL = 13
DEFAULT_PERCENTILE_LOOKBACK = 252
DEFAULT_SPIKE_Z = 2.0

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

class VolumeIndicatorError(Exception):
    """Custom exception for errors in volume indicator calculations."""
    pass

# ==============================================================================
# NUMBA JIT ACCELERATED CORE FUNCTIONS
# ==============================================================================

@jit(nopython=True, cache=True)
def _klinger_cm_jit(tr: np.ndarray, trend: np.ndarray) -> np.ndarray:
    cm = np.zeros_like(tr)
    if len(cm) > 0:
        cm[0] = 0.0
        for i in range(1, len(tr)):
            if trend[i] == trend[i-1]:
                cm[i] = cm[i-1] + tr[i]
            else:
                cm[i] = tr[i] + tr[i-1]
    return cm

@jit(nopython=True, cache=True)
def _evwma_jit(price: np.ndarray, vol: np.ndarray, sum_vol: np.ndarray, length: int) -> np.ndarray:
    evwma = np.full_like(price, np.nan)
    if len(price) >= length:
        evwma[length-1] = np.mean(price[:length])
        for i in range(length, len(price)):
            if sum_vol[i] == 0:
                evwma[i] = evwma[i-1]
            else:
                w1 = (sum_vol[i] - vol[i]) / sum_vol[i]
                w2 = vol[i] / sum_vol[i]
                evwma[i] = w1 * evwma[i-1] + w2 * price[i]
    return evwma

@jit(nopython=True, cache=True)
def _vol_profile_jit(price: np.ndarray, vol: np.ndarray, length: int, bins: int):
    n = len(price)
    poc = np.full(n, np.nan)
    vah = np.full(n, np.nan)
    val = np.full(n, np.nan)
    hvn = np.full(n, np.nan)
    lvn = np.full(n, np.nan)
    
    if n >= length:
        for i in range(length - 1, n):
            p_win = price[i - length + 1 : i + 1]
            v_win = vol[i - length + 1 : i + 1]
            
            p_min = np.nanmin(p_win)
            p_max = np.nanmax(p_win)
            
            if np.isnan(p_min) or np.isnan(p_max) or p_min == p_max:
                poc[i] = vah[i] = val[i] = hvn[i] = lvn[i] = p_win[-1]
                continue
            
            bin_edges = np.linspace(p_min, p_max, bins + 1)
            vol_profile = np.zeros(bins)
            
            for j in range(length):
                b = 0
                while b < bins and p_win[j] >= bin_edges[b+1]:
                    b += 1
                if b == bins: b -= 1
                vol_profile[b] += v_win[j]
                
            poc_idx = np.argmax(vol_profile)
            poc[i] = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2.0
            hvn[i] = poc[i]
            
            non_zero_vols = vol_profile[vol_profile > 0]
            if len(non_zero_vols) > 0:
                min_v = np.min(non_zero_vols)
                for b in range(bins):
                    if vol_profile[b] == min_v:
                        lvn[i] = (bin_edges[b] + bin_edges[b + 1]) / 2.0
                        break
            
            target_vol = np.sum(vol_profile) * 0.70
            curr_vol = vol_profile[poc_idx]
            up_idx = poc_idx + 1
            dn_idx = poc_idx - 1
            
            while curr_vol < target_vol and (up_idx < bins or dn_idx >= 0):
                v_up = vol_profile[up_idx] if up_idx < bins else -1.0
                v_dn = vol_profile[dn_idx] if dn_idx >= 0 else -1.0
                
                if v_up > v_dn:
                    curr_vol += v_up
                    up_idx += 1
                else:
                    curr_vol += v_dn
                    dn_idx -= 1
                    
            vah[i] = bin_edges[min(up_idx, bins)]
            val[i] = bin_edges[max(0, dn_idx)]
            
    return poc, vah, val, hvn, lvn

@jit(nopython=True, cache=True)
def _session_vol_profile_jit(price: np.ndarray, vol: np.ndarray, start_idx: np.ndarray, 
                             day_min: np.ndarray, day_max: np.ndarray, bins: int):
    n = len(price)
    poc = np.full(n, np.nan)
    vah = np.full(n, np.nan)
    val = np.full(n, np.nan)
    
    for i in range(n):
        s_idx = start_idx[i]
        p_min = day_min[i]
        p_max = day_max[i]
        
        if np.isnan(p_min) or np.isnan(p_max) or p_min == p_max:
            poc[i] = vah[i] = val[i] = price[i]
            continue
            
        bin_edges = np.linspace(p_min, p_max, bins + 1)
        vol_profile = np.zeros(bins)
        
        for j in range(s_idx, i + 1):
            b = 0
            while b < bins and price[j] >= bin_edges[b+1]:
                b += 1
            if b == bins: b -= 1
            vol_profile[b] += vol[j]
            
        poc_idx = np.argmax(vol_profile)
        poc[i] = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2.0
        
        target_vol = np.sum(vol_profile) * 0.70
        curr_vol = vol_profile[poc_idx]
        up_idx = poc_idx + 1
        dn_idx = poc_idx - 1
        
        while curr_vol < target_vol and (up_idx < bins or dn_idx >= 0):
            v_up = vol_profile[up_idx] if up_idx < bins else -1.0
            v_dn = vol_profile[dn_idx] if dn_idx >= 0 else -1.0
            
            if v_up > v_dn:
                curr_vol += v_up
                up_idx += 1
            else:
                curr_vol += v_dn
                dn_idx -= 1
                
        vah[i] = bin_edges[min(up_idx, bins)]
        val[i] = bin_edges[max(0, dn_idx)]
        
    return poc, vah, val

# ==============================================================================
# COMMON UTILITIES
# ==============================================================================

def validate_length(length: int, name: str = "length") -> None:
    if not isinstance(length, int) or length <= 0:
        logger.error(f"Invalid volume window length for {name}: {length}")
        raise VolumeIndicatorError(f"Length for {name} must be a positive integer, got {length}")

def get_price_source(data: Union[pd.DataFrame, pd.Series], source: str = 'close') -> pd.Series:
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
            raise VolumeIndicatorError(f"DataFrame is missing columns needed for '{src}': {missing}")
        return func(*[df[c] for c in req_cols])
            
    logger.error(f"Invalid source '{source}'.")
    raise VolumeIndicatorError(f"Invalid source '{source}' or column not found.")

def _finalize_output(output: Union[pd.Series, pd.DataFrame], offset: int, fillna: Any) -> Union[pd.Series, pd.DataFrame]:
    res = output.shift(offset) if offset != 0 else output
    if fillna is not None:
        res = res.fillna(fillna)
    return res

def _extract_volume(data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
    if isinstance(data, pd.Series):
        return data
    df = standardize_column_names(data)
    if 'volume' not in df.columns:
        raise VolumeIndicatorError("DataFrame must contain a 'volume' column.")
    return df['volume']

# ==============================================================================
# BASIC VOLUME
# ==============================================================================

def volume(data: Union[pd.DataFrame, pd.Series], offset: int = 0, fillna: Any = None) -> pd.Series:
    vol = _extract_volume(data)
    out = pd.Series(vol, name="Volume")
    return _finalize_output(out, offset, fillna)

def average_volume(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_VOL_LEN, 
                   offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Average Volume")
    vol = _extract_volume(data)
    out = rolling_mean(vol, window=length)
    out.name = f"AvgVol_{length}"
    return _finalize_output(out, offset, fillna)

def rolling_volume(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_VOL_LEN, 
                   offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Rolling Volume")
    vol = _extract_volume(data)
    out = rolling_sum(vol, window=length)
    out.name = f"RollVol_{length}"
    return _finalize_output(out, offset, fillna)

def relative_volume(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_VOL_LEN, 
                    offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "RVOL")
    vol = _extract_volume(data)
    avg_vol = rolling_mean(vol, window=length)
    rvol = safe_divide(vol.to_numpy(dtype=np.float64), avg_vol.to_numpy(dtype=np.float64), default=np.nan)
    out = pd.Series(rvol, index=vol.index, name=f"RVOL_{length}")
    return _finalize_output(out, offset, fillna)

def volume_ratio(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_VOL_LEN, 
                 offset: int = 0, fillna: Any = None) -> pd.Series:
    rvol = relative_volume(data, length=length) * 100.0
    rvol.name = f"VolRatio_{length}"
    return _finalize_output(rvol, offset, fillna)

def volume_sma(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_VOL_LEN, 
               offset: int = 0, fillna: Any = None) -> pd.Series:
    return average_volume(data, length, offset, fillna)

def volume_ema(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_VOL_LEN, 
               offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Volume EMA")
    vol = _extract_volume(data)
    out = vol.ewm(span=length, adjust=False).mean()
    out.name = f"VolEMA_{length}"
    return _finalize_output(out, offset, fillna)

def volume_zscore(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_VOL_LEN, 
                  offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Volume Z-Score")
    vol = _extract_volume(data)
    mean_vol = rolling_mean(vol, window=length)
    std_vol = rolling_std(vol, window=length)
    z = safe_divide((vol - mean_vol).to_numpy(dtype=np.float64), std_vol.to_numpy(dtype=np.float64), default=np.nan)
    out = pd.Series(z, index=vol.index, name=f"VolZ_{length}")
    return _finalize_output(out, offset, fillna)

def volume_percentile(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_PERCENTILE_LOOKBACK, 
                      offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Volume Percentile")
    vol = _extract_volume(data)
    ranks = vol.rolling(window=length).rank(pct=True) * 100.0
    out = pd.Series(ranks, index=vol.index, name=f"VolP_{length}")
    return _finalize_output(out, offset, fillna)

def volume_roc(data: Union[pd.DataFrame, pd.Series], length: int = 1, 
               offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Volume ROC")
    vol = _extract_volume(data)
    shifted = vol.shift(length)
    roc = 100.0 * safe_divide((vol - shifted).to_numpy(dtype=np.float64), shifted.to_numpy(dtype=np.float64), default=np.nan)
    out = pd.Series(roc, index=vol.index, name=f"VolROC_{length}")
    return _finalize_output(out, offset, fillna)

# ==============================================================================
# PRICE × VOLUME & VWAP VARIANTS
# ==============================================================================

def typical_price_volume(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    df = standardize_column_names(data)
    validate_ohlc(df)
    tp = hlc3(df['high'], df['low'], df['close'])
    vol = _extract_volume(df)
    out = pd.Series(tp * vol, index=df.index, name="TPV")
    return _finalize_output(out, offset, fillna)

def weighted_close_volume(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    df = standardize_column_names(data)
    validate_ohlc(df)
    wp = weighted_price(df['high'], df['low'], df['close'])
    vol = _extract_volume(df)
    out = pd.Series(wp * vol, index=df.index, name="WCV")
    return _finalize_output(out, offset, fillna)

def vwap(data: pd.DataFrame, source: str = 'hlc3', anchor: str = 'Cumulative', 
         offset: int = 0, fillna: Any = None) -> pd.Series:
    df = standardize_column_names(data)
    price = get_price_source(df, source)
    vol = _extract_volume(df)
    
    if anchor.lower() == 'cumulative':
        cum_pv = (price * vol).cumsum()
        cum_v = vol.cumsum()
    else:
        validate_datetime_index(df)
        if anchor.lower() in ['session', 'daily', 'd']:
            group_id = df.index.date
        elif anchor.lower() in ['weekly', 'w']:
            group_id = df.index.isocalendar().week + df.index.isocalendar().year * 100
        elif anchor.lower() in ['monthly', 'm']:
            group_id = df.index.month + df.index.year * 100
        else:
            raise VolumeIndicatorError(f"Unsupported VWAP anchor: {anchor}")
            
        cum_pv = (price * vol).groupby(group_id).cumsum()
        cum_v = vol.groupby(group_id).cumsum()
    
    res = safe_divide(cum_pv.to_numpy(dtype=np.float64), cum_v.to_numpy(dtype=np.float64), default=np.nan)
    out = pd.Series(res, index=df.index, name=f"VWAP_{anchor}")
    return _finalize_output(out, offset, fillna)

def vwap_bands(data: pd.DataFrame, source: str = 'hlc3', anchor: str = 'Cumulative', 
               std_mult: float = 2.0, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    df = standardize_column_names(data)
    price = get_price_source(df, source)
    vol = _extract_volume(df)
    
    if anchor.lower() == 'cumulative':
        group_id = pd.Series(1, index=df.index)
    else:
        validate_datetime_index(df)
        if anchor.lower() in ['session', 'daily', 'd']:
            group_id = df.index.date
        elif anchor.lower() in ['weekly', 'w']:
            group_id = df.index.isocalendar().week + df.index.isocalendar().year * 100
        elif anchor.lower() in ['monthly', 'm']:
            group_id = df.index.month + df.index.year * 100
        else:
            raise VolumeIndicatorError(f"Unsupported VWAP anchor: {anchor}")

    cum_pv = (price * vol).groupby(group_id).cumsum()
    cum_v = vol.groupby(group_id).cumsum()
    vwap_line = safe_divide(cum_pv.to_numpy(dtype=np.float64), cum_v.to_numpy(dtype=np.float64), default=np.nan)
    vwap_series = pd.Series(vwap_line, index=df.index)
    
    diff_sq = (price - vwap_series) ** 2
    cum_v_diff_sq = (vol * diff_sq).groupby(group_id).cumsum()
    vw_var = safe_divide(cum_v_diff_sq.to_numpy(dtype=np.float64), cum_v.to_numpy(dtype=np.float64), default=np.nan)
    vw_std = np.sqrt(vw_var)
    
    out = pd.DataFrame({
        "vwap": vwap_series,
        "upper": pd.Series(vwap_line + (std_mult * vw_std), index=df.index),
        "lower": pd.Series(vwap_line - (std_mult * vw_std), index=df.index)
    })
    return _finalize_output(out, offset, fillna)

def rolling_vwap(data: pd.DataFrame, length: int = DEFAULT_VOL_LEN, source: str = 'hlc3', 
                 offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Rolling VWAP")
    df = standardize_column_names(data)
    price = get_price_source(df, source)
    vol = _extract_volume(df)
    
    roll_pv = rolling_sum(price * vol, window=length)
    roll_v = rolling_sum(vol, window=length)
    
    res = safe_divide(roll_pv.to_numpy(dtype=np.float64), roll_v.to_numpy(dtype=np.float64), default=np.nan)
    out = pd.Series(res, index=df.index, name=f"RVWAP_{length}")
    return _finalize_output(out, offset, fillna)

def anchored_vwap(data: pd.DataFrame, anchor: Union[pd.Series, str, pd.Timestamp], source: str = 'hlc3', 
                  offset: int = 0, fillna: Any = None) -> pd.Series:
    df = standardize_column_names(data)
    price = get_price_source(df, source)
    vol = _extract_volume(df)
    
    if isinstance(anchor, pd.Series) and np.issubdtype(anchor.dtype, np.bool_):
        group_id = anchor.cumsum()
    else:
        validate_datetime_index(df)
        anchor_dt = pd.to_datetime(anchor)
        mask = pd.Series(df.index >= anchor_dt, index=df.index)
        group_id = mask.astype(int)
        
        pv = price * vol
        pv.loc[~mask] = 0
        v = vol.copy()
        v.loc[~mask] = 0
        
        cum_pv = pv.groupby(group_id).cumsum()
        cum_v = v.groupby(group_id).cumsum()
        
        res = safe_divide(cum_pv.to_numpy(dtype=np.float64), cum_v.to_numpy(dtype=np.float64), default=np.nan)
        res[~mask.to_numpy()] = np.nan
        
        out = pd.Series(res, index=df.index, name="AVWAP")
        return _finalize_output(out, offset, fillna)
        
    cum_pv = (price * vol).groupby(group_id).cumsum()
    cum_v = vol.groupby(group_id).cumsum()
    
    res = safe_divide(cum_pv.to_numpy(dtype=np.float64), cum_v.to_numpy(dtype=np.float64), default=np.nan)
    out = pd.Series(res, index=df.index, name="AVWAP")
    return _finalize_output(out, offset, fillna)

def anchored_vwap_bands(data: pd.DataFrame, anchor: Union[pd.Series, str, pd.Timestamp], source: str = 'hlc3', 
                        std_mult: float = 2.0, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    df = standardize_column_names(data)
    price = get_price_source(df, source)
    vol = _extract_volume(df)
    
    vwap_series = anchored_vwap(data, anchor=anchor, source=source)
    
    if isinstance(anchor, pd.Series) and np.issubdtype(anchor.dtype, np.bool_):
        group_id = anchor.cumsum()
        mask = pd.Series(True, index=df.index)
    else:
        validate_datetime_index(df)
        anchor_dt = pd.to_datetime(anchor)
        mask = pd.Series(df.index >= anchor_dt, index=df.index)
        group_id = mask.astype(int)
        
    v = vol.copy()
    v.loc[~mask] = 0
    diff_sq = (price - vwap_series) ** 2
    diff_sq.loc[~mask] = 0
    
    cum_v = v.groupby(group_id).cumsum()
    cum_v_diff_sq = (v * diff_sq).groupby(group_id).cumsum()
    
    vw_var = safe_divide(cum_v_diff_sq.to_numpy(dtype=np.float64), cum_v.to_numpy(dtype=np.float64), default=np.nan)
    vw_std = np.sqrt(vw_var)
    
    out = pd.DataFrame({
        "avwap": vwap_series,
        "upper": pd.Series(vwap_series + (std_mult * vw_std), index=df.index),
        "lower": pd.Series(vwap_series - (std_mult * vw_std), index=df.index)
    })
    return _finalize_output(out, offset, fillna)

def vwma(data: pd.DataFrame, length: int = DEFAULT_VOL_LEN, source: str = 'close', 
         offset: int = 0, fillna: Any = None) -> pd.Series:
    return rolling_vwap(data, length=length, source=source, offset=offset, fillna=fillna)

def vwema(data: pd.DataFrame, length: int = DEFAULT_VOL_LEN, source: str = 'close', 
          offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "VWEMA")
    df = standardize_column_names(data)
    price = get_price_source(df, source)
    vol = _extract_volume(df)
    
    pv_ema = (price * vol).ewm(span=length, adjust=False).mean()
    v_ema = vol.ewm(span=length, adjust=False).mean()
    
    res = safe_divide(pv_ema.to_numpy(dtype=np.float64), v_ema.to_numpy(dtype=np.float64), default=np.nan)
    out = pd.Series(res, index=df.index, name=f"VWEMA_{length}")
    return _finalize_output(out, offset, fillna)

# ==============================================================================
# MONEY FLOW
# ==============================================================================

def positive_money_flow(data: pd.DataFrame, source: str = 'hlc3', offset: int = 0, fillna: Any = None) -> pd.Series:
    df = standardize_column_names(data)
    price = get_price_source(df, source)
    vol = _extract_volume(df)
    
    change = price.diff()
    pmf = np.where(change > 0, price * vol, 0.0)
    out = pd.Series(pmf, index=df.index, name="PMF")
    return _finalize_output(out, offset, fillna)

def negative_money_flow(data: pd.DataFrame, source: str = 'hlc3', offset: int = 0, fillna: Any = None) -> pd.Series:
    df = standardize_column_names(data)
    price = get_price_source(df, source)
    vol = _extract_volume(df)
    
    change = price.diff()
    nmf = np.where(change < 0, price * vol, 0.0)
    out = pd.Series(nmf, index=df.index, name="NMF")
    return _finalize_output(out, offset, fillna)

def money_flow_index(data: pd.DataFrame, length: int = DEFAULT_MFI_LEN, source: str = 'hlc3', 
                     offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "MFI")
    df = standardize_column_names(data)
    price = get_price_source(df, source)
    vol = _extract_volume(df)
    
    change = price.diff()
    mf = price * vol
    
    pos_flow = pd.Series(np.where(change > 0, mf, 0.0), index=df.index)
    neg_flow = pd.Series(np.where(change < 0, mf, 0.0), index=df.index)
    
    pos_sum = rolling_sum(pos_flow, window=length)
    neg_sum = rolling_sum(neg_flow, window=length)
    
    ratio = safe_divide(pos_sum.to_numpy(dtype=np.float64), neg_sum.to_numpy(dtype=np.float64), default=np.nan)
    mfi = 100.0 - (100.0 / (1.0 + ratio))
    mfi = np.where(neg_sum == 0.0, np.where(pos_sum == 0.0, 50.0, 100.0), mfi)
    
    out = pd.Series(mfi, index=df.index, name=f"MFI_{length}")
    return _finalize_output(out, offset, fillna)

def chaikin_money_flow(data: pd.DataFrame, length: int = DEFAULT_CMF_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "CMF")
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    close = df['close'].to_numpy(dtype=np.float64)
    low = df['low'].to_numpy(dtype=np.float64)
    high = df['high'].to_numpy(dtype=np.float64)
    vol = _extract_volume(df).to_numpy(dtype=np.float64)
    
    num = (close - low) - (high - close)
    den = high - low
    den = np.where(den == 0.0, EPSILON, den)
    
    mfm = safe_divide(num, den, default=0.0)
    mfv = mfm * vol
    
    sum_mfv = rolling_sum(pd.Series(mfv, index=df.index), window=length)
    sum_vol = rolling_sum(pd.Series(vol, index=df.index), window=length)
    
    cmf = safe_divide(sum_mfv.to_numpy(dtype=np.float64), sum_vol.to_numpy(dtype=np.float64), default=np.nan)
    out = pd.Series(cmf, index=df.index, name=f"CMF_{length}")
    return _finalize_output(out, offset, fillna)

def accumulation_distribution_line(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    close = df['close'].to_numpy(dtype=np.float64)
    low = df['low'].to_numpy(dtype=np.float64)
    high = df['high'].to_numpy(dtype=np.float64)
    vol = _extract_volume(df).to_numpy(dtype=np.float64)
    
    num = (close - low) - (high - close)
    den = high - low
    den = np.where(den == 0.0, EPSILON, den)
    mfm = safe_divide(num, den, default=0.0)
    mfv = mfm * vol
    
    adl = pd.Series(mfv, index=df.index).cumsum()
    adl.name = "ADL"
    return _finalize_output(adl, offset, fillna)

def accumulation_distribution_oscillator(data: pd.DataFrame, fast_len: int = 3, slow_len: int = 10, 
                                         offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(fast_len, "AD Osc Fast")
    validate_length(slow_len, "AD Osc Slow")
    
    adl = accumulation_distribution_line(data)
    fast_ema = adl.ewm(span=fast_len, adjust=False).mean()
    slow_ema = adl.ewm(span=slow_len, adjust=False).mean()
    
    res = fast_ema - slow_ema
    res.name = f"ADOSC_{fast_len}_{slow_len}"
    return _finalize_output(res, offset, fillna)

def money_flow_multiplier(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    high = df['high'].to_numpy(dtype=np.float64)
    low = df['low'].to_numpy(dtype=np.float64)
    close = df['close'].to_numpy(dtype=np.float64)
    
    num = (close - low) - (high - close)
    den = high - low
    den = np.where(den == 0.0, EPSILON, den)
    
    mfm = safe_divide(num, den, default=0.0)
    out = pd.Series(mfm, index=df.index, name="MFM")
    return _finalize_output(out, offset, fillna)

def money_flow_volume(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    mfm = money_flow_multiplier(data)
    vol = _extract_volume(data)
    mfv = mfm * vol
    mfv.name = "MFV"
    return _finalize_output(mfv, offset, fillna)

# ==============================================================================
# ON BALANCE VOLUME (Exact Parity)
# ==============================================================================

def obv(data: pd.DataFrame, source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    """Exact TradingView OBV implementation."""
    df = standardize_column_names(data)
    price = get_price_source(df, source).to_numpy(dtype=np.float64)
    vol = _extract_volume(df).to_numpy(dtype=np.float64)
    
    prev_price = np.empty_like(price)
    prev_price[0] = np.nan
    prev_price[1:] = price[:-1]
    
    direction = np.zeros_like(price)
    direction[price > prev_price] = 1.0
    direction[price < prev_price] = -1.0
    direction[price == prev_price] = 0.0
    direction[np.isnan(price) | np.isnan(prev_price)] = 0.0
    
    obv_arr = np.cumsum(vol * direction)
    out = pd.Series(obv_arr, index=df.index, name="OBV")
    return _finalize_output(out, offset, fillna)

def obv_sma(data: pd.DataFrame, length: int = 20, source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "OBV SMA")
    obv_line = obv(data, source=source)
    res = rolling_mean(obv_line, window=length)
    res.name = f"OBV_SMA_{length}"
    return _finalize_output(res, offset, fillna)

def obv_ema(data: pd.DataFrame, length: int = 20, source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "OBV EMA")
    obv_line = obv(data, source=source)
    res = obv_line.ewm(span=length, adjust=False).mean()
    res.name = f"OBV_EMA_{length}"
    return _finalize_output(res, offset, fillna)

def obv_oscillator(data: pd.DataFrame, fast_len: int = 12, slow_len: int = 26, source: str = 'close', 
                   offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(fast_len, "OBV Osc Fast")
    validate_length(slow_len, "OBV Osc Slow")
    obv_line = obv(data, source=source)
    
    fast_ema = obv_line.ewm(span=fast_len, adjust=False).mean()
    slow_ema = obv_line.ewm(span=slow_len, adjust=False).mean()
    
    res = fast_ema - slow_ema
    res.name = f"OBV_OSC_{fast_len}_{slow_len}"
    return _finalize_output(res, offset, fillna)

def obv_roc(data: pd.DataFrame, length: int = 10, source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "OBV ROC")
    obv_line = obv(data, source=source)
    shifted = obv_line.shift(length)
    
    res = 100.0 * safe_divide((obv_line - shifted).to_numpy(dtype=np.float64), np.abs(shifted).to_numpy(dtype=np.float64), default=np.nan)
    out = pd.Series(res, index=obv_line.index, name=f"OBV_ROC_{length}")
    return _finalize_output(out, offset, fillna)

# ==============================================================================
# FORCE & FLOW INDICATORS
# ==============================================================================

def force_index(data: pd.DataFrame, source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    df = standardize_column_names(data)
    price = get_price_source(df, source)
    vol = _extract_volume(df)
    
    fi = price.diff() * vol
    out = pd.Series(fi, index=df.index, name="ForceIndex")
    return _finalize_output(out, offset, fillna)

def force_index_ema(data: pd.DataFrame, length: int = DEFAULT_FORCE_LEN, source: str = 'close', 
                    offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Force Index EMA")
    fi = force_index(data, source=source)
    res = fi.ewm(span=length, adjust=False).mean()
    res.name = f"EFI_{length}"
    return _finalize_output(res, offset, fillna)

def ease_of_movement(data: pd.DataFrame, divisor: float = 10000.0, offset: int = 0, fillna: Any = None) -> pd.Series:
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    high = df['high'].to_numpy(dtype=np.float64)
    low = df['low'].to_numpy(dtype=np.float64)
    vol = _extract_volume(df).to_numpy(dtype=np.float64)
    
    hl_avg = (high + low) / 2.0
    distance = np.diff(hl_avg, prepend=np.nan)
    
    box_ratio = safe_divide(vol, (high - low), default=0.0) / divisor
    eom = safe_divide(distance, box_ratio, default=np.nan)
    
    out = pd.Series(eom, index=df.index, name="EOM")
    return _finalize_output(out, offset, fillna)

def smoothed_eom(data: pd.DataFrame, length: int = DEFAULT_EOM_LEN, divisor: float = 10000.0, 
                 offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Smoothed EOM")
    eom = ease_of_movement(data, divisor=divisor)
    res = rolling_mean(eom, window=length)
    res.name = f"SEOM_{length}"
    return _finalize_output(res, offset, fillna)

def elastic_volume_weighted_momentum(data: pd.DataFrame, length: int = 14, source: str = 'close', 
                                     offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "EVWMA")
    df = standardize_column_names(data)
    price = get_price_source(df, source).to_numpy(dtype=np.float64)
    vol = _extract_volume(df).to_numpy(dtype=np.float64)
    
    sum_vol = pd.Series(vol).rolling(window=length).sum().to_numpy(dtype=np.float64)
    evwma = _evwma_jit(price, vol, sum_vol, length)
    
    out = pd.Series(evwma, index=df.index, name=f"EVWMA_{length}")
    return _finalize_output(out, offset, fillna)

# ==============================================================================
# ADVANCED VOLUME
# ==============================================================================

def volume_oscillator(data: Union[pd.DataFrame, pd.Series], fast_len: int = 14, slow_len: int = 28, 
                      offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(fast_len, "Volume Osc Fast")
    validate_length(slow_len, "Volume Osc Slow")
    vol = _extract_volume(data)
    
    fast_sma = rolling_mean(vol, window=fast_len)
    slow_sma = rolling_mean(vol, window=slow_len)
    
    vosc = 100.0 * safe_divide((fast_sma - slow_sma).to_numpy(dtype=np.float64), slow_sma.to_numpy(dtype=np.float64), default=np.nan)
    out = pd.Series(vosc, index=vol.index, name=f"VOSC_{fast_len}_{slow_len}")
    return _finalize_output(out, offset, fillna)

def percentage_volume_oscillator(data: Union[pd.DataFrame, pd.Series], fast_len: int = DEFAULT_PVO_FAST, 
                                 slow_len: int = DEFAULT_PVO_SLOW, signal_len: int = DEFAULT_PVO_SIGNAL, 
                                 offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    validate_length(fast_len, "PVO Fast")
    validate_length(slow_len, "PVO Slow")
    validate_length(signal_len, "PVO Signal")
    vol = _extract_volume(data)
    
    fast_ema = vol.ewm(span=fast_len, adjust=False).mean()
    slow_ema = vol.ewm(span=slow_len, adjust=False).mean()
    
    pvo_line = 100.0 * safe_divide((fast_ema - slow_ema).to_numpy(dtype=np.float64), slow_ema.to_numpy(dtype=np.float64), default=np.nan)
    pvo_series = pd.Series(pvo_line, index=vol.index)
    
    signal_line = pvo_series.ewm(span=signal_len, adjust=False).mean()
    histogram = pvo_series - signal_line
    
    out = pd.DataFrame({
        "pvo": pvo_series,
        "signal": signal_line,
        "histogram": histogram
    }, index=vol.index)
    return _finalize_output(out, offset, fillna)

def volume_macd(data: Union[pd.DataFrame, pd.Series], fast_len: int = DEFAULT_PVO_FAST, 
                slow_len: int = DEFAULT_PVO_SLOW, signal_len: int = DEFAULT_PVO_SIGNAL, 
                offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    validate_length(fast_len, "Vol MACD Fast")
    validate_length(slow_len, "Vol MACD Slow")
    validate_length(signal_len, "Vol MACD Signal")
    vol = _extract_volume(data)
    
    fast_ema = vol.ewm(span=fast_len, adjust=False).mean()
    slow_ema = vol.ewm(span=slow_len, adjust=False).mean()
    
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_len, adjust=False).mean()
    histogram = macd_line - signal_line
    
    out = pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram
    }, index=vol.index)
    return _finalize_output(out, offset, fillna)

def klinger_oscillator(data: pd.DataFrame, fast_len: int = DEFAULT_KLINGER_FAST, 
                       slow_len: int = DEFAULT_KLINGER_SLOW, signal_len: int = DEFAULT_KLINGER_SIGNAL, 
                       offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    validate_length(fast_len, "Klinger Fast")
    validate_length(slow_len, "Klinger Slow")
    validate_length(signal_len, "Klinger Signal")
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    tp = hlc3(df['high'], df['low'], df['close']).to_numpy(dtype=np.float64)
    vol = _extract_volume(df).to_numpy(dtype=np.float64)
    high = df['high'].to_numpy(dtype=np.float64)
    low = df['low'].to_numpy(dtype=np.float64)
    
    trend = np.zeros_like(tp)
    trend[1:] = np.where(tp[1:] > tp[:-1], 1, -1)
    
    tr = high - low
    cm = _klinger_cm_jit(tr, trend)
    
    vf = np.zeros_like(vol)
    mask = cm != 0
    vf[mask] = vol[mask] * np.abs(2 * (tr[mask] / cm[mask]) - 1) * trend[mask] * 100.0
    
    vf_series = pd.Series(vf, index=df.index)
    fast_ema = vf_series.ewm(span=fast_len, adjust=False).mean()
    slow_ema = vf_series.ewm(span=slow_len, adjust=False).mean()
    
    ko_line = fast_ema - slow_ema
    signal_line = ko_line.ewm(span=signal_len, adjust=False).mean()
    histogram = ko_line - signal_line
    
    out = pd.DataFrame({
        "klinger": ko_line,
        "signal": signal_line,
        "histogram": histogram
    }, index=df.index)
    return _finalize_output(out, offset, fillna)

def klinger_signal(data: pd.DataFrame, **kwargs) -> pd.Series:
    return klinger_oscillator(data, **kwargs)['signal']

def klinger_histogram(data: pd.DataFrame, **kwargs) -> pd.Series:
    return klinger_oscillator(data, **kwargs)['histogram']

# ==============================================================================
# SMART MONEY & WYCKOFF PROXIES
# ==============================================================================

def negative_volume_index(data: pd.DataFrame, source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    df = standardize_column_names(data)
    price = get_price_source(df, source)
    vol = _extract_volume(df)
    
    roc = price.pct_change().to_numpy(dtype=np.float64)
    vol_diff = vol.diff().to_numpy(dtype=np.float64)
    
    nvi_ret = np.where(vol_diff < 0, roc, 0.0)
    nvi_ret[np.isnan(nvi_ret)] = 0.0
    
    nvi = 1000.0 * np.cumprod(1.0 + nvi_ret)
    out = pd.Series(nvi, index=df.index, name="NVI")
    return _finalize_output(out, offset, fillna)

def positive_volume_index(data: pd.DataFrame, source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    df = standardize_column_names(data)
    price = get_price_source(df, source)
    vol = _extract_volume(df)
    
    roc = price.pct_change().to_numpy(dtype=np.float64)
    vol_diff = vol.diff().to_numpy(dtype=np.float64)
    
    pvi_ret = np.where(vol_diff > 0, roc, 0.0)
    pvi_ret[np.isnan(pvi_ret)] = 0.0
    
    pvi = 1000.0 * np.cumprod(1.0 + pvi_ret)
    out = pd.Series(pvi, index=df.index, name="PVI")
    return _finalize_output(out, offset, fillna)

def volume_trend(data: pd.DataFrame, source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    df = standardize_column_names(data)
    price = get_price_source(df, source)
    vol = _extract_volume(df)
    
    trend = np.sign(price.diff().fillna(0).to_numpy(dtype=np.float64))
    vt = (vol * trend).cumsum()
    out = pd.Series(vt, index=df.index, name="VolTrend")
    return _finalize_output(out, offset, fillna)

def price_volume_trend(data: pd.DataFrame, source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    df = standardize_column_names(data)
    price = get_price_source(df, source)
    vol = _extract_volume(df)
    
    roc = price.pct_change().fillna(0).to_numpy(dtype=np.float64)
    pvt = (vol * roc).cumsum()
    out = pd.Series(pvt, index=df.index, name="PVT")
    return _finalize_output(out, offset, fillna)

def price_volume_rank(data: pd.DataFrame, length: int = 252, source: str = 'close', 
                      offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "PV Rank")
    df = standardize_column_names(data)
    price = get_price_source(df, source)
    vol = _extract_volume(df)
    
    pv = (price * vol).to_numpy(dtype=np.float64)
    if len(pv) < length:
        out_arr = np.full_like(pv, np.nan)
    else:
        s_filled = pd.Series(pv).ffill().bfill().to_numpy(dtype=np.float64)
        views = np.lib.stride_tricks.sliding_window_view(s_filled, length)
        current = views[:, -1:]
        ranks = np.sum(views <= current, axis=1) / length * 100.0
        out_arr = np.full_like(pv, np.nan)
        out_arr[length - 1:] = ranks
        out_arr[np.isnan(pv)] = np.nan
        
    out = pd.Series(out_arr, index=df.index, name=f"PVR_{length}")
    return _finalize_output(out, offset, fillna)

def smart_money_index(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    o = df['open'].to_numpy(dtype=np.float64)
    c = df['close'].to_numpy(dtype=np.float64)
    
    prev_c = np.empty_like(c)
    prev_c[0] = np.nan
    prev_c[1:] = c[:-1]
    
    net_gap = o - prev_c
    net_close = c - o
    
    smi_diff = -net_gap + net_close
    smi_diff[0] = 0.0
    smi_diff[np.isnan(smi_diff)] = 0.0
    
    smi = np.cumsum(smi_diff)
    out = pd.Series(smi, index=df.index, name="SMI")
    return _finalize_output(out, offset, fillna)

def effort_vs_result(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    df = standardize_column_names(data)
    validate_ohlc(df)
    vol = _extract_volume(df).to_numpy(dtype=np.float64)
    spread = np.abs(df['close'].to_numpy(dtype=np.float64) - df['open'].to_numpy(dtype=np.float64))
    spread = np.where(spread == 0, EPSILON, spread)
    
    evr = safe_divide(vol, spread, default=np.nan)
    out = pd.Series(evr, index=df.index, name="EffortVsResult")
    return _finalize_output(out, offset, fillna)

def stopping_volume(data: pd.DataFrame, length: int = 20, offset: int = 0, fillna: Any = None) -> pd.Series:
    df = standardize_column_names(data)
    validate_ohlc(df)
    vol = _extract_volume(df)
    
    vol_high = vol == vol.rolling(length).max()
    is_down = df['close'] < df['open']
    lower_wick = df['close'] - df['low']
    rng = df['high'] - df['low']
    rng_safe = np.where(rng == 0, EPSILON, rng)
    
    wick_pct = lower_wick / rng_safe
    sv = vol_high & is_down & (wick_pct > 0.5)
    
    out = pd.Series(sv, index=df.index, name="StoppingVol")
    return _finalize_output(out, offset, fillna)

def no_demand_no_supply(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    df = standardize_column_names(data)
    validate_ohlc(df)
    vol = _extract_volume(df)
    
    spread = df['high'] - df['low']
    up_bar = df['close'] > df['open']
    dn_bar = df['close'] < df['open']
    
    vol_less_2 = (vol < vol.shift(1)) & (vol < vol.shift(2))
    narrow_spread = spread < spread.shift(1)
    
    no_demand = up_bar & vol_less_2 & narrow_spread
    no_supply = dn_bar & vol_less_2 & narrow_spread
    
    out = pd.DataFrame({"no_demand": no_demand, "no_supply": no_supply}, index=df.index)
    return _finalize_output(out, offset, fillna)

# ==============================================================================
# DEMAND / SUPPLY / LIQUIDITY (ORDER FLOW PROXIES)
# ==============================================================================

def buy_volume(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    df = standardize_column_names(data)
    validate_ohlc(df)
    vol = _extract_volume(df).to_numpy(dtype=np.float64)
    
    num = df['close'].to_numpy(dtype=np.float64) - df['low'].to_numpy(dtype=np.float64)
    den = df['high'].to_numpy(dtype=np.float64) - df['low'].to_numpy(dtype=np.float64)
    den = np.where(den == 0, EPSILON, den)
    
    ratio = safe_divide(num, den, default=0.5)
    buy_v = vol * ratio
    out = pd.Series(buy_v, index=df.index, name="BuyVol")
    return _finalize_output(out, offset, fillna)

def sell_volume(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    df = standardize_column_names(data)
    validate_ohlc(df)
    vol = _extract_volume(df).to_numpy(dtype=np.float64)
    
    num = df['high'].to_numpy(dtype=np.float64) - df['close'].to_numpy(dtype=np.float64)
    den = df['high'].to_numpy(dtype=np.float64) - df['low'].to_numpy(dtype=np.float64)
    den = np.where(den == 0, EPSILON, den)
    
    ratio = safe_divide(num, den, default=0.5)
    sell_v = vol * ratio
    out = pd.Series(sell_v, index=df.index, name="SellVol")
    return _finalize_output(out, offset, fillna)

def delta_volume(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    buy_v = buy_volume(data)
    sell_v = sell_volume(data)
    delta = buy_v - sell_v
    delta.name = "DeltaVol"
    return _finalize_output(delta, offset, fillna)

def amihud_illiquidity(data: pd.DataFrame, length: int = 20, offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Amihud Illiquidity")
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    price = df['close'].to_numpy(dtype=np.float64)
    prev_price = np.empty_like(price)
    prev_price[0] = np.nan
    prev_price[1:] = price[:-1]
    
    ret = np.abs(safe_divide(price - prev_price, prev_price, default=np.nan))
    dv = price * _extract_volume(df).to_numpy(dtype=np.float64)
    
    amihud_raw = safe_divide(ret, dv, default=np.nan)
    amihud = rolling_mean(pd.Series(amihud_raw, index=df.index), window=length)
    amihud.name = f"Amihud_{length}"
    return _finalize_output(amihud, offset, fillna)

def vpin(data: pd.DataFrame, length: int = 50, offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "VPIN")
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    bv = buy_volume(df).to_numpy(dtype=np.float64)
    sv = sell_volume(df).to_numpy(dtype=np.float64)
    vol = _extract_volume(df).to_numpy(dtype=np.float64)
    
    v_imb = np.abs(bv - sv)
    v_imb_roll = rolling_sum(pd.Series(v_imb), window=length).to_numpy(dtype=np.float64)
    vol_roll = rolling_sum(pd.Series(vol), window=length).to_numpy(dtype=np.float64)
    
    vpin_val = safe_divide(v_imb_roll, vol_roll, default=np.nan)
    out = pd.Series(vpin_val, index=df.index, name=f"VPIN_{length}")
    return _finalize_output(out, offset, fillna)

def time_segmented_volume(data: pd.DataFrame, start_time: str, end_time: str, offset: int = 0, fillna: Any = None) -> pd.Series:
    df = standardize_column_names(data)
    validate_datetime_index(df)
    vol = _extract_volume(df).copy()
    
    mask = (df.index.strftime('%H:%M') >= start_time) & (df.index.strftime('%H:%M') <= end_time)
    vol.loc[~mask] = 0.0
    vol.name = f"TSV_{start_time}_{end_time}"
    return _finalize_output(vol, offset, fillna)

def intraday_rvol(data: pd.DataFrame, length: int = 20, offset: int = 0, fillna: Any = None) -> pd.Series:
    df = standardize_column_names(data)
    validate_datetime_index(df)
    vol = _extract_volume(df)
    
    time_groups = df.index.time
    avg_vol_at_time = vol.groupby(time_groups).transform(lambda x: x.rolling(window=length, min_periods=1).mean())
    rvol = safe_divide(vol.to_numpy(dtype=np.float64), avg_vol_at_time.to_numpy(dtype=np.float64), default=np.nan)
    
    out = pd.Series(rvol, index=df.index, name=f"IntradayRVOL_{length}")
    return _finalize_output(out, offset, fillna)

# ==============================================================================
# VOLUME PROFILE HELPERS
# ==============================================================================

def volume_profile(data: pd.DataFrame, length: int = 20, bins: int = 10, source: str = 'hlc3', 
                   offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    validate_length(length, "Volume Profile")
    df = standardize_column_names(data)
    price = get_price_source(df, source).to_numpy(dtype=np.float64)
    vol = _extract_volume(df).to_numpy(dtype=np.float64)
    
    poc, vah, val, hvn, lvn = _vol_profile_jit(price, vol, length, bins)
    
    out = pd.DataFrame({
        "poc": poc,
        "vah": vah,
        "val": val,
        "hvn": hvn,
        "lvn": lvn
    }, index=df.index)
    return _finalize_output(out, offset, fillna)

def session_volume_profile(data: pd.DataFrame, bins: int = 10, source: str = 'hlc3', 
                           offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    df = standardize_column_names(data)
    validate_datetime_index(df)
    price = get_price_source(df, source).to_numpy(dtype=np.float64)
    vol = _extract_volume(df).to_numpy(dtype=np.float64)
    
    dates = df.index.date
    # Find indices where a new session starts
    start_idx = np.zeros(len(price), dtype=np.int64)
    day_min = np.full(len(price), np.nan, dtype=np.float64)
    day_max = np.full(len(price), np.nan, dtype=np.float64)
    
    curr_start = 0
    curr_min = price[0]
    curr_max = price[0]
    
    for i in range(len(price)):
        if i > 0 and dates[i] != dates[i-1]:
            curr_start = i
            curr_min = price[i]
            curr_max = price[i]
        else:
            if not np.isnan(price[i]):
                if np.isnan(curr_min) or price[i] < curr_min: curr_min = price[i]
                if np.isnan(curr_max) or price[i] > curr_max: curr_max = price[i]
                
        start_idx[i] = curr_start
        day_min[i] = curr_min
        day_max[i] = curr_max
        
    poc, vah, val = _session_vol_profile_jit(price, vol, start_idx, day_min, day_max, bins)

    out = pd.DataFrame({"poc": poc, "vah": vah, "val": val}, index=df.index)
    return _finalize_output(out, offset, fillna)

# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "VolumeIndicatorError",
    "validate_length",
    "get_price_source",
    "volume",
    "average_volume",
    "rolling_volume",
    "relative_volume",
    "volume_ratio",
    "volume_sma",
    "volume_ema",
    "volume_zscore",
    "volume_percentile",
    "volume_roc",
    "typical_price_volume",
    "weighted_close_volume",
    "vwap",
    "vwap_bands",
    "rolling_vwap",
    "anchored_vwap",
    "anchored_vwap_bands",
    "vwma",
    "vwema",
    "positive_money_flow",
    "negative_money_flow",
    "money_flow_index",
    "chaikin_money_flow",
    "accumulation_distribution_line",
    "accumulation_distribution_oscillator",
    "money_flow_multiplier",
    "money_flow_volume",
    "obv",
    "obv_sma",
    "obv_ema",
    "obv_oscillator",
    "obv_roc",
    "force_index",
    "force_index_ema",
    "ease_of_movement",
    "smoothed_eom",
    "elastic_volume_weighted_momentum",
    "volume_oscillator",
    "percentage_volume_oscillator",
    "volume_macd",
    "klinger_oscillator",
    "klinger_signal",
    "klinger_histogram",
    "negative_volume_index",
    "positive_volume_index",
    "volume_trend",
    "price_volume_trend",
    "price_volume_rank",
    "smart_money_index",
    "effort_vs_result",
    "stopping_volume",
    "no_demand_no_supply",
    "buy_volume",
    "sell_volume",
    "delta_volume",
    "amihud_illiquidity",
    "vpin",
    "time_segmented_volume",
    "intraday_rvol",
    "volume_profile",
    "session_volume_profile"
]
