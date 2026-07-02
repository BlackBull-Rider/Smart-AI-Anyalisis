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
    rolling_sum,
    rolling_std,
    EPSILON
)

# ==============================================================================
# LOGGING & CONSTANTS
# ==============================================================================

logger = logging.getLogger(__name__)

DEFAULT_INTERNAL_LEN = 5
DEFAULT_EXTERNAL_LEN = 20
DEFAULT_OTE_MIN = 0.618
DEFAULT_OTE_MAX = 0.786

# SMC State Machine Constants
STATE_NONE = 0
STATE_FRESH = 1
STATE_MITIGATED = 2
STATE_FILLED = 3
STATE_INVALIDATED = 4
STATE_INVERTED = 5       # Breaker Block / Inverse FVG
STATE_RECLAIMED = 6      # Price crosses back over a Breaker
STATE_MITIGATION_BLK = 7 # Failed OB without liquidity sweep

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

class SmartMoneyIndicatorError(Exception):
    """Custom exception for errors in Smart Money calculations."""
    pass

# ==============================================================================
# NUMBA ACCELERATED CORE ENGINES
# ==============================================================================

@jit(nopython=True, cache=True)
def _atr_jit(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    n = len(high)
    atr = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i-1])
        tr3 = np.abs(low[i] - close[i-1])
        tr[i] = max(tr1, tr2, tr3)
    
    # SMA of TR for ATR proxy
    for i in range(length, n):
        atr[i] = np.mean(tr[i-length+1:i+1])
    return atr

@jit(nopython=True, cache=True)
def _structure_engine_jit(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(high)
    sh, sl = np.full(n, np.nan), np.full(n, np.nan)
    hh, hl, lh, ll = np.zeros(n, dtype=np.bool_), np.zeros(n, dtype=np.bool_), np.zeros(n, dtype=np.bool_), np.zeros(n, dtype=np.bool_)
    bos, choch = np.zeros(n, dtype=np.int8), np.zeros(n, dtype=np.int8)
    trend = np.zeros(n, dtype=np.int8)
    
    last_sh_val, last_sl_val = np.nan, np.nan
    prev_sh_val, prev_sl_val = np.nan, np.nan
    curr_trend = 0
    
    for i in range(length, n - length):
        is_sh = True
        for j in range(1, length + 1):
            if high[i-j] > high[i] or high[i+j] >= high[i]:
                is_sh = False; break
        if is_sh:
            sh[i] = high[i]
            
        is_sl = True
        for j in range(1, length + 1):
            if low[i-j] < low[i] or low[i+j] <= low[i]:
                is_sl = False; break
        if is_sl:
            sl[i] = low[i]

    for i in range(length, n):
        if not np.isnan(sh[i]):
            prev_sh_val = last_sh_val
            last_sh_val = sh[i]
            if not np.isnan(prev_sh_val):
                if last_sh_val > prev_sh_val: hh[i] = True
                else: lh[i] = True
                
        if not np.isnan(sl[i]):
            prev_sl_val = last_sl_val
            last_sl_val = sl[i]
            if not np.isnan(prev_sl_val):
                if last_sl_val > prev_sl_val: hl[i] = True
                else: ll[i] = True
                
        bos[i] = 0
        choch[i] = 0
        
        if curr_trend == 1:
            if not np.isnan(last_sh_val) and close[i] > last_sh_val and close[i-1] <= last_sh_val:
                bos[i] = 1
            elif not np.isnan(last_sl_val) and close[i] < last_sl_val and close[i-1] >= last_sl_val:
                choch[i] = -1
                curr_trend = -1
        elif curr_trend == -1:
            if not np.isnan(last_sl_val) and close[i] < last_sl_val and close[i-1] >= last_sl_val:
                bos[i] = -1
            elif not np.isnan(last_sh_val) and close[i] > last_sh_val and close[i-1] <= last_sh_val:
                choch[i] = 1
                curr_trend = 1
        else:
            if not np.isnan(last_sh_val) and close[i] > last_sh_val:
                choch[i] = 1
                curr_trend = 1
            elif not np.isnan(last_sl_val) and close[i] < last_sl_val:
                choch[i] = -1
                curr_trend = -1
                
        trend[i] = curr_trend
        
    return sh, sl, hh, hl, lh, ll, bos, choch, trend

@jit(nopython=True, cache=True)
def _liquidity_engine_jit(high: np.ndarray, low: np.ndarray, close: np.ndarray, sh: np.ndarray, sl: np.ndarray, atr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(high)
    bsl_sw, ssl_sw = np.zeros(n, dtype=np.bool_), np.zeros(n, dtype=np.bool_)
    eqh, eql = np.zeros(n, dtype=np.bool_), np.zeros(n, dtype=np.bool_)
    rest_bsl, rest_ssl = np.full(n, np.nan), np.full(n, np.nan)
    
    active_sh_list = []
    active_sl_list = []
    
    for i in range(n):
        if not np.isnan(sh[i]):
            active_sh_list.append(sh[i])
            # EqH check
            eps = atr[i] * 0.1
            for j in range(max(0, len(active_sh_list)-5), len(active_sh_list)-1):
                if np.abs(sh[i] - active_sh_list[j]) <= eps:
                    eqh[i] = True
                    break
                    
        if not np.isnan(sl[i]):
            active_sl_list.append(sl[i])
            # EqL check
            eps = atr[i] * 0.1
            for j in range(max(0, len(active_sl_list)-5), len(active_sl_list)-1):
                if np.abs(sl[i] - active_sl_list[j]) <= eps:
                    eql[i] = True
                    break
        
        # Resting Liquidity
        if len(active_sh_list) > 0:
            mx = -np.inf
            for val in active_sh_list:
                if val > mx: mx = val
            rest_bsl[i] = mx
            
        if len(active_sl_list) > 0:
            mn = np.inf
            for val in active_sl_list:
                if val < mn: mn = val
            rest_ssl[i] = mn
            
        # Sweeps
        new_sh = []
        for val in active_sh_list:
            if high[i] > val:
                if close[i] <= val: bsl_sw[i] = True
            else:
                new_sh.append(val)
        active_sh_list = new_sh
        
        new_sl = []
        for val in active_sl_list:
            if low[i] < val:
                if close[i] >= val: ssl_sw[i] = True
            else:
                new_sl.append(val)
        active_sl_list = new_sl
        
    return bsl_sw, ssl_sw, eqh, eql, rest_bsl, rest_ssl

@jit(nopython=True, cache=True)
def _ob_fvg_lifecycle_jit(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, v: np.ndarray, atr: np.ndarray, bos: np.ndarray, bsl_sw: np.ndarray, ssl_sw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(h)
    
    fvg_state = np.zeros(n, dtype=np.int8)
    fvg_top, fvg_bot = np.full(n, np.nan), np.full(n, np.nan)
    
    ob_state = np.zeros(n, dtype=np.int8)
    ob_top, ob_bot = np.full(n, np.nan), np.full(n, np.nan)
    ob_age = np.zeros(n, dtype=np.float64)
    fvg_age = np.zeros(n, dtype=np.float64)
    mss = np.zeros(n, dtype=np.int8)
    
    vol_sma = np.zeros(n, dtype=np.float64)
    for i in range(20, n):
        vol_sma[i] = np.mean(v[i-20:i])
    
    for i in range(2, n):
        # Displacement & FVG criteria
        is_disp_bull = (c[i] > o[i]) and ((h[i] - l[i]) > atr[i] * 1.5) and (v[i] > vol_sma[i] * 1.2)
        is_disp_bear = (c[i] < o[i]) and ((h[i] - l[i]) > atr[i] * 1.5) and (v[i] > vol_sma[i] * 1.2)
        
        fvg_bull = (l[i] > h[i-2]) and ((l[i] - h[i-2]) > atr[i] * 0.1)
        fvg_bear = (h[i] < l[i-2]) and ((l[i-2] - h[i]) > atr[i] * 0.1)
        
        # FVG Creation
        if fvg_bull and is_disp_bull:
            fvg_state[i] = STATE_FRESH
            fvg_top[i] = l[i]
            fvg_bot[i] = h[i-2]
            if bos[i] == 1: mss[i] = 1 # MSS confirmed
        elif fvg_bear and is_disp_bear:
            fvg_state[i] = -STATE_FRESH
            fvg_top[i] = l[i-2]
            fvg_bot[i] = h[i]
            if bos[i] == -1: mss[i] = -1
            
        # Order Block Creation (Strict ICT)
        if bos[i] == 1 and fvg_bull:
            swept = False
            for k in range(max(0, i-10), i):
                if ssl_sw[k]: swept = True; break
            
            for j in range(i-1, max(-1, i-15), -1):
                if c[j] < o[j]: # Last down candle
                    ob_state[j] = STATE_FRESH
                    ob_top[j], ob_bot[j] = h[j], l[j]
                    if not swept: ob_state[j] = STATE_MITIGATION_BLK # Mapped to specific state
                    break
                    
        if bos[i] == -1 and fvg_bear:
            swept = False
            for k in range(max(0, i-10), i):
                if bsl_sw[k]: swept = True; break
                
            for j in range(i-1, max(-1, i-15), -1):
                if c[j] > o[j]: # Last up candle
                    ob_state[j] = -STATE_FRESH
                    ob_top[j], ob_bot[j] = h[j], l[j]
                    if not swept: ob_state[j] = -STATE_MITIGATION_BLK
                    break

        # Lifecycle Updates
        for k in range(max(0, i-50), i):
            # FVG Lifecycle
            if fvg_state[k] == STATE_FRESH or fvg_state[k] == STATE_MITIGATED:
                fvg_age[k] += 1
                if l[i] < fvg_top[k]:
                    if c[i] < fvg_bot[k]: fvg_state[k] = STATE_INVERTED
                    elif l[i] <= fvg_bot[k]: fvg_state[k] = STATE_FILLED
                    else: 
                        fvg_state[k] = STATE_MITIGATED
                        fvg_top[k] = l[i]
            elif fvg_state[k] == -STATE_FRESH or fvg_state[k] == -STATE_MITIGATED:
                fvg_age[k] += 1
                if h[i] > fvg_bot[k]:
                    if c[i] > fvg_top[k]: fvg_state[k] = -STATE_INVERTED
                    elif h[i] >= fvg_top[k]: fvg_state[k] = -STATE_FILLED
                    else:
                        fvg_state[k] = -STATE_MITIGATED
                        fvg_bot[k] = h[i]
                        
            # OB Lifecycle
            if ob_state[k] == STATE_FRESH or ob_state[k] == STATE_MITIGATED:
                ob_age[k] += 1
                if c[i] < ob_bot[k]: ob_state[k] = STATE_INVERTED # Valid OB broken -> Breaker
                elif l[i] < ob_top[k]: ob_state[k] = STATE_MITIGATED
            elif ob_state[k] == -STATE_FRESH or ob_state[k] == -STATE_MITIGATED:
                ob_age[k] += 1
                if c[i] > ob_top[k]: ob_state[k] = -STATE_INVERTED # Valid OB broken -> Breaker
                elif h[i] > ob_bot[k]: ob_state[k] = -STATE_MITIGATED
                
            # Mitigation block handling
            if ob_state[k] == STATE_MITIGATION_BLK:
                if c[i] < ob_bot[k]: ob_state[k] = STATE_INVALIDATED
            elif ob_state[k] == -STATE_MITIGATION_BLK:
                if c[i] > ob_top[k]: ob_state[k] = -STATE_INVALIDATED
                
            # Reclaimed handling
            if ob_state[k] == STATE_INVERTED and c[i] > ob_top[k]: ob_state[k] = STATE_RECLAIMED
            if ob_state[k] == -STATE_INVERTED and c[i] < ob_bot[k]: ob_state[k] = -STATE_RECLAIMED

    return fvg_state, fvg_top, fvg_bot, fvg_age, ob_state, ob_top, ob_bot, ob_age, mss

# ==============================================================================
# VALIDATION & EXTRACTION HELPERS
# ==============================================================================

def validate_length(length: int, name: str = "length") -> None:
    if not isinstance(length, int) or length <= 0:
        logger.error(f"Invalid window length for {name}: {length}")
        raise SmartMoneyIndicatorError(f"Length for {name} must be a positive integer, got {length}")

def _finalize_output(output: Union[pd.Series, pd.DataFrame], offset: int, fillna: Any) -> Union[pd.Series, pd.DataFrame]:
    res = output.shift(offset) if offset != 0 else output
    if fillna is not None:
        res = res.fillna(fillna)
    return res

def _extract_ohlcv(data: Union[pd.DataFrame, pd.Series]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    if isinstance(data, pd.Series):
        raise SmartMoneyIndicatorError("OHLC DataFrame required.")
    df = standardize_column_names(data)
    validate_ohlc(df)
    o = df['open'].to_numpy(dtype=np.float64)
    h = df['high'].to_numpy(dtype=np.float64)
    l = df['low'].to_numpy(dtype=np.float64)
    c = df['close'].to_numpy(dtype=np.float64)
    
    if 'volume' in df.columns:
        v = df['volume'].to_numpy(dtype=np.float64)
    else:
        v = np.ones_like(c)
        
    return o, h, l, c, v, df.index

def get_price_source(data: Union[pd.DataFrame, pd.Series], source: str = 'close') -> pd.Series:
    if isinstance(data, pd.Series): return data
    df = standardize_column_names(data)
    src = source.lower().strip()
    if src in df.columns: return df[src]
    if src in COMPUTED_SOURCES:
        func, req_cols = COMPUTED_SOURCES[src]
        return func(*[df[c] for c in req_cols])
    raise SmartMoneyIndicatorError(f"Invalid source '{source}'.")

# ==============================================================================
# CORE EXECUTION WRAPPERS (With Global Memory Cache)
# ==============================================================================

_SMC_GLOBAL_CACHE = {}

def _run_core_smc(data: pd.DataFrame, length: int) -> dict:
    """Runs the core JIT engines with Global Memory Caching to prevent redundant executions."""
    global _SMC_GLOBAL_CACHE
    
    # Create a unique signature for the current dataframe state
    cache_key = (id(data), len(data), length)
    
    if cache_key in _SMC_GLOBAL_CACHE:
        return _SMC_GLOBAL_CACHE[cache_key]
        
    # Prevent memory leaks by keeping cache small
    if len(_SMC_GLOBAL_CACHE) > 10:
        _SMC_GLOBAL_CACHE.clear()

    o, h, l, c, v, idx = _extract_ohlcv(data)
    atr = _atr_jit(h, l, c, 14)
    
    sh, sl, hh, hl, lh, ll, bos, choch, trend = _structure_engine_jit(h, l, c, length)
    bsl_sw, ssl_sw, eqh, eql, rest_bsl, rest_ssl = _liquidity_engine_jit(h, l, c, sh, sl, atr)
    fvg_s, f_t, f_b, f_age, ob_s, o_t, o_b, o_age, mss = _ob_fvg_lifecycle_jit(o, h, l, c, v, atr, bos, bsl_sw, ssl_sw)
    
    result = {
        "idx": idx, "sh": sh, "sl": sl, "hh": hh, "hl": hl, "lh": lh, "ll": ll, 
        "bos": bos, "choch": choch, "trend": trend, "bsl_sw": bsl_sw, "ssl_sw": ssl_sw, 
        "eqh": eqh, "eql": eql, "rest_bsl": rest_bsl, "rest_ssl": rest_ssl,
        "fvg_s": fvg_s, "f_t": f_t, "f_b": f_b, "f_age": f_age,
        "ob_s": ob_s, "o_t": o_t, "o_b": o_b, "o_age": o_age, "mss": mss
    }
    
    _SMC_GLOBAL_CACHE[cache_key] = result
    return result

# ==============================================================================
# 1. MARKET STRUCTURE
# ==============================================================================

def swing_high(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = _run_core_smc(data, length)
    return _finalize_output(pd.Series(res['sh'], index=res['idx'], name="SwingHigh"), offset, fillna)

def swing_low(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = _run_core_smc(data, length)
    return _finalize_output(pd.Series(res['sl'], index=res['idx'], name="SwingLow"), offset, fillna)

def higher_high(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = _run_core_smc(data, length)
    return _finalize_output(pd.Series(res['hh'], index=res['idx'], name="HigherHigh"), offset, fillna)

def higher_low(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = _run_core_smc(data, length)
    return _finalize_output(pd.Series(res['hl'], index=res['idx'], name="HigherLow"), offset, fillna)

def lower_high(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = _run_core_smc(data, length)
    return _finalize_output(pd.Series(res['lh'], index=res['idx'], name="LowerHigh"), offset, fillna)

def lower_low(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = _run_core_smc(data, length)
    return _finalize_output(pd.Series(res['ll'], index=res['idx'], name="LowerLow"), offset, fillna)

def bos(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = _run_core_smc(data, length)
    return _finalize_output(pd.Series(res['bos'], index=res['idx'], name="BOS"), offset, fillna)

def internal_bos(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return bos(data, DEFAULT_INTERNAL_LEN, offset, fillna).rename("InternalBOS")
def external_bos(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return bos(data, DEFAULT_EXTERNAL_LEN, offset, fillna).rename("ExternalBOS")

def choch(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = _run_core_smc(data, length)
    return _finalize_output(pd.Series(res['choch'], index=res['idx'], name="CHOCH"), offset, fillna)

def internal_choch(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return choch(data, DEFAULT_INTERNAL_LEN, offset, fillna).rename("InternalCHOCH")
def external_choch(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return choch(data, DEFAULT_EXTERNAL_LEN, offset, fillna).rename("ExternalCHOCH")

def market_structure_shift(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = _run_core_smc(data, length)
    return _finalize_output(pd.Series(res['mss'], index=res['idx'], name="MSS"), offset, fillna)

def trend_state(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = _run_core_smc(data, DEFAULT_INTERNAL_LEN)
    return _finalize_output(pd.Series(res['trend'], index=res['idx'], name="TrendState"), offset, fillna)

def trend_bias(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = _run_core_smc(data, DEFAULT_EXTERNAL_LEN)
    return _finalize_output(pd.Series(res['trend'], index=res['idx'], name="TrendBias"), offset, fillna)

def structure_strength(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    int_t = _run_core_smc(data, DEFAULT_INTERNAL_LEN)['trend']
    ext_t = _run_core_smc(data, DEFAULT_EXTERNAL_LEN)['trend']
    strength = np.where(int_t == ext_t, 100.0, 50.0)
    return _finalize_output(pd.Series(strength, index=data.index, name="StructureStrength"), offset, fillna)

# ==============================================================================
# 2. LIQUIDITY
# ==============================================================================

def buy_side_liquidity(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = _run_core_smc(data, length)
    return _finalize_output(pd.Series(res['rest_bsl'], index=res['idx'], name="BSL"), offset, fillna)

def sell_side_liquidity(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = _run_core_smc(data, length)
    return _finalize_output(pd.Series(res['rest_ssl'], index=res['idx'], name="SSL"), offset, fillna)

def internal_liquidity(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return buy_side_liquidity(data, DEFAULT_INTERNAL_LEN, offset, fillna).rename("InternalLiquidity")
def external_liquidity(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return buy_side_liquidity(data, DEFAULT_EXTERNAL_LEN, offset, fillna).rename("ExternalLiquidity")

def liquidity_pool(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    res = _run_core_smc(data, length)
    out = pd.DataFrame({"BSL": res['rest_bsl'], "SSL": res['rest_ssl']}, index=res['idx'])
    return _finalize_output(out, offset, fillna)

def liquidity_sweep(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = _run_core_smc(data, length)
    sweep = np.where(res['bsl_sw'], -1, np.where(res['ssl_sw'], 1, 0))
    return _finalize_output(pd.Series(sweep, index=res['idx'], name="LiquiditySweep"), offset, fillna)

def liquidity_grab(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return liquidity_sweep(data, offset=offset, fillna=fillna).rename("LiquidityGrab")
def stop_hunt(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return liquidity_sweep(data, offset=offset, fillna=fillna).rename("StopHunt")

def equal_high(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = _run_core_smc(data, length)
    return _finalize_output(pd.Series(res['eqh'], index=res['idx'], name="EqualHigh"), offset, fillna)

def equal_low(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = _run_core_smc(data, length)
    return _finalize_output(pd.Series(res['eql'], index=res['idx'], name="EqualLow"), offset, fillna)

def resting_liquidity(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.DataFrame: return liquidity_pool(data, offset=offset, fillna=fillna)

def liquidity_density(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = _run_core_smc(data, length)
    dens = rolling_sum(pd.Series(res['eqh'].astype(int) + res['eql'].astype(int)), 20).to_numpy(dtype=np.float64)
    return _finalize_output(pd.Series(dens, index=res['idx'], name="LiquidityDensity"), offset, fillna)

def liquidity_strength(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    dens = liquidity_density(data, length).to_numpy()
    strength = np.clip(dens / 10.0 * 100.0, 0, 100.0)
    return _finalize_output(pd.Series(strength, index=data.index, name="LiquidityStrength"), offset, fillna)

# ==============================================================================
# 3. ORDER BLOCKS
# ==============================================================================

def bullish_order_block(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    res = _run_core_smc(data, length)
    active = (res['ob_s'] == STATE_FRESH) | (res['ob_s'] == STATE_MITIGATED)
    out = pd.DataFrame({"active": active, "top": np.where(active, res['o_t'], np.nan), "bottom": np.where(active, res['o_b'], np.nan)}, index=res['idx'])
    return _finalize_output(out, offset, fillna)

def bearish_order_block(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    res = _run_core_smc(data, length)
    active = (res['ob_s'] == -STATE_FRESH) | (res['ob_s'] == -STATE_MITIGATED)
    out = pd.DataFrame({"active": active, "top": np.where(active, res['o_t'], np.nan), "bottom": np.where(active, res['o_b'], np.nan)}, index=res['idx'])
    return _finalize_output(out, offset, fillna)

def fresh_order_block(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = _run_core_smc(data, length)
    return _finalize_output(pd.Series(np.abs(res['ob_s']) == STATE_FRESH, index=res['idx'], name="FreshOB"), offset, fillna)

def mitigated_order_block(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = _run_core_smc(data, length)
    return _finalize_output(pd.Series(np.abs(res['ob_s']) == STATE_MITIGATED, index=res['idx'], name="MitigatedOB"), offset, fillna)

def invalidated_order_block(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = _run_core_smc(data, length)
    return _finalize_output(pd.Series(np.abs(res['ob_s']) == STATE_INVALIDATED, index=res['idx'], name="InvalidatedOB"), offset, fillna)

def reclaimed_order_block(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = _run_core_smc(data, length)
    return _finalize_output(pd.Series(np.abs(res['ob_s']) == STATE_RECLAIMED, index=res['idx'], name="ReclaimedOB"), offset, fillna)

def breaker_block(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = _run_core_smc(data, length)
    return _finalize_output(pd.Series(np.abs(res['ob_s']) == STATE_INVERTED, index=res['idx'], name="BreakerBlock"), offset, fillna)

def mitigation_block(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = _run_core_smc(data, length)
    return _finalize_output(pd.Series(np.abs(res['ob_s']) == STATE_MITIGATION_BLK, index=res['idx'], name="MitigationBlock"), offset, fillna)

def order_block_strength(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, v, idx = _extract_ohlcv(data)
    res = _run_core_smc(data, length)
    size = res['o_t'] - res['o_b']
    size = np.where(size == 0, EPSILON, size)
    strength = np.where(res['ob_s'] != 0, v / size, 0.0)
    return _finalize_output(pd.Series(strength, index=idx, name="OBStrength"), offset, fillna)

def order_block_score(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    obs = order_block_strength(data, length).to_numpy()
    score = np.clip(safe_divide(obs, rolling_mean(pd.Series(obs), 50).to_numpy(), default=1.0) * 10.0, 0, 100.0)
    return _finalize_output(pd.Series(score, index=data.index, name="OBScore"), offset, fillna)

def order_block_age(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = _run_core_smc(data, length)
    return _finalize_output(pd.Series(res['o_age'], index=res['idx'], name="OBAge"), offset, fillna)

# ==============================================================================
# 4. FAIR VALUE GAPS
# ==============================================================================

def bullish_fvg(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    res = _run_core_smc(data, length)
    valid = (res['fvg_s'] == STATE_FRESH) | (res['fvg_s'] == STATE_MITIGATED)
    out = pd.DataFrame({"active": valid, "top": np.where(valid, res['f_t'], np.nan), "bottom": np.where(valid, res['f_b'], np.nan)}, index=res['idx'])
    return _finalize_output(out, offset, fillna)

def bearish_fvg(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    res = _run_core_smc(data, length)
    valid = (res['fvg_s'] == -STATE_FRESH) | (res['fvg_s'] == -STATE_MITIGATED)
    out = pd.DataFrame({"active": valid, "top": np.where(valid, res['f_t'], np.nan), "bottom": np.where(valid, res['f_b'], np.nan)}, index=res['idx'])
    return _finalize_output(out, offset, fillna)

def inverse_fvg(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    res = _run_core_smc(data, length)
    valid = (np.abs(res['fvg_s']) == STATE_INVERTED)
    out = pd.DataFrame({"active": valid, "top": np.where(valid, res['f_t'], np.nan), "bottom": np.where(valid, res['f_b'], np.nan)}, index=res['idx'])
    return _finalize_output(out, offset, fillna)

def active_fvg(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = _run_core_smc(data, length)
    active = (np.abs(res['fvg_s']) == STATE_FRESH) | (np.abs(res['fvg_s']) == STATE_MITIGATED)
    return _finalize_output(pd.Series(active, index=res['idx'], name="ActiveFVG"), offset, fillna)

def mitigated_fvg(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = _run_core_smc(data, length)
    return _finalize_output(pd.Series(np.abs(res['fvg_s']) == STATE_MITIGATED, index=res['idx'], name="MitigatedFVG"), offset, fillna)

def filled_fvg(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = _run_core_smc(data, length)
    return _finalize_output(pd.Series(np.abs(res['fvg_s']) == STATE_FILLED, index=res['idx'], name="FilledFVG"), offset, fillna)

def fvg_strength(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = _run_core_smc(data, length)
    strength = np.where(res['fvg_s'] != 0, np.abs(res['f_t'] - res['f_b']), 0.0)
    return _finalize_output(pd.Series(strength, index=res['idx'], name="FVGStrength"), offset, fillna)

def fvg_width(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return fvg_strength(data, offset=offset, fillna=fillna).rename("FVGWidth")

def fvg_age(data: pd.DataFrame, length: int = DEFAULT_INTERNAL_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = _run_core_smc(data, length)
    return _finalize_output(pd.Series(res['f_age'], index=res['idx'], name="FVGAge"), offset, fillna)

# ==============================================================================
# 5. PREMIUM / DISCOUNT ZONES
# ==============================================================================

def premium_zone(data: pd.DataFrame, length: int = 20, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, v, idx = _extract_ohlcv(data)
    rhh = pd.Series(h).rolling(length).max().to_numpy()
    rll = pd.Series(l).rolling(length).min().to_numpy()
    return _finalize_output(pd.Series(c > (rhh + rll)/2.0, index=idx, name="PremiumZone"), offset, fillna)

def discount_zone(data: pd.DataFrame, length: int = 20, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, v, idx = _extract_ohlcv(data)
    rhh = pd.Series(h).rolling(length).max().to_numpy()
    rll = pd.Series(l).rolling(length).min().to_numpy()
    return _finalize_output(pd.Series(c < (rhh + rll)/2.0, index=idx, name="DiscountZone"), offset, fillna)

def equilibrium(data: pd.DataFrame, length: int = 20, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, v, idx = _extract_ohlcv(data)
    rhh = pd.Series(h).rolling(length).max().to_numpy()
    rll = pd.Series(l).rolling(length).min().to_numpy()
    return _finalize_output(pd.Series((rhh + rll)/2.0, index=idx, name="Equilibrium"), offset, fillna)

def ote_zone(data: pd.DataFrame, length: int = 20, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    o, h, l, c, v, idx = _extract_ohlcv(data)
    rhh = pd.Series(h).rolling(length).max().to_numpy()
    rll = pd.Series(l).rolling(length).min().to_numpy()
    rng = rhh - rll
    out = pd.DataFrame({"ote_lower": rll + rng * DEFAULT_OTE_MIN, "ote_upper": rll + rng * DEFAULT_OTE_MAX}, index=idx)
    return _finalize_output(out, offset, fillna)

def premium_score(data: pd.DataFrame, length: int = 20, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, v, idx = _extract_ohlcv(data)
    rhh = pd.Series(h).rolling(length).max().to_numpy()
    rll = pd.Series(l).rolling(length).min().to_numpy()
    rng = np.where((rhh - rll) == 0, EPSILON, rhh - rll)
    score = np.clip((c - rll) / rng * 100.0, 0, 100)
    return _finalize_output(pd.Series(score, index=idx, name="PremiumScore"), offset, fillna)

def discount_score(data: pd.DataFrame, length: int = 20, offset: int = 0, fillna: Any = None) -> pd.Series:
    return _finalize_output((100.0 - premium_score(data, length)).rename("DiscountScore"), offset, fillna)

# ==============================================================================
# 6. INSTITUTIONAL FOOTPRINTS
# ==============================================================================

def displacement_candle(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, v, idx = _extract_ohlcv(data)
    atr = _atr_jit(h, l, c, 14)
    res = np.abs(c - o) > (atr * 1.5)
    return _finalize_output(pd.Series(res, index=idx, name="DisplacementCandle"), offset, fillna)

def impulse_candle(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return displacement_candle(data, offset=offset, fillna=fillna).rename("ImpulseCandle")
def expansion_candle(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return displacement_candle(data, offset=offset, fillna=fillna).rename("ExpansionCandle")

def absorption_candle(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, v, idx = _extract_ohlcv(data)
    v_sma = rolling_mean(pd.Series(v), 20).to_numpy()
    res = (v > v_sma * 1.5) & (np.abs(c - o) < (h - l) * 0.3)
    return _finalize_output(pd.Series(res, index=idx, name="AbsorptionCandle"), offset, fillna)

def delta_proxy(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, v, idx = _extract_ohlcv(data)
    rng = np.where((h - l) == 0, EPSILON, h - l)
    delta = v * (((c - l) / rng) - ((h - c) / rng))
    return _finalize_output(pd.Series(delta, index=idx, name="DeltaProxy"), offset, fillna)

def institutional_volume_proxy(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, v, idx = _extract_ohlcv(data)
    return _finalize_output(pd.Series(v * (h - l), index=idx, name="InstVolProxy"), offset, fillna)

def smart_money_footprint(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    o, h, l, c, v, idx = _extract_ohlcv(data)
    disp = displacement_candle(data).to_numpy()
    v_sma = rolling_mean(pd.Series(v), 20).to_numpy()
    res = disp & (v > v_sma * 2.0)
    return _finalize_output(pd.Series(res, index=idx, name="SmartMoneyFootprint"), offset, fillna)

# ==============================================================================
# 7. CONFLUENCE & SCORING
# ==============================================================================

def bos_score(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return _finalize_output((np.abs(bos(data)) * 100.0).rename("BOSScore"), offset, fillna)
def choch_score(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return _finalize_output((np.abs(choch(data)) * 100.0).rename("CHOCHScore"), offset, fillna)
def liquidity_score(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return _finalize_output((np.abs(liquidity_sweep(data)) * 100.0).rename("LiquidityScore"), offset, fillna)
def fvg_score(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    w = fvg_width(data).to_numpy()
    mx = np.nanmax(w) if not np.isnan(np.nanmax(w)) and np.nanmax(w) != 0 else EPSILON
    res = np.where(active_fvg(data).to_numpy(), np.clip(w / mx, 0, 1) * 100.0, 0.0)
    return _finalize_output(pd.Series(res, index=data.index, name="FVGScore"), offset, fillna)

def trend_score(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series: return structure_strength(data, offset=offset, fillna=fillna).rename("TrendScore")

def institutional_score(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    iv = institutional_volume_proxy(data).to_numpy()
    sma = rolling_mean(pd.Series(iv), 50).to_numpy()
    sma = np.where(sma == 0, EPSILON, sma)
    return _finalize_output(pd.Series(np.clip((iv / sma) * 20.0, 0, 100), index=data.index, name="InstScore"), offset, fillna)

def smart_money_score(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    res = _run_core_smc(data, DEFAULT_INTERNAL_LEN) # Extracted once
    
    b_score = np.abs(res['bos']) * 15.0
    c_score = np.abs(res['choch']) * 15.0
    l_score = np.abs(np.where(res['bsl_sw'], -1, np.where(res['ssl_sw'], 1, 0))) * 20.0
    o_score = np.where(res['ob_s'] != 0, 20.0, 0.0)
    f_score = np.where(res['fvg_s'] != 0, 10.0, 0.0)
    t_score = np.where(res['trend'] != 0, 10.0, 0.0)
    
    iv = _extract_ohlcv(data)[4] * (_extract_ohlcv(data)[1] - _extract_ohlcv(data)[2])
    sma = rolling_mean(pd.Series(iv), 50).to_numpy()
    sma = np.where(sma == 0, EPSILON, sma)
    i_score = np.clip((iv / sma) * 2.0, 0, 10.0)
    
    total = np.clip(b_score + c_score + l_score + o_score + f_score + t_score + i_score, 0, 100.0)
    return _finalize_output(pd.Series(total, index=res['idx'], name="SmartMoneyScore"), offset, fillna)

def confidence_score(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    return smart_money_score(data, offset=offset, fillna=fillna).rename("ConfidenceScore")

# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "SmartMoneyIndicatorError", "validate_length", "get_price_source",
    "swing_high", "swing_low", "higher_high", "higher_low", "lower_high", "lower_low",
    "bos", "internal_bos", "external_bos", "choch", "internal_choch", "external_choch",
    "market_structure_shift", "trend_state", "trend_bias", "structure_strength",
    "buy_side_liquidity", "sell_side_liquidity", "internal_liquidity", "external_liquidity",
    "liquidity_pool", "liquidity_sweep", "liquidity_grab", "stop_hunt",
    "equal_high", "equal_low", "resting_liquidity", "liquidity_density", "liquidity_strength",
    "bullish_order_block", "bearish_order_block", "fresh_order_block", "mitigated_order_block",
    "invalidated_order_block", "reclaimed_order_block", "breaker_block", "mitigation_block",
    "order_block_strength", "order_block_score", "order_block_age",
    "bullish_fvg", "bearish_fvg", "inverse_fvg", "active_fvg", "mitigated_fvg", "filled_fvg",
    "fvg_strength", "fvg_width", "fvg_age",
    "premium_zone", "discount_zone", "equilibrium", "ote_zone", "premium_score", "discount_score",
    "displacement_candle", "impulse_candle", "expansion_candle", "absorption_candle",
    "delta_proxy", "institutional_volume_proxy", "smart_money_footprint",
    "bos_score", "choch_score", "liquidity_score", "fvg_score", "trend_score", 
    "institutional_score", "smart_money_score", "confidence_score"
]
