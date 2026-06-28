import logging
import numpy as np
import pandas as pd
from typing import Union, Optional, Any

from backend.indicators.helper import (
    ensure_series,
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
    EPSILON
)

# ==============================================================================
# LOGGING & CONSTANTS
# ==============================================================================

logger = logging.getLogger(__name__)

DEFAULT_RSI_LEN = 14
DEFAULT_SLOPE_LEN = 1
DEFAULT_FAST_LEN = 12
DEFAULT_SLOW_LEN = 26
DEFAULT_SIGNAL_LEN = 9
DEFAULT_ADX_LEN = 14
DEFAULT_ROC_LEN = 12
DEFAULT_CCI_LEN = 20
DEFAULT_CCI_CONSTANT = 0.015
DEFAULT_MOM_LEN = 10
DEFAULT_TRIX_LEN = 18
DEFAULT_DPO_LEN = 20
DEFAULT_STOCH_K = 14
DEFAULT_STOCH_D = 3
DEFAULT_STOCH_SMOOTH = 3
DEFAULT_WILLIAMS_LEN = 14
DEFAULT_ULT_LEN1 = 7
DEFAULT_ULT_LEN2 = 14
DEFAULT_ULT_LEN3 = 28
DEFAULT_ULT_W1 = 4.0
DEFAULT_ULT_W2 = 2.0
DEFAULT_ULT_W3 = 1.0

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

class MomentumIndicatorError(Exception):
    """Custom exception for errors in momentum indicator calculations."""
    pass

# ==============================================================================
# COMMON UTILITIES
# ==============================================================================

def validate_length(length: int, name: str = "length") -> None:
    if not isinstance(length, int) or length <= 0:
        logger.error(f"Invalid momentum window length for {name}: {length}")
        raise MomentumIndicatorError(f"Length for {name} must be a positive integer, got {length}")

def get_price_source(data: Union[pd.DataFrame, pd.Series], source: str = 'close') -> pd.Series:
    """Extracts price source. Excludes DatetimeIndex validation to support RangeIndex."""
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
            raise MomentumIndicatorError(f"DataFrame is missing columns needed for '{src}': {missing}")
        return func(*[df[c] for c in req_cols])
            
    logger.error(f"Invalid source '{source}'.")
    raise MomentumIndicatorError(f"Invalid source '{source}' or column not found.")

def _finalize_output(output: Union[pd.Series, pd.DataFrame], offset: int, fillna: Any) -> Union[pd.Series, pd.DataFrame]:
    res = output.shift(offset) if offset != 0 else output
    if fillna is not None:
        res = res.fillna(fillna)
    return res

def _wilder_rma(arr: np.ndarray, length: int) -> np.ndarray:
    """
    Wilder's Smoothing (Running Moving Average).
    Vectorized contiguous window search for precise SMA seeding.
    """
    out = np.full_like(arr, np.nan)
    valid_mask = ~np.isnan(arr)
    
    # Vectorized search for the first contiguous block of 'length' valid values
    window_sums = np.convolve(valid_mask.astype(int), np.ones(length, dtype=int), mode='valid')
    valid_starts = np.where(window_sums == length)[0]
    
    if len(valid_starts) == 0:
        return out
        
    seed_idx = valid_starts[0] + length - 1
    
    # Seed with SMA of the contiguous window
    out[seed_idx] = np.mean(arr[valid_starts[0] : seed_idx + 1])
    alpha = 1.0 / length
    
    # FUTURE_OPTIMIZATION: Candidate for Numba @jit
    for i in range(seed_idx + 1, len(arr)):
        if np.isnan(arr[i]):
            out[i] = out[i-1]
        else:
            out[i] = alpha * arr[i] + (1.0 - alpha) * out[i-1]
            
    return out

def _rolling_mad(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Ultra-fast vectorized Rolling Mean Absolute Deviation (MAD) using stride tricks.
    """
    if len(arr) < window:
        return np.full_like(arr, np.nan)
        
    # Safely pad NaNs to prevent stride trick poisoning
    s_filled = pd.Series(arr).ffill().bfill().to_numpy(dtype=np.float64)
    
    views = np.lib.stride_tricks.sliding_window_view(s_filled, window)
    means = np.mean(views, axis=1, keepdims=True)
    mads = np.mean(np.abs(views - means), axis=1)
    
    out = np.full_like(arr, np.nan)
    out[window - 1:] = mads
    out[np.isnan(arr)] = np.nan
    return out

# ==============================================================================
# INDICATOR IMPLEMENTATIONS
# ==============================================================================

def rsi(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_RSI_LEN, source: str = 'close',
        offset: int = 0, fillna: Any = None) -> pd.Series:
    """Relative Strength Index (RSI). Uses precise Wilder's RMA for TradingView parity."""
    validate_length(length, "RSI")
    s = get_price_source(data, source)
    arr = s.to_numpy(dtype=np.float64)
    
    delta = np.diff(arr, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # NaN seeding for exact parity
    gain[0] = np.nan
    loss[0] = np.nan
    
    avg_gain = _wilder_rma(gain, length)
    avg_loss = _wilder_rma(loss, length)
    
    rs = safe_divide(avg_gain, avg_loss, default=np.nan)
    res = 100.0 - (100.0 / (1.0 + rs))
    
    res = np.where(avg_loss == 0.0, np.where(avg_gain == 0.0, 50.0, 100.0), res)
    
    out = pd.Series(res, index=s.index, name=f"RSI_{length}")
    return _finalize_output(out, offset, fillna)


def rsi_slope(data: Union[pd.DataFrame, pd.Series], rsi_length: int = DEFAULT_RSI_LEN, 
              slope_length: int = DEFAULT_SLOPE_LEN, source: str = 'close', offset: int = 0, 
              fillna: Any = None) -> pd.Series:
    """Derivative (Momentum) of the RSI curve."""
    validate_length(slope_length, "RSI Slope")
    rsi_series = rsi(data, length=rsi_length, source=source)
    
    res = rsi_series.diff(periods=slope_length)
    res.name = f"RSISlope_{rsi_length}_{slope_length}"
    return _finalize_output(res, offset, fillna)


def macd(data: Union[pd.DataFrame, pd.Series], fast_length: int = DEFAULT_FAST_LEN,
         slow_length: int = DEFAULT_SLOW_LEN, signal_length: int = DEFAULT_SIGNAL_LEN,
         source: str = 'close', offset: int = 0, fillna: Any = None, 
         min_periods: Optional[int] = None) -> pd.DataFrame:
    """Moving Average Convergence Divergence (MACD). Returns macd, signal, histogram."""
    validate_length(fast_length, "MACD Fast")
    validate_length(slow_length, "MACD Slow")
    validate_length(signal_length, "MACD Signal")
    
    s = get_price_source(data, source)
    
    fast_min_p = min_periods if min_periods is not None else fast_length
    slow_min_p = min_periods if min_periods is not None else slow_length
    sig_min_p = min_periods if min_periods is not None else signal_length
    
    fast_ema = s.ewm(span=fast_length, min_periods=fast_min_p, adjust=False).mean()
    slow_ema = s.ewm(span=slow_length, min_periods=slow_min_p, adjust=False).mean()
    
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_length, min_periods=sig_min_p, adjust=False).mean()
    histogram = macd_line - signal_line
    
    out = pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram
    }, index=s.index)
    return _finalize_output(out, offset, fillna)


def macd_signal(data: Union[pd.DataFrame, pd.Series], **kwargs) -> pd.Series:
    """Standalone wrapper extracting MACD Signal Line."""
    return macd(data, **kwargs)['signal']


def macd_histogram(data: Union[pd.DataFrame, pd.Series], **kwargs) -> pd.Series:
    """Standalone wrapper extracting MACD Histogram."""
    return macd(data, **kwargs)['histogram']


def adx(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_ADX_LEN, offset: int = 0, 
        fillna: Any = None) -> pd.DataFrame:
    """Average Directional Index (ADX). Uses precise Wilder's RMA. Returns adx, plus_di, minus_di, dx."""
    validate_length(length, "ADX")
    if isinstance(data, pd.Series):
        raise MomentumIndicatorError("ADX calculation requires full OHLC DataFrame.")
        
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    high = df['high'].to_numpy(dtype=np.float64)
    low = df['low'].to_numpy(dtype=np.float64)
    close = df['close'].to_numpy(dtype=np.float64)
    
    up_move = np.diff(high, prepend=np.nan)
    down_move = -np.diff(low, prepend=np.nan)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # NaN seeding for exact parity
    plus_dm[0] = np.nan
    minus_dm[0] = np.nan
    
    prev_close = np.empty_like(close)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]
    
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    smooth_tr = _wilder_rma(tr, length)
    smooth_plus_dm = _wilder_rma(plus_dm, length)
    smooth_minus_dm = _wilder_rma(minus_dm, length)
    
    plus_di_arr = 100.0 * safe_divide(smooth_plus_dm, smooth_tr, default=0.0)
    minus_di_arr = 100.0 * safe_divide(smooth_minus_dm, smooth_tr, default=0.0)
    
    di_sum = plus_di_arr + minus_di_arr
    di_sum_safe = np.where(di_sum == 0.0, EPSILON, di_sum)
    dx = 100.0 * np.abs(plus_di_arr - minus_di_arr) / di_sum_safe
    
    adx_arr = _wilder_rma(dx, length)
    
    out = pd.DataFrame({
        "adx": adx_arr,
        "plus_di": plus_di_arr,
        "minus_di": minus_di_arr,
        "dx": dx
    }, index=df.index)
    return _finalize_output(out, offset, fillna)


def plus_di(data: Union[pd.DataFrame, pd.Series], **kwargs) -> pd.Series:
    """Standalone wrapper extracting DI+ from ADX."""
    return adx(data, **kwargs)['plus_di']


def minus_di(data: Union[pd.DataFrame, pd.Series], **kwargs) -> pd.Series:
    """Standalone wrapper extracting DI- from ADX."""
    return adx(data, **kwargs)['minus_di']


def roc(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_ROC_LEN, source: str = 'close',
        offset: int = 0, fillna: Any = None) -> pd.Series:
    """Rate of Change (ROC)."""
    validate_length(length, "ROC")
    s = get_price_source(data, source)
    
    shifted = s.shift(length)
    res = 100.0 * safe_divide((s - shifted).to_numpy(), shifted.to_numpy(), default=np.nan)
    
    out = pd.Series(res, index=s.index, name=f"ROC_{length}")
    return _finalize_output(out, offset, fillna)


def cci(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_CCI_LEN, offset: int = 0, 
        fillna: Any = None) -> pd.Series:
    """Commodity Channel Index (CCI). Uses ultra-fast sliding window view for MAD."""
    validate_length(length, "CCI")
    
    if isinstance(data, pd.Series):
        tp = data
    else:
        df = standardize_column_names(data)
        validate_ohlc(df)
        tp = hlc3(df['high'], df['low'], df['close'])
        
    tp_series = ensure_series(tp)
    tp_arr = tp_series.to_numpy(dtype=np.float64)
    
    sma_arr = rolling_mean(tp_series, window=length).to_numpy(dtype=np.float64)
    mad_arr = _rolling_mad(tp_arr, window=length)
    
    res = safe_divide((tp_arr - sma_arr), (DEFAULT_CCI_CONSTANT * mad_arr), default=np.nan)
    
    out = pd.Series(res, index=tp_series.index, name=f"CCI_{length}")
    return _finalize_output(out, offset, fillna)


def momentum(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_MOM_LEN, source: str = 'close',
             offset: int = 0, fillna: Any = None) -> pd.Series:
    """Absolute Momentum (Price minus Price n-periods ago)."""
    validate_length(length, "Momentum")
    s = get_price_source(data, source)
    
    res = s - s.shift(length)
    res.name = f"MOM_{length}"
    return _finalize_output(res, offset, fillna)


def trix(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_TRIX_LEN, source: str = 'close',
         offset: int = 0, fillna: Any = None) -> pd.Series:
    """TRIX (1-period rate of change of a triple exponentially smoothed EMA)."""
    validate_length(length, "TRIX")
    s = get_price_source(data, source)
    
    ema1 = s.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    ema3 = ema2.ewm(span=length, adjust=False).mean()
    
    res = ema3.pct_change(periods=1) * 100.0
    res.name = f"TRIX_{length}"
    return _finalize_output(res, offset, fillna)


def ppo(data: Union[pd.DataFrame, pd.Series], fast_length: int = DEFAULT_FAST_LEN,
        slow_length: int = DEFAULT_SLOW_LEN, signal_length: int = DEFAULT_SIGNAL_LEN, 
        source: str = 'close', offset: int = 0, fillna: Any = None,
        min_periods: Optional[int] = None) -> pd.DataFrame:
    """Percentage Price Oscillator (PPO). Returns ppo, signal, histogram."""
    validate_length(fast_length, "PPO Fast")
    validate_length(slow_length, "PPO Slow")
    validate_length(signal_length, "PPO Signal")
    
    s = get_price_source(data, source)
    
    fast_min_p = min_periods if min_periods is not None else fast_length
    slow_min_p = min_periods if min_periods is not None else slow_length
    sig_min_p = min_periods if min_periods is not None else signal_length
    
    fast_ema = s.ewm(span=fast_length, min_periods=fast_min_p, adjust=False).mean()
    slow_ema = s.ewm(span=slow_length, min_periods=slow_min_p, adjust=False).mean()
    
    ppo_line = 100.0 * safe_divide((fast_ema - slow_ema).to_numpy(), slow_ema.to_numpy(), default=np.nan)
    ppo_series = pd.Series(ppo_line, index=s.index)
    
    signal_line = ppo_series.ewm(span=signal_length, min_periods=sig_min_p, adjust=False).mean()
    histogram = ppo_series - signal_line
    
    out = pd.DataFrame({
        "ppo": ppo_series,
        "signal": signal_line,
        "histogram": histogram
    }, index=s.index)
    return _finalize_output(out, offset, fillna)


def dpo(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_DPO_LEN, source: str = 'close',
        offset: int = 0, fillna: Any = None) -> pd.Series:
    """Detrended Price Oscillator (DPO)."""
    validate_length(length, "DPO")
    s = get_price_source(data, source)
    
    shift_len = int((length / 2) + 1)
    sma_back = rolling_mean(s, window=length).shift(shift_len)
    
    res = s - sma_back
    res.name = f"DPO_{length}"
    return _finalize_output(res, offset, fillna)


def stochastic(data: pd.DataFrame, length_k: int = DEFAULT_STOCH_K, 
               length_d: int = DEFAULT_STOCH_D, smooth_k: int = DEFAULT_STOCH_SMOOTH,
               offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    """Stochastic Oscillator (TradingView matching logic). Returns k and d."""
    validate_length(length_k, "Stochastic K Window")
    validate_length(length_d, "Stochastic D Window")
    validate_length(smooth_k, "Stochastic K Smooth")
    
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    lowest_low = df['low'].rolling(window=length_k).min()
    highest_high = df['high'].rolling(window=length_k).max()
    
    num = (df['close'] - lowest_low).to_numpy(dtype=np.float64)
    den = (highest_high - lowest_low).to_numpy(dtype=np.float64)
    
    fast_k = 100.0 * safe_divide(num, den, default=np.nan)
    fast_k_series = pd.Series(fast_k, index=df.index)
    
    k_line = rolling_mean(fast_k_series, window=smooth_k) if smooth_k > 1 else fast_k_series
    d_line = rolling_mean(k_line, window=length_d)
    
    out = pd.DataFrame({"k": k_line, "d": d_line}, index=df.index)
    return _finalize_output(out, offset, fillna)


def stochastic_rsi(data: Union[pd.DataFrame, pd.Series], length_rsi: int = DEFAULT_RSI_LEN,
                   length_stoch: int = DEFAULT_STOCH_K, smooth_k: int = DEFAULT_STOCH_SMOOTH, 
                   smooth_d: int = DEFAULT_STOCH_D, source: str = 'close', offset: int = 0,
                   fillna: Any = None) -> pd.DataFrame:
    """Stochastic RSI. Matches TradingView logic strictly."""
    validate_length(length_rsi, "StochRSI Base RSI")
    validate_length(length_stoch, "StochRSI Length")
    validate_length(smooth_k, "StochRSI K Smooth")
    validate_length(smooth_d, "StochRSI D Smooth")
    
    rsi_series = rsi(data, length=length_rsi, source=source)
    
    lowest_rsi = rsi_series.rolling(window=length_stoch).min()
    highest_rsi = rsi_series.rolling(window=length_stoch).max()
    
    num = (rsi_series - lowest_rsi).to_numpy(dtype=np.float64)
    den = (highest_rsi - lowest_rsi).to_numpy(dtype=np.float64)
    
    stoch_rsi = 100.0 * safe_divide(num, den, default=np.nan)
    stoch_rsi_series = pd.Series(stoch_rsi, index=rsi_series.index)
    
    k_line = rolling_mean(stoch_rsi_series, window=smooth_k) if smooth_k > 1 else stoch_rsi_series
    d_line = rolling_mean(k_line, window=smooth_d)
    
    out = pd.DataFrame({"k": k_line, "d": d_line}, index=rsi_series.index)
    return _finalize_output(out, offset, fillna)


def williams_r(data: pd.DataFrame, length: int = DEFAULT_WILLIAMS_LEN, offset: int = 0,
               fillna: Any = None) -> pd.Series:
    """Williams %R."""
    validate_length(length, "Williams %R")
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    highest_high = df['high'].rolling(window=length).max()
    lowest_low = df['low'].rolling(window=length).min()
    
    num = (highest_high - df['close']).to_numpy(dtype=np.float64)
    den = (highest_high - lowest_low).to_numpy(dtype=np.float64)
    
    res = -100.0 * safe_divide(num, den, default=np.nan)
    
    out = pd.Series(res, index=df.index, name=f"WilliamsR_{length}")
    return _finalize_output(out, offset, fillna)


def ultimate_oscillator(data: pd.DataFrame, length1: int = DEFAULT_ULT_LEN1, length2: int = DEFAULT_ULT_LEN2,
                        length3: int = DEFAULT_ULT_LEN3, weight1: float = DEFAULT_ULT_W1,
                        weight2: float = DEFAULT_ULT_W2, weight3: float = DEFAULT_ULT_W3,
                        offset: int = 0, fillna: Any = None) -> pd.Series:
    """Ultimate Oscillator (Larry Williams)."""
    validate_length(length1, "Ult Osc Length 1")
    validate_length(length2, "Ult Osc Length 2")
    validate_length(length3, "Ult Osc Length 3")
    
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    prev_close = df['close'].shift(1)
    
    bp = df['close'] - np.minimum(df['low'], prev_close)
    tr = np.maximum(df['high'], prev_close) - np.minimum(df['low'], prev_close)
    
    bp_series = ensure_series(bp)
    tr_series = ensure_series(tr)
    
    avg1 = safe_divide(rolling_sum(bp_series, length1).to_numpy(), rolling_sum(tr_series, length1).to_numpy(), default=0.0)
    avg2 = safe_divide(rolling_sum(bp_series, length2).to_numpy(), rolling_sum(tr_series, length2).to_numpy(), default=0.0)
    avg3 = safe_divide(rolling_sum(bp_series, length3).to_numpy(), rolling_sum(tr_series, length3).to_numpy(), default=0.0)
    
    weight_sum = weight1 + weight2 + weight3
    res = 100.0 * ((weight1 * avg1) + (weight2 * avg2) + (weight3 * avg3)) / weight_sum
    
    out = pd.Series(res, index=df.index, name="ULTOSC")
    return _finalize_output(out, offset, fillna)

# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "MomentumIndicatorError",
    "validate_length",
    "get_price_source",
    "rsi",
    "rsi_slope",
    "macd",
    "macd_signal",
    "macd_histogram",
    "adx",
    "plus_di",
    "minus_di",
    "roc",
    "cci",
    "momentum",
    "trix",
    "ppo",
    "dpo",
    "stochastic",
    "stochastic_rsi",
    "williams_r",
    "ultimate_oscillator"
]
