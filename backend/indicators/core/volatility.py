import logging
import numpy as np
import pandas as pd
from typing import Union, Optional, Any, Tuple

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
    rolling_std,
    EPSILON
)

# ==============================================================================
# LOGGING & CONSTANTS
# ==============================================================================

logger = logging.getLogger(__name__)

DEFAULT_ATR_LEN = 14
DEFAULT_BB_LEN = 20
DEFAULT_BB_STD = 2.0
DEFAULT_DONCHIAN_LEN = 20
DEFAULT_KELTNER_LEN = 20
DEFAULT_KELTNER_ATR_LEN = 10
DEFAULT_KELTNER_MULT = 2.0
DEFAULT_HV_LEN = 21
DEFAULT_ANNUALIZATION = 252.0
DEFAULT_CHAIKIN_LEN = 10
DEFAULT_CHAIKIN_ROC = 10
DEFAULT_ULCER_LEN = 14
DEFAULT_MASS_LEN = 25
DEFAULT_MASS_EMA = 9
DEFAULT_VR_LEN = 14
DEFAULT_PERCENTILE_LOOKBACK = 252
DEFAULT_CHOP_LEN = 14
DEFAULT_VHF_LEN = 28
DEFAULT_REI_LEN = 8

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

class VolatilityIndicatorError(Exception):
    """Custom exception for errors in volatility indicator calculations."""
    pass

# ==============================================================================
# COMMON UTILITIES
# ==============================================================================

def validate_length(length: int, name: str = "length") -> None:
    if not isinstance(length, int) or length <= 0:
        logger.error(f"Invalid volatility window length for {name}: {length}")
        raise VolatilityIndicatorError(f"Length for {name} must be a positive integer, got {length}")

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
            raise VolatilityIndicatorError(f"DataFrame is missing columns needed for '{src}': {missing}")
        return func(*[df[c] for c in req_cols])
            
    logger.error(f"Invalid source '{source}'.")
    raise VolatilityIndicatorError(f"Invalid source '{source}' or column not found.")

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
    
    window_sums = np.convolve(valid_mask.astype(int), np.ones(length, dtype=int), mode='valid')
    valid_starts = np.where(window_sums == length)[0]
    
    if len(valid_starts) == 0:
        return out
        
    seed_idx = valid_starts[0] + length - 1
    out[seed_idx] = np.mean(arr[valid_starts[0] : seed_idx + 1])
    alpha = 1.0 / length
    
    for i in range(seed_idx + 1, len(arr)):
        if np.isnan(arr[i]):
            out[i] = out[i-1]
        else:
            out[i] = alpha * arr[i] + (1.0 - alpha) * out[i-1]
            
    return out

# ==============================================================================
# INDICATOR IMPLEMENTATIONS
# ==============================================================================

def true_range(data: pd.DataFrame, offset: int = 0, fillna: Any = None) -> pd.Series:
    """True Range (TR). Mathematically exact max of High-Low, abs(High-PrevClose), abs(Low-PrevClose)."""
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    high = df['high'].to_numpy(dtype=np.float64)
    low = df['low'].to_numpy(dtype=np.float64)
    close = df['close'].to_numpy(dtype=np.float64)
    
    prev_close = np.empty_like(close)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]
    
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    out = pd.Series(tr, index=df.index, name="TR")
    return _finalize_output(out, offset, fillna)


def atr(data: pd.DataFrame, length: int = DEFAULT_ATR_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    """Average True Range (ATR). Uses Wilder's RMA for exact TradingView compatibility."""
    validate_length(length, "ATR")
    tr_series = true_range(data)
    tr_arr = tr_series.to_numpy(dtype=np.float64)
    
    atr_arr = _wilder_rma(tr_arr, length)
    
    out = pd.Series(atr_arr, index=tr_series.index, name=f"ATR_{length}")
    return _finalize_output(out, offset, fillna)


def natr(data: pd.DataFrame, length: int = DEFAULT_ATR_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    """Normalized Average True Range (NATR). (100 * ATR / Close)"""
    validate_length(length, "NATR")
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    atr_series = atr(data, length=length)
    close_series = df['close']
    
    natr_arr = 100.0 * safe_divide(atr_series.to_numpy(), close_series.to_numpy(), default=np.nan)
    
    out = pd.Series(natr_arr, index=df.index, name=f"NATR_{length}")
    return _finalize_output(out, offset, fillna)


def bollinger_bands(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_BB_LEN, 
                    std_mult: float = DEFAULT_BB_STD, source: str = 'close', 
                    offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    """Bollinger Bands. Returns middle, upper, lower."""
    validate_length(length, "Bollinger Bands")
    s = get_price_source(data, source)
    
    middle = rolling_mean(s, window=length)
    std = rolling_std(s, window=length)
    
    upper = middle + (std_mult * std)
    lower = middle - (std_mult * std)
    
    out = pd.DataFrame({
        "middle": middle,
        "upper": upper,
        "lower": lower
    }, index=s.index)
    return _finalize_output(out, offset, fillna)


def bollinger_width(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_BB_LEN, 
                    std_mult: float = DEFAULT_BB_STD, percentage: bool = True,
                    source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    """Bollinger Bands Width. Returns standard ratio or percentage scale."""
    bb = bollinger_bands(data, length=length, std_mult=std_mult, source=source)
    
    num = (bb['upper'] - bb['lower']).to_numpy()
    den = bb['middle'].to_numpy()
    
    res = safe_divide(num, den, default=np.nan)
    if percentage:
        res *= 100.0
        
    out = pd.Series(res, index=bb.index, name=f"BBW_{length}_{std_mult}")
    return _finalize_output(out, offset, fillna)


def bollinger_percent_b(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_BB_LEN, 
                        std_mult: float = DEFAULT_BB_STD, source: str = 'close', 
                        offset: int = 0, fillna: Any = None) -> pd.Series:
    """Bollinger Bands %B. (Close - Lower) / (Upper - Lower)."""
    s = get_price_source(data, source)
    bb = bollinger_bands(data, length=length, std_mult=std_mult, source=source)
    
    num = (s - bb['lower']).to_numpy()
    den = (bb['upper'] - bb['lower']).to_numpy()
    
    res = safe_divide(num, den, default=np.nan)
    out = pd.Series(res, index=s.index, name=f"PercentB_{length}_{std_mult}")
    return _finalize_output(out, offset, fillna)


def donchian_channel(data: pd.DataFrame, length: int = DEFAULT_DONCHIAN_LEN, 
                     offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    """Donchian Channel. Returns upper, middle, lower."""
    validate_length(length, "Donchian Channel")
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    upper = df['high'].rolling(window=length).max()
    lower = df['low'].rolling(window=length).min()
    middle = (upper + lower) / 2.0
    
    out = pd.DataFrame({
        "upper": upper,
        "middle": middle,
        "lower": lower
    }, index=df.index)
    return _finalize_output(out, offset, fillna)


def keltner_channel(data: pd.DataFrame, length: int = DEFAULT_KELTNER_LEN, 
                    atr_length: int = DEFAULT_KELTNER_ATR_LEN, mult: float = DEFAULT_KELTNER_MULT, 
                    source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    """Keltner Channel. Uses EMA for middle band and ATR for range."""
    validate_length(length, "Keltner Middle")
    validate_length(atr_length, "Keltner ATR")
    df = standardize_column_names(data)
    validate_ohlc(df)
    s = get_price_source(df, source)
    
    middle = s.ewm(span=length, adjust=False).mean()
    atr_val = atr(df, length=atr_length)
    
    upper = middle + (mult * atr_val)
    lower = middle - (mult * atr_val)
    
    out = pd.DataFrame({
        "middle": middle,
        "upper": upper,
        "lower": lower
    }, index=df.index)
    return _finalize_output(out, offset, fillna)


def historical_volatility(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_HV_LEN, 
                          annualization: float = DEFAULT_ANNUALIZATION, source: str = 'close', 
                          offset: int = 0, fillna: Any = None) -> pd.Series:
    """Annualized Historical Volatility based on log returns (Index preserved)."""
    validate_length(length, "Historical Volatility")
    s = get_price_source(data, source)
    
    shifted = s.shift(1)
    log_returns = np.log(safe_divide(s.to_numpy(), shifted.to_numpy(), default=np.nan))
    
    rolling_stdev = pd.Series(log_returns, index=s.index).rolling(window=length).std().to_numpy()
    hv = rolling_stdev * np.sqrt(annualization) * 100.0
    
    out = pd.Series(hv, index=s.index, name=f"HV_{length}")
    return _finalize_output(out, offset, fillna)


def chaikin_volatility(data: pd.DataFrame, length: int = DEFAULT_CHAIKIN_LEN, 
                       roc_length: int = DEFAULT_CHAIKIN_ROC, offset: int = 0, fillna: Any = None) -> pd.Series:
    """Chaikin Volatility. ROC of EMA(High-Low)."""
    validate_length(length, "Chaikin EMA")
    validate_length(roc_length, "Chaikin ROC")
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    hl = df['high'] - df['low']
    ema_hl = hl.ewm(span=length, adjust=False).mean()
    
    shifted_ema = ema_hl.shift(roc_length)
    num = (ema_hl - shifted_ema).to_numpy()
    den = shifted_ema.to_numpy()
    
    cv = 100.0 * safe_divide(num, den, default=np.nan)
    
    out = pd.Series(cv, index=df.index, name=f"CV_{length}_{roc_length}")
    return _finalize_output(out, offset, fillna)


def ulcer_index(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_ULCER_LEN, 
                source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    """Ulcer Index. RMS of percentage drawdowns over a rolling window."""
    validate_length(length, "Ulcer Index")
    s = get_price_source(data, source)
    
    highest_close = s.rolling(window=length).max()
    
    drawdown = 100.0 * safe_divide((s - highest_close).to_numpy(), highest_close.to_numpy(), default=0.0)
    sq_drawdown = drawdown ** 2
    
    mean_sq_dd = pd.Series(sq_drawdown).rolling(window=length).mean().to_numpy()
    ulcer = np.sqrt(mean_sq_dd)
    
    out = pd.Series(ulcer, index=s.index, name=f"UI_{length}")
    return _finalize_output(out, offset, fillna)


def standard_deviation(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_BB_LEN, 
                       source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    """Rolling Standard Deviation."""
    validate_length(length, "Standard Deviation")
    s = get_price_source(data, source)
    
    std_val = rolling_std(s, window=length)
    
    out = pd.Series(std_val, index=s.index, name=f"STD_{length}")
    return _finalize_output(out, offset, fillna)


def variance(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_BB_LEN, 
             source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    """Rolling Variance."""
    validate_length(length, "Variance")
    s = get_price_source(data, source)
    
    std_val = rolling_std(s, window=length)
    var_val = std_val ** 2
    
    out = pd.Series(var_val, index=s.index, name=f"VAR_{length}")
    return _finalize_output(out, offset, fillna)


def mass_index(data: pd.DataFrame, length: int = DEFAULT_MASS_LEN, ema_length: int = DEFAULT_MASS_EMA, 
               offset: int = 0, fillna: Any = None) -> pd.Series:
    """Mass Index. Uses EMA of EMA of High-Low."""
    validate_length(length, "Mass Index Sum")
    validate_length(ema_length, "Mass Index EMA")
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    hl = df['high'] - df['low']
    ema1 = hl.ewm(span=ema_length, adjust=False).mean()
    ema2 = ema1.ewm(span=ema_length, adjust=False).mean()
    
    ratio = safe_divide(ema1.to_numpy(), ema2.to_numpy(), default=np.nan)
    ratio_series = pd.Series(ratio, index=df.index)
    
    mass = rolling_sum(ratio_series, window=length)
    
    out = pd.Series(mass, index=df.index, name=f"MASS_{length}_{ema_length}")
    return _finalize_output(out, offset, fillna)


def volatility_ratio(data: pd.DataFrame, length: int = DEFAULT_VR_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    """Volatility Ratio. Current True Range / Rolling ATR."""
    validate_length(length, "Volatility Ratio")
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    tr_series = true_range(df)
    atr_series = atr(df, length=length)
    
    vr = safe_divide(tr_series.to_numpy(), atr_series.to_numpy(), default=np.nan)
    
    out = pd.Series(vr, index=df.index, name=f"VR_{length}")
    return _finalize_output(out, offset, fillna)


def atr_percentile(data: pd.DataFrame, length: int = DEFAULT_ATR_LEN, 
                   lookback: int = DEFAULT_PERCENTILE_LOOKBACK, offset: int = 0, fillna: Any = None) -> pd.Series:
    """Rolling Percentile Rank of ATR (TradingView math parity)."""
    validate_length(length, "ATR Base Length")
    validate_length(lookback, "ATR Percentile Lookback")
    
    atr_series = atr(data, length=length)
    arr = atr_series.to_numpy(dtype=np.float64)
    
    if len(arr) < lookback:
        out = np.full_like(arr, np.nan)
    else:
        s_filled = pd.Series(arr).ffill().bfill().to_numpy(dtype=np.float64)
        views = np.lib.stride_tricks.sliding_window_view(s_filled, lookback)
        current = views[:, -1:]
        
        ranks = np.sum(views <= current, axis=1) / lookback * 100.0
        
        out = np.full_like(arr, np.nan)
        out[lookback - 1:] = ranks
        out[np.isnan(arr)] = np.nan
        
    res = pd.Series(out, index=atr_series.index, name=f"ATRP_{length}_{lookback}")
    return _finalize_output(res, offset, fillna)


def bollinger_squeeze(data: pd.DataFrame, length: int = DEFAULT_BB_LEN, bb_mult: float = DEFAULT_BB_STD, 
                      kc_mult: float = 1.5, source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    """Bollinger Squeeze. Returns True when Bollinger Bands are completely inside Keltner Channels."""
    validate_length(length, "Squeeze Length")
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    bb = bollinger_bands(df, length=length, std_mult=bb_mult, source=source)
    kc = keltner_channel(df, length=length, atr_length=length, mult=kc_mult, source=source)
    
    squeeze_on = (bb['lower'] > kc['lower']) & (bb['upper'] < kc['upper'])
    
    out = pd.Series(squeeze_on, index=df.index, name=f"SQZ_{length}")
    return _finalize_output(out, offset, fillna)


def expansion_index(data: pd.DataFrame, length: int = DEFAULT_ATR_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    """Volatility Expansion Index. ROC of True Range."""
    validate_length(length, "Expansion Index")
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    tr_series = true_range(df)
    shifted = tr_series.shift(length)
    
    ei = 100.0 * safe_divide((tr_series - shifted).to_numpy(), shifted.to_numpy(), default=np.nan)
    
    out = pd.Series(ei, index=df.index, name=f"EI_{length}")
    return _finalize_output(out, offset, fillna)


def volatility_oscillator(data: pd.DataFrame, fast_len: int = 14, slow_len: int = 28, 
                          offset: int = 0, fillna: Any = None) -> pd.Series:
    """Volatility Oscillator. Normalized percentage difference of Fast ATR vs Slow ATR."""
    validate_length(fast_len, "Volatility Osc Fast")
    validate_length(slow_len, "Volatility Osc Slow")
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    atr_fast = atr(df, length=fast_len)
    atr_slow = atr(df, length=slow_len)
    
    vosc = 100.0 * safe_divide((atr_fast - atr_slow).to_numpy(), atr_slow.to_numpy(), default=np.nan)
    
    out = pd.Series(vosc, index=df.index, name=f"VOSC_{fast_len}_{slow_len}")
    return _finalize_output(out, offset, fillna)


def adaptive_atr(data: pd.DataFrame, length: int = DEFAULT_ATR_LEN, fast_len: int = 2, 
                 slow_len: int = 30, offset: int = 0, fillna: Any = None) -> pd.Series:
    """
    Adaptive ATR. Applies KAMA smoothing logic directly to True Range.
    FUTURE_OPTIMIZATION: Candidate for Numba @jit
    """
    validate_length(length, "Adaptive ATR")
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    tr_series = true_range(df)
    arr = tr_series.to_numpy(dtype=np.float64)
    
    if len(arr) < length:
        out = pd.Series(np.full(len(arr), np.nan), index=tr_series.index)
        return _finalize_output(out, offset, fillna)
        
    change = np.abs(pd.Series(arr).diff(length).to_numpy(dtype=np.float64))
    volatility = pd.Series(arr).diff(1).abs().rolling(window=length).sum().to_numpy(dtype=np.float64)
    
    er = safe_divide(change, volatility, default=0.0)
    fast_sc = 2.0 / (fast_len + 1.0)
    slow_sc = 2.0 / (slow_len + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama_arr = np.full_like(arr, np.nan)
    valid_mask = ~np.isnan(arr)
    
    window_sums = np.convolve(valid_mask.astype(int), np.ones(length, dtype=int), mode='valid')
    valid_starts = np.where(window_sums == length)[0]
    
    if len(valid_starts) == 0:
        out = pd.Series(kama_arr, index=tr_series.index)
        return _finalize_output(out, offset, fillna)
        
    first_valid = valid_starts[0] + length - 1
    kama_arr[first_valid] = arr[first_valid]
    
    for i in range(first_valid + 1, len(arr)):
        if np.isnan(sc[i]) or np.isnan(arr[i]):
            kama_arr[i] = kama_arr[i-1]
        else:
            kama_arr[i] = kama_arr[i-1] + sc[i] * (arr[i] - kama_arr[i-1])
            
    out = pd.Series(kama_arr, index=tr_series.index, name=f"AATR_{length}")
    return _finalize_output(out, offset, fillna)

# ==============================================================================
# INSTITUTIONAL VOLATILITY MODELS
# ==============================================================================

def parkinson_volatility(data: pd.DataFrame, length: int = DEFAULT_HV_LEN, 
                         annualization: float = DEFAULT_ANNUALIZATION, offset: int = 0, fillna: Any = None) -> pd.Series:
    """Parkinson Volatility. Uses High/Low ratio to estimate annualized volatility."""
    validate_length(length, "Parkinson Volatility")
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    hl_ratio_sq = np.log(safe_divide(df['high'].to_numpy(), df['low'].to_numpy(), default=np.nan)) ** 2
    roll_sum = pd.Series(hl_ratio_sq).rolling(window=length).sum().to_numpy()
    
    factor = 1.0 / (4.0 * length * np.log(2.0))
    park = np.sqrt(factor * roll_sum) * np.sqrt(annualization) * 100.0
    
    out = pd.Series(park, index=df.index, name=f"PARK_{length}")
    return _finalize_output(out, offset, fillna)


def garman_klass_volatility(data: pd.DataFrame, length: int = DEFAULT_HV_LEN, 
                            annualization: float = DEFAULT_ANNUALIZATION, offset: int = 0, fillna: Any = None) -> pd.Series:
    """Garman-Klass Volatility. Incorporates OHLC data for refined volatility estimation."""
    validate_length(length, "Garman-Klass")
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    log_hl_sq = np.log(safe_divide(df['high'].to_numpy(), df['low'].to_numpy(), default=np.nan)) ** 2
    log_co_sq = np.log(safe_divide(df['close'].to_numpy(), df['open'].to_numpy(), default=np.nan)) ** 2
    
    term = 0.5 * log_hl_sq - (2.0 * np.log(2.0) - 1.0) * log_co_sq
    gk = np.sqrt(pd.Series(term).rolling(window=length).mean().to_numpy()) * np.sqrt(annualization) * 100.0
    
    out = pd.Series(gk, index=df.index, name=f"GK_{length}")
    return _finalize_output(out, offset, fillna)


def rogers_satchell_volatility(data: pd.DataFrame, length: int = DEFAULT_HV_LEN, 
                               annualization: float = DEFAULT_ANNUALIZATION, offset: int = 0, fillna: Any = None) -> pd.Series:
    """Rogers-Satchell Volatility. Accounts for non-zero mean drift."""
    validate_length(length, "Rogers-Satchell")
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    h = df['high'].to_numpy()
    l = df['low'].to_numpy()
    o = df['open'].to_numpy()
    c = df['close'].to_numpy()
    
    term1 = np.log(safe_divide(h, c, default=np.nan)) * np.log(safe_divide(h, o, default=np.nan))
    term2 = np.log(safe_divide(l, c, default=np.nan)) * np.log(safe_divide(l, o, default=np.nan))
    
    rs = np.sqrt(pd.Series(term1 + term2).rolling(window=length).mean().to_numpy()) * np.sqrt(annualization) * 100.0
    
    out = pd.Series(rs, index=df.index, name=f"RS_{length}")
    return _finalize_output(out, offset, fillna)


def yang_zhang_volatility(data: pd.DataFrame, length: int = DEFAULT_HV_LEN, 
                          annualization: float = DEFAULT_ANNUALIZATION, offset: int = 0, fillna: Any = None) -> pd.Series:
    """Yang-Zhang Volatility. The optimal drift-independent volatility estimator."""
    validate_length(length, "Yang-Zhang")
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    o = df['open'].to_numpy()
    c = df['close'].to_numpy()
    h = df['high'].to_numpy()
    l = df['low'].to_numpy()
    
    prev_c = np.empty_like(c)
    prev_c[0] = np.nan
    prev_c[1:] = c[:-1]
    
    log_ho = np.log(safe_divide(h, o, default=np.nan))
    log_lo = np.log(safe_divide(l, o, default=np.nan))
    log_co = np.log(safe_divide(c, o, default=np.nan))
    
    log_oc_prev = np.log(safe_divide(o, prev_c, default=np.nan))
    log_cc_prev = np.log(safe_divide(c, prev_c, default=np.nan))
    
    rs_var = pd.Series(log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(window=length).mean().to_numpy()
    o_var = pd.Series(log_oc_prev).rolling(window=length).var(ddof=1).to_numpy()
    c_var = pd.Series(log_cc_prev).rolling(window=length).var(ddof=1).to_numpy()
    
    k = 0.34 / (1.34 + (length + 1) / max(1, length - 1))
    yz_var = o_var + k * c_var + (1 - k) * rs_var
    yz = np.sqrt(yz_var) * np.sqrt(annualization) * 100.0
    
    out = pd.Series(yz, index=df.index, name=f"YZ_{length}")
    return _finalize_output(out, offset, fillna)


def choppiness_index(data: pd.DataFrame, length: int = DEFAULT_CHOP_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    """Choppiness Index (CHOP). Measures market trendiness vs chop."""
    validate_length(length, "Choppiness Index")
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    tr_sum = true_range(df).rolling(window=length).sum().to_numpy()
    max_h = df['high'].rolling(window=length).max().to_numpy()
    min_l = df['low'].rolling(window=length).min().to_numpy()
    
    range_hl = max_h - min_l
    ratio = safe_divide(tr_sum, range_hl, default=np.nan)
    
    chop = 100.0 * np.log10(ratio) / np.log10(length)
    
    out = pd.Series(chop, index=df.index, name=f"CHOP_{length}")
    return _finalize_output(out, offset, fillna)


def vhf(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_VHF_LEN, source: str = 'close', 
        offset: int = 0, fillna: Any = None) -> pd.Series:
    """Vertical Horizontal Filter (VHF). Determines whether prices are in a trending phase."""
    validate_length(length, "VHF")
    s = get_price_source(data, source)
    
    num = np.abs(s - s.shift(length)).to_numpy()
    den = s.diff().abs().rolling(window=length).sum().to_numpy()
    
    vhf_val = safe_divide(num, den, default=np.nan)
    
    out = pd.Series(vhf_val, index=s.index, name=f"VHF_{length}")
    return _finalize_output(out, offset, fillna)


def bb_squeeze_momentum(data: pd.DataFrame, length: int = DEFAULT_BB_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    """BB Squeeze Momentum. Linear Regression of Price deviation from Donchian & BB Midpoints."""
    validate_length(length, "BB Squeeze Momentum")
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    c = df['close']
    middle = rolling_mean(c, length)
    upper_dc = df['high'].rolling(window=length).max()
    lower_dc = df['low'].rolling(window=length).min()
    
    avg_dc = (upper_dc + lower_dc) / 2.0
    val = (c - ((avg_dc + middle) / 2.0)).to_numpy(dtype=np.float64)
    
    weights = np.arange(1, length + 1, dtype=np.float64)
    weights = (6 * weights - 2 * (length + 1)) / (length * (length + 1))
    
    s_filled = pd.Series(val).ffill().bfill().to_numpy()
    mom = np.convolve(s_filled, weights[::-1], mode='valid')
    
    out_mom = np.full(len(val), np.nan)
    out_mom[length - 1:] = mom
    out_mom[np.isnan(val)] = np.nan
    
    out = pd.Series(out_mom, index=df.index, name=f"SQZ_MOM_{length}")
    return _finalize_output(out, offset, fillna)


def atr_stop_distance(data: pd.DataFrame, length: int = DEFAULT_ATR_LEN, mult: float = 3.0, 
                      offset: int = 0, fillna: Any = None) -> pd.Series:
    """ATR Stop Distance (Multiplier * ATR)."""
    atr_val = atr(data, length=length)
    res = atr_val * mult
    res.name = f"ATR_STOP_DIST_{length}_{mult}"
    return _finalize_output(res, offset, fillna)


def volatility_stop(data: pd.DataFrame, length: int = DEFAULT_ATR_LEN, mult: float = 3.0, 
                    offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    """
    Volatility Stop / Chandelier Exit logic.
    FUTURE_OPTIMIZATION: Candidate for Numba @jit
    Returns DataFrame with 'stop_price' and 'is_long' direction.
    """
    validate_length(length, "Volatility Stop")
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    atr_val = atr(df, length=length).to_numpy()
    c = df['close'].to_numpy()
    
    vstop = np.full_like(c, np.nan)
    is_long = np.ones_like(c, dtype=bool)
    
    if len(c) > length:
        vstop[length-1] = c[length-1] - mult * atr_val[length-1]
        for i in range(length, len(c)):
            if np.isnan(c[i]) or np.isnan(atr_val[i]):
                vstop[i] = vstop[i-1]
                is_long[i] = is_long[i-1]
                continue
                
            prev_stop = vstop[i-1]
            prev_is_long = is_long[i-1]
            
            if prev_is_long:
                stop = max(prev_stop, c[i] - mult * atr_val[i])
                if c[i] < stop:
                    is_long[i] = False
                    stop = c[i] + mult * atr_val[i]
                else:
                    is_long[i] = True
                vstop[i] = stop
            else:
                stop = min(prev_stop, c[i] + mult * atr_val[i])
                if c[i] > stop:
                    is_long[i] = True
                    stop = c[i] - mult * atr_val[i]
                else:
                    is_long[i] = False
                vstop[i] = stop

    out = pd.DataFrame({"stop_price": vstop, "is_long": is_long}, index=df.index)
    return _finalize_output(out, offset, fillna)


def standard_error(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_BB_LEN, 
                   source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    """Standard Error. Rolling StdDev / sqrt(Length)."""
    validate_length(length, "Standard Error")
    s = get_price_source(data, source)
    
    se = rolling_std(s, window=length) / np.sqrt(length)
    
    out = pd.Series(se, index=s.index, name=f"SE_{length}")
    return _finalize_output(out, offset, fillna)


def rei(data: pd.DataFrame, length: int = DEFAULT_REI_LEN, offset: int = 0, fillna: Any = None) -> pd.Series:
    """Range Expansion Index (Thomas DeMark's REI). Vectorized implementation."""
    validate_length(length, "REI")
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    h = df['high']
    l = df['low']
    h2 = h.shift(2)
    l2 = l.shift(2)
    
    cond1 = (h >= l2) | (h.shift(1) >= l.shift(3))
    cond2 = (l <= h2) | (l.shift(1) <= h.shift(3))
    valid_overlap = cond1 & cond2
    
    num = np.where(valid_overlap, (h - h2) + (l - l2), 0.0)
    den = np.where(valid_overlap, np.abs(h - h2) + np.abs(l - l2), 0.0)
    
    num_sum = pd.Series(num).rolling(window=length).sum().to_numpy()
    den_sum = pd.Series(den).rolling(window=length).sum().to_numpy()
    
    rei_val = 100.0 * safe_divide(num_sum, den_sum, default=np.nan)
    
    out = pd.Series(rei_val, index=df.index, name=f"REI_{length}")
    return _finalize_output(out, offset, fillna)

# ==============================================================================
# WRAPPERS & ADVANCED LOGIC
# ==============================================================================

def bollinger_upper(data: Union[pd.DataFrame, pd.Series], **kwargs) -> pd.Series:
    return bollinger_bands(data, **kwargs)['upper']

def bollinger_lower(data: Union[pd.DataFrame, pd.Series], **kwargs) -> pd.Series:
    return bollinger_bands(data, **kwargs)['lower']

def bollinger_middle(data: Union[pd.DataFrame, pd.Series], **kwargs) -> pd.Series:
    return bollinger_bands(data, **kwargs)['middle']

def donchian_upper(data: pd.DataFrame, **kwargs) -> pd.Series:
    return donchian_channel(data, **kwargs)['upper']

def donchian_lower(data: pd.DataFrame, **kwargs) -> pd.Series:
    return donchian_channel(data, **kwargs)['lower']

def donchian_middle(data: pd.DataFrame, **kwargs) -> pd.Series:
    return donchian_channel(data, **kwargs)['middle']

def keltner_upper(data: pd.DataFrame, **kwargs) -> pd.Series:
    return keltner_channel(data, **kwargs)['upper']

def keltner_lower(data: pd.DataFrame, **kwargs) -> pd.Series:
    return keltner_channel(data, **kwargs)['lower']

def keltner_middle(data: pd.DataFrame, **kwargs) -> pd.Series:
    return keltner_channel(data, **kwargs)['middle']

def atr_trailing_stop(data: pd.DataFrame, **kwargs) -> pd.Series:
    """Wrapper returning Volatility Stop price line."""
    return volatility_stop(data, **kwargs)['stop_price']

def supertrend(data: pd.DataFrame, length: int = 10, multiplier: float = 3.0, offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    """
    SuperTrend Indicator (TradingView Standard).
    
    Returns a DataFrame containing:
    - supertrend: The actual trailing stop line
    - trend: 1 for Bullish, -1 for Bearish
    - upper_band: Final Upper Band
    - lower_band: Final Lower Band
    """
    validate_length(length, "Supertrend")
    df = standardize_column_names(data)
    validate_ohlc(df)
    
    # Get ATR and HL2
    atr_arr = atr(df, length=length).to_numpy()
    hl2_arr = hl2(df['high'], df['low']).to_numpy()
    close_arr = df['close'].to_numpy()
    
    n = len(close_arr)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    trend = np.full(n, 1)  # Default to bullish
    st_line = np.full(n, np.nan)
    
    basic_upper = hl2_arr + (multiplier * atr_arr)
    basic_lower = hl2_arr - (multiplier * atr_arr)
    
    # Find the first index where ATR is valid (not NaN)
    first_valid = -1
    for i in range(n):
        if not np.isnan(atr_arr[i]):
            first_valid = i
            break
            
    if first_valid != -1 and first_valid < n:
        # Initialize base values at first valid index
        final_upper[first_valid] = basic_upper[first_valid]
        final_lower[first_valid] = basic_lower[first_valid]
        trend[first_valid] = 1 if close_arr[first_valid] > hl2_arr[first_valid] else -1
        st_line[first_valid] = final_lower[first_valid] if trend[first_valid] == 1 else final_upper[first_valid]
        
        # O(n) Calculation
        for i in range(first_valid + 1, n):
            # Guard against unexpected mid-series NaNs
            if np.isnan(atr_arr[i]) or np.isnan(hl2_arr[i]):
                final_upper[i] = final_upper[i-1]
                final_lower[i] = final_lower[i-1]
                trend[i] = trend[i-1]
                st_line[i] = st_line[i-1]
                continue

            # Calculate Final Upper Band
            if basic_upper[i] < final_upper[i-1] or close_arr[i-1] > final_upper[i-1]:
                final_upper[i] = basic_upper[i]
            else:
                final_upper[i] = final_upper[i-1]
                
            # Calculate Final Lower Band
            if basic_lower[i] > final_lower[i-1] or close_arr[i-1] < final_lower[i-1]:
                final_lower[i] = basic_lower[i]
            else:
                final_lower[i] = final_lower[i-1]
                
            # TradingView Strict Trend Switching Logic
            if trend[i-1] == -1 and close_arr[i] > final_upper[i-1]:
                trend[i] = 1
            elif trend[i-1] == 1 and close_arr[i] < final_lower[i-1]:
                trend[i] = -1
            else:
                trend[i] = trend[i-1]
                
            # Assign Final SuperTrend Line (Trailing Stop)
            if trend[i] == 1:
                st_line[i] = final_lower[i]
            else:
                st_line[i] = final_upper[i]

    out = pd.DataFrame({
        "supertrend": st_line,
        "trend": trend,
        "upper_band": final_upper,
        "lower_band": final_lower
    }, index=df.index)
    
    return _finalize_output(out, offset, fillna)

# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "VolatilityIndicatorError",
    "validate_length",
    "get_price_source",
    "true_range",
    "atr",
    "natr",
    "bollinger_bands",
    "bollinger_width",
    "bollinger_percent_b",
    "donchian_channel",
    "keltner_channel",
    "historical_volatility",
    "chaikin_volatility",
    "ulcer_index",
    "standard_deviation",
    "variance",
    "mass_index",
    "volatility_ratio",
    "atr_percentile",
    "bollinger_squeeze",
    "expansion_index",
    "volatility_oscillator",
    "adaptive_atr",
    "parkinson_volatility",
    "garman_klass_volatility",
    "rogers_satchell_volatility",
    "yang_zhang_volatility",
    "choppiness_index",
    "vhf",
    "bb_squeeze_momentum",
    "atr_stop_distance",
    "volatility_stop",
    "standard_error",
    "rei",
    "bollinger_upper",
    "bollinger_lower",
    "bollinger_middle",
    "donchian_upper",
    "donchian_lower",
    "donchian_middle",
    "keltner_upper",
    "keltner_lower",
    "keltner_middle",
    "supertrend",
    "atr_trailing_stop"
]
