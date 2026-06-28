import logging
import numpy as np
import pandas as pd
from typing import Union, Optional, Any, Tuple
import math

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

DEFAULT_STAT_LEN = 20
DEFAULT_REG_LEN = 20
DEFAULT_CORR_LEN = 20
DEFAULT_Z_LEN = 20
DEFAULT_OUTLIER_Z = 3.0
DEFAULT_BINS = 10

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

class StatisticsIndicatorError(Exception):
    """Custom exception for errors in statistics indicator calculations."""
    pass

# ==============================================================================
# NUMBA JIT ACCELERATED CORE FUNCTIONS
# ==============================================================================

@jit(nopython=True, cache=True)
def _rolling_linreg_jit(y: np.ndarray, w: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rolling Linear Regression calculating slope, intercept, r-squared, and regression line endpoint."""
    n = len(y)
    slope = np.full(n, np.nan)
    intercept = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    line = np.full(n, np.nan)

    if n < w:
        return slope, intercept, r2, line

    x = np.arange(w, dtype=np.float64)
    x_mean = (w - 1) / 2.0
    x_var = np.sum((x - x_mean)**2)

    for i in range(w - 1, n):
        y_win = y[i - w + 1 : i + 1]
        
        has_nan = False
        for j in range(w):
            if np.isnan(y_win[j]):
                has_nan = True
                break
                
        if has_nan:
            continue

        y_mean = np.mean(y_win)
        cov_xy = np.sum((x - x_mean) * (y_win - y_mean))
        var_y = np.sum((y_win - y_mean)**2)

        b1 = cov_xy / x_var if x_var > 0 else 0.0
        b0 = y_mean - b1 * x_mean

        slope[i] = b1
        intercept[i] = b0
        line[i] = b0 + b1 * (w - 1) 

        if var_y > 0:
            r2[i] = (b1**2 * x_var) / var_y
        else:
            r2[i] = 1.0

    return slope, intercept, r2, line

@jit(nopython=True, cache=True)
def _rolling_shannon_entropy_jit(y: np.ndarray, w: int, bins: int) -> np.ndarray:
    """Rolling Shannon Entropy using dynamic histogram binning."""
    n = len(y)
    entropy = np.full(n, np.nan)
    
    if n < w:
        return entropy
        
    for i in range(w - 1, n):
        y_win = y[i - w + 1 : i + 1]
        
        y_min, y_max = np.nanmin(y_win), np.nanmax(y_win)
        if np.isnan(y_min) or np.isnan(y_max) or y_min == y_max:
            entropy[i] = 0.0
            continue
            
        bin_edges = np.linspace(y_min, y_max, bins + 1)
        hist = np.zeros(bins, dtype=np.float64)
        
        for j in range(w):
            val = y_win[j]
            if np.isnan(val):
                continue
            b = 0
            while b < bins and val >= bin_edges[b+1]:
                b += 1
            if b == bins: b -= 1
            hist[b] += 1.0
            
        probs = hist / w
        e = 0.0
        for p in probs:
            if p > 0:
                e -= p * math.log(p)
        entropy[i] = e
        
    return entropy

@jit(nopython=True, cache=True)
def _rolling_mad_jit(y: np.ndarray, w: int) -> np.ndarray:
    """Rolling Median Absolute Deviation (MAD)."""
    n = len(y)
    mad = np.full(n, np.nan)
    if n < w: return mad
    
    for i in range(w - 1, n):
        y_win = y[i - w + 1 : i + 1]
        med = np.nanmedian(y_win)
        if np.isnan(med):
            continue
        mad[i] = np.nanmedian(np.abs(y_win - med))
    return mad

# ==============================================================================
# COMMON UTILITIES
# ==============================================================================

def validate_length(length: int, name: str = "length") -> None:
    if not isinstance(length, int) or length <= 0:
        logger.error(f"Invalid statistics window length for {name}: {length}")
        raise StatisticsIndicatorError(f"Length for {name} must be a positive integer, got {length}")

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
            raise StatisticsIndicatorError(f"DataFrame is missing columns needed for '{src}': {missing}")
        return func(*[df[c] for c in req_cols])
            
    logger.error(f"Invalid source '{source}'.")
    raise StatisticsIndicatorError(f"Invalid source '{source}' or column not found.")

def _finalize_output(output: Union[pd.Series, pd.DataFrame], offset: int, fillna: Any) -> Union[pd.Series, pd.DataFrame]:
    res = output.shift(offset) if offset != 0 else output
    if fillna is not None:
        res = res.fillna(fillna)
    return res

def _broadcast_scalar(scalar_val: float, index: pd.Index, name: str) -> pd.Series:
    """Helper to broadcast a global scalar metric across a full Series."""
    return pd.Series(np.full(len(index), scalar_val), index=index, name=name)

# ==============================================================================
# DESCRIPTIVE & ROLLING STATISTICS
# ==============================================================================

def mean(data: Union[pd.DataFrame, pd.Series], source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    s = get_price_source(data, source)
    return _finalize_output(_broadcast_scalar(s.mean(), s.index, "Mean"), offset, fillna)

def rolling_mean(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_STAT_LEN, source: str = 'close', 
                 offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Rolling Mean")
    s = get_price_source(data, source)
    out = s.rolling(window=length).mean()
    out.name = f"RollMean_{length}"
    return _finalize_output(out, offset, fillna)

def median(data: Union[pd.DataFrame, pd.Series], source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    s = get_price_source(data, source)
    return _finalize_output(_broadcast_scalar(s.median(), s.index, "Median"), offset, fillna)

def rolling_median(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_STAT_LEN, source: str = 'close', 
                   offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Rolling Median")
    s = get_price_source(data, source)
    out = s.rolling(window=length).median()
    out.name = f"RollMedian_{length}"
    return _finalize_output(out, offset, fillna)

def mode(data: Union[pd.DataFrame, pd.Series], source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    s = get_price_source(data, source)
    mode_val = s.mode().iloc[0] if not s.mode().empty else np.nan
    return _finalize_output(_broadcast_scalar(mode_val, s.index, "Mode"), offset, fillna)

def variance(data: Union[pd.DataFrame, pd.Series], source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    s = get_price_source(data, source)
    return _finalize_output(_broadcast_scalar(s.var(), s.index, "Variance"), offset, fillna)

def rolling_variance(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_STAT_LEN, source: str = 'close', 
                     offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Rolling Variance")
    s = get_price_source(data, source)
    out = s.rolling(window=length).var()
    out.name = f"RollVar_{length}"
    return _finalize_output(out, offset, fillna)

def standard_deviation(data: Union[pd.DataFrame, pd.Series], source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    s = get_price_source(data, source)
    return _finalize_output(_broadcast_scalar(s.std(), s.index, "StdDev"), offset, fillna)

def rolling_standard_deviation(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_STAT_LEN, source: str = 'close', 
                               offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Rolling Std")
    s = get_price_source(data, source)
    out = s.rolling(window=length).std()
    out.name = f"RollStd_{length}"
    return _finalize_output(out, offset, fillna)

def mean_absolute_deviation(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_STAT_LEN, source: str = 'close', 
                            offset: int = 0, fillna: Any = None) -> pd.Series:
    """Rolling Mean Absolute Deviation (MAD based on Mean)."""
    validate_length(length, "MAD (Mean)")
    s = get_price_source(data, source)
    sma = s.rolling(window=length).mean()
    mad = (s - sma).abs().rolling(window=length).mean()
    mad.name = f"MAD_Mean_{length}"
    return _finalize_output(mad, offset, fillna)

def median_absolute_deviation(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_STAT_LEN, source: str = 'close', 
                              offset: int = 0, fillna: Any = None) -> pd.Series:
    """Rolling Median Absolute Deviation (MAD based on Median). Numba accelerated."""
    validate_length(length, "MAD (Median)")
    s = get_price_source(data, source)
    arr = s.to_numpy(dtype=np.float64)
    mad_arr = _rolling_mad_jit(arr, length)
    out = pd.Series(mad_arr, index=s.index, name=f"MAD_Median_{length}")
    return _finalize_output(out, offset, fillna)

def root_mean_square(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_STAT_LEN, source: str = 'close', 
                     offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "RMS")
    s = get_price_source(data, source)
    rms = np.sqrt((s**2).rolling(window=length).mean())
    rms.name = f"RMS_{length}"
    return _finalize_output(rms, offset, fillna)

def coefficient_of_variation(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_STAT_LEN, source: str = 'close', 
                             offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "CV")
    s = get_price_source(data, source)
    mean_s = s.rolling(window=length).mean()
    std_s = s.rolling(window=length).std()
    cv = safe_divide(std_s.to_numpy(dtype=np.float64), mean_s.to_numpy(dtype=np.float64), default=np.nan)
    out = pd.Series(cv, index=s.index, name=f"CV_{length}")
    return _finalize_output(out, offset, fillna)

def range_stat(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_STAT_LEN, source: str = 'close', 
               offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Range")
    s = get_price_source(data, source)
    rng = s.rolling(window=length).max() - s.rolling(window=length).min()
    rng.name = f"Range_{length}"
    return _finalize_output(rng, offset, fillna)

def interquartile_range(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_STAT_LEN, source: str = 'close', 
                        offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "IQR")
    s = get_price_source(data, source)
    q75 = s.rolling(window=length).quantile(0.75)
    q25 = s.rolling(window=length).quantile(0.25)
    iqr = q75 - q25
    iqr.name = f"IQR_{length}"
    return _finalize_output(iqr, offset, fillna)

def quantile(data: Union[pd.DataFrame, pd.Series], q: float = 0.5, length: int = DEFAULT_STAT_LEN, source: str = 'close', 
             offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Quantile")
    s = get_price_source(data, source)
    out = s.rolling(window=length).quantile(q)
    out.name = f"Quantile_{q}_{length}"
    return _finalize_output(out, offset, fillna)

def percentile(data: Union[pd.DataFrame, pd.Series], p: float = 50.0, length: int = DEFAULT_STAT_LEN, source: str = 'close', 
               offset: int = 0, fillna: Any = None) -> pd.Series:
    """Wrapper around quantile for percentage scale."""
    return quantile(data, q=p/100.0, length=length, source=source, offset=offset, fillna=fillna)

def rolling_percentile(data: Union[pd.DataFrame, pd.Series], p: float = 50.0, length: int = DEFAULT_STAT_LEN, source: str = 'close', 
                       offset: int = 0, fillna: Any = None) -> pd.Series:
    """Alias for percentile."""
    return percentile(data, p=p, length=length, source=source, offset=offset, fillna=fillna)

def percentile_rank(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_STAT_LEN, source: str = 'close', 
                    offset: int = 0, fillna: Any = None) -> pd.Series:
    """Evaluates the percentile rank of the current value inside the rolling window."""
    validate_length(length, "Percentile Rank")
    s = get_price_source(data, source)
    ranks = s.rolling(window=length).rank(pct=True) * 100.0
    ranks.name = f"PctRank_{length}"
    return _finalize_output(ranks, offset, fillna)

# ==============================================================================
# NORMALIZATION & SCALING
# ==============================================================================

def z_score(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_Z_LEN, source: str = 'close', 
            offset: int = 0, fillna: Any = None) -> pd.Series:
    """Rolling Z-Score. (Price - SMA) / StdDev."""
    validate_length(length, "Z-Score")
    s = get_price_source(data, source)
    sma = s.rolling(window=length).mean()
    std = s.rolling(window=length).std()
    z = safe_divide((s - sma).to_numpy(dtype=np.float64), std.to_numpy(dtype=np.float64), default=np.nan)
    out = pd.Series(z, index=s.index, name=f"ZScore_{length}")
    return _finalize_output(out, offset, fillna)

def rolling_z_score(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_Z_LEN, source: str = 'close', 
                    offset: int = 0, fillna: Any = None) -> pd.Series:
    """Alias for z_score."""
    return z_score(data, length=length, source=source, offset=offset, fillna=fillna)

def modified_z_score(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_Z_LEN, source: str = 'close', 
                     offset: int = 0, fillna: Any = None) -> pd.Series:
    """Modified Z-Score using Median and Median Absolute Deviation (MAD)."""
    validate_length(length, "Modified Z-Score")
    s = get_price_source(data, source)
    med = s.rolling(window=length).median()
    mad = median_absolute_deviation(data, length=length, source=source)
    
    mad_adj = np.where(mad == 0.0, EPSILON, mad)
    mod_z = 0.6745 * safe_divide((s - med).to_numpy(dtype=np.float64), mad_adj, default=np.nan)
    
    out = pd.Series(mod_z, index=s.index, name=f"ModZ_{length}")
    return _finalize_output(out, offset, fillna)

def min_max_scaling(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_STAT_LEN, source: str = 'close', 
                    offset: int = 0, fillna: Any = None) -> pd.Series:
    """Rolling Min-Max Normalization (Stochastic equivalent)."""
    validate_length(length, "Min-Max")
    s = get_price_source(data, source)
    low = s.rolling(window=length).min()
    high = s.rolling(window=length).max()
    
    rng = high - low
    rng_safe = np.where(rng == 0.0, EPSILON, rng)
    mm = safe_divide((s - low).to_numpy(dtype=np.float64), rng_safe, default=np.nan)
    
    out = pd.Series(mm, index=s.index, name=f"MinMax_{length}")
    return _finalize_output(out, offset, fillna)

def normalization(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_STAT_LEN, source: str = 'close', 
                  offset: int = 0, fillna: Any = None) -> pd.Series:
    """Alias for min_max_scaling."""
    return min_max_scaling(data, length=length, source=source, offset=offset, fillna=fillna)

def robust_scaling(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_STAT_LEN, source: str = 'close', 
                   offset: int = 0, fillna: Any = None) -> pd.Series:
    """Rolling Robust Scaler. (Price - Median) / IQR."""
    validate_length(length, "Robust Scaling")
    s = get_price_source(data, source)
    med = s.rolling(window=length).median()
    iqr = interquartile_range(data, length=length, source=source)
    
    iqr_safe = np.where(iqr == 0.0, EPSILON, iqr)
    rs = safe_divide((s - med).to_numpy(dtype=np.float64), iqr_safe, default=np.nan)
    
    out = pd.Series(rs, index=s.index, name=f"RobustScale_{length}")
    return _finalize_output(out, offset, fillna)

def winsorization(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_STAT_LEN, limits: Tuple[float, float] = (0.05, 0.95), 
                  source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    """Rolling Winsorization. Caps values at given lower and upper percentiles."""
    validate_length(length, "Winsorization")
    s = get_price_source(data, source)
    
    lower_bound = s.rolling(window=length).quantile(limits[0])
    upper_bound = s.rolling(window=length).quantile(limits[1])
    
    win = s.clip(lower=lower_bound, upper=upper_bound)
    win.name = f"Winsor_{length}_{limits[0]}_{limits[1]}"
    return _finalize_output(win, offset, fillna)

# ==============================================================================
# DISTRIBUTION STATISTICS
# ==============================================================================

def skewness(data: Union[pd.DataFrame, pd.Series], source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    s = get_price_source(data, source)
    return _finalize_output(_broadcast_scalar(s.skew(), s.index, "Skewness"), offset, fillna)

def rolling_skewness(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_STAT_LEN, source: str = 'close', 
                     offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Rolling Skewness")
    s = get_price_source(data, source)
    out = s.rolling(window=length).skew()
    out.name = f"RollSkew_{length}"
    return _finalize_output(out, offset, fillna)

def kurtosis(data: Union[pd.DataFrame, pd.Series], source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    s = get_price_source(data, source)
    return _finalize_output(_broadcast_scalar(s.kurt(), s.index, "Kurtosis"), offset, fillna)

def rolling_kurtosis(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_STAT_LEN, source: str = 'close', 
                     offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Rolling Kurtosis")
    s = get_price_source(data, source)
    out = s.rolling(window=length).kurt()
    out.name = f"RollKurt_{length}"
    return _finalize_output(out, offset, fillna)

def entropy(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_STAT_LEN, bins: int = DEFAULT_BINS, 
            source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    """Shannon Entropy over a rolling window utilizing dynamic histogram bins."""
    validate_length(length, "Entropy")
    s = get_price_source(data, source)
    arr = s.to_numpy(dtype=np.float64)
    ent_arr = _rolling_shannon_entropy_jit(arr, length, bins)
    out = pd.Series(ent_arr, index=s.index, name=f"Entropy_{length}")
    return _finalize_output(out, offset, fillna)

def shannon_entropy(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_STAT_LEN, bins: int = DEFAULT_BINS, 
                    source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    """Alias for standard rolling entropy."""
    return entropy(data, length=length, bins=bins, source=source, offset=offset, fillna=fillna)

def jarque_bera(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_STAT_LEN, source: str = 'close', 
                offset: int = 0, fillna: Any = None) -> pd.Series:
    """Rolling Jarque-Bera Test Statistic. JB = (N/6) * (S^2 + 0.25 * (K-3)^2)."""
    validate_length(length, "Jarque Bera")
    skew = rolling_skewness(data, length=length, source=source)
    kurt = rolling_kurtosis(data, length=length, source=source) # Pandas kurtosis is excess kurtosis (K-3)
    
    jb = (length / 6.0) * (skew**2 + (kurt**2) / 4.0)
    jb.name = f"JarqueBera_{length}"
    return _finalize_output(jb, offset, fillna)

def normality_score(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_STAT_LEN, source: str = 'close', 
                    offset: int = 0, fillna: Any = None) -> pd.Series:
    """Inverse mapping of Jarque Bera to evaluate normality (0 = Not Normal, 1 = Perfect Normal)."""
    jb = jarque_bera(data, length=length, source=source)
    ns = np.exp(-jb)
    ns.name = f"Normality_{length}"
    return _finalize_output(ns, offset, fillna)

# ==============================================================================
# CORRELATION & COVARIANCE
# ==============================================================================

def pearson_correlation(data: Union[pd.DataFrame, pd.Series], benchmark: Union[pd.DataFrame, pd.Series], 
                        source: str = 'close', bench_source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    s = get_price_source(data, source)
    b = get_price_source(benchmark, bench_source)
    return _finalize_output(_broadcast_scalar(s.corr(b), s.index, "Pearson"), offset, fillna)

def rolling_pearson(data: Union[pd.DataFrame, pd.Series], benchmark: Union[pd.DataFrame, pd.Series], 
                    length: int = DEFAULT_CORR_LEN, source: str = 'close', bench_source: str = 'close', 
                    offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Rolling Pearson")
    s = get_price_source(data, source)
    b = get_price_source(benchmark, bench_source)
    out = s.rolling(window=length).corr(b)
    out.name = f"Pearson_{length}"
    return _finalize_output(out, offset, fillna)

def spearman_correlation(data: Union[pd.DataFrame, pd.Series], benchmark: Union[pd.DataFrame, pd.Series], 
                         source: str = 'close', bench_source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    s = get_price_source(data, source)
    b = get_price_source(benchmark, bench_source)
    return _finalize_output(_broadcast_scalar(s.corr(b, method='spearman'), s.index, "Spearman"), offset, fillna)

def rolling_spearman(data: Union[pd.DataFrame, pd.Series], benchmark: Union[pd.DataFrame, pd.Series], 
                     length: int = DEFAULT_CORR_LEN, source: str = 'close', bench_source: str = 'close', 
                     offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Rolling Spearman")
    s = get_price_source(data, source)
    b = get_price_source(benchmark, bench_source)
    out = s.rolling(window=length).corr(b, method='spearman')
    out.name = f"Spearman_{length}"
    return _finalize_output(out, offset, fillna)

def kendall_correlation(data: Union[pd.DataFrame, pd.Series], benchmark: Union[pd.DataFrame, pd.Series], 
                        source: str = 'close', bench_source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    s = get_price_source(data, source)
    b = get_price_source(benchmark, bench_source)
    return _finalize_output(_broadcast_scalar(s.corr(b, method='kendall'), s.index, "Kendall"), offset, fillna)

def rolling_kendall(data: Union[pd.DataFrame, pd.Series], benchmark: Union[pd.DataFrame, pd.Series], 
                    length: int = DEFAULT_CORR_LEN, source: str = 'close', bench_source: str = 'close', 
                    offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Rolling Kendall")
    s = get_price_source(data, source)
    b = get_price_source(benchmark, bench_source)
    out = s.rolling(window=length).corr(b, method='kendall')
    out.name = f"Kendall_{length}"
    return _finalize_output(out, offset, fillna)

def autocorrelation(data: Union[pd.DataFrame, pd.Series], lag: int = 1, source: str = 'close', 
                    offset: int = 0, fillna: Any = None) -> pd.Series:
    s = get_price_source(data, source)
    return _finalize_output(_broadcast_scalar(s.autocorr(lag=lag), s.index, f"AutoCorr_{lag}"), offset, fillna)

def lag_correlation(data: Union[pd.DataFrame, pd.Series], benchmark: Union[pd.DataFrame, pd.Series], 
                    lag: int = 1, length: int = DEFAULT_CORR_LEN, source: str = 'close', bench_source: str = 'close', 
                    offset: int = 0, fillna: Any = None) -> pd.Series:
    """Rolling correlation between Data and Lagged Benchmark."""
    validate_length(length, "Lag Correlation")
    s = get_price_source(data, source)
    b = get_price_source(benchmark, bench_source).shift(lag)
    out = s.rolling(window=length).corr(b)
    out.name = f"LagCorr_{length}_{lag}"
    return _finalize_output(out, offset, fillna)

def cross_correlation(data: Union[pd.DataFrame, pd.Series], benchmark: Union[pd.DataFrame, pd.Series], 
                      length: int = DEFAULT_CORR_LEN, source: str = 'close', bench_source: str = 'close', 
                      offset: int = 0, fillna: Any = None) -> pd.Series:
    """Alias for rolling_pearson."""
    return rolling_pearson(data, benchmark, length=length, source=source, bench_source=bench_source, offset=offset, fillna=fillna)

def covariance(data: Union[pd.DataFrame, pd.Series], benchmark: Union[pd.DataFrame, pd.Series], 
               source: str = 'close', bench_source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    s = get_price_source(data, source)
    b = get_price_source(benchmark, bench_source)
    return _finalize_output(_broadcast_scalar(s.cov(b), s.index, "Covariance"), offset, fillna)

def rolling_covariance(data: Union[pd.DataFrame, pd.Series], benchmark: Union[pd.DataFrame, pd.Series], 
                       length: int = DEFAULT_CORR_LEN, source: str = 'close', bench_source: str = 'close', 
                       offset: int = 0, fillna: Any = None) -> pd.Series:
    validate_length(length, "Rolling Covariance")
    s = get_price_source(data, source)
    b = get_price_source(benchmark, bench_source)
    out = s.rolling(window=length).cov(b)
    out.name = f"Cov_{length}"
    return _finalize_output(out, offset, fillna)

# ==============================================================================
# REGRESSION
# ==============================================================================

def rolling_regression(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_REG_LEN, source: str = 'close', 
                       offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    """
    Rolling Linear Regression against a normalized time vector [0, 1, 2, ..., w-1].
    Returns highly optimized Dataframe: value (line endpoint), slope, intercept, r2.
    """
    validate_length(length, "Rolling Regression")
    s = get_price_source(data, source)
    arr = s.to_numpy(dtype=np.float64)
    
    slope, intercept, r2, line = _rolling_linreg_jit(arr, length)
    
    out = pd.DataFrame({
        "value": line,
        "slope": slope,
        "intercept": intercept,
        "r2": r2
    }, index=s.index)
    return _finalize_output(out, offset, fillna)

def linear_regression(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_REG_LEN, source: str = 'close', 
                      offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    """Alias for rolling_regression."""
    return rolling_regression(data, length=length, source=source, offset=offset, fillna=fillna)

def regression_line(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_REG_LEN, source: str = 'close', 
                    offset: int = 0, fillna: Any = None) -> pd.Series:
    """Extracts the Regression Line endpoint."""
    return rolling_regression(data, length=length, source=source, offset=offset, fillna=fillna)['value']

def regression_value(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_REG_LEN, source: str = 'close', 
                     offset: int = 0, fillna: Any = None) -> pd.Series:
    return regression_line(data, length=length, source=source, offset=offset, fillna=fillna)

def regression_intercept(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_REG_LEN, source: str = 'close', 
                         offset: int = 0, fillna: Any = None) -> pd.Series:
    return rolling_regression(data, length=length, source=source, offset=offset, fillna=fillna)['intercept']

def regression_slope(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_REG_LEN, source: str = 'close', 
                     offset: int = 0, fillna: Any = None) -> pd.Series:
    return rolling_regression(data, length=length, source=source, offset=offset, fillna=fillna)['slope']

def rolling_slope(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_REG_LEN, source: str = 'close', 
                  offset: int = 0, fillna: Any = None) -> pd.Series:
    return regression_slope(data, length=length, source=source, offset=offset, fillna=fillna)

def r_squared(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_REG_LEN, source: str = 'close', 
              offset: int = 0, fillna: Any = None) -> pd.Series:
    return rolling_regression(data, length=length, source=source, offset=offset, fillna=fillna)['r2']

def adjusted_r_squared(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_REG_LEN, source: str = 'close', 
                       offset: int = 0, fillna: Any = None) -> pd.Series:
    """Adj R^2 = 1 - (1 - R^2) * (N - 1) / (N - k - 1). Here k=1 (Time)."""
    r2 = r_squared(data, length=length, source=source)
    adj_r2 = 1.0 - safe_divide((1.0 - r2).to_numpy() * (length - 1), (length - 2), default=np.nan)
    out = pd.Series(adj_r2, index=r2.index, name=f"AdjR2_{length}")
    return _finalize_output(out, offset, fillna)

def residual(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_REG_LEN, source: str = 'close', 
             offset: int = 0, fillna: Any = None) -> pd.Series:
    """Current value minus the calculated regression line value."""
    s = get_price_source(data, source)
    line = regression_line(data, length=length, source=source)
    res = s - line
    res.name = f"Residual_{length}"
    return _finalize_output(res, offset, fillna)

def residual_standard_error(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_REG_LEN, source: str = 'close', 
                            offset: int = 0, fillna: Any = None) -> pd.Series:
    """Standard error of the residuals."""
    res = residual(data, length=length, source=source)
    rse = np.sqrt((res**2).rolling(window=length).sum() / max(1, length - 2))
    rse.name = f"RSE_{length}"
    return _finalize_output(rse, offset, fillna)

def regression_channel(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_REG_LEN, std_mult: float = 2.0, 
                       source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.DataFrame:
    """Regression Line with Upper/Lower bands based on standard error."""
    line = regression_line(data, length=length, source=source)
    rse = residual_standard_error(data, length=length, source=source)
    
    upper = line + (std_mult * rse)
    lower = line - (std_mult * rse)
    
    out = pd.DataFrame({
        "middle": line,
        "upper": upper,
        "lower": lower
    }, index=line.index)
    return _finalize_output(out, offset, fillna)

# ==============================================================================
# MARKET RELATIONSHIP
# ==============================================================================

def beta(data: Union[pd.DataFrame, pd.Series], benchmark: Union[pd.DataFrame, pd.Series], 
         source: str = 'close', bench_source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    """Global Beta: Cov(Asset, Bench) / Var(Bench)."""
    cov_val = covariance(data, benchmark, source=source, bench_source=bench_source).iloc[-1]
    var_val = variance(benchmark, source=bench_source).iloc[-1]
    b = cov_val / var_val if var_val != 0 else np.nan
    s = get_price_source(data, source)
    return _finalize_output(_broadcast_scalar(b, s.index, "Beta"), offset, fillna)

def rolling_beta(data: Union[pd.DataFrame, pd.Series], benchmark: Union[pd.DataFrame, pd.Series], 
                 length: int = DEFAULT_CORR_LEN, source: str = 'close', bench_source: str = 'close', 
                 offset: int = 0, fillna: Any = None) -> pd.Series:
    """Rolling Beta."""
    rcov = rolling_covariance(data, benchmark, length=length, source=source, bench_source=bench_source)
    rvar = rolling_variance(benchmark, length=length, source=bench_source)
    b = safe_divide(rcov.to_numpy(dtype=np.float64), rvar.to_numpy(dtype=np.float64), default=np.nan)
    out = pd.Series(b, index=rcov.index, name=f"RollBeta_{length}")
    return _finalize_output(out, offset, fillna)

def alpha(data: Union[pd.DataFrame, pd.Series], benchmark: Union[pd.DataFrame, pd.Series], risk_free_rate: float = 0.0,
          source: str = 'close', bench_source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    """Global Jensen's Alpha."""
    s = get_price_source(data, source)
    b = get_price_source(benchmark, bench_source)
    
    ret_s = s.pct_change().mean()
    ret_b = b.pct_change().mean()
    b_val = beta(data, benchmark, source=source, bench_source=bench_source).iloc[-1]
    
    a = (ret_s - risk_free_rate) - b_val * (ret_b - risk_free_rate)
    return _finalize_output(_broadcast_scalar(a, s.index, "Alpha"), offset, fillna)

def rolling_alpha(data: Union[pd.DataFrame, pd.Series], benchmark: Union[pd.DataFrame, pd.Series], 
                  length: int = DEFAULT_CORR_LEN, risk_free_rate: float = 0.0, source: str = 'close', 
                  bench_source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    s = get_price_source(data, source)
    b = get_price_source(benchmark, bench_source)
    
    ret_s = s.pct_change().rolling(window=length).mean()
    ret_b = b.pct_change().rolling(window=length).mean()
    roll_b = rolling_beta(data, benchmark, length=length, source=source, bench_source=bench_source)
    
    a = (ret_s - risk_free_rate) - roll_b * (ret_b - risk_free_rate)
    a.name = f"RollAlpha_{length}"
    return _finalize_output(a, offset, fillna)

def tracking_error(data: Union[pd.DataFrame, pd.Series], benchmark: Union[pd.DataFrame, pd.Series], 
                   length: int = DEFAULT_CORR_LEN, source: str = 'close', bench_source: str = 'close', 
                   offset: int = 0, fillna: Any = None) -> pd.Series:
    """Rolling Standard Deviation of active return (Data Return - Benchmark Return)."""
    validate_length(length, "Tracking Error")
    s = get_price_source(data, source).pct_change()
    b = get_price_source(benchmark, bench_source).pct_change()
    
    active_ret = s - b
    te = active_ret.rolling(window=length).std()
    te.name = f"TE_{length}"
    return _finalize_output(te, offset, fillna)

def information_ratio(data: Union[pd.DataFrame, pd.Series], benchmark: Union[pd.DataFrame, pd.Series], 
                      length: int = DEFAULT_CORR_LEN, source: str = 'close', bench_source: str = 'close', 
                      offset: int = 0, fillna: Any = None) -> pd.Series:
    """Rolling Information Ratio: Active Return / Tracking Error."""
    s = get_price_source(data, source).pct_change()
    b = get_price_source(benchmark, bench_source).pct_change()
    
    active_ret = (s - b).rolling(window=length).mean()
    te = tracking_error(data, benchmark, length=length, source=source, bench_source=bench_source)
    
    ir = safe_divide(active_ret.to_numpy(dtype=np.float64), te.to_numpy(dtype=np.float64), default=np.nan)
    out = pd.Series(ir, index=s.index, name=f"IR_{length}")
    return _finalize_output(out, offset, fillna)

# ==============================================================================
# OUTLIER DETECTION
# ==============================================================================

def outlier_detection(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_Z_LEN, z_thresh: float = DEFAULT_OUTLIER_Z, 
                      source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    """Boolean mask: True if absolute Z-Score > threshold."""
    z = rolling_z_score(data, length=length, source=source)
    outliers = z.abs() > z_thresh
    outliers.name = f"Outlier_{length}_{z_thresh}"
    return _finalize_output(outliers, offset, fillna)

def three_sigma_rule(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_Z_LEN, source: str = 'close', 
                     offset: int = 0, fillna: Any = None) -> pd.Series:
    """Outlier detection using strict 3-sigma boundaries."""
    return outlier_detection(data, length=length, z_thresh=3.0, source=source, offset=offset, fillna=fillna)

def modified_z_outlier(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_Z_LEN, z_thresh: float = 3.5, 
                       source: str = 'close', offset: int = 0, fillna: Any = None) -> pd.Series:
    """Robust outlier detection using Modified Z-Score."""
    mod_z = modified_z_score(data, length=length, source=source)
    outliers = mod_z.abs() > z_thresh
    outliers.name = f"ModZOutlier_{length}_{z_thresh}"
    return _finalize_output(outliers, offset, fillna)

# ==============================================================================
# TREND
# ==============================================================================

def linear_trend_strength(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_REG_LEN, source: str = 'close', 
                          offset: int = 0, fillna: Any = None) -> pd.Series:
    """Trend Strength Proxy: Absolute Slope normalized by Price."""
    s = get_price_source(data, source)
    slope = rolling_slope(data, length=length, source=source)
    ts = 100.0 * safe_divide(np.abs(slope).to_numpy(dtype=np.float64), s.to_numpy(dtype=np.float64), default=np.nan)
    out = pd.Series(ts, index=s.index, name=f"TrendStr_{length}")
    return _finalize_output(out, offset, fillna)

def trend_angle(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_REG_LEN, source: str = 'close', 
                offset: int = 0, fillna: Any = None) -> pd.Series:
    """Angle of the Regression Line in degrees."""
    slope = rolling_slope(data, length=length, source=source)
    angle = np.degrees(np.arctan(slope))
    angle.name = f"TrendAngle_{length}"
    return _finalize_output(angle, offset, fillna)

def slope_percentage(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_REG_LEN, source: str = 'close', 
                     offset: int = 0, fillna: Any = None) -> pd.Series:
    """Slope expressed as percentage of the Price."""
    s = get_price_source(data, source)
    slope = rolling_slope(data, length=length, source=source)
    pct = 100.0 * safe_divide(slope.to_numpy(dtype=np.float64), s.to_numpy(dtype=np.float64), default=np.nan)
    out = pd.Series(pct, index=s.index, name=f"SlopePct_{length}")
    return _finalize_output(out, offset, fillna)

# ==============================================================================
# VOLATILITY STATISTICS
# ==============================================================================

def coefficient_of_dispersion(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_STAT_LEN, source: str = 'close', 
                              offset: int = 0, fillna: Any = None) -> pd.Series:
    """Quartile Coefficient of Dispersion: (Q3 - Q1) / (Q3 + Q1)."""
    s = get_price_source(data, source)
    q75 = s.rolling(window=length).quantile(0.75)
    q25 = s.rolling(window=length).quantile(0.25)
    
    num = (q75 - q25).to_numpy(dtype=np.float64)
    den = (q75 + q25).to_numpy(dtype=np.float64)
    
    cod = safe_divide(num, den, default=np.nan)
    out = pd.Series(cod, index=s.index, name=f"COD_{length}")
    return _finalize_output(out, offset, fillna)

def relative_standard_deviation(data: Union[pd.DataFrame, pd.Series], length: int = DEFAULT_STAT_LEN, source: str = 'close', 
                                offset: int = 0, fillna: Any = None) -> pd.Series:
    """Alias for Coefficient of Variation (CV) expressed as percentage."""
    cv = coefficient_of_variation(data, length=length, source=source) * 100.0
    cv.name = f"RSD_{length}"
    return _finalize_output(cv, offset, fillna)

# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "StatisticsIndicatorError",
    "validate_length",
    "get_price_source",
    "mean",
    "rolling_mean",
    "median",
    "rolling_median",
    "mode",
    "variance",
    "rolling_variance",
    "standard_deviation",
    "rolling_standard_deviation",
    "mean_absolute_deviation",
    "median_absolute_deviation",
    "root_mean_square",
    "coefficient_of_variation",
    "range_stat",
    "interquartile_range",
    "quantile",
    "percentile",
    "rolling_percentile",
    "percentile_rank",
    "z_score",
    "rolling_z_score",
    "modified_z_score",
    "normalization",
    "min_max_scaling",
    "robust_scaling",
    "winsorization",
    "skewness",
    "rolling_skewness",
    "kurtosis",
    "rolling_kurtosis",
    "entropy",
    "shannon_entropy",
    "jarque_bera",
    "normality_score",
    "pearson_correlation",
    "rolling_pearson",
    "spearman_correlation",
    "rolling_spearman",
    "kendall_correlation",
    "rolling_kendall",
    "autocorrelation",
    "lag_correlation",
    "cross_correlation",
    "covariance",
    "rolling_covariance",
    "linear_regression",
    "regression_line",
    "regression_value",
    "regression_intercept",
    "regression_slope",
    "rolling_regression",
    "rolling_slope",
    "residual",
    "residual_standard_error",
    "r_squared",
    "adjusted_r_squared",
    "regression_channel",
    "beta",
    "rolling_beta",
    "alpha",
    "rolling_alpha",
    "information_ratio",
    "tracking_error",
    "outlier_detection",
    "three_sigma_rule",
    "modified_z_outlier",
    "linear_trend_strength",
    "trend_angle",
    "slope_percentage",
    "coefficient_of_dispersion",
    "relative_standard_deviation"
]
