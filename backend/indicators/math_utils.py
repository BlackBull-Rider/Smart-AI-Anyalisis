import logging
import functools
import numpy as np
import pandas as pd
from typing import Any, Tuple, Union, Optional

# ==============================================================================
# LOGGING & CONSTANTS
# ==============================================================================

logger = logging.getLogger(__name__)

EPSILON = np.finfo(float).eps
DEFAULT_FILL = 0.0
MAX_FLOAT = np.finfo(np.float64).max
MIN_FLOAT = np.finfo(np.float64).min
MAX_EXP = np.log(MAX_FLOAT)

# ==============================================================================
# CUSTOM EXCEPTIONS
# ==============================================================================

class MathUtilsError(Exception): pass
class ValidationError(MathUtilsError): pass
class NumericalError(MathUtilsError): pass

# ==============================================================================
# DECORATORS & VALIDATION
# ==============================================================================

def validate_window(window: int):
    """Validates rolling window size."""
    if not isinstance(window, int) or window <= 0:
        logger.error(f"Invalid window size: {window}. Must be a positive integer.")
        raise ValidationError(f"Invalid window size: {window}")

def ensure_numpy_input(func):
    """Decorator to ensure the first argument is a float64 NumPy array."""
    @functools.wraps(func)
    def wrapper(data, *args, **kwargs):
        try:
            if isinstance(data, (pd.Series, pd.DataFrame)):
                arr = data.to_numpy(dtype=np.float64)
            else:
                arr = np.asarray(data, dtype=np.float64)
            return func(arr, *args, **kwargs)
        except Exception as e:
            logger.error(f"Failed to convert input to float64 numpy array in {func.__name__}: {e}")
            raise ValidationError(f"Invalid input dtype for {func.__name__}")
    return wrapper

def ensure_numpy_dual_input(func):
    """Decorator to ensure both primary arguments are float64 NumPy arrays."""
    @functools.wraps(func)
    def wrapper(a, b, *args, **kwargs):
        try:
            arr_a = a.to_numpy(dtype=np.float64) if isinstance(a, (pd.Series, pd.DataFrame)) else np.asarray(a, dtype=np.float64)
            arr_b = b.to_numpy(dtype=np.float64) if isinstance(b, (pd.Series, pd.DataFrame)) else np.asarray(b, dtype=np.float64)
            return func(arr_a, arr_b, *args, **kwargs)
        except Exception as e:
            logger.error(f"Failed dual conversion in {func.__name__}: {e}")
            raise ValidationError(f"Invalid input dtype for {func.__name__}")
    return wrapper

def validate_numeric(func):
    """Decorator to validate that the primary input is numeric before processing."""
    @functools.wraps(func)
    def wrapper(data, *args, **kwargs):
        arr = np.asarray(data)
        if not np.issubdtype(arr.dtype, np.number):
            logger.warning(f"Non-numeric data passed to {func.__name__}. Attempting cast.")
            try:
                arr = arr.astype(np.float64)
            except ValueError:
                logger.error(f"Data must be numeric in {func.__name__}")
                raise ValidationError(f"Data must be numeric in {func.__name__}")
        return func(arr, *args, **kwargs)
    return wrapper

# ==============================================================================
# FAST MATH (PERFORMANCE HELPERS)
# ==============================================================================

@ensure_numpy_input
def fast_relu(a: np.ndarray) -> np.ndarray:
    return np.maximum(0, a)

@ensure_numpy_input
def fast_log(a: np.ndarray) -> np.ndarray:
    return np.log(a)

@ensure_numpy_input
def fast_clip(a: np.ndarray, a_min: float, a_max: float) -> np.ndarray:
    return np.clip(a, a_min, a_max)

@ensure_numpy_input
def fast_square(a: np.ndarray) -> np.ndarray:
    return np.square(a)

@ensure_numpy_input
def fast_cube(a: np.ndarray) -> np.ndarray:
    return a * a * a

@ensure_numpy_input
def fast_abs(a: np.ndarray) -> np.ndarray:
    return np.abs(a)

@ensure_numpy_input
def fast_sigmoid(a: np.ndarray) -> np.ndarray:
    # Reusing stable logic to prevent large negative overflow
    return stable_sigmoid(a)

# ==============================================================================
# SAFE MATHEMATICAL OPERATIONS
# ==============================================================================

@ensure_numpy_dual_input
def safe_divide(a: np.ndarray, b: np.ndarray, default: float = DEFAULT_FILL) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(b != 0.0, a / b, default)
    return np.nan_to_num(result, nan=default, posinf=default, neginf=default)

@ensure_numpy_input
def safe_log(x: np.ndarray, default: float = DEFAULT_FILL) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(x > 0, np.log(x), default)
    return np.nan_to_num(result, nan=default, posinf=default, neginf=default)

@ensure_numpy_input
def safe_log10(x: np.ndarray, default: float = DEFAULT_FILL) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(x > 0, np.log10(x), default)
    return np.nan_to_num(result, nan=default, posinf=default, neginf=default)

def safe_ln(x: Any, default: float = DEFAULT_FILL) -> np.ndarray:
    return safe_log(x, default)

@ensure_numpy_input
def safe_exp(x: np.ndarray, default: float = MAX_FLOAT) -> np.ndarray:
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.where(x < MAX_EXP, np.exp(x), default)
    return np.nan_to_num(result, nan=default, posinf=default, neginf=default)

@ensure_numpy_input
def safe_power(x: np.ndarray, p: float, default: float = DEFAULT_FILL) -> np.ndarray:
    with np.errstate(invalid='ignore', over='ignore'):
        result = np.power(x, p)
        result = np.where(np.isfinite(result), result, default)
    return np.nan_to_num(result, nan=default, posinf=default, neginf=default)

@ensure_numpy_input
def safe_sqrt(x: np.ndarray, default: float = DEFAULT_FILL) -> np.ndarray:
    with np.errstate(invalid='ignore'):
        result = np.where(x >= 0, np.sqrt(x), default)
    return np.nan_to_num(result, nan=default, posinf=default, neginf=default)

@ensure_numpy_input
def safe_inverse(x: np.ndarray, default: float = DEFAULT_FILL) -> np.ndarray:
    return safe_divide(1.0, x, default)

@ensure_numpy_dual_input
def safe_percentage_change(current: np.ndarray, previous: np.ndarray, default: float = DEFAULT_FILL) -> np.ndarray:
    diff = np.subtract(current, previous)
    return safe_divide(diff, previous, default)

@ensure_numpy_dual_input
def safe_ratio(num: np.ndarray, den: np.ndarray, default: float = DEFAULT_FILL) -> np.ndarray:
    return safe_divide(num, den, default)

@validate_numeric
@ensure_numpy_input
def safe_min(x: np.ndarray, default: float = DEFAULT_FILL) -> float:
    arr = x[~np.isnan(x)]
    return float(np.min(arr)) if arr.size > 0 else default

@validate_numeric
@ensure_numpy_input
def safe_max(x: np.ndarray, default: float = DEFAULT_FILL) -> float:
    arr = x[~np.isnan(x)]
    return float(np.max(arr)) if arr.size > 0 else default

@validate_numeric
@ensure_numpy_input
def safe_prod(x: np.ndarray, default: float = DEFAULT_FILL) -> float:
    arr = x[~np.isnan(x)]
    return float(np.prod(arr)) if arr.size > 0 else default

@validate_numeric
@ensure_numpy_input
def safe_clip(x: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    return np.clip(np.nan_to_num(x, nan=0.0, posinf=max_val, neginf=min_val), min_val, max_val)

@validate_numeric
@ensure_numpy_input
def safe_mean(x: np.ndarray, default: float = DEFAULT_FILL) -> float:
    arr = x[~np.isnan(x)]
    return float(np.mean(arr)) if arr.size > 0 else default

@validate_numeric
@ensure_numpy_input
def safe_std(x: np.ndarray, default: float = DEFAULT_FILL) -> float:
    arr = x[~np.isnan(x)]
    return float(np.std(arr)) if arr.size > 1 else default

@validate_numeric
@ensure_numpy_input
def safe_sum(x: np.ndarray, default: float = DEFAULT_FILL) -> float:
    arr = x[~np.isnan(x)]
    return float(np.sum(arr)) if arr.size > 0 else default

@validate_numeric
@ensure_numpy_input
def safe_median(x: np.ndarray, default: float = DEFAULT_FILL) -> float:
    arr = x[~np.isnan(x)]
    return float(np.median(arr)) if arr.size > 0 else default

# ==============================================================================
# ADVANCED STATISTICAL HELPERS (PURE NUMPY)
# ==============================================================================

@validate_numeric
@ensure_numpy_input
def mode(data: np.ndarray) -> float:
    arr = data[~np.isnan(data)]
    if arr.size == 0: return np.nan
    vals, counts = np.unique(arr, return_counts=True)
    return float(vals[np.argmax(counts)])

@validate_numeric
@ensure_numpy_input
def variance(data: np.ndarray, ddof: int = 1) -> float:
    return float(np.nanvar(data, ddof=ddof))

@validate_numeric
@ensure_numpy_input
def std(data: np.ndarray, ddof: int = 1) -> float:
    return float(np.nanstd(data, ddof=ddof))

@validate_numeric
@ensure_numpy_input
def mad(data: np.ndarray) -> float:
    arr = data[~np.isnan(data)]
    if arr.size == 0: return np.nan
    return float(np.mean(np.abs(arr - np.mean(arr))))

@validate_numeric
@ensure_numpy_input
def iqr(data: np.ndarray) -> float:
    arr = data[~np.isnan(data)]
    if arr.size == 0: return np.nan
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))

@validate_numeric
@ensure_numpy_input
def percentile(data: np.ndarray, q: float) -> float:
    arr = data[~np.isnan(data)]
    if arr.size == 0: return np.nan
    return float(np.percentile(arr, q))

@validate_numeric
@ensure_numpy_input
def percentile_rank(data: np.ndarray, score: float) -> float:
    arr = data[~np.isnan(data)]
    if arr.size == 0: return np.nan
    return float((np.count_nonzero(arr < score) + 0.5 * np.count_nonzero(arr == score)) / len(arr) * 100)

@validate_numeric
@ensure_numpy_input
def zscore(data: np.ndarray) -> np.ndarray:
    arr_std = np.nanstd(data)
    if is_close(arr_std, 0.0): return np.zeros_like(data)
    return (data - np.nanmean(data)) / arr_std

@validate_numeric
@ensure_numpy_input
def robust_zscore(data: np.ndarray) -> np.ndarray:
    arr_median = np.nanmedian(data)
    arr_mad = mad(data)
    if is_close(arr_mad, 0.0): return np.zeros_like(data)
    # Using 1.4826 scale factor for normal distribution consistency
    return (data - arr_median) / (arr_mad * 1.4826)

@validate_numeric
@ensure_numpy_input
def normalize_minmax(data: np.ndarray) -> np.ndarray:
    arr_min = np.nanmin(data)
    arr_max = np.nanmax(data)
    if is_close(arr_min, arr_max): return np.zeros_like(data)
    return (data - arr_min) / (arr_max - arr_min)

@validate_numeric
@ensure_numpy_input
def standardize_zero_mean(data: np.ndarray) -> np.ndarray:
    return zscore(data)

@validate_numeric
@ensure_numpy_input
def winsorize(data: np.ndarray, limits: Tuple[float, float] = (0.05, 0.05)) -> np.ndarray:
    arr = data.copy()
    lower_bound = np.nanpercentile(arr, limits[0] * 100)
    upper_bound = np.nanpercentile(arr, 100 - (limits[1] * 100))
    return np.clip(arr, lower_bound, upper_bound)

@validate_numeric
@ensure_numpy_input
def remove_outliers(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    z_scores = np.abs(zscore(data))
    arr = data.copy()
    arr[z_scores > threshold] = np.nan
    return arr

@validate_numeric
@ensure_numpy_input
def cumulative_return(data: np.ndarray) -> np.ndarray:
    # Assumes data is percentage returns
    return np.nancumprod(1 + data) - 1.0

@validate_numeric
@ensure_numpy_input
def coefficient_of_variation(data: np.ndarray) -> float:
    arr_mean = np.nanmean(data)
    if is_close(arr_mean, 0.0): return np.nan
    return float(np.nanstd(data) / np.abs(arr_mean))

@validate_numeric
@ensure_numpy_input
def standard_error(data: np.ndarray) -> float:
    arr = data[~np.isnan(data)]
    if arr.size == 0: return np.nan
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr)))

@validate_numeric
@ensure_numpy_input
def entropy(data: np.ndarray) -> float:
    arr = data[~np.isnan(data)]
    if arr.size == 0: return np.nan
    _, counts = np.unique(arr, return_counts=True)
    probs = counts / len(arr)
    return float(-np.sum(probs * np.log2(probs)))

@validate_numeric
@ensure_numpy_input
def sharpe_like_score(data: np.ndarray, risk_free_rate: float = 0.0) -> float:
    arr = data[~np.isnan(data)]
    if arr.size < 2: return 0.0
    arr_std = np.std(arr)
    if is_close(arr_std, 0.0): return 0.0
    return float((np.mean(arr) - risk_free_rate) / arr_std)

@validate_numeric
@ensure_numpy_input
def signal_to_noise_ratio(data: np.ndarray) -> float:
    arr = data[~np.isnan(data)]
    if arr.size < 2: return 0.0
    arr_std = np.std(arr)
    if is_close(arr_std, 0.0): return 0.0
    return float(np.mean(arr) / arr_std)

@validate_numeric
@ensure_numpy_input
def geometric_mean(data: np.ndarray) -> float:
    arr = data[~np.isnan(data)]
    if arr.size == 0 or np.any(arr <= 0): return np.nan
    return float(np.exp(np.mean(np.log(arr))))

@validate_numeric
@ensure_numpy_input
def harmonic_mean(data: np.ndarray) -> float:
    arr = data[~np.isnan(data)]
    if arr.size == 0 or np.any(arr == 0): return np.nan
    return float(len(arr) / np.sum(1.0 / arr))

# ==============================================================================
# VECTOR OPERATIONS
# ==============================================================================

@ensure_numpy_dual_input
def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

@ensure_numpy_dual_input
def cross_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cross(a, b)

@ensure_numpy_input
def vector_norm(a: np.ndarray) -> float:
    return float(np.linalg.norm(a))

@ensure_numpy_input
def l1_norm(a: np.ndarray) -> float:
    return float(np.sum(np.abs(a)))

@ensure_numpy_input
def l2_norm(a: np.ndarray) -> float:
    return float(np.linalg.norm(a))

@ensure_numpy_input
def normalize_vector(a: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(a)
    return a / norm if not is_close(norm, 0.0) else np.zeros_like(a)

@ensure_numpy_dual_input
def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

@ensure_numpy_dual_input
def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))

@ensure_numpy_dual_input
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot_val = np.dot(a, b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if is_close(norm_a * norm_b, 0.0): return 0.0
    return float(dot_val / (norm_a * norm_b))

@ensure_numpy_dual_input
def vector_angle_cosine(a: np.ndarray, b: np.ndarray) -> float:
    cos_sim = np.clip(cosine_similarity(a, b), -1.0, 1.0)
    return float(np.arccos(cos_sim))

@ensure_numpy_dual_input
def projection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    b_dot_b = np.dot(b, b)
    if is_close(b_dot_b, 0.0): return np.zeros_like(b)
    return (np.dot(a, b) / b_dot_b) * b

# ==============================================================================
# ROLLING WINDOW OPERATIONS
# ==============================================================================

def _apply_rolling(data: Any, window: int, func_name: str, **kwargs) -> Any:
    validate_window(window)
    is_series = isinstance(data, pd.Series)
    s = pd.Series(data) if not is_series else data
    res = getattr(s.rolling(window=window, **kwargs), func_name)()
    return res if is_series else res.to_numpy()

def rolling_mean(data: Any, window: int) -> Any:
    return _apply_rolling(data, window, 'mean')

def rolling_sum(data: Any, window: int) -> Any:
    return _apply_rolling(data, window, 'sum')

def rolling_std(data: Any, window: int) -> Any:
    return _apply_rolling(data, window, 'std')

def rolling_var(data: Any, window: int) -> Any:
    return _apply_rolling(data, window, 'var')

def rolling_min(data: Any, window: int) -> Any:
    return _apply_rolling(data, window, 'min')

def rolling_max(data: Any, window: int) -> Any:
    return _apply_rolling(data, window, 'max')

def rolling_median(data: Any, window: int) -> Any:
    return _apply_rolling(data, window, 'median')

def rolling_quantile(data: Any, window: int, quantile: float) -> Any:
    validate_window(window)
    is_series = isinstance(data, pd.Series)
    s = pd.Series(data) if not is_series else data
    res = s.rolling(window=window).quantile(quantile)
    return res if is_series else res.to_numpy()

def rolling_percentile(data: Any, window: int, percentile: float) -> Any:
    # Wrapper for consistency (percentile 0-100 to quantile 0-1)
    return rolling_quantile(data, window, percentile / 100.0)

def rolling_iqr(data: Any, window: int) -> Any:
    validate_window(window)
    is_series = isinstance(data, pd.Series)
    s = pd.Series(data) if not is_series else data
    res = s.rolling(window=window).quantile(0.75) - s.rolling(window=window).quantile(0.25)
    return res if is_series else res.to_numpy()

def rolling_mad(data: Any, window: int) -> Any:
    # Note: Mean Absolute Deviation rolling is computationally heavy in Pandas.
    validate_window(window)
    is_series = isinstance(data, pd.Series)
    s = pd.Series(data) if not is_series else data
    res = s.rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return res if is_series else res.to_numpy()

def rolling_rank(data: Any, window: int) -> Any:
    """Optimized C-backend rolling rank (Pandas >= 1.4 support)."""
    validate_window(window)
    is_series = isinstance(data, pd.Series)
    s = pd.Series(data) if not is_series else data
    try:
        # Native pandas rolling rank (ultra-fast C backend)
        res = s.rolling(window=window).rank()
    except AttributeError:
        # Fallback for older pandas versions
        logger.warning("Native rolling.rank() not found. Falling back to slow apply.")
        res = s.rolling(window=window).apply(lambda x: pd.Series(x).rank().iloc[-1], raw=False)
    return res if is_series else res.to_numpy()

def rolling_zscore(data: Any, window: int) -> Any:
    validate_window(window)
    is_series = isinstance(data, pd.Series)
    s = pd.Series(data) if not is_series else data
    mean = s.rolling(window=window).mean()
    std = s.rolling(window=window).std()
    res = (s - mean) / std.replace(0, np.nan)
    return res if is_series else res.to_numpy()

def rolling_skew(data: Any, window: int) -> Any:
    return _apply_rolling(data, window, 'skew')

def rolling_kurtosis(data: Any, window: int) -> Any:
    return _apply_rolling(data, window, 'kurt')

def rolling_return(data: Any, window: int) -> Any:
    validate_window(window)
    is_series = isinstance(data, pd.Series)
    s = pd.Series(data) if not is_series else data
    res = (s / s.shift(window)) - 1.0
    return res if is_series else res.to_numpy()

def rolling_volatility(data: Any, window: int, trading_periods: int = 252) -> Any:
    validate_window(window)
    is_series = isinstance(data, pd.Series)
    s = pd.Series(data) if not is_series else data
    res = s.rolling(window=window).std() * np.sqrt(trading_periods)
    return res if is_series else res.to_numpy()

def rolling_drawdown(data: Any, window: int) -> Any:
    validate_window(window)
    is_series = isinstance(data, pd.Series)
    s = pd.Series(data) if not is_series else data
    peak = s.rolling(window=window, min_periods=1).max()
    res = (s - peak) / peak.replace(0, np.nan)
    return res if is_series else res.to_numpy()

def rolling_correlation(data_a: Any, data_b: Any, window: int) -> Any:
    validate_window(window)
    is_series = isinstance(data_a, pd.Series)
    s_a = pd.Series(data_a) if not is_series else data_a
    s_b = pd.Series(data_b) if not isinstance(data_b, pd.Series) else data_b
    res = s_a.rolling(window=window).corr(s_b)
    return res if is_series else res.to_numpy()

def rolling_covariance(data_a: Any, data_b: Any, window: int) -> Any:
    validate_window(window)
    is_series = isinstance(data_a, pd.Series)
    s_a = pd.Series(data_a) if not is_series else data_a
    s_b = pd.Series(data_b) if not isinstance(data_b, pd.Series) else data_b
    res = s_a.rolling(window=window).cov(s_b)
    return res if is_series else res.to_numpy()

def rolling_difference(data: Any, window: int = 1) -> Any:
    validate_window(window)
    is_series = isinstance(data, pd.Series)
    s = pd.Series(data) if not is_series else data
    res = s.diff(periods=window)
    return res if is_series else res.to_numpy()

def rolling_ratio(data: Any, window: int = 1) -> Any:
    validate_window(window)
    is_series = isinstance(data, pd.Series)
    s = pd.Series(data) if not is_series else data
    res = s / s.shift(window)
    return res if is_series else res.to_numpy()

# ==============================================================================
# MEMORY OPTIMIZATION & DATAFRAME HELPERS
# ==============================================================================

def optimize_float_dtype(data: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    df = data if inplace else data.copy()
    for col in df.select_dtypes(include=['float']):
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df

def optimize_integer_dtype(data: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    df = data if inplace else data.copy()
    for col in df.select_dtypes(include=['int']):
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

def convert_object_to_category(data: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    df = data if inplace else data.copy()
    for col in df.select_dtypes(include=['object']):
        # nunique with dropna=False is significantly faster on large datasets
        if df[col].nunique(dropna=False) / len(df[col]) < 0.5:
            df[col] = df[col].astype('category')
    return df

def reduce_dataframe_memory(data: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    df = data if inplace else data.copy()
    df = optimize_float_dtype(df, inplace=True)
    df = optimize_integer_dtype(df, inplace=True)
    df = convert_object_to_category(df, inplace=True)
    return df

# ==============================================================================
# NUMERICAL STABILITY & CACHING
# ==============================================================================

@functools.lru_cache(maxsize=1)
def epsilon() -> float:
    return EPSILON

def is_close(a: Any, b: Any, rtol: float = 1e-05, atol: float = 1e-08) -> Union[bool, np.ndarray]:
    if isinstance(a, (float, int)) and isinstance(b, (float, int)):
        return bool(np.isclose(a, b, rtol=rtol, atol=atol))
    return np.isclose(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), rtol=rtol, atol=atol)

@ensure_numpy_input
def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shift_x = x - np.max(x, axis=axis, keepdims=True)
    exps = np.exp(shift_x)
    return exps / np.sum(exps, axis=axis, keepdims=True)

@ensure_numpy_input
def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

@ensure_numpy_input
def stable_tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(np.clip(x, -20.0, 20.0))

@ensure_numpy_input
def log_sum_exp(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    max_val = np.max(x, axis=axis, keepdims=True)
    res = max_val + np.log(np.sum(np.exp(x - max_val), axis=axis, keepdims=True))
    return res if keepdims else np.squeeze(res, axis=axis)

# ==============================================================================
# __ALL__ EXPORTS
# ==============================================================================

__all__ = [
    # Decorators & Validation
    "validate_window", "ensure_numpy_input", "ensure_numpy_dual_input", "validate_numeric",
    
    # Fast Math
    "fast_relu", "fast_log", "fast_clip", "fast_square", "fast_cube", "fast_sigmoid", "fast_abs",
    
    # Safe Math
    "safe_divide", "safe_log", "safe_log10", "safe_ln", "safe_exp", "safe_power", "safe_sqrt",
    "safe_inverse", "safe_percentage_change", "safe_ratio", "safe_min", "safe_max", "safe_prod",
    "safe_clip", "safe_mean", "safe_std", "safe_sum", "safe_median",
    
    # Advanced Statistical Helpers
    "mode", "variance", "std", "mad", "iqr", "percentile", "percentile_rank", "zscore",
    "robust_zscore", "normalize_minmax", "standardize_zero_mean", "winsorize", "remove_outliers",
    "cumulative_return", "coefficient_of_variation", "standard_error", "entropy",
    "sharpe_like_score", "signal_to_noise_ratio", "geometric_mean", "harmonic_mean",
    
    # Vector Operations
    "dot_product", "cross_product", "vector_norm", "l1_norm", "l2_norm", "normalize_vector",
    "euclidean_distance", "manhattan_distance", "cosine_similarity", "vector_angle_cosine",
    "projection",
    
    # Rolling Operations
    "rolling_mean", "rolling_sum", "rolling_std", "rolling_var", "rolling_min", "rolling_max",
    "rolling_median", "rolling_quantile", "rolling_percentile", "rolling_iqr", "rolling_mad",
    "rolling_rank", "rolling_zscore", "rolling_skew", "rolling_kurtosis", "rolling_return",
    "rolling_volatility", "rolling_drawdown", "rolling_correlation", "rolling_covariance",
    "rolling_difference", "rolling_ratio",
    
    # Memory Optimization
    "optimize_float_dtype", "optimize_integer_dtype", "convert_object_to_category",
    "reduce_dataframe_memory",
    
    # Numerical Stability
    "epsilon", "is_close", "stable_softmax", "stable_sigmoid", "stable_tanh", "log_sum_exp"
]
