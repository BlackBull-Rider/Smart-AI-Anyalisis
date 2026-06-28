import logging
import numpy as np
import pandas as pd
from typing import Union, List, Optional, Tuple, Any, Callable
from pandas.api.types import is_numeric_dtype, is_bool_dtype

# ==============================================================================
# LOGGING
# ==============================================================================

logger = logging.getLogger(__name__)

# ==============================================================================
# CUSTOM EXCEPTIONS
# ==============================================================================

class IndicatorHelperError(Exception):
    """Base exception for indicator helper errors."""
    pass

class ValidationError(IndicatorHelperError):
    """Exception raised for data validation failures."""
    pass

class CleaningError(IndicatorHelperError):
    """Exception raised during data cleaning processes."""
    pass

# ==============================================================================
# INPUT CONVERSION & STANDARDIZATION
# ==============================================================================

def ensure_dataframe(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    try:
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Cannot convert to DataFrame: {e}")
        raise ValidationError(f"Invalid type for DataFrame conversion: {type(data)}")

def ensure_series(data: Any, name: str = 'series') -> pd.Series:
    if isinstance(data, pd.Series):
        return data
    if isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            return data.iloc[:, 0]
        raise ValidationError("DataFrame has multiple columns; cannot cast to Series implicitly.")
    try:
        return pd.Series(data, name=name)
    except Exception as e:
        logger.error(f"Cannot convert to Series: {e}")
        raise ValidationError(f"Invalid type for Series conversion: {type(data)}")

def ensure_numpy(data: Any) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, (pd.Series, pd.DataFrame)):
        return data.to_numpy()
    try:
        return np.asarray(data)
    except Exception as e:
        logger.error(f"Cannot convert to NumPy array: {e}")
        raise ValidationError(f"Invalid type for NumPy conversion: {type(data)}")

def ensure_float(data: Any) -> np.ndarray:
    arr = ensure_numpy(data)
    if arr.dtype != np.float64:
        return arr.astype(np.float64, copy=False)
    return arr

def ensure_integer(data: Any) -> np.ndarray:
    arr = ensure_numpy(data)
    if arr.dtype != np.int64:
        return arr.astype(np.int64, copy=False)
    return arr

def ensure_datetime(index: Any, tz: Optional[str] = None) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        try:
            index = pd.to_datetime(index)
        except Exception as e:
            logger.error(f"Cannot convert index to DatetimeIndex: {e}")
            raise ValidationError("Index must be convertible to datetime.")
    
    if tz is not None:
        if index.tz is None:
            index = index.tz_localize('UTC').tz_convert(tz)
        else:
            index = index.tz_convert(tz)
    return index

def ensure_copy(data: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    return data.copy(deep=True)

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strips whitespace and converts column names to lowercase."""
    df.columns = df.columns.str.strip().str.lower()
    return df

# ==============================================================================
# OHLCV VALIDATION
# ==============================================================================

def validate_required_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    df_cols = set(df.columns)
    missing = [c for c in required_cols if c not in df_cols]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        raise ValidationError(f"Missing required columns: {missing}")

def validate_numeric_columns(df: pd.DataFrame) -> None:
    for col in df.columns:
        if not is_numeric_dtype(df[col]) or is_bool_dtype(df[col]):
            logger.error(f"Column '{col}' is not strictly numeric.")
            raise ValidationError(f"DataFrame must contain only numeric data. Column '{col}' failed.")

def validate_index(df: pd.DataFrame) -> None:
    if df.index is None or len(df.index) != len(df):
        logger.error("Invalid or mismatched index.")
        raise ValidationError("DataFrame index is invalid or mismatched.")

def validate_datetime_index(df: pd.DataFrame) -> None:
    validate_index(df)
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("Index is not a DatetimeIndex.")
        raise ValidationError("DataFrame must have a DatetimeIndex.")

def validate_monotonic_index(df: pd.DataFrame) -> None:
    if not df.index.is_monotonic_increasing:
        logger.error("Index is not monotonically increasing.")
        raise ValidationError("Index must be sorted chronologically.")

def validate_length(df: Union[pd.DataFrame, pd.Series], min_len: int = 1) -> None:
    if len(df) < min_len:
        logger.error(f"Data length {len(df)} is less than required minimum {min_len}.")
        raise ValidationError(f"Insufficient data length. Required: {min_len}, Got: {len(df)}")

def validate_no_duplicates(df: pd.DataFrame) -> None:
    if df.index.duplicated().any():
        logger.error("Duplicate index values detected.")
        raise ValidationError("DataFrame index contains duplicates.")

def validate_missing_data(df: pd.DataFrame) -> None:
    if df.isna().any().any():
        logger.error("NaN values detected in DataFrame.")
        raise ValidationError("DataFrame contains missing (NaN) values.")

def validate_price(series: pd.Series) -> None:
    if not is_numeric_dtype(series) or is_bool_dtype(series):
        raise ValidationError(f"Price series '{series.name}' must be numeric.")
    if (series <= 0).any():
        logger.error(f"Negative or zero prices detected in '{series.name}'.")
        raise ValidationError(f"Prices must be strictly positive in '{series.name}'.")

def validate_high_low(df: pd.DataFrame) -> None:
    validate_required_columns(df, ['high', 'low'])
    if (df['high'] < df['low']).any():
        logger.error("High price is less than Low price in some rows.")
        raise ValidationError("Invalid OHLC data: High cannot be less than Low.")

def validate_volume(df: pd.DataFrame) -> None:
    validate_required_columns(df, ['volume'])
    if (df['volume'] < 0).any():
        logger.error("Negative volume detected.")
        raise ValidationError("Volume cannot be negative.")

def validate_ohlc(df: pd.DataFrame) -> None:
    df = ensure_dataframe(df)
    validate_required_columns(df, ['open', 'high', 'low', 'close'])
    validate_numeric_columns(df[['open', 'high', 'low', 'close']])
    validate_price(df['open'])
    validate_price(df['high'])
    validate_price(df['low'])
    validate_price(df['close'])
    validate_high_low(df)

def validate_ohlcv(df: pd.DataFrame) -> None:
    validate_ohlc(df)
    validate_volume(df)

def validate_dataframe(df: pd.DataFrame) -> None:
    df = ensure_dataframe(df)
    validate_index(df)
    validate_datetime_index(df)
    validate_length(df, min_len=1)

# ==============================================================================
# DATA CLEANING
# ==============================================================================

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    dup_count = df.index.duplicated().sum()
    if dup_count > 0:
        logger.warning(f"Removing {dup_count} duplicate index entries.")
    return df[~df.index.duplicated(keep='last')]

def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    original_len = len(df)
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df_clean) < original_len:
        logger.warning(f"Removed {original_len - len(df_clean)} rows containing NaN/Inf.")
    return df_clean

def remove_zero_volume(df: pd.DataFrame) -> pd.DataFrame:
    if 'volume' in df.columns:
        original_len = len(df)
        df = df[df['volume'] > 0]
        if len(df) < original_len:
            logger.info(f"Removed {original_len - len(df)} rows with zero volume.")
    return df

def remove_negative_prices(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c in ['open', 'high', 'low', 'close']]
    if not cols:
        return df
    mask = (df[cols] > 0).all(axis=1)
    return df[mask]

def replace_invalid_values(df: pd.DataFrame, fill_value: float = 0.0) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf, np.nan], fill_value)

def forward_fill(df: pd.DataFrame) -> pd.DataFrame:
    return df.ffill()

def backward_fill(df: pd.DataFrame) -> pd.DataFrame:
    return df.bfill()

def fill_missing(df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
    if method == 'ffill':
        return forward_fill(df)
    elif method == 'bfill':
        return backward_fill(df)
    elif method == 'interpolate':
        return df.interpolate(method='linear')
    else:
        logger.error(f"Unsupported fill method: {method}")
        raise CleaningError(f"Unsupported fill method: {method}")

def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()

def clip_prices(df: pd.DataFrame, min_val: float, max_val: float) -> pd.DataFrame:
    cols = [c for c in df.columns if c in ['open', 'high', 'low', 'close']]
    df_out = df.copy()
    df_out[cols] = df_out[cols].clip(lower=min_val, upper=max_val)
    return df_out

def sort_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_index()

def remove_outliers(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    """Removes statistical outliers using Z-score, calculating mask across all columns first."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if numeric_cols.empty:
        return df
    z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std(ddof=0))
    mask = (z_scores < z_thresh).all(axis=1)
    dropped = len(df) - mask.sum()
    if dropped > 0:
        logger.info(f"Removed {dropped} outlier rows.")
    return df[mask]

def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standard production-grade cleaning pipeline for OHLCV data.
    """
    df = ensure_dataframe(df)
    df = standardize_column_names(df)
    df = sort_dataframe(df)
    df = remove_duplicates(df)
    df = replace_invalid_values(df, fill_value=np.nan) 
    df = remove_invalid_rows(df)
    validate_required_columns(df, ['open', 'high', 'low', 'close', 'volume'])
    df = remove_negative_prices(df)
    df = remove_zero_volume(df)
    validate_ohlcv(df)
    validate_monotonic_index(df)
    return df

# ==============================================================================
# ROLLING HELPERS
# ==============================================================================

def rolling_apply(series: pd.Series, window: int, func: Callable, raw: bool = True) -> pd.Series:
    return series.rolling(window=window).apply(func, raw=raw)

def rolling_shift(series: pd.Series, periods: int = 1) -> pd.Series:
    return series.shift(periods)

def rolling_difference(series: pd.Series, periods: int = 1) -> pd.Series:
    return series.diff(periods=periods)

def rolling_ratio(series: pd.Series, periods: int = 1) -> pd.Series:
    shifted = series.shift(periods)
    return np.where(shifted != 0, series / shifted, np.nan)

def rolling_return(series: pd.Series, periods: int = 1) -> pd.Series:
    return series.pct_change(periods=periods)

def rolling_window_view(arr: np.ndarray, window: int) -> np.ndarray:
    arr = ensure_numpy(arr)
    if len(arr) < window:
        logger.error(f"Array length {len(arr)} is smaller than window {window}.")
        raise ValidationError(f"Array length {len(arr)} is smaller than window {window}.")
    return np.lib.stride_tricks.sliding_window_view(arr, window_shape=window)

def rolling_min_periods(series: pd.Series, window: int, min_periods: int) -> pd.core.window.rolling.Rolling:
    return series.rolling(window=window, min_periods=min_periods)

def expanding_apply(series: pd.Series, func: Callable, min_periods: int = 1, raw: bool = True) -> pd.Series:
    return series.expanding(min_periods=min_periods).apply(func, raw=raw)

def expanding_mean(series: pd.Series, min_periods: int = 1) -> pd.Series:
    return series.expanding(min_periods=min_periods).mean()

def expanding_std(series: pd.Series, min_periods: int = 1) -> pd.Series:
    return series.expanding(min_periods=min_periods).std()

# ==============================================================================
# TIMEFRAME & TIMEZONE HELPERS
# ==============================================================================

def convert_timezone(df: pd.DataFrame, tz: str = 'Asia/Kolkata') -> pd.DataFrame:
    validate_datetime_index(df)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert(tz)
    else:
        df.index = df.index.tz_convert(tz)
    return df

def infer_timezone(df: pd.DataFrame) -> Optional[str]:
    validate_datetime_index(df)
    if df.index.tz is not None:
        return str(df.index.tz)
    return None

def validate_timezone(df: pd.DataFrame) -> None:
    validate_datetime_index(df)
    if df.index.tz is None:
        logger.warning("DataFrame index has no timezone. Expected tz-aware index.")

def detect_timeframe(df: pd.DataFrame) -> pd.Timedelta:
    validate_datetime_index(df)
    if len(df) < 2:
        logger.error("Need at least 2 rows to detect timeframe.")
        raise ValidationError("Need at least 2 rows to detect timeframe.")
    deltas = df.index.to_series().diff().dropna()
    return deltas.median()

def convert_timeframe(td: pd.Timedelta) -> str:
    seconds = td.total_seconds()
    if seconds >= 86400:
        return f"{int(seconds // 86400)}D"
    elif seconds >= 3600:
        return f"{int(seconds // 3600)}h"
    elif seconds >= 60:
        return f"{int(seconds // 60)}min"
    return f"{int(seconds)}s"

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    validate_datetime_index(df)
    df = standardize_column_names(df)
    validate_required_columns(df, ['open', 'high', 'low', 'close', 'volume'])
    
    resampler = df.resample(timeframe)
    resampled = pd.DataFrame({
        'open': resampler['open'].first(),
        'high': resampler['high'].max(),
        'low': resampler['low'].min(),
        'close': resampler['close'].last(),
        'volume': resampler['volume'].sum(min_count=1)
    })
    
    # Drop rows where all price data is missing
    resampled.dropna(subset=['open', 'high', 'low', 'close'], how='all', inplace=True)
    # Fill remaining missing volume with 0
    resampled['volume'] = resampled['volume'].fillna(0.0)
    
    return resampled

def is_intraday(df: pd.DataFrame) -> bool:
    return detect_timeframe(df).total_seconds() < 86400

def is_daily(df: pd.DataFrame) -> bool:
    return detect_timeframe(df).total_seconds() == 86400

def is_weekly(df: pd.DataFrame) -> bool:
    return detect_timeframe(df).total_seconds() == 604800

def is_monthly(df: pd.DataFrame) -> bool:
    seconds = detect_timeframe(df).total_seconds()
    return 2419200 <= seconds <= 2678400

def bars_per_day(df: pd.DataFrame) -> float:
    td = detect_timeframe(df).total_seconds()
    if td >= 86400:
        return 1.0
    return 22500 / td  # 6.25 hours standard Indian market

def bars_per_week(df: pd.DataFrame) -> float:
    return bars_per_day(df) * 5

def bars_per_month(df: pd.DataFrame) -> float:
    return bars_per_day(df) * 21

def annualization_factor(df: pd.DataFrame, trading_days: float = 252.0) -> float:
    td = detect_timeframe(df).total_seconds()
    if td >= 86400:
        return trading_days
    return trading_days * bars_per_day(df)

# ==============================================================================
# COMMON UTILITIES
# ==============================================================================

def price_change(series: pd.Series) -> pd.Series:
    return series.diff(1)

def percent_change(series: pd.Series) -> pd.Series:
    return series.pct_change(1) * 100.0

def log_return(series: pd.Series) -> pd.Series:
    return np.log(series / series.shift(1))

def simple_return(series: pd.Series) -> pd.Series:
    return series.pct_change(1)

def true_range_inputs(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1)

def hl2(high: pd.Series, low: pd.Series) -> pd.Series:
    return (high + low) / 2.0

def hlc3(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return (high + low + close) / 3.0

def ohlc4(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return (open_ + high + low + close) / 4.0

def typical_price(df: pd.DataFrame) -> pd.Series:
    df = standardize_column_names(df)
    return hlc3(df['high'], df['low'], df['close'])

def weighted_price(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return (high + low + 2 * close) / 4.0

def median_price(df: pd.DataFrame) -> pd.Series:
    df = standardize_column_names(df)
    return hl2(df['high'], df['low'])

def price_range(high: pd.Series, low: pd.Series) -> pd.Series:
    return high - low

def body_size(open_: pd.Series, close: pd.Series) -> pd.Series:
    return (close - open_).abs()

def upper_shadow(open_: pd.Series, high: pd.Series, close: pd.Series) -> pd.Series:
    return high - np.maximum(open_, close)

def lower_shadow(open_: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return np.minimum(open_, close) - low

def candle_direction(open_: pd.Series, close: pd.Series) -> pd.Series:
    return np.sign(close - open_)

# ==============================================================================
# DATA ALIGNMENT
# ==============================================================================

def align_series(s1: pd.Series, s2: pd.Series, join: str = 'inner') -> Tuple[pd.Series, pd.Series]:
    return s1.align(s2, join=join)

def align_dataframe(df1: pd.DataFrame, df2: pd.DataFrame, join: str = 'inner') -> Tuple[pd.DataFrame, pd.DataFrame]:
    return df1.align(df2, join=join, axis=0)

def common_index(*dfs: Union[pd.DataFrame, pd.Series]) -> pd.Index:
    if not dfs:
        return pd.Index([])
    idx = dfs[0].index
    for df in dfs[1:]:
        idx = idx.intersection(df.index)
    return idx

def intersection_index(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.Index:
    return df1.index.intersection(df2.index)

def union_index(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.Index:
    return df1.index.union(df2.index)

# ==============================================================================
# SAFE HELPERS
# ==============================================================================

def safe_shift(data: Union[pd.DataFrame, pd.Series], periods: int, fill_value: float = 0.0) -> Union[pd.DataFrame, pd.Series]:
    return data.shift(periods).fillna(fill_value)

def safe_slice(data: Union[pd.DataFrame, pd.Series], start: int, end: int) -> Union[pd.DataFrame, pd.Series]:
    length = len(data)
    start = max(0, min(start, length))
    end = max(0, min(end, length))
    return data.iloc[start:end]

def safe_select(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    existing_cols = [c for c in columns if c in df.columns]
    if not existing_cols:
        logger.warning("None of the requested columns exist in the DataFrame.")
    return df[existing_cols]

def safe_merge(df1: pd.DataFrame, df2: pd.DataFrame, how: str = 'inner') -> pd.DataFrame:
    """Merges two dataframes on index, cleanly dropping pre-existing duplicates first."""
    df1_clean = df1[~df1.index.duplicated(keep='last')]
    df2_clean = df2[~df2.index.duplicated(keep='last')]
    return df1_clean.merge(df2_clean, left_index=True, right_index=True, how=how)

def safe_concat(dfs: List[Union[pd.DataFrame, pd.Series]], axis: int = 1) -> Union[pd.DataFrame, pd.Series]:
    """Safely concatenates objects, ignoring Nones and empty objects."""
    valid_dfs = [df for df in dfs if df is not None and not df.empty]
    if not valid_dfs:
        logger.error("All provided objects for concatenation are empty or None.")
        raise CleaningError("All provided objects for concatenation are empty or None.")
    return pd.concat(valid_dfs, axis=axis)

def safe_copy(data: Any) -> Any:
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.copy(deep=True)
    if isinstance(data, np.ndarray):
        return np.copy(data)
    import copy
    return copy.deepcopy(data)

# ==============================================================================
# __ALL__ EXPORTS
# ==============================================================================

__all__ = [
    "IndicatorHelperError", "ValidationError", "CleaningError",
    "ensure_dataframe", "ensure_series", "ensure_numpy", "ensure_float", "ensure_integer",
    "ensure_datetime", "ensure_copy", "standardize_column_names", "validate_required_columns",
    "validate_numeric_columns", "validate_index", "validate_datetime_index", 
    "validate_monotonic_index", "validate_length", "validate_no_duplicates",
    "validate_missing_data", "validate_price", "validate_high_low", "validate_volume",
    "validate_ohlc", "validate_ohlcv", "validate_dataframe", "remove_duplicates",
    "remove_invalid_rows", "remove_zero_volume", "remove_negative_prices",
    "replace_invalid_values", "forward_fill", "backward_fill", "fill_missing", "drop_missing",
    "clip_prices", "sort_dataframe", "remove_outliers", "clean_ohlcv", "rolling_apply",
    "rolling_shift", "rolling_difference", "rolling_ratio", "rolling_return", "rolling_window_view",
    "rolling_min_periods", "expanding_apply", "expanding_mean", "expanding_std", 
    "convert_timezone", "infer_timezone", "validate_timezone", "detect_timeframe",
    "convert_timeframe", "resample_ohlcv", "is_intraday", "is_daily", "is_weekly", "is_monthly",
    "bars_per_day", "bars_per_week", "bars_per_month", "annualization_factor", "price_change",
    "percent_change", "log_return", "simple_return", "true_range_inputs", "hl2", "hlc3", "ohlc4",
    "typical_price", "weighted_price", "median_price", "price_range", "body_size", "upper_shadow",
    "lower_shadow", "candle_direction", "align_series", "align_dataframe", "common_index",
    "intersection_index", "union_index", "safe_shift", "safe_slice", "safe_select", "safe_merge",
    "safe_concat", "safe_copy"
]
