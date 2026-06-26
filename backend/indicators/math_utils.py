import numpy as np
import pandas as pd

def calculate_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Fast rolling mean (SMA/Base) calculation."""
    return series.rolling(window=window, min_periods=1).mean()

def calculate_rolling_std(series: pd.Series, window: int) -> pd.Series:
    """Fast rolling standard deviation calculation."""
    return series.rolling(window=window, min_periods=1).std()

def calculate_ema(series: pd.Series, window: int) -> pd.Series:
    """Fast Exponential Moving Average calculation."""
    return series.ewm(span=window, adjust=False).mean()

def handle_nan_values(series_or_df):
    """Memory optimized NaN handling using forward fill then backward fill."""
    return series_or_df.ffill().bfill()

def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Vectorized safe division to prevent ZeroDivisionError."""
    # denominator 0 হলে 0 বসাবে, না হলে ভাগফল বসাবে
    return pd.Series(np.where(denominator == 0, 0, numerator / denominator), index=numerator.index)
