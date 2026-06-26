import pandas as pd
import numpy as np

# ==========================================
# 1. DESCRIPTIVE STATISTICS
# ==========================================

def calculate_mean(series: pd.Series) -> float:
    return series.mean()

def calculate_median(series: pd.Series) -> float:
    return series.median()

def calculate_variance(series: pd.Series) -> float:
    return series.var()

def calculate_std_dev(series: pd.Series) -> float:
    return series.std()

def calculate_z_score(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std()

def calculate_percentile(series: pd.Series, q: float) -> float:
    """q is between 0 and 100"""
    return np.percentile(series.dropna(), q)

# ==========================================
# 2. RELATIONAL STATISTICS
# ==========================================

def calculate_correlation(s1: pd.Series, s2: pd.Series) -> float:
    return s1.corr(s2)

def calculate_covariance(s1: pd.Series, s2: pd.Series) -> float:
    return s1.cov(s2)

# ==========================================
# 3. REGRESSION & FINANCIAL MODELING
# ==========================================

def calculate_linear_regression(series: pd.Series) -> tuple:
    """Returns (slope, intercept)"""
    y = series.values
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept

def calculate_regression_slope(series: pd.Series, period: int = 20) -> pd.Series:
    """Rolling Regression Slope"""
    return series.rolling(period).apply(lambda y: np.polyfit(np.arange(len(y)), y, 1)[0])

def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """Beta = Cov(Asset, Market) / Var(Market)"""
    covariance = asset_returns.cov(market_returns)
    variance = market_returns.var()
    return covariance / variance

def calculate_alpha(asset_returns: pd.Series, market_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Alpha = AssetReturn - (RiskFree + Beta * (MarketReturn - RiskFree))"""
    beta = calculate_beta(asset_returns, market_returns)
    alpha = asset_returns.mean() - (risk_free_rate + beta * (market_returns.mean() - risk_free_rate))
    return alpha

# ==========================================
# 4. DISTRIBUTION ANALYSIS
# ==========================================

def calculate_skewness(series: pd.Series) -> float:
    return series.skew()

def calculate_kurtosis(series: pd.Series) -> float:
    return series.kurtosis()
