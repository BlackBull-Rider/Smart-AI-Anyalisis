import pandas as pd
import numpy as np

def calculate_candle_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Candlestick metrics computation for analysis.
    """
    metrics = pd.DataFrame(index=df.index)
    
    # Body and Range
    metrics['body_size'] = (df['open'] - df['close']).abs()
    metrics['candle_range'] = df['high'] - df['low']
    
    # Wick Analysis
    metrics['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    metrics['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    
    # Power Analysis
    # Bull Power: High - (Closer of Open/Close) - Simplifying to High-Close for simplicity
    metrics['bull_power'] = df['high'] - df[['open', 'close']].max(axis=1)
    metrics['bear_power'] = df[['open', 'close']].min(axis=1) - df['low']
    
    # Gap Analysis
    gap = df['open'] - df['close'].shift(1)
    metrics['gap_up'] = np.where(gap > 0, gap, 0)
    metrics['gap_down'] = np.where(gap < 0, gap.abs(), 0)
    
    # Pattern Detection
    # Doji: Body size is less than 10% of total range
    metrics['doji'] = (metrics['body_size'] <= (metrics['candle_range'] * 0.1)).astype(int)
    
    # Marubozu: Wicks are less than 10% of total range
    total_wick = metrics['upper_wick'] + metrics['lower_wick']
    metrics['marubozu'] = (total_wick <= (metrics['candle_range'] * 0.1)).astype(int)
    
    return metrics
