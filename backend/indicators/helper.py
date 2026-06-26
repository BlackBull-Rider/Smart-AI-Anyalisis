import pandas as pd
import logging

logger = logging.getLogger("IndicatorHelper")

def validate_ohlcv(df: pd.DataFrame) -> bool:
    """চেক করবে ডেটাফ্রেমে open, high, low, close, volume আছে কি না।"""
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required_cols if col not in df.columns.str.lower()]
    
    if missing:
        logger.error(f"Missing Required Columns for Indicator Calculation: {missing}")
        return False
    return True

def prepare_data_for_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    ইন্ডিকেটর ক্যালকুলেশনের আগে ডেটা ক্লিন এবং ম্যাপ করে।
    স্পেশাল ম্যাপিং: 'AH' কলামকে signal এবং 'B' কলামকে close হিসেবে ধরবে।
    """
    if df.empty:
        return df
        
    df = df.copy()
    # সব কলামের নাম ছোট হাতের করে নেওয়া যাতে ক্যালকুলেশনে সুবিধা হয়
    df.columns = df.columns.str.lower()
    
    # Custom Mapping 
    if 'b' in df.columns:
        df['close'] = df['b']
        df['entry_price'] = df['b']
    if 'ah' in df.columns:
        df['signal'] = df['ah']
        df['entry_signal'] = df['ah']
        
    if validate_ohlcv(df):
        # NaN ভ্যালুগুলো ফিলাপ করা যাতে ম্যাথ ইঞ্জিন ক্র্যাশ না করে
        df = df.ffill().bfill()
        
    return df
