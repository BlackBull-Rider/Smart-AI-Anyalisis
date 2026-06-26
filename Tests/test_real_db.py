import pytest
import pandas as pd
import numpy as np
from Indicators.helper import fetch_real_ohlcv, get_existing_symbol

def test_real_db_symbol_exists():
    """টেস্ট ১: ডাটাবেস থেকে রিয়েল সিম্বল স্ক্যান করা যাচ্ছে কিনা"""
    symbol = get_existing_symbol()
    assert symbol is not None, "ফেইল: ডাটাবেসে কোনো সিম্বল পাওয়া যায়নি বা ডাটাবেস পাথ ভুল!"
    print(f"\n[+] Found REAL Symbol for testing: {symbol}")

def test_fetch_real_ohlcv_data():
    """টেস্ট ২: রিয়েল সিম্বলের OHLCV ডেটা ঠিকমতো ফেচ এবং ভ্যালিডেট হচ্ছে কিনা"""
    symbol = get_existing_symbol()
    if not symbol:
        pytest.skip("No symbol found, skipping data fetch test.")
        
    df = fetch_real_ohlcv(symbol, limit=100)
    
    assert not df.empty, f"ফেইল: {symbol} এর জন্য কোনো ডেটা আসেনি!"
    assert len(df) <= 100, "ফেইল: লিমিটের চেয়ে বেশি ডেটা চলে এসেছে!"
    
    # কলামগুলো ঠিক আছে কিনা চেক করা
    expected_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    for col in expected_cols:
        assert col in df.columns, f"ফেইল: রিয়েল ডেটায় '{col}' কলামটি নেই!"

def test_data_types_and_validation():
    """টেস্ট ৩: ম্যাথ ক্যালকুলেশনের জন্য ডেটা টাইপ float64 এ কনভার্ট হয়েছে কিনা"""
    symbol = get_existing_symbol()
    if not symbol:
        pytest.skip("No symbol found, skipping validation test.")
        
    df = fetch_real_ohlcv(symbol, limit=10)
    
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        assert df[col].dtype == np.float64, f"ফেইল: '{col}' কলামের ডেটা টাইপ float64 নয়!"
        
    # টাইম-সিরিজ পুরনো থেকে নতুন (Ascending) অর্ডারে আছে কিনা চেক করা
    assert df['date'].iloc[0] < df['date'].iloc[-1], "ফেইল: ডেটা টাইম-সিরিজ অনুযায়ী সোজা করা নেই (নতুন থেকে পুরনো হয়ে আছে)!"
