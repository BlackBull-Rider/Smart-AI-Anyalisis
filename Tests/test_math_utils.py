import pytest
import numpy as np
from Indicators.helper import fetch_real_ohlcv, get_existing_symbol
from Indicators.math_utils import enforce_writeable_float_array_fast, get_rolling_window, replace_nan_with_zero

def get_real_close_data():
    """ডাটাবেস থেকে রিয়েল ক্লোজ প্রাইস ফেচ করার হেল্পার ফাংশন"""
    symbol = get_existing_symbol()
    if not symbol:
        pytest.skip("No real data found in database to test.")
    df = fetch_real_ohlcv(symbol, limit=100)
    if df.empty or 'close' not in df.columns:
        pytest.skip("Invalid data fetched.")
    return df['close'].values

def test_real_data_memory_pointer():
    """টেস্ট ১: রিয়েল ডেটার ওপর মেমোরি পয়েন্টার এবং C-Contiguous চেক"""
    real_close = get_real_close_data()
    fast_arr = enforce_writeable_float_array_fast(real_close)

    assert isinstance(fast_arr, np.ndarray), "ফেইল: এটি নাম্পাই অ্যারে নয়"
    assert fast_arr.dtype == np.float64, "ফেইল: ডাটা টাইপ float64 নয়"
    assert fast_arr.flags['C_CONTIGUOUS'], "ফেইল: মেমোরি C-contiguous নয় (লাইভ মার্কেটে ল্যাগ হবে)"
    assert fast_arr.flags['WRITEABLE'], "ফেইল: ডাটা রাইটেবল নয়"

def test_real_data_rolling_window():
    """টেস্ট ২: রিয়েল ডেটার ওপর জিরো-ল্যাগ রোলিং উইন্ডো (Stride tricks) চেক (যেমন ১৪ পিরিয়ডের জন্য)"""
    real_close = get_real_close_data()
    window = 14
    
    if len(real_close) < window:
        pytest.skip("Not enough data for rolling window test.")

    rolling = get_rolling_window(real_close, window)

    expected_shape = (len(real_close) - window + 1, window)
    assert rolling.shape == expected_shape, "ফেইল: রিয়েল ডেটার রোলিং উইন্ডোর শেপ ভুল"
    
    # প্রথম উইন্ডো রিয়েল ডেটার প্রথম ১৪টি ক্যান্ডেলের সমান কিনা চেক করা
    assert np.array_equal(rolling[0], real_close[:window]), "ফেইল: প্রথম উইন্ডো ম্যাচ করেনি"

def test_real_data_nan_handling():
    """টেস্ট ৩: রিয়েল ডেটায় হঠাৎ মিসিং ভ্যালু (NaN) এলে ইঞ্জিন যেন ক্র্যাশ না করে"""
    real_close = get_real_close_data()
    
    # রিয়েল ডেটার একটি কপিতে জোর করে মার্কেটের গ্লিচ (NaN এবং Inf) ঢুকিয়ে চেক করা
    dirty_data = np.copy(real_close)
    dirty_data[1] = np.nan
    dirty_data[2] = np.inf

    clean_arr = replace_nan_with_zero(dirty_data)

    assert clean_arr[1] == 0.0, "ফেইল: রিয়েল ডেটার NaN জিরো দিয়ে রিপ্লেস হয়নি"
    assert clean_arr[2] == 0.0, "ফেইল: রিয়েল ডেটার Inf জিরো দিয়ে রিপ্লেস হয়নি"
    assert clean_arr[0] == real_close[0], "ফেইল: অরিজিনাল রিয়েল ডাটা করাপ্ট হয়ে গেছে"
