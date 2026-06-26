import pytest
import numpy as np
import pandas as pd
from Indicators.helper import fetch_real_ohlcv, get_existing_symbol
from Indicators.volatility import calculate_true_range, calculate_atr, calculate_bollinger_bands

def get_real_market_data():
    symbol = get_existing_symbol()
    if not symbol:
        pytest.skip("No real data found in database to test.")
    df = fetch_real_ohlcv(symbol, limit=100)
    if df.empty or len(df) < 50:
        pytest.skip("Not enough data to run Volatility tests.")
    return df, symbol

def test_true_range_and_atr():
    """টেস্ট ১: True Range এবং ATR ক্যালকুলেশন চেক"""
    df, symbol = get_real_market_data()
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    tr = calculate_true_range(high, low, close)
    atr = calculate_atr(high, low, close, period=14)
    
    assert len(tr) == len(close), "ফেইল: TR অ্যারে সাইজ ইনপুটের সমান নয়!"
    assert len(atr) == len(close), "ফেইল: ATR অ্যারে সাইজ ভুল!"
    assert atr[-1] >= 0, "ফেইল: লেটেস্ট ATR ভ্যালু নেগেটিভ হতে পারে না!"

def test_bollinger_bands_and_show_results():
    """টেস্ট ২: Bollinger Bands ভ্যালিডেশন এবং টার্মিনালে লেটেস্ট রেজাল্ট প্রিন্ট"""
    df, symbol = get_real_market_data()
    close = df['close'].values
    dates = df['date'].values
    
    middle, upper, lower = calculate_bollinger_bands(close, period=20, num_std=2.0)
    
    assert len(middle) == len(close), "ফেইল: মিডল ব্যান্ডের সাইজ মিসম্যাচ!"
    assert len(upper) == len(close), "ফেইল: আপার ব্যান্ডের সাইজ মিসম্যাচ!"
    assert len(lower) == len(close), "ফেইল: লোয়ার ব্যান্ডের সাইজ মিসম্যাচ!"
    assert (upper >= lower).all(), "ফেইল: আপার ব্যান্ড লোয়ার ব্যান্ডের নিচে নেমে গেছে!"

    # রেজাল্ট প্রিন্ট করার জন্য টেবিল তৈরি
    result_df = pd.DataFrame({
        'Date': dates,
        'Close': close,
        'Lower_BB': lower,
        'Middle_BB': middle,
        'Upper_BB': upper
    })

    # টার্মিনালে লেটেস্ট ৫ দিনের ডেটা প্রিন্ট
    print(f"\n\n[+] --- {symbol} LATEST 5 DAYS VOLATILITY RESULTS ---")
    print(result_df.tail(5).to_string(index=False))
    print("-------------------------------------------------------\n")
