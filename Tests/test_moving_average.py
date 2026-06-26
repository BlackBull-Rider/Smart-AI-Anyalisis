import pytest
import numpy as np
import pandas as pd
from Indicators.helper import fetch_real_ohlcv, get_existing_symbol
from Indicators.moving_average import (
    calculate_sma, calculate_ema, calculate_wma, calculate_vwma,
    calculate_dema, calculate_tema, calculate_hma, calculate_zlema,
    calculate_kama, calculate_alma, calculate_t3, moving_average_matrix
)

def get_real_market_data():
    symbol = get_existing_symbol()
    if not symbol:
        pytest.skip("No real data found in database to test.")
    df = fetch_real_ohlcv(symbol, limit=100)
    if df.empty or len(df) < 50:
        pytest.skip("Not enough data to run Moving Average tests.")
    return df, symbol

def test_classic_and_vwma():
    df, symbol = get_real_market_data()
    close, volume = df['close'].values, df['volume'].values
    period = 14
    
    sma = calculate_sma(close, period)
    vwma = calculate_vwma(close, volume, period)
    
    assert len(sma) == len(close), "ফেইল: SMA অ্যারে সাইজ ইনপুটের সমান নয়!"
    assert len(vwma) == len(close), "ফেইল: VWMA অ্যারে সাইজ ভুল!"

def test_zero_lag_smoothers():
    df, symbol = get_real_market_data()
    close = df['close'].values
    period = 20
    
    dema = calculate_dema(close, period)
    hma = calculate_hma(close, period)
    
    assert len(dema) == len(close), "ফেইল: DEMA সাইজ মিসম্যাচ!"
    assert hma[-1] > 0, "ফেইল: HMA-এর লেটেস্ট ভ্যালু জেনারেট হয়নি!"

def test_adaptive_institutional_ma():
    df, symbol = get_real_market_data()
    close = df['close'].values
    
    kama = calculate_kama(close, period=10)
    t3 = calculate_t3(close, period=10)
    
    assert len(kama) == len(close), "ফেইল: KAMA সাইজ মিসম্যাচ!"
    assert not np.isnan(t3).any(), "ফেইল: T3-তে NaN ভ্যালু এসেছে!"

def test_master_matrix_and_show_results():
    """টেস্ট ৪: মাস্টার ম্যাট্রিক্স চেক এবং টার্মিনালে লেটেস্ট রেজাল্ট প্রিন্ট করা"""
    df, symbol = get_real_market_data()
    close = df['close'].values
    volume = df['volume'].values
    dates = df['date'].values

    periods = [14, 50]
    matrix = moving_average_matrix(close, volume, periods)

    # রেজাল্ট প্রিন্ট করার জন্য টেবিল তৈরি
    result_df = pd.DataFrame({
        'Date': dates,
        'Close': close,
        'SMA_14': matrix['SMA_14'],
        'EMA_14': matrix['EMA_14'],
        'KAMA_10': matrix['KAMA_10'],
        'ZLEMA_20': matrix['ZLEMA_20']
    })

    # টার্মিনালে চোখের সামনে লেটেস্ট ৫ দিনের ডেটা প্রিন্ট
    print(f"\n\n[+] --- {symbol} LATEST 5 DAYS INDICATOR RESULTS ---")
    print(result_df.tail(5).to_string(index=False))
    print("-------------------------------------------------------\n")

    assert 'SMA_50' in matrix, "ফেইল: ম্যাট্রিক্সে SMA_50 নেই!"
    assert len(matrix['EMA_14']) == len(close), "ফেইল: ম্যাট্রিক্সের ভেতরের অ্যারে সাইজ ভুল!"
