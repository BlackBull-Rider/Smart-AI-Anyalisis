"""
test_gbr_pipeline.py

এই স্ক্রিপ্টটি তোর Layer-1 এবং Layer-2 এর মধ্যে ব্রিজ হিসেবে কাজ করবে।
"""

import yfinance as yf
import pandas as pd
import traceback

# Layer 2 Imports
from backend.analyzers.trend_analyzer import TrendAnalyzer
from backend.analyzers.momentum_analyzer import MomentumAnalyzer

# Layer 1 Imports (তোর ৭টি ইন্ডিকেটর ফাইল)
import backend.indicators.moving_average as ma
import backend.indicators.momentum as mom
import backend.indicators.volatility as vol
import backend.indicators.volume as vol_u
import backend.indicators.statistics as stat
import backend.indicators.smart_money as smc
import backend.indicators.support_resistance as sr

def fetch_real_data(symbol):
    df = yf.download(symbol, period="100d", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return df[['open', 'high', 'low', 'close', 'volume']].dropna()

def prepare_layer1_data(df):
    """তোর ৭টি ফাইল থেকে ডেটা ক্যালকুলেট করছে"""
    w_df = df.copy()
    
    # Moving Averages (moving_average.py)
    w_df['ema_20'] = ma.ema(w_df, length=20)
    w_df['ema_50'] = ma.ema(w_df, length=50)
    w_df['ema_200'] = ma.ema(w_df, length=200)
    
    # Momentum (momentum.py)
    macd = mom.macd(w_df)
    w_df['macd_line'] = macd['macd_line']
    w_df['macd_histogram'] = macd['macd_histogram']
    w_df['rsi'] = mom.rsi(w_df, length=14)
    w_df['roc'] = mom.roc(w_df, length=10)
    w_df['momentum'] = mom.absolute_momentum(w_df, length=10)
    
    # Volatility (volatility.py)
    w_df['atr_14'] = vol.atr(w_df, length=14)
    w_df['adx'] = vol.adx(w_df, length=14)['adx']
    
    # Statistics (statistics.py)
    reg = stat.linear_regression(w_df, length=14)
    w_df['linreg_slope'] = reg['slope']
    w_df['linreg_r2'] = reg['r2']
    
    # Smart Money (smart_money.py)
    w_df['bos'] = smc.bos_level(w_df)
    w_df['choch'] = smc.choch_level(w_df)
    w_df['fvg_active'] = smc.active_fvg(w_df)
    
    return w_df.dropna()

def run_tests():
    trend_engine = TrendAnalyzer()
    mom_engine = MomentumAnalyzer()
    
    symbols = ['RELIANCE.NS', 'TCS.NS']
    
    for s in symbols:
        print(f"\n--- Testing: {s} ---")
        try:
            raw_data = fetch_real_data(s)
            processed_data = prepare_layer1_data(raw_data)
            
            # Layer 2 Test
            t_res = trend_engine.analyze(processed_data)
            m_res = mom_engine.analyze(processed_data)
            
            print(f"Trend Result: {t_res['direction']['status']} (Conf: {t_res['direction']['confidence']}%)")
            print(f"Momentum Stage: {m_res['momentum_cycle']['stage']} | Readiness: {m_res['swing_readiness']['state']}")
            print("✅ Layer 1 to Layer 2 data flow successful!")
        except Exception as e:
            print(f"❌ Error at Layer 2: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    run_tests()
