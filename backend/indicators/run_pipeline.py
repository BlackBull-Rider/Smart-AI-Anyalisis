"""
run_pipeline.py

True Architecture Test for GBR v101.
FLOW: Data Fetcher -> Layer 1 (Your exact .py files) -> Layer 2 (Analyzers)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import traceback
from scipy.stats import linregress

# ==============================================================================
# IMPORT LAYER 2 ANALYZERS
# ==============================================================================
from backend.analyzers.trend_analyzer import TrendAnalyzer
from backend.analyzers.momentum_analyzer import MomentumAnalyzer

# ==============================================================================
# IMPORT LAYER 1 INDICATORS (Mapped to your exact 'ls' output)
# ==============================================================================
try:
    import backend.indicators.moving_average as ma
    import backend.indicators.momentum as mom
    import backend.indicators.volatility as vol
    import backend.indicators.volume as v_vol
    import backend.indicators.statistics as stat
    import backend.indicators.smart_money as smc
    LAYER1_MAPPED = True
except ImportError as e:
    print(f"⚠️ Layer 1 Mapping Warning: {e}")
    LAYER1_MAPPED = False

# ==============================================================================
# 1. DATA FETCHER
# ==============================================================================
class DataFetcher:
    @staticmethod
    def get_real_data(symbol: str, period: str = "200d", interval: str = "1d") -> pd.DataFrame:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty: raise ValueError(f"No data fetched for {symbol}")
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.columns = [c.lower() for c in df.columns]
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.dropna(inplace=True)
        return df

# ==============================================================================
# 2. LAYER 1: BULLETPROOF ENGINE
# ==============================================================================
class Layer1Engine:
    @staticmethod
    def build_features(df: pd.DataFrame) -> pd.DataFrame:
        w_df = df.copy()
        
        # --- 1. Moving Averages ---
        try:
            w_df['ema_20'] = ma.ema(w_df, length=20)
            w_df['ema_50'] = ma.ema(w_df, length=50)
            w_df['ema_200'] = ma.ema(w_df, length=200)
        except:
            w_df['ema_20'] = ta.ema(w_df['close'], length=20)
            w_df['ema_50'] = ta.ema(w_df['close'], length=50)
            w_df['ema_200'] = ta.ema(w_df['close'], length=200)

        # --- 2. Volume (VWAP) ---
        try:
            w_df['vwap'] = v_vol.vwap(w_df)
        except:
            w_df['vwap'] = ta.vwap(w_df['high'], w_df['low'], w_df['close'], w_df['volume'])

        # --- 3. Volatility (ATR & Supertrend) ---
        try:
            w_df['atr_14'] = vol.atr(w_df, length=14)
        except:
            w_df['atr_14'] = ta.atr(w_df['high'], w_df['low'], w_df['close'], length=14)
            
        try:
            w_df['supertrend'] = vol.supertrend(w_df, length=10, multiplier=3.0)
        except:
            st = ta.supertrend(w_df['high'], w_df['low'], w_df['close'], length=10, multiplier=3.0)
            w_df['supertrend'] = st['SUPERTd_10_3.0'] if st is not None else 1

        # --- 4. Momentum (MACD, RSI, ROC, ADX) ---
        try:
            macd_res = mom.macd(w_df)
            w_df['macd_line'] = macd_res['macd_line'] if isinstance(macd_res, pd.DataFrame) else macd_res[0]
            w_df['macd_signal'] = macd_res['macd_signal'] if isinstance(macd_res, pd.DataFrame) else macd_res[1]
            w_df['macd_histogram'] = macd_res['macd_histogram'] if isinstance(macd_res, pd.DataFrame) else macd_res[2]
        except:
            macd_df = ta.macd(w_df['close'])
            w_df['macd_line'] = macd_df['MACD_12_26_9']
            w_df['macd_signal'] = macd_df['MACDs_12_26_9']
            w_df['macd_histogram'] = macd_df['MACDh_12_26_9']

        try:
            w_df['rsi'] = mom.rsi(w_df, length=14)
            w_df['roc'] = mom.roc(w_df, length=10)
            w_df['momentum'] = mom.absolute_momentum(w_df, length=10)
        except:
            w_df['rsi'] = ta.rsi(w_df['close'], length=14)
            w_df['roc'] = ta.roc(w_df['close'], length=10)
            w_df['momentum'] = w_df['close'].diff(10)

        try:
            adx_res = mom.adx(w_df, length=14) # Or wherever your ADX is
            w_df['adx'] = adx_res['adx']
            w_df['dmi_plus'] = adx_res['dmi_plus']
            w_df['dmi_minus'] = adx_res['dmi_minus']
        except:
            adx_df = ta.adx(w_df['high'], w_df['low'], w_df['close'], length=14)
            w_df['adx'] = adx_df['ADX_14']
            w_df['dmi_plus'] = adx_df['DMP_14']
            w_df['dmi_minus'] = adx_df['DMN_14']

        # --- 5. Statistics (LinReg) ---
        try:
            reg_res = stat.linear_regression(w_df, length=14)
            w_df['linreg_slope'] = reg_res['slope']
            w_df['linreg_r2'] = reg_res['r2']
        except:
            slopes, r2s = np.full(len(w_df), np.nan), np.full(len(w_df), np.nan)
            close_vals = w_df['close'].values
            for i in range(14, len(w_df)):
                y, x = close_vals[i-14:i], np.arange(14)
                slope, _, r_val, _, _ = linregress(x, y)
                slopes[i], r2s[i] = slope, r_val**2
            w_df['linreg_slope'] = slopes
            w_df['linreg_r2'] = r2s

        # --- 6. Smart Money Concepts ---
        try:
            w_df['bos'] = smc.bos(w_df)
            w_df['choch'] = smc.choch(w_df)
            w_df['liq_sweep'] = smc.liquidity_sweep(w_df)
            w_df['fvg_active'] = smc.active_fvg(w_df)
            w_df['ob_active'] = smc.order_block_score(w_df) > 0
        except:
            w_df['bos'], w_df['choch'], w_df['liq_sweep'] = 0, 0, 0
            w_df['fvg_active'], w_df['ob_active'] = False, False

        w_df.dropna(inplace=True)
        return w_df

# ==============================================================================
# 3. LAYER 2: ANALYZER ENGINE
# ==============================================================================
def test_institutional_pipeline():
    print("="*70)
    print("🚀 GBR v101 TRUE ARCHITECTURE TEST (Mapped to 'ls' output)")
    print("="*70)
    
    stocks = ['RELIANCE.NS', 'HDFCBANK.NS', 'TCS.NS']
    
    trend_analyzer = TrendAnalyzer()
    momentum_analyzer = MomentumAnalyzer()
    
    for symbol in stocks:
        print(f"\n[+] Processing: {symbol}")
        try:
            raw_df = DataFetcher.get_real_data(symbol)
            l1_df = Layer1Engine.build_features(raw_df)
            
            trend_report = trend_analyzer.analyze(l1_df)
            mom_report = momentum_analyzer.analyze(l1_df)
            
            print("\n  [ GREEN BULL RIDER - INTELLIGENCE REPORT ]")
            print(f"  ▪ Trend State    : {trend_report['direction']['status']} (Conf: {trend_report['direction']['confidence']}%)")
            print(f"  ▪ Market Phase   : {trend_report['regime']['phase']}")
            print(f"  ▪ Momentum Shift : {mom_report['momentum_shift']['status']} | {mom_report['momentum_ignition']['stage']}")
            print(f"  ▪ Swing Ready    : {mom_report['swing_readiness']['state']} (Score: {mom_report['swing_readiness']['score']})")
            
        except Exception as e:
            print(f"  └─ ❌ Pipeline Failed for {symbol}: {str(e)}")

if __name__ == "__main__":
    test_institutional_pipeline()
