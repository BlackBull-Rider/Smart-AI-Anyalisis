"""
gbr_pipeline_test.py

True Architecture Test for GBR v101.
FLOW: Data Fetcher (Raw) -> Layer 1 (Indicators) -> Layer 2 (Analyzers)
"""

import yfinance as yf
import pandas as pd
import json

# ==============================================================================
# IMPORT LAYER 2 ANALYZERS
# ==============================================================================
from backend.analyzers.trend_analyzer import TrendAnalyzer
from backend.analyzers.momentum_analyzer import MomentumAnalyzer

# ==============================================================================
# IMPORT LAYER 1 INDICATORS (Your custom GBR indicators)
# ==============================================================================
# NOTE: Assuming you have these Layer 1 modules built in backend/indicators/
try:
    from backend.indicators.trend import ema, supertrend, adx
    from backend.indicators.momentum import macd, rsi, roc, absolute_momentum
    from backend.indicators.volatility import atr
    from backend.indicators.volume import vwap
    from backend.indicators.statistics import linear_regression
    from backend.indicators.support_resistance import market_structure_shift, bos_level, choch_level, liquidity_sweep, active_fvg, order_blocks
except ImportError as e:
    print(f"⚠️ Warning: Layer 1 Import Error. Ensure all Layer 1 files exist. Details: {e}")

# ==============================================================================
# 1. DATA FETCHER (RAW DATA ONLY)
# ==============================================================================
class DataFetcher:
    @staticmethod
    def get_real_data(symbol: str, period: str = "200d", interval: str = "1d") -> pd.DataFrame:
        """Fetches purely raw OHLCV data."""
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty:
            raise ValueError(f"No data fetched for {symbol}")
        
        # Ensure lowercase standard
        df.columns = [c.lower() for c in df.columns]
        return df

# ==============================================================================
# 2. LAYER 1: INDICATOR ENGINE (Calculates features via GBR indicator files)
# ==============================================================================
class Layer1Engine:
    @staticmethod
    def build_features(df: pd.DataFrame) -> pd.DataFrame:
        """Passes raw data through Layer 1 Indicator files to build the feature set."""
        working_df = df.copy()
        
        # --- Standard Technicals ---
        working_df['ema_20'] = ema(working_df, length=20)
        working_df['ema_50'] = ema(working_df, length=50)
        working_df['ema_200'] = ema(working_df, length=200)
        working_df['vwap'] = vwap(working_df)
        working_df['supertrend'] = supertrend(working_df, length=10, multiplier=3.0)
        
        # --- Momentum & Volatility ---
        macd_res = macd(working_df)
        working_df['macd_line'] = macd_res['macd_line']
        working_df['macd_signal'] = macd_res['macd_signal']
        working_df['macd_histogram'] = macd_res['macd_histogram']
        
        working_df['rsi'] = rsi(working_df, length=14)
        working_df['roc'] = roc(working_df, length=10)
        working_df['momentum'] = absolute_momentum(working_df, length=10)
        
        adx_res = adx(working_df, length=14)
        working_df['adx'] = adx_res['adx']
        working_df['dmi_plus'] = adx_res['dmi_plus']
        working_df['dmi_minus'] = adx_res['dmi_minus']
        
        working_df['atr_14'] = atr(working_df, length=14)
        
        # --- Statistics ---
        reg_res = linear_regression(working_df, length=14)
        working_df['linreg_slope'] = reg_res['slope']
        working_df['linreg_r2'] = reg_res['r2']
        
        # --- Smart Money Concepts (From your support_resistance.py) ---
        working_df['bos'] = bos_level(working_df)
        working_df['choch'] = choch_level(working_df)
        working_df['liq_sweep'] = liquidity_sweep(working_df)
        working_df['fvg_active'] = active_fvg(working_df)
        working_df['ob_active'] = order_blocks(working_df)['active']

        # Add MTF mock routing if Layer 1 MTF engine is separate
        # (Assuming Layer 1 handles _W, _M suffixes, otherwise generate them here via resample)

        working_df.dropna(inplace=True)
        return working_df

# ==============================================================================
# 3. LAYER 2: ANALYZER ENGINE
# ==============================================================================
def test_institutional_pipeline():
    print("="*60)
    print("GBR v101 TRUE ARCHITECTURE PIPELINE TEST")
    print("="*60)
    
    stocks = ['RELIANCE.NS', 'HDFCBANK.NS', 'TCS.NS']
    
    trend_analyzer = TrendAnalyzer()
    momentum_analyzer = MomentumAnalyzer()
    
    for symbol in stocks:
        print(f"\n[+] Processing: {symbol}")
        try:
            # 1. Fetch
            raw_df = DataFetcher.get_real_data(symbol)
            
            # 2. Layer 1 Process
            l1_df = Layer1Engine.build_features(raw_df)
            
            # 3. Layer 2 Analysis
            trend_report = trend_analyzer.analyze(l1_df)
            mom_report = momentum_analyzer.analyze(l1_df)
            
            print(f"  └─ Trend State   : {trend_report['direction']['status']} (Conf: {trend_report['direction']['confidence']}%)")
            print(f"  └─ Market Regime : {trend_report['regime']['regime']} | {trend_report['regime']['phase']}")
            print(f"  └─ Swing Ready   : {mom_report['swing_readiness']['state']} (Score: {mom_report['swing_readiness']['score']})")
            print(f"  └─ Shift/Ignition: {mom_report['momentum_shift']['status']} | {mom_report['momentum_ignition']['stage']}")
            
        except Exception as e:
            print(f"  └─ ❌ Pipeline Failed: {str(e)}")

if __name__ == "__main__":
    test_institutional_pipeline()
