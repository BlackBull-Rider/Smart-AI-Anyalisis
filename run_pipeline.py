"""
run_pipeline.py

True Architecture Test for GBR v101.
FLOW: Data Fetcher -> Layer 1 (Direct Python Module Imports) -> Layer 2 (Analyzers)
"""

import yfinance as yf
import pandas as pd
import traceback

# ==============================================================================
# IMPORT LAYER 2 ANALYZERS
# ==============================================================================
from backend.analyzers.trend_analyzer import TrendAnalyzer
from backend.analyzers.momentum_analyzer import MomentumAnalyzer

# ==============================================================================
# IMPORT LAYER 1 INDICATORS (Directly from your .py files)
# ==============================================================================
try:
    # Importing individual functions from your flat backend/indicators structure
    import backend.indicators.moving_average as ma
    import backend.indicators.volume as vol_mod
    import backend.indicators.volatility as vol_util
    import backend.indicators.momentum as mom
    import backend.indicators.statistics as stat
    import backend.indicators.smart_money as smc
    from backend.indicators.support_resistance import *
    
    LAYER1_READY = True
except ImportError as e:
    print(f"❌ Layer 1 Direct Import Error: {e}")
    LAYER1_READY = False

# ==============================================================================
# 1. DATA FETCHER (RAW DATA ONLY)
# ==============================================================================
class DataFetcher:
    @staticmethod
    def get_real_data(symbol: str, period: str = "200d", interval: str = "1d") -> pd.DataFrame:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty:
            raise ValueError(f"No data fetched for {symbol}")
        
        # YFinance MultiIndex fix
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.columns = [c.lower() for c in df.columns]
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.dropna(inplace=True)
        return df

# ==============================================================================
# 2. LAYER 1: ENGINE (Connecting your custom scripts)
# ==============================================================================
class Layer1Engine:
    @staticmethod
    def build_features(df: pd.DataFrame) -> pd.DataFrame:
        if not LAYER1_READY:
            raise RuntimeError("Layer 1 modules could not be mapped correctly.")
            
        w_df = df.copy()
        
        try:
            # --- Moving Averages ---
            w_df['ema_20'] = ma.ema(w_df, length=20) if hasattr(ma, 'ema') else w_df['close'].ewm(span=20, adjust=False).mean()
            w_df['ema_50'] = ma.ema(w_df, length=50) if hasattr(ma, 'ema') else w_df['close'].ewm(span=50, adjust=False).mean()
            w_df['ema_200'] = ma.ema(w_df, length=200) if hasattr(ma, 'ema') else w_df['close'].ewm(span=200, adjust=False).mean()
            
            # --- Volume & Volatility ---
            w_df['vwap'] = vol_mod.vwap(w_df) if hasattr(vol_mod, 'vwap') else (w_df['close'] * w_df['volume']).cumsum() / w_df['volume'].cumsum()
            w_df['supertrend'] = vol_util.supertrend(w_df, length=10, multiplier=3.0) if hasattr(vol_util, 'supertrend') else 1.0
            w_df['atr_14'] = vol_util.atr(w_df, length=14) if hasattr(vol_util, 'atr') else w_df['high'] - w_df['low']
            
            # --- Momentum & Oscillators ---
            macd_res = mom.macd(w_df)
            w_df['macd_line'] = macd_res.get('macd_line', macd_res[0]) if not isinstance(macd_res, tuple) else macd_res[0]
            w_df['macd_signal'] = macd_res.get('macd_signal', macd_res[1]) if not isinstance(macd_res, tuple) else macd_res[1]
            w_df['macd_histogram'] = macd_res.get('macd_histogram', macd_res[2]) if not isinstance(macd_res, tuple) else macd_res[2]
            
            w_df['rsi'] = mom.rsi(w_df, length=14)
            w_df['roc'] = mom.roc(w_df, length=10)
            w_df['momentum'] = mom.absolute_momentum(w_df, length=10)
            
            adx_res = mom.adx(w_df, length=14)
            w_df['adx'] = adx_res.get('adx', adx_res[0]) if not isinstance(adx_res, tuple) else adx_res[0]
            w_df['dmi_plus'] = adx_res.get('dmi_plus', adx_res[1]) if not isinstance(adx_res, tuple) else adx_res[1]
            w_df['dmi_minus'] = adx_res.get('dmi_minus', adx_res[2]) if not isinstance(adx_res, tuple) else adx_res[2]

            # --- Statistics ---
            reg_res = stat.linear_regression(w_df, length=14)
            w_df['linreg_slope'] = reg_res['slope']
            w_df['linreg_r2'] = reg_res['r2']
            
            # --- Smart Money Concepts ---
            w_df['bos'] = smc.bos_level(w_df)
            w_df['choch'] = smc.choch_level(w_df)
            w_df['liq_sweep'] = smc.liquidity_sweep(w_df)
            w_df['fvg_active'] = smc.active_fvg(w_df)
            
            # Check if order blocks structure is df or series dictionary
            ob_data = smc.order_blocks(w_df)
            w_df['ob_active'] = ob_data['active'] if isinstance(ob_data, dict) and 'active' in ob_data else False
            
            w_df.dropna(inplace=True)
            return w_df
            
        except Exception as e:
            logger.error(f"Layer 1 Feature Build Failure: {e}")
            traceback.print_exc()
            raise

# ==============================================================================
# 3. LAYER 2: ANALYZER ENGINE
# ==============================================================================
def test_institutional_pipeline():
    print("="*70)
    print("🚀 GBR v101 TRUE ARCHITECTURE PIPELINE TEST (NATIVE LAYER-1)")
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
            print(f"  ▪ Market Regime  : {trend_report['regime']['regime']} | Phase: {trend_report['regime']['phase']}")
            print(f"  ▪ Trend Quality  : {trend_report['quality']['status']} (Score: {trend_report['quality']['score']})")
            print(f"  ▪ MTF Alignment  : {trend_report['multi_timeframe']['alignment']}")
            print(f"  ▪ Momentum Shift : {mom_report['momentum_shift']['status']} | {mom_report['momentum_ignition']['stage']}")
            print(f"  ▪ Swing Ready    : {mom_report['swing_readiness']['state']} (Score: {mom_report['swing_readiness']['score']})")
            print(f"  ▪ Divergence     : {mom_report['divergence']['status']} ({mom_report['divergence'].get('divergence_type', 'None')})")
            
        except Exception as e:
            print(f"  └─ ❌ Pipeline Failed for {symbol}: {str(e)}")

if __name__ == "__main__":
    test_institutional_pipeline()
