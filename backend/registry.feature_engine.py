import pandas as pd
import numpy as np
import inspect
import logging

# ইন্ডিকেটর মডিউল ইম্পোর্ট
from backend.indicators.core import volume, smart_money, moving_average, statistics, volatility, momentum, support_resistance, candle

logger = logging.getLogger(__name__)

# যে ফাংশনগুলো ইন্ডিকেটর নয়, সেগুলো বাদ দেওয়ার জন্য এক্সক্লুশন লিস্ট
EXCLUDED_FUNCS = {
    'validate_length', 'get_price_source', 'prepare_series', 'validate_ma_input', 
    'standardize_column_names', 'extract_ohlc', 'vola_std', 'vola_tr', 'vola_var',
    'cdl_tr', 'cdl_gap_up', 'cdl_gap_down', 'cdl_absorption', 'cdl_expansion', 
    'cdl_impulse', 'sr_bear_fvg', 'sr_bull_fvg', 'sr_breaker', 'sr_bsl', 'sr_dc', 
    'sr_el', 'sr_hh', 'sr_hl', 'sr_il', 'sr_lp', 'sr_lh', 'sr_ll', 'sr_mss', 
    'sr_mit_fvg', 'sr_ssl', 'sr_sh', 'sr_sl', 'smc_absorption', 'smc_bear_fvg', 
    'smc_breaker', 'smc_bull_fvg', 'smc_bsl', 'smc_expansion', 'smc_el', 'smc_hh', 
    'smc_hl', 'smc_impulse', 'smc_il', 'smc_lp', 'smc_lh', 'smc_ll', 'smc_mss', 
    'smc_mit_fvg', 'smc_ssl', 'smc_sh', 'smc_sl', 'vol_volume', 'vol_vwma', 'stat_std', 'stat_var'
}

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Empty DataFrame provided to Feature Engine")
    
    df = df.copy()
    
    # ইন্ডিকেটর মডিউলের লিস্ট
    modules = [volume, smart_money, moving_average, statistics, volatility, momentum, support_resistance, candle]

    # প্রতিটি মডিউল থেকে ফাংশন অটোমেটিক্যালি লোড করা
    for module in modules:
        for name, func in inspect.getmembers(module, inspect.isfunction):
            # স্কিপ লিস্ট চেক
            if name in EXCLUDED_FUNCS or name.startswith('_'):
                continue
            
            try:
                result = func(df)
                
                # আউটপুট হ্যান্ডলিং
                if isinstance(result, pd.DataFrame):
                    for col in result.columns:
                        if col not in df.columns:
                            df[col] = result[col]
                elif isinstance(result, pd.Series):
                    df[name] = result
                else:
                    df[name] = result
            except Exception as e:
                logger.debug(f"Indicator {name} failed: {e}")
                continue

    # ==============================================================================
    # MANDATORY COMPATIBILITY ALIAS LAYER (Fixes Naming Mismatches)
    # ==============================================================================
    # ATR Fix
    df["atr"] = df["atr_14"] if "atr_14" in df.columns else df.get("atr", np.nan)
    
    # Name Fixes (অরিজিনাল নামের সাথে সিনক্রোনাইজেশন)
    if "close_location_value" in df.columns: df["clv"] = df["close_location_value"]
    if "body_percent" in df.columns: df["body_pct"] = df["body_percent"]
    if "natr" in df.columns: df["normalized_volatility"] = df["natr"]
    
    # Logic Fixes
    if "ema_20" in df.columns and "ema_50" in df.columns:
        df["trend_direction"] = np.sign(df["ema_20"] - df["ema_50"]).fillna(0)
    
    if "gap_size" in df.columns:
        df["gap_up"] = (df["gap_size"] > 0).astype(int)
        df["gap_down"] = (df["gap_size"] < 0).astype(int)

    # Efficiency Ratio
    if "ema_20" in df.columns and "ema_50" in df.columns and "atr" in df.columns:
        df["efficiency_ratio"] = ((df["ema_20"] - df["ema_50"]).abs() / df["atr"].replace(0, np.nan)).clip(0, 1).fillna(0)

    # Sequence Generation
    is_bull = (df["close"] > df["open"]).astype(int)
    df["bull_sequence"] = is_bull.groupby((is_bull == 0).cumsum()).cumsum()
    is_bear = (df["close"] < df["open"]).astype(int)
    df["bear_sequence"] = is_bear.groupby((is_bear == 0).cumsum()).cumsum()

    # ==============================================================================
    # CLEANUP
    # ==============================================================================
    df = df.loc[:, ~df.columns.duplicated()]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.ffill().bfill().fillna(0)

    return df
