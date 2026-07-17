"""
GREEN BULL AI
Master Feature Adapter
Dynamically maps 571 Database Columns to Layer 2 Analyzer Expected Schemas.
"""

import pandas as pd
import numpy as np
import warnings
from pandas.errors import PerformanceWarning

# Mute Pandas Warnings
warnings.simplefilter("ignore", PerformanceWarning)
pd.set_option('future.no_silent_downcasting', True)

class MasterFeatureAdapter:
    def __init__(self):
        # ==============================================================
        # EXPLICIT MAPPING DICTIONARY
        # "Analyzer_Expected_Name" : "Database_Column_Name"
        # ==============================================================
        self.mapping = {
            "macd_line": "MACD",
            "macd_signal": "MACD_SIGNAL",
            "macd_hist": "MACD_HIST",
            "macd_histogram": "MACD_HIST",
            "rsi": "RSI_14",
            "adx": "ADX_14",
            "linreg_slope": "REGRESSION_SLOPE",
            "linreg_r2": "R_SQUARED",
            "atr": "ATR_14",
            "bbw_20_2.0": "BB_WIDTH",
            "hv_21": "HV_21",
            "sqz_20": "BB_SQUEEZE",
            "ei_14": "EXPANSION_INDEX",
            "chop_14": "CHOPPINESS",
            "relative_volume": "RVOL_20",
            "volume_percentile": "VOL_PERCENTILE",
            "volume_zscore": "VOL_ZSCORE",
            "cmf": "CMF_20",
            "adl": "ADL",
            "advance_decline_line": "ADL",
            "vpt": "PVT",
            "mfi": "MFI_14",
            "money_flow": "SMART_MONEY_INDEX",
            "force_index": "FORCE_INDEX",
            "accdist": "ADOSC",
            "roc": "ROC_12",
            "roc_20": "ROC_12",
            "momentum": "MOM_10",
            "supertrend": "SUPERTREND",
            "supertrend_direction": "SUPERTREND_TREND",
            "vix_proxy": "HV_21"  # Proxy for VIX
        }

    def combine_directional_features(self, adapted_data: dict, raw_df: pd.DataFrame):
        """Combines split DB columns (UP/DOWN) into single Polarity columns (-1.0, 0.0, 1.0)"""
        
        # 1. Break of Structure (BOS)
        if "BOS_UP" in raw_df.columns and "BOS_DOWN" in raw_df.columns:
            up = raw_df["BOS_UP"].fillna(0).astype(float)
            dn = raw_df["BOS_DOWN"].fillna(0).astype(float)
            adapted_data["bos"] = np.where(up > 0, 1.0, np.where(dn > 0, -1.0, 0.0))

        # 2. Change of Character (CHOCH)
        if "CHOCH_UP" in raw_df.columns and "CHOCH_DOWN" in raw_df.columns:
            up = raw_df["CHOCH_UP"].fillna(0).astype(float)
            dn = raw_df["CHOCH_DOWN"].fillna(0).astype(float)
            adapted_data["choch"] = np.where(up > 0, 1.0, np.where(dn > 0, -1.0, 0.0))
            
        # 3. Fair Value Gap (FVG)
        if "BULLISH_FVG" in raw_df.columns and "BEARISH_FVG" in raw_df.columns:
            up = raw_df["BULLISH_FVG"].fillna(0).astype(float)
            dn = raw_df["BEARISH_FVG"].fillna(0).astype(float)
            adapted_data["fair_value_gap"] = np.where(up > 0, 1.0, np.where(dn > 0, -1.0, 0.0))
            adapted_data["fvg_active"] = adapted_data["fair_value_gap"]

        # 4. Liquidity Sweep (SMC)
        if "SELL_SIDE_LIQUIDITY" in raw_df.columns and "BUY_SIDE_LIQUIDITY" in raw_df.columns:
            # Sweeping Sell-Side is Bullish (1.0), Sweeping Buy-Side is Bearish (-1.0)
            sell_side = raw_df["SELL_SIDE_LIQUIDITY"].fillna(0).astype(float)
            buy_side = raw_df["BUY_SIDE_LIQUIDITY"].fillna(0).astype(float)
            adapted_data["liquidity_sweep"] = np.where(sell_side > 0, 1.0, np.where(buy_side > 0, -1.0, 0.0))
            
        # 5. Order Blocks (Proxy)
        if "FRESH_OB" in raw_df.columns:
            adapted_data["ob_active"] = raw_df["FRESH_OB"].fillna(0).astype(float)
            adapted_data["order_block"] = raw_df["FRESH_OB"].fillna(0).astype(float)

    def adapt(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        adapted_data = {}

        # 1. Base auto-mapping (Lowercase all DB columns)
        for col in df.columns:
            adapted_data[col.lower()] = df[col].values

        # 2. Apply Explicit Dictionary Mapping
        for expected_name, db_name in self.mapping.items():
            if db_name in df.columns:
                adapted_data[expected_name] = df[db_name].values

        # 3. Apply Polarity Merging (BOS, CHOCH, FVG)
        self.combine_directional_features(adapted_data, df)

        # 4. Fast Dictionary to DataFrame Conversion (Zero Fragmentation)
        return pd.DataFrame(adapted_data, index=df.index)

