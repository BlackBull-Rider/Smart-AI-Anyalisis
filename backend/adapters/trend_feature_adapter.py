"""
backend/adapters/trend_feature_adapter.py

Trend Analyzer Feature Adapter

Converts Green Bull Data Engine feature_history +
financial_data into TrendAnalyzer compatible dataframe.
"""

from __future__ import annotations
import warnings
from pandas.errors import PerformanceWarning
warnings.simplefilter("ignore", PerformanceWarning)

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class TrendFeatureAdapter:
    """
    Adapter for TrendAnalyzer.

    Input
    -----
    feature_df : feature_history dataframe
    financial  : optional financial_data row

    Output
    ------
    DataFrame compatible with TrendAnalyzer
    """

    COLUMN_MAP = {

        # -----------------------
        # OHLCV
        # -----------------------

        "OPEN": "open",
        "HIGH": "high",
        "LOW": "low",
        "CLOSE": "close",
        "VOLUME": "volume",

        # -----------------------
        # EMA
        # -----------------------

        "EMA_20": "ema_20",
        "EMA_50": "ema_50",
        "EMA_200": "ema_200",

        # -----------------------
        # VWAP
        # -----------------------

        "VWAP": "vwap",

        # -----------------------
        # Trend
        # -----------------------

        "SUPERTREND": "supertrend",

        # -----------------------
        # Momentum
        # -----------------------

        "ADX_14": "adx",

        "MACD": "macd_line",
        "MACD_SIGNAL": "macd_signal",

        "RSI_14": "rsi",

        "ATR_14": "atr_14",

        # -----------------------
        # Regression
        # -----------------------

        "REGRESSION_SLOPE": "linreg_slope",
        "R_SQUARED": "linreg_r2",

        # -----------------------
        # Smart Money
        # -----------------------

        "SM_BOS": "bos",

        "SM_CHOCH": "choch",

        "LIQUIDITY_SWEEP": "liq_sweep",

        "FVG_ACTIVE": "fvg_active",

        "FRESH_OB": "ob_active",
    }

    BOOL_COLUMNS = [

        "bos",
        "choch",
        "liq_sweep",

        "fvg_active",
        "ob_active",

    ]

    REQUIRED = [

        "open",
        "high",
        "low",
        "close",
        "volume",

        "ema_20",
        "ema_50",
        "ema_200",

        "vwap",

        "supertrend",

        "adx",

        "macd_line",
        "macd_signal",

        "rsi",

        "atr_14",

        "linreg_slope",
        "linreg_r2",

    ]

    def adapt(
        self,
        feature_df: pd.DataFrame,
        financial: pd.Series | None = None,
    ) -> pd.DataFrame:

        df = feature_df.copy()

        # ---------------------------------------
        # Rename Columns
        # ---------------------------------------

        rename_map = {}

        for src, dst in self.COLUMN_MAP.items():

            if src in df.columns:

                rename_map[src] = dst

        df = df.rename(columns=rename_map)

        # ---------------------------------------
        # Boolean Conversion
        # ---------------------------------------

        for col in self.BOOL_COLUMNS:

            if col in df.columns:

                df[col] = (
                    df[col]
                    .fillna(0)
                    .astype(int)
                )

        # ---------------------------------------
        # Numeric Conversion
        # ---------------------------------------

        for col in df.columns:

            if col == "symbol":

                continue

            if col == "date":

                continue

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

        # ---------------------------------------
        # Missing Required
        # ---------------------------------------

        for col in self.REQUIRED:

            if col not in df.columns:

                logger.warning(
                    "Missing feature : %s",
                    col,
                )

                df[col] = 0.0

        # ---------------------------------------
        # Financial Merge
        # ---------------------------------------

        if financial is not None:

            for k, v in financial.items():

                if k not in df.columns:

                    df[k.lower()] = v

        # ---------------------------------------
        # Cleanup
        # ---------------------------------------

        df = df.sort_values("date")

        df = df.ffill()

        df = df.bfill()

        df = df.reset_index(drop=True)

        return df


__all__ = [
    "TrendFeatureAdapter",
]
