"""
feature_adapter.py
PART 1 / 3
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureAdapter:
    """
    Converts feature_history columns
    into analyzer compatible columns.
    """

    def __init__(self) -> None:

        # ------------------------------------------------------------------
        # Raw OHLCV
        # ------------------------------------------------------------------

        self.base_columns: List[str] = [
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        # ------------------------------------------------------------------
        # Trend Analyzer Mapping
        # ------------------------------------------------------------------

        self.trend_mapping: Dict[str, str] = {

            # Moving Averages
            "EMA_20": "ema_20",
            "EMA_50": "ema_50",
            "EMA_100": "ema_100",
            "EMA_200": "ema_200",

            "SMA_20": "sma_20",
            "SMA_50": "sma_50",
            "SMA_100": "sma_100",
            "SMA_200": "sma_200",

            # Trend
            "SUPERTREND": "supertrend",
            "SUPERTREND_TREND": "supertrend_trend",

            # Momentum
            "RSI_14": "rsi",

            "MACD": "macd_line",
            "MACD_SIGNAL": "macd_signal",
            "MACD_HIST": "macd_hist",

            "ADX_14": "adx",
            "PLUS_DI": "plus_di",
            "MINUS_DI": "minus_di",

            # Volatility
            "ATR_14": "atr_14",

            # Volume
            "VWAP": "vwap",

            # Regression
            "REGRESSION_SLOPE": "linreg_slope",
            "R_SQUARED": "linreg_r2",
        }

        # ------------------------------------------------------------------
        # Smart Money Mapping
        # ------------------------------------------------------------------

        self.smc_mapping: Dict[str, str] = {

            "BOS_UP": "bos_up",
            "BOS_DOWN": "bos_down",

            "CHOCH_UP": "choch_up",
            "CHOCH_DOWN": "choch_down",

            "LIQUIDITY_SWEEP": "liq_sweep",

            "FVG_ACTIVE": "fvg_active",

            "FRESH_OB": "ob_active",

            "SM_SMART_MONEY_SCORE": "smart_money_score",
        }

        # ------------------------------------------------------------------
        # Merge Mapping
        # ------------------------------------------------------------------

        self.mapping: Dict[str, str] = {}
        self.mapping.update(self.trend_mapping)
        self.mapping.update(self.smc_mapping)

    def available_columns(self, df: pd.DataFrame) -> List[str]:
        return [c for c in self.mapping if c in df.columns]

    def missing_columns(self, df: pd.DataFrame) -> List[str]:
        return [c for c in self.mapping if c not in df.columns]

    # ------------------------------------------------------------------
    # Rename Columns
    # ------------------------------------------------------------------

    def rename(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Database column
                ↓
        Analyzer column
        """

        working = df.copy()

        cols = {
            k: v
            for k, v in self.mapping.items()
            if k in working.columns
        }

        working.rename(columns=cols, inplace=True)

        return working

    # ------------------------------------------------------------------
    # Build Derived Columns
    # ------------------------------------------------------------------

    def build(self, df: pd.DataFrame) -> pd.DataFrame:

        df = self.rename(df)

        # -------------------------
        # BOS
        # -------------------------

        if "bos" not in df.columns:

            if "bos_up" in df.columns and "bos_down" in df.columns:

                df["bos"] = (
                    df["bos_up"].fillna(0)
                    - df["bos_down"].fillna(0)
                )

        # -------------------------
        # CHOCH
        # -------------------------

        if "choch" not in df.columns:

            if "choch_up" in df.columns and "choch_down" in df.columns:

                df["choch"] = (
                    df["choch_up"].fillna(0)
                    - df["choch_down"].fillna(0)
                )

        # -------------------------
        # Order Block
        # -------------------------

        if "ob_active" in df.columns:

            df["ob_active"] = (
                df["ob_active"]
                .fillna(0)
                .astype(bool)
            )

        # -------------------------
        # Fair Value Gap
        # -------------------------

        if "fvg_active" in df.columns:

            df["fvg_active"] = (
                df["fvg_active"]
                .fillna(0)
                .astype(bool)
            )

        # -------------------------
        # Liquidity Sweep
        # -------------------------

        if "liq_sweep" in df.columns:

            df["liq_sweep"] = (
                df["liq_sweep"]
                .fillna(0)
            )

        # -------------------------
        # Numeric Cleanup
        # -------------------------

        numeric = df.select_dtypes(include=np.number).columns

        df[numeric] = (
            df[numeric]
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
        )

        return df

    # ------------------------------------------------------------------
    # Validate Analyzer Input
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> None:

        required = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "ema_20",
            "ema_50",
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

        missing = [
            c for c in required
            if c not in df.columns
        ]

        if missing:
            raise ValueError(
                f"FeatureAdapter Missing Columns: {missing}"
            )

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def adapt(self, df: pd.DataFrame) -> pd.DataFrame:

        df = self.build(df)

        self.validate(df)

        return df

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, df: pd.DataFrame) -> None:

        available = self.available_columns(df)
        missing = self.missing_columns(df)

        logger.info("=" * 70)
        logger.info("FEATURE ADAPTER")
        logger.info("=" * 70)
        logger.info("Database Columns : %d", len(df.columns))
        logger.info("Mapped Columns   : %d", len(available))
        logger.info("Missing Columns  : %d", len(missing))

        if missing:
            logger.warning(
                "Missing -> %s",
                ", ".join(missing),
            )


# ----------------------------------------------------------------------
# Helper Function
# ----------------------------------------------------------------------

def adapt_features(df: pd.DataFrame) -> pd.DataFrame:

    adapter = FeatureAdapter()

    adapter.summary(df)

    return adapter.adapt(df)


__all__ = [
    "FeatureAdapter",
    "adapt_features",
]
