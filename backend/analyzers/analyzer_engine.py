import logging
import pandas as pd
from typing import Dict, Any

# =====================================================================
# Importing Repository for Database Fetching
# =====================================================================
from backend.repository.stock_repository import repository

# =====================================================================
# Importing all 12 Analyzers
# =====================================================================
from backend.analyzers.trend_analyzer import TrendAnalyzer
from backend.analyzers.fundamental_analyzer import FundamentalAnalyzer
from backend.analyzers.volume_analyzer import VolumeAnalyzer
from backend.analyzers.volatility_analyzer import VolatilityAnalyzer
from backend.analyzers.support_resistance_analyzer import SupportResistanceAnalyzer
from backend.analyzers.smart_money_analyzer import SmartMoneyAnalyzer
from backend.analyzers.pattern_analyzer import PatternAnalyzer
from backend.analyzers.momentum_analyzer import MomentumAnalyzer
from backend.analyzers.market_regime_analyzer import MarketRegimeAnalyzer
from backend.analyzers.ipo_analyzer import IPOAnalyzer
from backend.analyzers.candle_analyzer import CandleAnalyzer
from backend.analyzers.institutional_analyzer import InstitutionalAnalyzer

logger = logging.getLogger(__name__)

class AnalyzerEngine:
    """
    Master Plugin Pipeline for executing all 12 Layer-2 specific analyzers.
    Returns strictly the RAW JSON outputs from the engines.
    """
    def __init__(self):
        self.engines = {
            "trend": TrendAnalyzer(),
            "momentum": MomentumAnalyzer(),
            "volatility": VolatilityAnalyzer(),
            "volume": VolumeAnalyzer(),
            "pattern": PatternAnalyzer(),
            "support_resistance": SupportResistanceAnalyzer(),
            "smart_money": SmartMoneyAnalyzer(),
            "market_regime": MarketRegimeAnalyzer(),
            "fundamental": FundamentalAnalyzer(),
            "ipo": IPOAnalyzer(),
            "candle": CandleAnalyzer(),
            "institutional": InstitutionalAnalyzer()
        }

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Data Normalization & Anti-Crash Zero-Padding.
        """
        working_df = df.copy()

        # 1. Lowercase all columns
        working_df.columns = [str(c).lower().strip() for c in working_df.columns]
        
        # 2. Fuzzy Column Mapper for common indicators
        aliases = {
            'rsi': ['rsi_14', 'rsi_'],
            'atr_14': ['atr_14', 'atr_', 'atrp_'],
            'atr': ['atr_14', 'atr_'],
            'macd_line': ['macd_line', 'macd_'],
            'macd_signal': ['macd_signal', 'macds_'],
            'macd_histogram': ['macd_hist', 'macdh_'],
            'adx': ['adx_14', 'adx_'],
            'roc': ['roc_14', 'roc_'],
            'momentum': ['mom_14', 'mom_', 'momentum'],
            'linreg_slope': ['linreg_slope', 'slope'],
            'linreg_r2': ['linreg_r2', 'r2'],
            'vwap': ['vwap'],
            'supertrend': ['supertrend', 'st_']
        }

        for target_col, possible_names in aliases.items():
            if target_col not in working_df.columns:
                for col in working_df.columns:
                    if any(alias in col for alias in possible_names):
                        working_df[target_col] = working_df[col]
                        break

        # 3. Anti-Crash Zero-Padding
        for name, engine in self.engines.items():
            if hasattr(engine, 'EXPECTED_SCHEMA'):
                for feat in engine.EXPECTED_SCHEMA:
                    if feat not in working_df.columns:
                        working_df[feat] = 0.0

        return working_df

    def run(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes all 12 engines and returns EXACTLY their raw JSON outputs.
        Fetches external DB data ONCE and injects it into specific analyzers.
        """
        if df is None or df.empty:
            logger.warning(f"[{symbol}] Analyzer Engine received empty DataFrame. Aborting.")
            return {}

        normalized_df = self._normalize_columns(df)
        results = {}

        # Fetch external data ONCE to save database trips
        try:
            fundamental = repository.get_fundamental(symbol)
            profile = repository.get_company_profile(symbol)
            financials = repository.get_financials(symbol)
            shareholding = repository.get_shareholding(symbol)
            earnings = repository.get_earnings(symbol)
            corporate_actions = repository.get_corporate_actions(symbol)
        except Exception as db_err:
            logger.error(f"[{symbol}] Database fetch failed: {db_err}")
            fundamental = profile = financials = shareholding = earnings = corporate_actions = {}

        for name, engine in self.engines.items():
            try:
                # Conditional Data Injection
                if name == "fundamental":
                    results[name] = engine.analyze(
                        normalized_df,
                        fundamental=fundamental,
                        financials=financials,
                        profile=profile,
                        earnings=earnings
                    )
                elif name == "institutional":
                    results[name] = engine.analyze(
                        normalized_df,
                        shareholding=shareholding,
                    )
                elif name == "ipo":
                    results[name] = engine.analyze(
                        normalized_df,
                        corporate_actions=corporate_actions,
                    )
                else:
                    results[name] = engine.analyze(normalized_df)
                    
            except Exception as e:
                logger.error(f"[{symbol}] Engine '{name}' Failed: {e}")
                results[name] = {
                    "error": str(e),
                    "status": "FAILED"
                }

        return results

# Global Singleton Instance
analyzer_engine = AnalyzerEngine()
