import logging
import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

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

        self.ALIAS_MAP = {
            'close': ['close', 'ltp', 'c', 'regular_market_price'],
            'open': ['open', 'o', 'regular_market_open'],
            'high': ['high', 'h', 'day_high', 'regular_market_day_high'],
            'low': ['low', 'l', 'day_low', 'regular_market_day_low'],
            'volume': ['volume', 'vol', 'regular_market_volume'],

            'ema_20': ['ema_20'],
            'ema_50': ['ema_50'],
            'ema_200': ['ema_200'],
            'sma_200': ['sma_200'],
            'vwap': ['vwap', 'rolling_vwap'],
            'supertrend': ['supertrend'],
            'supertrend_direction': ['supertrend_trend'],
            'bollinger_upper': ['bb_upper'],
            'bollinger_lower': ['bb_lower'],
            'donchian_upper': ['donchian_upper'],
            'donchian_lower': ['donchian_lower'],

            'macd': ['macd'],
            'macd_line': ['macd'],
            'macd_signal': ['macd_signal'],
            'macd_hist': ['macd_hist', 'macd_histogram'],
            'macd_histogram': ['macd_hist'],
            'rsi': ['rsi_14', 'rsi'],
            'adx': ['adx_14', 'adx'],
            'roc': ['roc_12', 'roc_20', 'roc'],
            'roc_20': ['roc_20', 'roc'],
            'momentum': ['mom_10', 'momentum'],
            'linreg_r2': ['regression_r2', 'linear_regression_r2', 'r_squared'],
            'linreg_slope': ['regression_slope', 'linear_regression_slope', 'slope'],
            
            'atr': ['atr_14', 'atr'],
            'atr_14': ['atr_14', 'atr'],
            'hv_21': ['hv_21'],
            'bbw_20_2.0': ['bb_width'],
            'ei_14': ['expansion_index'],
            'sqz_20': ['bb_squeeze'],
            'chop_14': ['choppiness'],
            'normalized_volatility': ['stat_normalized', 'normalized_volatility'],

            'obv': ['obv'],
            'cmf': ['cmf_20', 'cmf'],
            'adl': ['adl', 'advance_decline_line'],
            'advance_decline_line': ['adl'],
            'vpt': ['pvt', 'volume_trend'],
            'mfi': ['mfi_14'],
            'money_flow': ['cmf_20'],
            'force_index': ['efi_13', 'force_index'],
            'accdist': ['adl'],
            'nvi': ['nvi'],
            'pvi': ['pvi'],
            'relative_volume': ['rvol_20', 'volume_ratio'],
            'volume_ratio': ['rvol_20'],
            'volume_zscore': ['vol_zscore'],
            'volume_percentile': ['vol_percentile'],
            'volume_ma_20': ['avg_volume_20', 'vol_ema_20'],
            'delivery_percent': ['delivery_percent'],
            'delivery_quantity': ['delivery_quantity'],
            'delivery_trend': ['delivery_trend'],

            'bos': ['bos', 'sm_bos'],
            'major_bos': ['major_bos'],
            'minor_bos': ['minor_bos'],
            'internal_bos': ['internal_bos'],
            'external_bos': ['external_bos'],
            'choch': ['choch', 'sm_choch'],
            'internal_choch': ['internal_choch'],
            'external_choch': ['external_choch'],
            'liquidity_sweep': ['liquidity_sweep', 'turtle_soup'],
            'swing_failure': ['sm_swing_high', 'turtle_soup'],
            'protected_high': ['confirmed_swing_high'],
            'protected_low': ['confirmed_swing_low'],
            'violation_level': ['level_strength'],
            'structure_score': ['sm_institutional_score'],
            'bullish_ob_detected': ['sm_fresh_ob'],
            'bearish_ob_detected': ['sm_fresh_ob'],
            'active_order_block': ['active_order_block', 'sm_fresh_ob'],
            'order_block': ['sm_fresh_ob'],
            'ob_active': ['sm_fresh_ob'],
            'mitigation_block': ['sm_mitigated_ob'],
            'mitigation': ['sm_mitigated_ob'],
            'breaker_block': ['breaker_block'],
            'breaker': ['breaker_block'],
            'ob_age': ['ob_age'],
            'ob_efficiency': ['ob_efficiency'],
            'reaction_count': ['bounce_count', 'reaction_count'],
            'fair_value_gap': ['fvg', 'fvg_active'],
            'fvg': ['fvg', 'fvg_active'],
            'fvg_active': ['fvg_active'],
            'inverse_fvg': ['inverse_fvg'],
            'nested_fvg': ['nested_fvg'],
            'fvg_stack': ['fvg_stack'],
            'fvg_width': ['fvg_width'],
            'fvg_fill_ratio': ['fvg_fill_ratio'],
            'fvg_age': ['fvg_age'],
            'imbalance_strength': ['imbalance_strength'],
            'htf_alignment': ['htf_alignment'],
            'equal_highs': ['equal_highs'],
            'equal_lows': ['equal_lows'],
            'equal_highs_strength': ['equal_highs_strength'],
            'equal_lows_strength': ['equal_lows_strength'],
            'liquidity_pool_strength': ['liquidity_pool'],
            'sweep_strength': ['sweep_strength'],
            'liquidity_age': ['liquidity_age'],
            'internal_liquidity': ['internal_liquidity'],
            'external_liquidity': ['external_liquidity'],
            'resting_liquidity': ['resting_liquidity'],
            'liquidity_void': ['liquidity_void'],
            'liquidity_exhaustion': ['liquidity_exhaustion'],
            'liquidity_consumption': ['liquidity_consumption'],
            'inducement': ['inducement'],
            'smart_money_flow': ['smart_money_flow'],
            'institutional_volume_score': ['institutional_volume_score', 'institutional_score'],

            'clv': ['clv'],
            'efficiency_ratio': ['efficiency_ratio'],
            'body_pct': ['body_percent', 'body_ratio'],
            'bull_sequence': ['bull_sequence'],
            'bear_sequence': ['bear_sequence'],
            'gap_up': ['gap_up'],
            'gap_down': ['gap_down'],
            'upper_wick': ['upper_wick'],
            'lower_wick': ['lower_wick'],
            'trend_direction': ['trend_direction', 'smc_trend'],
            'trend_strength': ['trend_strength', 'linear_trend_strength'],

            'breakout_strength': ['breakout_strength', 'break_strength'],
            'volume_confirmation': ['volume_confirmation', 'effort_result'],
            'support_strength': ['support_strength'],
            'resistance_strength': ['resistance_strength'],
            'pattern_confidence': ['pattern_confidence'],
            'swing_high': ['swing_high', 'sm_swing_high'],
            'swing_low': ['swing_low', 'sm_swing_low'],
            'neckline': ['neckline'],
            'triangle_upper': ['triangle_upper'],
            'triangle_lower': ['triangle_lower'],
            'channel_upper': ['channel_upper'],
            'channel_lower': ['channel_lower'],
            'rectangle_upper': ['rectangle_upper'],
            'rectangle_lower': ['rectangle_lower'],
            'triangle_detected': ['triangle_detected'],
            'triangle_type': ['triangle_type'],
            'apex_distance': ['apex_distance'],
            'breakout_pressure': ['breakout_pressure'],
            'channel_detected': ['channel_detected'],
            'channel_type': ['channel_type'],
            'channel_width': ['channel_width'],
            'rectangle_detected': ['rectangle_detected'],
            'rectangle_width': ['rectangle_width'],
            'flag_detected': ['flag_detected'],
            'pennant_detected': ['pennant_detected'],
            'flag_quality': ['flag_quality'],
            'pole_length': ['pole_length'],
            'retracement_depth': ['retracement_depth'],
            'volume_decay': ['volume_decay'],
            'wedge_detected': ['wedge_detected'],
            'wedge_type': ['wedge_type'],
            'cup_detected': ['cup_detected'],
            'handle_detected': ['handle_detected'],
            'rounding_top_detected': ['rounding_top_detected'],
            'rounding_bottom_detected': ['rounding_bottom_detected'],
            'hs_detected': ['hs_detected'],
            'ihs_detected': ['ihs_detected'],
            'hs_neckline_slope': ['hs_neckline_slope'],
            'hs_breakout_confirmed': ['hs_breakout_confirmed'],
            'double_top_detected': ['double_top_detected'],
            'double_bottom_detected': ['double_bottom_detected'],
            'triple_top_detected': ['triple_top_detected'],
            'triple_bottom_detected': ['triple_bottom_detected'],
            'compression_pct': ['compression_pct'],
            'market_phase': ['market_phase'],
            'market_regime': ['market_regime'],
            'pattern_family': ['pattern_family'],

            'sales': ['total_revenue', 'sales'],
            'net_income': ['net_income', 'net_income_to_common'],
            'total_assets': ['total_assets'],
            'total_equity': ['total_equity', 'stockholders_equity', 'common_stock_equity'],
            'ebitda': ['ebitda'],
            'current_liabilities': ['current_liabilities'],
            'free_cash_flow': ['free_cash_flow', 'free_cashflow'],
            'roic': ['roic'],
            'roe': ['roe', 'return_on_equity'],
            'roce': ['roce'],
            'roa': ['roa', 'return_on_assets'],
            'gross_margin': ['gross_margin', 'gross_margins'],
            'operating_margin': ['operating_margin', 'operating_margins'],
            'net_margin': ['net_margin', 'profit_margins'],
            'operating_cash_flow': ['operating_cash_flow', 'operating_cashflow'],
            'working_capital': ['working_capital'],
            'retained_earnings': ['retained_earnings'],
            'ebit': ['ebit'],
            'market_cap': ['market_cap', 'non_diluted_market_cap'],
            'total_liabilities': ['total_liabilities'],
            'debt_to_equity': ['debt_to_equity'],
            'current_ratio': ['current_ratio'],
            'shares_outstanding': ['shares_outstanding', 'implied_shares_outstanding'],
            'receivables': ['accounts_receivable', 'net_receivables'],
            'current_assets': ['current_assets'],
            'depreciation': ['depreciation', 'depreciation_and_amortization'],
            'sga_expense': ['sga_expense', 'selling_general_and_administration'],
            'long_term_debt': ['long_term_debt'],
            'interest_expense': ['interest_expense'],
            'total_debt': ['total_debt'],
            'quick_ratio': ['quick_ratio'],
            'interest_coverage': ['interest_coverage'],
            'revenue_growth_yoy': ['revenue_growth_yoy'],
            'profit_growth_yoy': ['profit_growth_yoy'],
            'eps_growth_yoy': ['eps_growth_yoy'],
            'fcf_growth_yoy': ['fcf_growth_yoy'],
            'public_holding': ['public_holding'],
            'promoter_pledge': ['promoter_pledge'],
            'dividend_yield': ['dividend_yield', 'trailing_annual_dividend_yield'],
            'eps': ['eps', 'trailing_eps'],
            'book_value_per_share': ['book_value_per_share', 'book_value'],
            'capex': ['capital_expenditure'],
            'beta': ['beta'],
            'goodwill': ['goodwill'],
            'sector_avg_pe': ['sector_avg_pe'],
            'sector_avg_pb': ['sector_avg_pb'],
            'sector_avg_ev_ebitda': ['sector_avg_ev_ebitda'],
            'days_sales_outstanding': ['days_sales_outstanding'],
            'days_inventory_outstanding': ['days_inventory_outstanding'],
            'days_payable_outstanding': ['days_payable_outstanding'],
            'pe_ratio': ['pe_ratio', 'trailing_pe', 'pe'],
            'pb_ratio': ['pb_ratio', 'pb'],
            'peg_ratio': ['peg_ratio', 'trailing_peg_ratio'],
            'ev_ebitda': ['ev_ebitda', 'enterprise_to_ebitda'],
            'price_to_sales': ['price_to_sales', 'price_to_sales_trailing12_months'],

            'fii_change': ['fii_change'],
            'fii_holding': ['fii_holding'],
            'dii_change': ['dii_change'],
            'dii_holding': ['dii_holding'],
            'promoter_holding': ['promoter_holding'],
            'promoter_change': ['promoter_change'],
            'block_deal': ['block_deal'],
            'block_deal_value': ['block_deal_value'],
            'block_deal_buy': ['block_deal_buy'],
            'block_deal_sell': ['block_deal_sell'],
            'bulk_deal': ['bulk_deal'],
            'bulk_deal_buy': ['bulk_deal_buy'],
            'bulk_deal_sell': ['bulk_deal_sell'],
            'institutional_holding': ['institutional_holding'],
            'ownership_concentration': ['ownership_concentration'],
            'free_float': ['free_float'],
            'issue_price': ['issue_price'],
            'ipo_size': ['ipo_size', 'issue_size'],
            'gmp': ['gmp'],
            'promoter_holding_pre': ['promoter_holding_pre'],
            'promoter_holding_post': ['promoter_holding_post'],
            'listing_price': ['listing_price'],
            'intraday_high': ['intraday_high', 'day_high'],
            'opening_auction_volume': ['opening_auction_volume'],
            'listing_volume': ['listing_volume'],
            'float_shares': ['float_shares'],
            'cash_equivalents': ['cash_and_cash_equivalents', 'cash', 'total_cash'],
            'sector_pe': ['sector_pe', 'sector_avg_pe'],
            'sector_ev_ebitda': ['sector_ev_ebitda', 'sector_avg_ev_ebitda'],
            'sector_ev_sales': ['sector_ev_sales'],
            'profit_growth': ['profit_growth'],
            'sector_risk_score': ['sector_risk_score'],
            'market_sentiment_score': ['market_sentiment_score'],
            'macro_liquidity_index': ['macro_liquidity_index'],
            'subscription_qib': ['subscription_qib'],
            'subscription_hni': ['subscription_hni'],
            'subscription_retail': ['subscription_retail'],
            'subscription_velocity': ['subscription_velocity'],
            'last_day_spike': ['last_day_spike'],
            'category_concentration': ['category_concentration'],
            'anchor_allocation': ['anchor_allocation'],
            'top_anchor_concentration': ['top_anchor_concentration'],
            'mf_anchor_pct': ['mf_anchor_pct'],
            'sovereign_anchor_pct': ['sovereign_anchor_pct'],
            'domestic_vs_foreign_ratio': ['domestic_vs_foreign_ratio'],
            'lockin_days': ['lockin_days'],

            'vix_proxy': ['vix_proxy'],
            'credit_spread_proxy': ['credit_spread_proxy'],
            'yield_10y': ['yield_10y'],
            'gold_ret': ['gold_ret'],
            'dxy_ret': ['dxy_ret'],
            'oil_ret': ['oil_ret'],
            'distribution_days': ['distribution_days']
        }

        mtf_suffixes = ['_5m', '_15m', '_1H', '_4H', '_D', '_W', '_M']
        for sfx in mtf_suffixes:
            self.ALIAS_MAP[f'close{sfx}'] = [f'close{sfx}', f'c{sfx}']
            self.ALIAS_MAP[f'open{sfx}'] = [f'open{sfx}', f'o{sfx}']
            self.ALIAS_MAP[f'high{sfx}'] = [f'high{sfx}', f'h{sfx}']
            self.ALIAS_MAP[f'low{sfx}'] = [f'low{sfx}', f'l{sfx}']
            self.ALIAS_MAP[f'volume{sfx}'] = [f'volume{sfx}', f'vol{sfx}']
            self.ALIAS_MAP[f'ema_50{sfx}'] = [f'ema_50{sfx}']
            self.ALIAS_MAP[f'roc{sfx}'] = [f'roc_12{sfx}', f'roc{sfx}']
            self.ALIAS_MAP[f'rsi{sfx}'] = [f'rsi_14{sfx}', f'rsi{sfx}']
            self.ALIAS_MAP[f'macd_histogram{sfx}'] = [f'macd_hist{sfx}', f'macd_histogram{sfx}']
            self.ALIAS_MAP[f'momentum{sfx}'] = [f'mom_10{sfx}', f'momentum{sfx}']
            self.ALIAS_MAP[f'adx{sfx}'] = [f'adx_14{sfx}', f'adx{sfx}']
            self.ALIAS_MAP[f'obv{sfx}'] = [f'obv{sfx}']
            self.ALIAS_MAP[f'cmf{sfx}'] = [f'cmf_20{sfx}', f'cmf{sfx}']
            self.ALIAS_MAP[f'relative_volume{sfx}'] = [f'rvol_20{sfx}', f'relative_volume{sfx}']
            self.ALIAS_MAP[f'vwap{sfx}'] = [f'vwap{sfx}', f'rolling_vwap{sfx}']
            self.ALIAS_MAP[f'volume_ma_20{sfx}'] = [f'avg_volume_20{sfx}', f'volume_ma_20{sfx}']

    def _get_db_connection(self) -> sqlite3.Connection:
        try:
            from backend.config.settings import settings
            db_path = Path(settings.database_path)
        except Exception:
            db_path = Path.home() / "Green-Bull-Data-Engine" / "database" / "market.db"
            
        conn = sqlite3.connect(str(db_path), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _table_exists(self, cursor: sqlite3.Cursor, table_name: str) -> bool:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
        return cursor.fetchone() is not None

    def _fetch_single_row(self, conn: sqlite3.Connection, table: str, symbol: Optional[str] = None) -> dict:
        cursor = conn.cursor()
        if not self._table_exists(cursor, table):
            return {}
        try:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [info[1].lower() for info in cursor.fetchall()]
        except Exception:
            return {}
        if not columns:
            return {}

        order_by = "rowid DESC"
        if "updated_at" in columns: order_by = "updated_at DESC"
        elif "date" in columns: order_by = "date DESC"
        elif "fiscal_year" in columns and "fiscal_quarter" in columns: order_by = "fiscal_year DESC, fiscal_quarter DESC"

        query = f"SELECT * FROM {table} "
        params = []
        if "symbol" in columns and symbol:
            query += "WHERE symbol = ? "
            params.append(symbol)
        query += f"ORDER BY {order_by}"

        try:
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            if not rows:
                return {col: np.nan for col in columns}

            composite_row = {col: np.nan for col in columns}
            filled_count = 0
            total_columns = len(columns)
            
            for row in rows:
                row_dict = dict(zip([col[0] for col in cursor.description], row))
                for col, val in row_dict.items():
                    if pd.isna(composite_row[col]) and val is not None:
                        composite_row[col] = float(val) if isinstance(val, (int, float, bool)) else val
                        filled_count += 1
                if filled_count == total_columns:
                    break
            return composite_row
        except Exception as e:
            logger.error(f"Query failed for {table}: {e}")
        return {}

    def _fetch_feature_history(self, conn: sqlite3.Connection, symbol: str) -> pd.DataFrame:
        query = "SELECT * FROM feature_history WHERE symbol = ? ORDER BY date DESC LIMIT 300"
        try:
            df = pd.read_sql_query(query, conn, params=(symbol,))
        except Exception:
            return pd.DataFrame()

        if df.empty: return df

        df = df.iloc[::-1].reset_index(drop=True)
        df.columns = [str(c).lower().strip() for c in df.columns]
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df.set_index('date', inplace=True, drop=False)

        if 'mtf_data' in df.columns:
            def parse_mtf(val):
                if pd.isna(val) or not val: return {}
                try: return json.loads(val) if isinstance(val, str) else val
                except: return {}
            
            mtf_dicts = df['mtf_data'].apply(parse_mtf)
            mtf_rows = []
            for d in mtf_dicts:
                row_data = {}
                for timeframe, indicators in d.items():
                    for ind_name, ind_val in indicators.items():
                        row_data[f"{ind_name}{timeframe}"] = ind_val
                mtf_rows.append(row_data)
            
            mtf_df = pd.DataFrame(mtf_rows, index=df.index)
            df = pd.concat([df, mtf_df], axis=1)

        # 🔴 The Overwrite Bug is fixed here. ALIAS_MAP runs LATER in run().
        
        for col in df.columns:
            if col in ['symbol', 'date', 'timestamp', 'mtf_data', 'triangle_type', 'channel_type', 'wedge_type', 'market_phase', 'market_regime', 'pattern_family', 'htf_alignment']:
                continue
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].ffill()

        return df

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        symbol = input_data.get("symbol")
        if not symbol:
            return {"status": "FAILED", "error": "Missing symbol in payload."}

        conn = self._get_db_connection()
        try:
            df = self._fetch_feature_history(conn, symbol)
            
            ctx_dicts = [
                self._fetch_single_row(conn, "fundamental_data", symbol),
                self._fetch_single_row(conn, "financial_data", symbol),
                self._fetch_single_row(conn, "company_profile", symbol),
                self._fetch_single_row(conn, "ipo_data", symbol),
                self._fetch_single_row(conn, "corporate_actions", symbol),
                self._fetch_single_row(conn, "shareholding_data", symbol),
                self._fetch_single_row(conn, "analyst_data", symbol),
                self._fetch_single_row(conn, "earnings_history", symbol),
                self._fetch_single_row(conn, "fundamental_snapshot", symbol),
                self._fetch_single_row(conn, "macro_environment")
            ]
            
            # 🔴 CAREFUL MERGE: Don't overwrite valid existing columns with NaNs
            for ctx in ctx_dicts:
                for k, v in ctx.items():
                    if k in ['symbol', 'date', 'updated_at']:
                        continue
                    if pd.notna(v):
                        df[k] = v
                    elif k not in df.columns:
                        df[k] = np.nan

            # 🔴 LATE MAPPING: Run ALIAS_MAP only after all data is safely merged
            for target_col, aliases in self.ALIAS_MAP.items():
                if target_col not in df.columns or pd.isna(df[target_col].iloc[-1] if not df.empty else np.nan):
                    for alias in aliases:
                        if alias in df.columns and pd.notna(df[alias].iloc[-1] if not df.empty else np.nan):
                            df[target_col] = df[alias]
                            break
                if target_col not in df.columns:
                    df[target_col] = np.nan
            
        except Exception as e:
            return {"status": "FAILED", "error": f"Database Load Error: {str(e)}"}
        finally:
            conn.close()

        results = {}
        for name, engine in self.engines.items():
            try:
                analyzer_df = df.copy() if not df.empty else pd.DataFrame()
                results[name] = engine.analyze(analyzer_df)
            except Exception as e:
                logger.error(f"Analyzer '{name}' execution failed for {symbol}: {str(e)}", exc_info=True)
                results[name] = {"status": "FAILED", "error": str(e)}

        return results

analyzer_engine = AnalyzerEngine()
