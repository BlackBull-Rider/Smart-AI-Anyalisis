"""
GREEN BULL RIDER V6
Module: backend/providers/yahoo_provider.py

Yahoo Finance Market Data Provider
Python 3.13 Compatible
"""

import logging
import time
import math
import random
import threading
from datetime import datetime, date, timedelta
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="yfinance",
)

from backend.config.settings import settings
from backend.providers.base_provider import BaseProvider


class YahooProvider(BaseProvider):

    def __init__(self) -> None:
        self.name = "YahooProvider"
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Cache with TTL & Thread Safety
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.Lock()
        self.cache_ttl = 3600  # 1 hour
        
        # Let yfinance manage its own session
        self.session = None
        
        self._tickers: Dict[str, yf.Ticker] = {}
        
        # Provider Statistics
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "retries": 0,
            "api_calls": 0
        }
        
    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()
            self._tickers.clear()
            self.logger.info("Cache and Tickers cleared.")

    def get_statistics(self) -> Dict[str, int]:
        with self._cache_lock:
            return self.stats.copy()

    # =========================================================================
    # SAFE HELPERS
    # =========================================================================

    @staticmethod
    def safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            val = float(value)
            if math.isnan(val) or math.isinf(val):
                return None
            return val
        except (ValueError, TypeError):
            return None

    @staticmethod
    def safe_int(value: Any) -> Optional[int]:
        val = YahooProvider.safe_float(value)
        return int(val) if val is not None else None

    @staticmethod
    def safe_bool(value: Any) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            val_lower = value.strip().lower()
            if val_lower in ('true', '1', 'yes', 'y'):
                return True
            if val_lower in ('false', '0', 'no', 'n'):
                return False
        return bool(value) if value else None

    @staticmethod
    def safe_str(value: Any) -> Optional[str]:
        if value is None or pd.isna(value):
            return None
        val = str(value).strip()
        if not val or val.lower() in ('nan', 'none', 'nat'):
            return None
        return val

    @staticmethod
    def safe_date(value: Any) -> Optional[str]:
        if value is None or pd.isna(value):
            return None
        try:
            if isinstance(value, (datetime, date)):
                return value.strftime("%Y-%m-%d")
            dt = pd.to_datetime(value)
            if pd.isna(dt):
                return None
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return None

    @staticmethod
    def safe_dataframe(df: Any) -> Optional[pd.DataFrame]:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return None
        return df

    @staticmethod
    def safe_series(series: Any) -> Optional[pd.Series]:
        if series is None or not isinstance(series, pd.Series) or series.empty:
            return None
        return series

    @staticmethod
    def safe_dict(d: Any) -> Dict[str, Any]:
        if d is None or not isinstance(d, dict):
            return {}
        return d

    def _get_first_valid(self, series: pd.Series, keys: List[str]) -> Optional[float]:
        for k in keys:
            val = series.get(k)
            if val is not None and not pd.isna(val):
                return self.safe_float(val)
        return None

    # =========================================================================
    # INTERNAL CORE
    # =========================================================================

    def _symbol(self, symbol: str) -> str:
        if not symbol:
            return ""
        clean = str(symbol).strip().upper().replace('.', '-')
        if clean == "^NSEI":
            return clean
        if "." not in clean and "^" not in clean:
            clean = f"{clean}.NS"
        return clean

    def _ticker(self, symbol: str) -> yf.Ticker:
        clean_symbol = self._symbol(symbol)
        with self._cache_lock:
            if clean_symbol not in self._tickers:
                self._tickers[clean_symbol] = yf.Ticker(clean_symbol)
            return self._tickers[clean_symbol]

    def _retry(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        # Integrated with settings.py
        max_retries = settings.provider.max_retries
        delay = settings.retry.base_delay
        backoff = settings.provider.retry_backoff
        max_delay = settings.retry.max_delay
        
        func_name = getattr(func, '__name__', str(func))
        
        for attempt in range(1, max_retries + 1):
            try:
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                exec_time = time.perf_counter() - start_time
                self.logger.debug(f"Execution Time: {func_name} took {exec_time:.3f}s")
                return result
            except Exception as e:
                with self._cache_lock:
                    self.stats["retries"] += 1
                self.logger.warning(f"Retry {attempt}/{max_retries} for {func_name} failed: {str(e)}")
                if attempt == max_retries:
                    self.logger.error(f"Failure: Exhausted all retries for {func_name}")
                    return None
                
                # Adding random jitter for exponential backoff
                jitter = random.uniform(0.8, 1.2)
                sleep_time = delay * jitter
                time.sleep(sleep_time)
                
                delay = min(delay * backoff, max_delay)
                
        return None

    def _fetch_cached(self, symbol: str, key: str, fetch_func: Callable) -> Any:
        cache_key = f"{self._symbol(symbol)}_{key}"
        
        with self._cache_lock:
            cached_item = self._cache.get(cache_key)
            if cached_item and time.time() < cached_item['expiry']:
                self.stats["cache_hits"] += 1
                self.logger.debug(f"Cache Hit: {cache_key}")
                return cached_item['data']
            self.stats["cache_misses"] += 1
            
        self.logger.debug(f"Cache Miss: {cache_key}")
        start_time = time.perf_counter()
        
        with self._cache_lock:
            self.stats["api_calls"] += 1
            
        data = self._retry(fetch_func)
        exec_time = time.perf_counter() - start_time
        
        if data is not None:
            self.logger.info(f"Download: {cache_key} succeeded in {exec_time:.3f}s")
            with self._cache_lock:
                self._cache[cache_key] = {
                    'data': data,
                    'expiry': time.time() + self.cache_ttl
                }
        else:
            self.logger.debug("No data available for %s", cache_key)
            
        return data

    def _safe_info(self, symbol: str) -> Dict[str, Any]:
        def fetch() -> Dict[str, Any]:
            ticker = self._ticker(symbol)
            return self.safe_dict(ticker.info)
        return self._fetch_cached(symbol, "info", fetch) or {}


    def _safe_fast_info(self, symbol: str) -> Dict[str, Any]:
        def fetch() -> Dict[str, Any]:
            ticker = self._ticker(symbol)

            try:
                fi = ticker.fast_info
            except Exception:
                return {}

            result: Dict[str, Any] = {}

            for key in (
                "market_cap",
                "shares",
                "last_price",
                "currency",
                "timezone",
            ):
                try:
                    result[key] = fi.get(key)
                except Exception:
                    continue

            return result

        return self._fetch_cached(symbol, "fast_info", fetch) or {}

    # =========================================================================
    # HISTORY & VALIDATION
    # =========================================================================

    def is_available(self) -> bool:
        try:
            hist = yf.Ticker("RELIANCE.NS").history(
                period="5d",
                auto_adjust=False
            )
            return hist is not None and not hist.empty
        except Exception as e:
            self.logger.exception(f"Provider availability check failed: {e}")
            return False

    def validate_history(self, df: pd.DataFrame) -> pd.DataFrame:
        df_safe = self.safe_dataframe(df)
        if df_safe is None:
            return pd.DataFrame()
            
        df_safe = df_safe.copy()
        
        if df_safe.index.name == 'Date' or isinstance(df_safe.index, pd.DatetimeIndex):
            df_safe = df_safe.reset_index()
            
        df_safe.columns = [str(c).lower().strip() for c in df_safe.columns]
        
        if 'date' not in df_safe.columns:
            return pd.DataFrame()
            
        df_safe['date'] = pd.to_datetime(df_safe['date'], utc=True).dt.tz_localize(None)
        
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df_safe.columns:
                return pd.DataFrame()
            df_safe[col] = pd.to_numeric(df_safe[col], errors='coerce')
            
        df_safe = df_safe.dropna(subset=required_cols)
        df_safe = df_safe.drop_duplicates(subset=['date'])
        df_safe = df_safe.sort_values('date').reset_index(drop=True)
        
        if 'volume' in df_safe.columns:
            df_safe['volume'] = pd.to_numeric(df_safe['volume'], errors='coerce').fillna(0)
        else:
            df_safe['volume'] = 0.0
            
        # Ensure we only return the maximum allowed trading candles from settings
        limit = settings.sync.trading_candles
        return df_safe[['date', 'open', 'high', 'low', 'close', 'volume']].tail(limit).reset_index(drop=True)

    def get_history(self, symbol: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        def fetch() -> Optional[pd.DataFrame]:
            ticker = self._ticker(symbol)
            
            # Dynamic fallback using settings.py
            nonlocal start_date, end_date
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if start_date is None:
                dt_end = pd.to_datetime(end_date)
                dt_start = dt_end - timedelta(days=settings.sync.history_days)
                start_date = dt_start.strftime("%Y-%m-%d")

            df = ticker.history(start=start_date, end=end_date, auto_adjust=False)
            return self.validate_history(df)
            
        # Create safe cache keys in case dates are dynamically resolved
        sd_key = start_date or "default"
        ed_key = end_date or "default"
        cache_key = f"history_{sd_key}_{ed_key}"
        return self._fetch_cached(symbol, cache_key, fetch)

    # =========================================================================
    # RAW DATA EXTRACTION
    # =========================================================================

    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        info = self._safe_info(symbol)
        return self.normalize_company_profile(info)

    def get_fast_info(self, symbol: str) -> Dict[str, Any]:
        return self._safe_fast_info(symbol)

    def get_actions(self, symbol: str) -> List[Dict[str, Any]]:
        def fetch() -> Optional[pd.DataFrame]:
            return self.safe_dataframe(self._ticker(symbol).actions)

        df = self._fetch_cached(symbol, "actions", fetch)
        return self.normalize_actions(df)


    def get_dividends(self, symbol: str) -> Optional[pd.Series]:
        def fetch() -> Optional[pd.Series]:
            return self.safe_series(self._ticker(symbol).dividends)
        return self._fetch_cached(symbol, "dividends", fetch)

    def get_splits(self, symbol: str) -> Optional[pd.Series]:
        def fetch() -> Optional[pd.Series]:
            return self.safe_series(self._ticker(symbol).splits)
        return self._fetch_cached(symbol, "splits", fetch)

    def get_income_statement(self, symbol: str) -> Optional[pd.DataFrame]:
        def fetch() -> Optional[pd.DataFrame]:
            return self.safe_dataframe(self._ticker(symbol).income_stmt)
        return self._fetch_cached(symbol, "income_stmt", fetch)

    def get_balance_sheet(self, symbol: str) -> Optional[pd.DataFrame]:
        def fetch() -> Optional[pd.DataFrame]:
            return self.safe_dataframe(self._ticker(symbol).balance_sheet)
        return self._fetch_cached(symbol, "balance_sheet", fetch)

    def get_cashflow(self, symbol: str) -> Optional[pd.DataFrame]:
        def fetch() -> Optional[pd.DataFrame]:
            return self.safe_dataframe(self._ticker(symbol).cashflow)
        return self._fetch_cached(symbol, "cashflow", fetch)

    def get_quarterly_income_statement(self, symbol: str) -> Optional[pd.DataFrame]:
        def fetch() -> Optional[pd.DataFrame]:
            return self.safe_dataframe(self._ticker(symbol).quarterly_income_stmt)
        return self._fetch_cached(symbol, "quarterly_income_stmt", fetch)

    def get_quarterly_balance_sheet(self, symbol: str) -> Optional[pd.DataFrame]:
        def fetch() -> Optional[pd.DataFrame]:
            return self.safe_dataframe(self._ticker(symbol).quarterly_balance_sheet)
        return self._fetch_cached(symbol, "quarterly_balance_sheet", fetch)

    def get_quarterly_cashflow(self, symbol: str) -> Optional[pd.DataFrame]:
        def fetch() -> Optional[pd.DataFrame]:
            return self.safe_dataframe(self._ticker(symbol).quarterly_cashflow)
        return self._fetch_cached(symbol, "quarterly_cashflow", fetch)

    def get_share_holders(self, symbol: str) -> List[Dict[str, Any]]:
        def fetch() -> Dict[str, Any]:
            ticker = self._ticker(symbol)
            return {
                "institutional": self.safe_dataframe(ticker.institutional_holders),
                "mutualfund": self.safe_dataframe(ticker.mutualfund_holders),
                "major": self.safe_dataframe(ticker.major_holders),
            }

        raw = self._fetch_cached(symbol, "shareholders", fetch) or {}
        return self.normalize_shareholders(raw)


    def get_analyst_targets(self, symbol: str) -> Dict[str, Any]:
        info = self._safe_info(symbol)
        return self.normalize_analyst(info)

    def get_earnings_dates(self, symbol: str) -> Optional[pd.DataFrame]:
        def fetch() -> Optional[pd.DataFrame]:
            return self.safe_dataframe(self._ticker(symbol).earnings_dates)
        return self._fetch_cached(symbol, "earnings_dates", fetch)

    def get_earnings_history(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Yahoo Finance earnings sync is temporarily disabled.
        """
        return None

    def get_earnings(self, symbol: str) -> Optional[pd.DataFrame]:
        def fetch() -> Optional[pd.DataFrame]:
            return self.safe_dataframe(self._ticker(symbol).earnings)
        return self._fetch_cached(symbol, "earnings", fetch)

    def get_quarterly_results(self, symbol: str) -> Optional[pd.DataFrame]:
        def fetch() -> Optional[pd.DataFrame]:
            return self.safe_dataframe(self._ticker(symbol).quarterly_earnings)
        return self._fetch_cached(symbol, "quarterly_earnings", fetch)

    def get_recommendations(self, symbol: str) -> List[Dict[str, Any]]:
        def fetch() -> Optional[pd.DataFrame]:
            return self.safe_dataframe(self._ticker(symbol).recommendations)
        df = self._fetch_cached(symbol, "recommendations", fetch)
        return self.normalize_recommendation_summary(df)

    def get_recommendation_summary(self, symbol: str) -> List[Dict[str, Any]]:
        def fetch() -> Optional[pd.DataFrame]:
            return self.safe_dataframe(self._ticker(symbol).recommendations_summary)
        df = self._fetch_cached(symbol, "recommendations_summary", fetch)
        return self.normalize_recommendation_summary(df)

    def get_insider_transactions(self, symbol: str) -> Optional[pd.DataFrame]:
        def fetch() -> Optional[pd.DataFrame]:
            return self.safe_dataframe(self._ticker(symbol).insider_transactions)
        return self._fetch_cached(symbol, "insider_transactions", fetch)

    def get_insider_purchases(self, symbol: str) -> Optional[pd.DataFrame]:
        def fetch() -> Optional[pd.DataFrame]:
            return self.safe_dataframe(self._ticker(symbol).insider_purchases)
        return self._fetch_cached(symbol, "insider_purchases", fetch)

    def get_calendar(self, symbol: str) -> Dict[str, Any]:
        def fetch() -> Dict[str, Any]:
            cal = self._ticker(symbol).calendar
            return self.safe_dict(cal) if isinstance(cal, dict) else {}
        cal_raw = self._fetch_cached(symbol, "calendar", fetch) or {}
        return self.normalize_calendar(cal_raw)

    def get_news(self, symbol: str) -> List[Dict[str, Any]]:
        def fetch() -> List[Any]:
            news = self._ticker(symbol).news
            return news if isinstance(news, list) else []
        news_raw = self._fetch_cached(symbol, "news", fetch) or []
        return self.normalize_news(news_raw)

    def get_sustainability(self, symbol: str) -> Dict[str, Any]:
        def fetch() -> Optional[pd.DataFrame]:
            return self.safe_dataframe(self._ticker(symbol).sustainability)
        df = self._fetch_cached(symbol, "sustainability", fetch)
        return self.normalize_sustainability(df)

    # =========================================================================
    # NORMALIZATION / ENTERPRISE MAPPERS
    # =========================================================================

    def normalize_company_profile(self, info: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "company_name": self.safe_str(info.get("longName", info.get("shortName"))),
            "short_name": self.safe_str(info.get("shortName")),
            "long_name": self.safe_str(info.get("longName")),
            "exchange": self.safe_str(info.get("exchange")),
            "exchange_code": self.safe_str(info.get("exchangeTimezoneShortName")),
            "isin": self.safe_str(info.get("isin")),
            "sector": self.safe_str(info.get("sector")),
            "industry": self.safe_str(info.get("industry")),
            "sub_industry": self.safe_str(info.get("industryKey")),
            "market": self.safe_str(info.get("market")),
            "currency": self.safe_str(info.get("currency", info.get("financialCurrency"))),
            "country": self.safe_str(info.get("country")),
            "state": self.safe_str(info.get("state")),
            "city": self.safe_str(info.get("city")),
            "address": self.safe_str(info.get("address1")),
            "zipcode": self.safe_str(info.get("zip")),
            "website": self.safe_str(info.get("website")),
            "phone": self.safe_str(info.get("phone")),
            "email": None,
            "ceo": None,
            "cfo": None,
            "chairman": None,
            "employees": self.safe_int(info.get("fullTimeEmployees")),
            "founded_year": None,
            "business_summary": self.safe_str(info.get("longBusinessSummary")),
            "logo_url": self.safe_str(info.get("logo_url")),
            "timezone": self.safe_str(info.get("exchangeTimezoneName")),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def get_fundamentals(self, symbol: str) -> Dict[str, Any]:
        info = self._safe_info(symbol)
        fast_info = self._safe_fast_info(symbol)
        
        return {
            "market_cap": self.safe_float(info.get("marketCap", fast_info.get("market_cap"))),
            "enterprise_value": self.safe_float(info.get("enterpriseValue")),
            "pe": self.safe_float(info.get("trailingPE")),
            "pb": self.safe_float(info.get("priceToBook")),
            "roe": self.safe_float(info.get("returnOnEquity")),
            "roce": None,
            "eps": self.safe_float(info.get("trailingEps")),
            "forward_eps": self.safe_float(info.get("forwardEps")),
            "book_value": self.safe_float(info.get("bookValue")),
            "dividend_yield": self.safe_float(info.get("dividendYield")),
            "dividend_rate": self.safe_float(info.get("dividendRate")),
            "payout_ratio": self.safe_float(info.get("payoutRatio")),
            "beta": self.safe_float(info.get("beta")),
            "current_ratio": self.safe_float(info.get("currentRatio")),
            "quick_ratio": self.safe_float(info.get("quickRatio")),
            "debt_to_equity": self.safe_float(info.get("debtToEquity")),
            "free_cash_flow": self.safe_float(info.get("freeCashflow")),
            "operating_cash_flow": self.safe_float(info.get("operatingCashflow")),
            "revenue_growth": self.safe_float(info.get("revenueGrowth")),
            "earnings_growth": self.safe_float(info.get("earningsGrowth")),
            "profit_margin": self.safe_float(info.get("profitMargins")),
            "gross_margin": self.safe_float(info.get("grossMargins")),
            "operating_margin": self.safe_float(info.get("operatingMargins")),
            "shares_outstanding": self.safe_float(info.get("sharesOutstanding", fast_info.get("shares"))),
            "held_percent_insiders": self.safe_float(info.get("heldPercentInsiders")),
            "held_percent_institutions": self.safe_float(info.get("heldPercentInstitutions")),
            "sector": self.safe_str(info.get("sector")),
            "industry": self.safe_str(info.get("industry")),
            "country": self.safe_str(info.get("country")),
            "currency": self.safe_str(info.get("financialCurrency")),
            "exchange": self.safe_str(info.get("exchange")),
            "website": self.safe_str(info.get("website")),
            "employees": self.safe_int(info.get("fullTimeEmployees")),
            "business_summary": self.safe_str(info.get("longBusinessSummary")),
            "target_price": self.safe_float(info.get("targetMeanPrice")),
            "recommendation": self.safe_str(info.get("recommendationKey"))
        }

    def normalize_income_statement(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        df_safe = self.safe_dataframe(df)
        if df_safe is None:
            return []
            
        results = []
        for date_col in df_safe.columns:
            series = df_safe[date_col]
            dt = pd.to_datetime(date_col)
            results.append({
                "fiscal_year": self.safe_int(dt.year),
                "fiscal_quarter": self.safe_int((dt.month - 1) // 3 + 1),
                "total_revenue": self._get_first_valid(series, ["Total Revenue", "Operating Revenue"]),
                "cost_of_revenue": self._get_first_valid(series, ["Cost Of Revenue", "Cost of Revenue"]),
                "gross_profit": self._get_first_valid(series, ["Gross Profit"]),
                "operating_expense": self._get_first_valid(series, ["Operating Expense", "Operating Expenses"]),
                "operating_income": self._get_first_valid(series, ["Operating Income"]),
                "ebit": self._get_first_valid(series, ["EBIT"]),
                "ebitda": self._get_first_valid(series, ["EBITDA", "Normalized EBITDA"]),
                "pretax_income": self._get_first_valid(series, ["Pretax Income"]),
                "tax_expense": self._get_first_valid(series, ["Tax Provision", "Income Tax Expense"]),
                "net_income": self._get_first_valid(series, ["Net Income", "Net Income Common Stockholders"]),
                "basic_eps": self._get_first_valid(series, ["Basic EPS"]),
                "diluted_eps": self._get_first_valid(series, ["Diluted EPS"])
            })
        return results

    def normalize_balance_sheet(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        df_safe = self.safe_dataframe(df)
        if df_safe is None:
            return []
            
        results = []
        for date_col in df_safe.columns:
            series = df_safe[date_col]
            dt = pd.to_datetime(date_col)
            results.append({
                "fiscal_year": self.safe_int(dt.year),
                "fiscal_quarter": self.safe_int((dt.month - 1) // 3 + 1),
                "cash": self._get_first_valid(series, ["Cash And Cash Equivalents", "Cash"]),
                "cash_equivalents": self._get_first_valid(series, ["Cash Equivalents"]),
                "short_term_investments": self._get_first_valid(series, ["Other Short Term Investments"]),
                "accounts_receivable": self._get_first_valid(series, ["Accounts Receivable", "Net Receivables"]),
                "inventory": self._get_first_valid(series, ["Inventory"]),
                "current_assets": self._get_first_valid(series, ["Total Current Assets"]),
                "total_assets": self._get_first_valid(series, ["Total Assets", "Assets"]),
                "accounts_payable": self._get_first_valid(series, ["Accounts Payable"]),
                "current_liabilities": self._get_first_valid(series, ["Total Current Liabilities"]),
                "total_liabilities": self._get_first_valid(series, ["Total Liabilities Net Minority Interest", "Total Liabilities"]),
                "short_term_debt": self._get_first_valid(series, ["Current Debt"]),
                "long_term_debt": self._get_first_valid(series, ["Long Term Debt"]),
                "total_debt": self._get_first_valid(series, ["Total Debt"]),
                "shareholder_equity": self._get_first_valid(series, ["Stockholders Equity", "Total Stockholder Equity"]),
                "retained_earnings": self._get_first_valid(series, ["Retained Earnings"])
            })
        return results

    def normalize_cashflow(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        df_safe = self.safe_dataframe(df)
        if df_safe is None:
            return []
            
        results = []
        for date_col in df_safe.columns:
            series = df_safe[date_col]
            dt = pd.to_datetime(date_col)
            results.append({
                "fiscal_year": self.safe_int(dt.year),
                "fiscal_quarter": self.safe_int((dt.month - 1) // 3 + 1),
                "operating_cash_flow": self._get_first_valid(series, ["Operating Cash Flow", "Total Cash From Operating Activities"]),
                "investing_cash_flow": self._get_first_valid(series, ["Investing Cash Flow", "Total Cashflows From Investing Activities"]),
                "financing_cash_flow": self._get_first_valid(series, ["Financing Cash Flow", "Total Cash From Financing Activities"]),
                "capital_expenditure": self._get_first_valid(series, ["Capital Expenditure"]),
                "free_cash_flow": self._get_first_valid(series, ["Free Cash Flow"])
            })
        return results

    def get_financials(self, symbol: str) -> List[Dict[str, Any]]:
        info = self._safe_info(symbol)

        income = {
            (r["fiscal_year"], r["fiscal_quarter"]): r
            for r in self.normalize_income_statement(
                self.get_income_statement(symbol)
            )
        }

        balance = {
            (r["fiscal_year"], r["fiscal_quarter"]): r
            for r in self.normalize_balance_sheet(
                self.get_balance_sheet(symbol)
            )
        }

        cashflow = {
            (r["fiscal_year"], r["fiscal_quarter"]): r
            for r in self.normalize_cashflow(
                self.get_cashflow(symbol)
            )
        }

        keys = sorted(
            set(income) | set(balance) | set(cashflow),
            reverse=True
        )

        rows = []

        for fy, fq in keys:
            row = {
                "symbol": symbol,
                "fiscal_year": fy,
                "fiscal_quarter": str(fq),
                "currency": self.safe_str(info.get("financialCurrency")),
                "market_cap": self.safe_float(info.get("marketCap")),
                "enterprise_value": self.safe_float(info.get("enterpriseValue")),
                "shares_outstanding": self.safe_float(info.get("sharesOutstanding")),
                "beta": self.safe_float(info.get("beta")),
                "dividend_yield": self.safe_float(info.get("dividendYield")),
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            row.update(income.get((fy, fq), {}))
            row.update(balance.get((fy, fq), {}))
            row.update(cashflow.get((fy, fq), {}))

            rows.append(row)

        return rows

    def normalize_actions(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        df_safe = self.safe_dataframe(df)
        if df_safe is None:
            return []
            
        if df_safe.index.name == 'Date' or isinstance(df_safe.index, pd.DatetimeIndex):
            df_safe = df_safe.reset_index()
            
        results = []
        for _, row in df_safe.iterrows():
            date_val = self.safe_date(row.get("Date", row.get("date")))
            
            div = self.safe_float(row.get("Dividends"))
            if div and div > 0:
                results.append({
                    "action_date": date_val,
                    "action_type": "DIVIDEND",
                    "dividend": div,
                    "split_ratio": None,
                    "bonus_ratio": None,
                    "rights_ratio": None,
                    "face_value_change": None,
                    "buyback": None,
                    "merger": None,
                    "demerger": None,
                    "spin_off": None,
                    "description": f"Dividend: {div}",
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
            split = self.safe_float(row.get("Stock Splits"))
            if split and split > 0:
                # Basic bonus/split heuristic based on ratio if needed
                action_type = "SPLIT"
                desc = f"Stock Split: {split}"
                
                results.append({
                    "action_date": date_val,
                    "action_type": action_type,
                    "dividend": None,
                    "split_ratio": split,
                    "bonus_ratio": None,
                    "rights_ratio": None,
                    "face_value_change": None,
                    "buyback": None,
                    "merger": None,
                    "demerger": None,
                    "spin_off": None,
                    "description": desc,
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        return results

    def normalize_shareholders(self, shareholders_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        results = []
        
        insider_held = None
        inst_held = None
        
        major = self.safe_dataframe(shareholders_dict.get("major"))
        if major is not None and not major.empty:
            for _, row in major.iterrows():
                desc = str(row.iloc[1]).lower() if len(row) > 1 else ""
                val_raw = str(row.iloc[0]).replace('%', '') if len(row) > 0 else None
                val = self.safe_float(val_raw)
                
                if "insider" in desc:
                    insider_held = val
                elif "institutions" in desc:
                    inst_held = val

        inst = self.safe_dataframe(shareholders_dict.get("institutional"))
        if inst is not None and not inst.empty:
            for _, row in inst.iterrows():
                results.append({
                    "quarter": self.safe_date(row.get("Date Reported")),
                    "promoter_holding": None, 
                    "promoter_pledged": None,
                    "fii_holding": None,
                    "dii_holding": None,
                    "mutual_fund_holding": None,
                    "insurance_holding": None,
                    "government_holding": None,
                    "foreign_holding": None,
                    "retail_holding": None,
                    "public_holding": None,
                    "insider_holding": insider_held,
                    "others_holding": inst_held,
                    # "institution_name": self.safe_str(row.get("Holder")),
                    # "shares": self.safe_float(row.get("Shares")),
                    # "value": self.safe_float(row.get("Value")),
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        return results

    def normalize_analyst(self, info: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "target_price": self.safe_float(info.get("targetMeanPrice")),
            "target_high": self.safe_float(info.get("targetHighPrice")),
            "target_low": self.safe_float(info.get("targetLowPrice")),
            "target_mean": self.safe_float(info.get("targetMeanPrice")),
            "recommendation": self.safe_str(info.get("recommendationKey")),
            "recommendation_key": self.safe_str(info.get("recommendationKey")),
            "number_of_analysts": self.safe_int(info.get("numberOfAnalystOpinions")),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def normalize_earnings(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        df_safe = self.safe_dataframe(df)
        if df_safe is None:
            return []
            
        if df_safe.index.name == 'Earnings Date' or isinstance(df_safe.index, pd.DatetimeIndex):
            df_safe = df_safe.reset_index()
            
        results = []
        for _, row in df_safe.iterrows():
            dt = pd.to_datetime(row.get("Earnings Date", row.get("index")))
            quarter_str = None
            if not pd.isna(dt):
                quarter_str = f"{dt.year}-Q{(dt.month-1)//3 + 1}"
                
            results.append({
                "quarter": quarter_str,
                "estimate": self.safe_float(row.get("EPS Estimate")),
                "reported": self.safe_float(row.get("Reported EPS")),
                "surprise": self.safe_float(row.get("Surprise(%)")),
                "surprise_percent": self.safe_float(row.get("Surprise(%)")),
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        return results

    def normalize_recommendation_summary(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        df_safe = self.safe_dataframe(df)
        if df_safe is None:
            return []
            
        if df_safe.index.name == 'period' or isinstance(df_safe.index, pd.Index):
            df_safe = df_safe.reset_index()
            
        results = []
        for _, row in df_safe.iterrows():
            results.append({
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        return results

    def normalize_calendar(self, calendar: Dict[str, Any]) -> Dict[str, Any]:
        if not calendar:
            return {}
        return {
            "earnings_date": self.safe_date(calendar.get("Earnings Date")),
            "earnings_high": self.safe_float(calendar.get("Earnings High")),
            "earnings_low": self.safe_float(calendar.get("Earnings Low")),
            "revenue_high": self.safe_float(calendar.get("Revenue High")),
            "revenue_low": self.safe_float(calendar.get("Revenue Low")),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def normalize_news(self, news_list: List[Any]) -> List[Dict[str, Any]]:
        results = []
        for item in news_list:
            if not isinstance(item, dict):
                continue
            results.append({
                "title": self.safe_str(item.get("title")),
                "publisher": self.safe_str(item.get("publisher")),
                "link": self.safe_str(item.get("link")),
                "publish_time": self.safe_int(item.get("providerPublishTime")),
                "type": self.safe_str(item.get("type")),
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        return results

    def normalize_sustainability(self, df: pd.DataFrame) -> Dict[str, Any]:
        df_safe = self.safe_dataframe(df)
        if df_safe is None:
            return {}
            
        results = {}
        for idx, row in df_safe.iterrows():
            if "Value" in row:
                results[self.safe_str(idx)] = self.safe_float(row["Value"])
                
        results["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return results

    def get_listing_date(self, symbol: str) -> Optional[str]:
        """
        Return company listing (first trade) date in YYYY-MM-DD format.
        """
        try:
            info = self._safe_info(symbol)

            ts = (
                info.get("firstTradeDateEpochUtc")
                or info.get("firstTradeDateMilliseconds")
            )

            if ts is None:
                return None

            ts = float(ts)

            # Yahoo may return milliseconds
            if ts > 1e12:
                ts /= 1000.0

            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")

        except Exception as e:
            self.logger.warning(
                f"Listing date fetch failed for {symbol}: {e}"
            )
            return None


def _debug_provider(self):
    try:
        ticker = yf.Ticker("RELIANCE.NS")
        hist = ticker.history(period="5d", auto_adjust=False)

        print("=" * 80)
        print(hist)
        print("=" * 80)

    except Exception:
        import traceback
        traceback.print_exc()

