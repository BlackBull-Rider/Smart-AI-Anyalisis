"""
GREEN BULL RIDER V6
Module: backend/providers/yahoo_provider.py

Yahoo Finance Market Data Provider (Universal Dynamic Fetcher - Array Crash Proof)
Python 3.13 Compatible
"""

import logging
import time
import math
import random
import threading
import re
from datetime import datetime, date, timedelta
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yfinance as yf
import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="yfinance",
)

from backend.config.settings import settings
from backend.providers.base_provider import BaseProvider


class YahooProvider(BaseProvider):

    def get_universe(self):
        raise NotImplementedError(
            "YahooProvider does not support get_universe(); use NSEProvider."
        )

    def __init__(self) -> None:
        self.name = "YahooProvider"
        self.logger = logging.getLogger(self.__class__.__name__)

        # Cache with TTL & Thread Safety
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.Lock()
        self.cache_ttl = 3600  # 1 hour

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
    # SAFE HELPERS & DYNAMIC FORMATTERS (CRASH PROOF)
    # =========================================================================

    @staticmethod
    def _extract_scalar(value: Any) -> Any:
        """Extracts a single scalar value from arrays, lists, or series to prevent ambiguous truth errors."""
        if value is None:
            return None
        if isinstance(value, (list, tuple, set)):
            if len(value) == 0:
                return None
            return list(value)[0]
        if isinstance(value, (pd.Series, pd.Index, np.ndarray)):
            if len(value) == 0:
                return None
            return value[0] if isinstance(value, np.ndarray) else value.iloc[0]
        return value

    @staticmethod
    def _to_snake_case(name: str) -> str:
        """যেকোনো স্ট্রিং (Total Revenue, trailingPE) কে snake_case বানাবে"""
        if not name:
            return "unknown_key"
        try:
            if pd.isna(name):
                return "unknown_key"
        except Exception:
            pass # In case name is an array
            
        name = str(name).strip()
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        name = re.sub(r'[^a-z0-9_]', '_', name)
        return re.sub(r'_+', '_', name).strip('_')

    @staticmethod
    def safe_float(value: Any) -> Optional[float]:
        value = YahooProvider._extract_scalar(value)
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
        value = YahooProvider._extract_scalar(value)
        if value is None:
            return None
            
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass

        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            val_lower = value.strip().lower()
            if val_lower in ('true', '1', 'yes', 'y'):
                return True
            if val_lower in ('false', '0', 'no', 'n'):
                return False
        return bool(value)

    @staticmethod
    def safe_str(value: Any) -> Optional[str]:
        value = YahooProvider._extract_scalar(value)
        if value is None:
            return None
            
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
            
        val = str(value).strip()
        if not val or val.lower() in ('nan', 'none', 'nat'):
            return None
        return val

    @staticmethod
    def safe_date(value: Any) -> Optional[str]:
        value = YahooProvider._extract_scalar(value)
        if value is None:
            return None
            
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
            
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
            if val is not None:
                try:
                    if not pd.isna(val):
                        return self.safe_float(val)
                except Exception:
                    pass
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
            for key in ("market_cap", "shares", "last_price", "currency", "timezone"):
                try: result[key] = fi.get(key)
                except Exception: continue
            return result
        return self._fetch_cached(symbol, "fast_info", fetch) or {}

    # =========================================================================
    # HISTORY & VALIDATION
    # =========================================================================

    def is_available(self) -> bool:
        try:
            hist = yf.Ticker("RELIANCE.NS").history(period="5d", auto_adjust=False)
            return hist is not None and not hist.empty
        except Exception as e:
            self.logger.exception(f"Provider availability check failed: {e}")
            return False

    def validate_history(self, df: pd.DataFrame) -> pd.DataFrame:
        df_safe = self.safe_dataframe(df)
        if df_safe is None: return pd.DataFrame()
        df_safe = df_safe.copy()

        if df_safe.index.name == 'Date' or isinstance(df_safe.index, pd.DatetimeIndex):
            df_safe = df_safe.reset_index()

        df_safe.columns = [str(c).lower().strip() for c in df_safe.columns]
        if 'date' not in df_safe.columns: return pd.DataFrame()

        df_safe["date"] = pd.to_datetime(df_safe["date"]).dt.tz_localize(None)

        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df_safe.columns: return pd.DataFrame()
            df_safe[col] = pd.to_numeric(df_safe[col], errors='coerce')

        df_safe = df_safe.dropna(subset=required_cols)
        df_safe = df_safe.drop_duplicates(subset=['date'])
        df_safe = df_safe.sort_values('date').reset_index(drop=True)

        if 'volume' in df_safe.columns:
            df_safe['volume'] = pd.to_numeric(df_safe['volume'], errors='coerce').fillna(0)
        else:
            df_safe['volume'] = 0.0

        limit = settings.sync.trading_candles
        return df_safe[['date', 'open', 'high', 'low', 'close', 'volume']].tail(limit).reset_index(drop=True)

    def get_history(self, symbol: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        def fetch() -> Optional[pd.DataFrame]:
            ticker = self._ticker(symbol)
            nonlocal start_date, end_date
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if start_date is None:
                dt_end = pd.to_datetime(end_date)
                dt_start = dt_end - timedelta(days=settings.sync.history_days)
                start_date = dt_start.strftime("%Y-%m-%d")
            df = ticker.history(start=start_date, end=end_date, auto_adjust=False)
            return self.validate_history(df)

        sd_key = start_date or "default"
        ed_key = end_date or "default"
        cache_key = f"history_{sd_key}_{ed_key}"
        return self._fetch_cached(symbol, cache_key, fetch)

    # =========================================================================
    # RAW DATA EXTRACTION
    # =========================================================================

    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        info = self._safe_info(symbol)
        info["symbol"] = symbol
        info["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        info["company_name"] = str(info.get("longName", info.get("shortName", symbol)))
        return info

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
        return None

    def get_earnings(self, symbol: str) -> List[Dict[str, Any]]:
        def fetch() -> Optional[pd.DataFrame]:
            return self.safe_dataframe(self._ticker(symbol).earnings)
        df = self._fetch_cached(symbol, "earnings", fetch)
        return self.normalize_earnings(df)

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

    def get_listing_date(self, symbol: str) -> Optional[str]:
        try:
            info = self._safe_info(symbol)
            ts = info.get("firstTradeDateEpochUtc") or info.get("firstTradeDateMilliseconds")
            if ts is None: return None
            ts = float(ts)
            if ts > 1e12: ts /= 1000.0
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        except Exception as e:
            self.logger.warning(f"Listing date fetch failed for {symbol}: {e}")
            return None

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
            "employees": self.safe_int(info.get("fullTimeEmployees")),
            "business_summary": self.safe_str(info.get("longBusinessSummary")),
            "logo_url": self.safe_str(info.get("logo_url")),
            "timezone": self.safe_str(info.get("exchangeTimezoneName")),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def get_fundamentals(self, symbol: str) -> Dict[str, Any]:
        result = {"symbol": symbol, "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        result.update(self._safe_fast_info(symbol))
        result.update(self._safe_info(symbol))
        return result

    def get_financials(self, symbol: str) -> List[Dict[str, Any]]:
        info = self._safe_info(symbol)
        merged_data = {}

        def _process_statement(df):
            if df is None or df.empty: return
            df = df.loc[:, ~df.columns.duplicated(keep="first")]
            for date_col in df.columns:
                dt_val = date_col.iloc[0] if hasattr(date_col, "iloc") else (date_col[0] if isinstance(date_col, (list, tuple)) else date_col)
                dt = pd.to_datetime(dt_val)
                if pd.isna(dt): continue
                
                key = (int(dt.year), int((dt.month - 1) // 3 + 1))
                if key not in merged_data:
                    merged_data[key] = {
                        "symbol": symbol, "fiscal_year": key[0], "fiscal_quarter": str(key[1]),
                        "currency": str(info.get("financialCurrency", "INR")),
                        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                for index_name, value in df[date_col].items():
                    merged_data[key][str(index_name)] = value

        _process_statement(self._fetch_cached(symbol, "inc", lambda: getattr(self._ticker(symbol), "income_stmt", None)))
        _process_statement(self._fetch_cached(symbol, "bal", lambda: getattr(self._ticker(symbol), "balance_sheet", None)))
        _process_statement(self._fetch_cached(symbol, "cf", lambda: getattr(self._ticker(symbol), "cashflow", None)))
        
        keys = sorted(merged_data.keys(), reverse=True)
        return [merged_data[k] for k in keys]

    def normalize_actions(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        df_safe = self.safe_dataframe(df)
        if df_safe is None: return []
        if df_safe.index.name == 'Date' or isinstance(df_safe.index, pd.DatetimeIndex): df_safe = df_safe.reset_index()
        results = []
        for _, row in df_safe.iterrows():
            date_val = self.safe_date(row.get("Date", row.get("date")))
            div = self.safe_float(row.get("Dividends"))
            if div is not None and div > 0:
                results.append({"action_date": date_val, "action_type": "DIVIDEND", "dividend": div, "description": f"Dividend: {div}", "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
            split = self.safe_float(row.get("Stock Splits"))
            if split is not None and split > 0:
                results.append({"action_date": date_val, "action_type": "SPLIT", "split_ratio": split, "description": f"Stock Split: {split}", "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
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
                if "insider" in desc: insider_held = val
                elif "institutions" in desc: inst_held = val

        inst = self.safe_dataframe(shareholders_dict.get("institutional"))
        if inst is not None and not inst.empty:
            for _, row in inst.iterrows():
                row_data = {
                    "quarter": self.safe_date(row.get("Date Reported")),
                    "insider_holding": insider_held,
                    "others_holding": inst_held,
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                results.append(row_data)
                
        if not results and (insider_held is not None or inst_held is not None):
            results.append({
                "quarter": datetime.now().strftime("%Y-%m-%d"),
                "insider_holding": insider_held,
                "institutional_holding": inst_held,
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
        if df_safe is None: return []
        if df_safe.index.name == 'Earnings Date' or isinstance(df_safe.index, pd.DatetimeIndex): df_safe = df_safe.reset_index()
        results = []
        for _, row in df_safe.iterrows():
            dt = pd.to_datetime(self._extract_scalar(row.get("Earnings Date", row.get("index"))))
            quarter_str = f"{dt.year}-Q{(dt.month-1)//3 + 1}" if not pd.isna(dt) else None
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
        if df_safe is None: return []
        if df_safe.index.name == 'period' or isinstance(df_safe.index, pd.Index): df_safe = df_safe.reset_index()
        results = []
        for _, row in df_safe.iterrows():
            results.append({"updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        return results

    def normalize_calendar(self, calendar: Dict[str, Any]) -> Dict[str, Any]:
        if not calendar: return {}
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
            if not isinstance(item, dict): continue
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
        if df_safe is None: return {}
        results = {}
        for idx, row in df_safe.iterrows():
            if "Value" in row: results[self.safe_str(idx)] = self.safe_float(row["Value"])
        results["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return results

def _debug_provider(self):
    pass
