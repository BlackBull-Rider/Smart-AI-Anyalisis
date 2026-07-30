import os
content = r'''"""
GREEN BULL RIDER V6
Module: backend/providers/yahoo_provider.py
Yahoo Finance Market Data Provider - DB Synced
"""
import logging, time, math, random, threading
from datetime import datetime, date, timedelta
from typing import Any, Callable, Dict, List, Optional
import pandas as pd
import yfinance as yf
from backend.config.settings import settings
from backend.providers.base_provider import BaseProvider

class YahooProvider(BaseProvider):
    def __init__(self) -> None:
        self.name = "YahooProvider"
        self.logger = logging.getLogger(self.__class__.__name__)
        self._cache = {}; self._cache_lock = threading.Lock(); self.cache_ttl = 3600
        self._tickers = {}
        self.stats = {"cache_hits": 0, "cache_misses": 0, "retries": 0, "api_calls": 0}

    def _symbol(self, symbol: str) -> str:
        clean = str(symbol).strip().upper().replace('.', '-')
        return clean if clean == "^NSEI" or "." in clean or "^" in clean else f"{clean}.NS"

    def _ticker(self, symbol: str) -> yf.Ticker:
        clean = self._symbol(symbol)
        with self._cache_lock:
            if clean not in self._tickers: self._tickers[clean] = yf.Ticker(clean)
            return self._tickers[clean]

    def _fetch_cached(self, symbol: str, key: str, fetch_func: Callable) -> Any:
        cache_key = f"{self._symbol(symbol)}_{key}"
        with self._cache_lock:
            item = self._cache.get(cache_key)
            if item and time.time() < item['expiry']: return item['data']
        data = fetch_func()
        if data is not None:
            with self._cache_lock: self._cache[cache_key] = {'data': data, 'expiry': time.time() + self.cache_ttl}
        return data

    def validate_history(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty: return pd.DataFrame()
        df = df.copy().reset_index()
        df.columns = [c.lower() for c in df.columns]
        if 'date' not in df.columns: return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df[['date', 'open', 'high', 'low', 'close', 'volume']].dropna().drop_duplicates().sort_values('date').tail(settings.sync.trading_candles)

    def get_history(self, symbol: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        def fetch():
            ticker = self._ticker(symbol)
            if end_date is None: e = datetime.now().strftime("%Y-%m-%d")
            else: e = end_date
            if start_date is None: s = (pd.to_datetime(e) - timedelta(days=settings.sync.history_days)).strftime("%Y-%m-%d")
            else: s = start_date
            df = ticker.history(start=s, end=e, auto_adjust=False)
            df = self.validate_history(df)
            if not df.empty:
                df['symbol'] = symbol
            return df
        return self._fetch_cached(symbol, f"history_{start_date or 'def'}_{end_date or 'def'}", fetch)

    def get_fundamentals(self, symbol: str) -> Dict[str, Any]:
        info = self._fetch_cached(symbol, "info", lambda: self._ticker(symbol).info) or {}
        return {
            "market_cap": info.get("marketCap"), "pe_ratio": info.get("trailingPE"),
            "roe": info.get("returnOnEquity"), "shares_outstanding": info.get("sharesOutstanding"),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def get_financials(self, symbol: str) -> List[Dict[str, Any]]:
        return []
'''
with open('backend/providers/yahoo_provider.py', 'w') as f: f.write(content)
print("File Updated!")
