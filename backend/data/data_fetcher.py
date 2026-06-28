"""
backend/data/data_fetcher.py

Green Bull Rider Data Fetcher
Raw data access layer for Green Bull Data Engine.
"""

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path.home() / "Green-Bull-Data-Engine" / "database" / "market.db"


class DataFetcherError(Exception):
    pass


def _connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise DataFetcherError(f"Database not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_ohlcv(symbol: str, limit: int = 500) -> pd.DataFrame:
    """Fetch historical OHLCV."""

    query = """
    SELECT
        date,
        open,
        high,
        low,
        close,
        volume
    FROM historical_data
    WHERE symbol = ?
    ORDER BY date ASC
    LIMIT ?
    """

    with _connect() as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params=(symbol.upper(), limit),
            parse_dates=["date"]
        )

    if df.empty:
        raise DataFetcherError(f"No OHLCV data found for {symbol}")

    df.set_index("date", inplace=True)
    return df


def fetch_fundamental(symbol: str) -> pd.Series:
    """Fetch fundamental data."""

    query = """
    SELECT *
    FROM fundamental_data
    WHERE symbol = ?
    LIMIT 1
    """

    with _connect() as conn:
        df = pd.read_sql_query(query, conn, params=(symbol.upper(),))

    if df.empty:
        raise DataFetcherError(f"No fundamental data found for {symbol}")

    return df.iloc[0]


def fetch_ipo(symbol: str) -> pd.Series:
    """Fetch IPO information."""

    query = """
    SELECT *
    FROM ipo_data
    WHERE symbol = ?
    LIMIT 1
    """

    with _connect() as conn:
        df = pd.read_sql_query(query, conn, params=(symbol.upper(),))

    if df.empty:
        raise DataFetcherError(f"No IPO data found for {symbol}")

    return df.iloc[0]


def fetch_stock_master(symbol: str) -> pd.Series:
    """Fetch stock master information."""

    query = """
    SELECT *
    FROM stock_master
    WHERE symbol = ?
    LIMIT 1
    """

    with _connect() as conn:
        df = pd.read_sql_query(query, conn, params=(symbol.upper(),))

    if df.empty:
        raise DataFetcherError(f"No stock master data found for {symbol}")

    return df.iloc[0]
