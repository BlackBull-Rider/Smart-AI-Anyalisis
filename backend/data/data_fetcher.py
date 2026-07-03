import os
import sqlite3
import pandas as pd
from pathlib import Path

# Config-driven DB Path with default fallback
DEFAULT_DB_PATH = Path.home() / "Green-Bull-Data-Engine" / "database" / "market.db"
DB_PATH = Path(os.getenv("GREEN_BULL_DB_PATH", DEFAULT_DB_PATH))

class DataFetcherError(Exception):
    pass

def _connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise DataFetcherError(f"Database not found at: {DB_PATH}")

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        raise DataFetcherError(f"Database connection failed: {e}")

def fetch_ohlcv(symbol: str, limit: int = 500) -> pd.DataFrame:
    """
    Fetch the LATEST historical OHLCV data.
    Uses subquery to get the most recent rows first, then sorts chronologically.
    """
    query = """
    SELECT * FROM (
        SELECT
            date,
            open,
            high,
            low,
            close,
            volume
        FROM historical_data
        WHERE symbol = ?
        ORDER BY date DESC
        LIMIT ?
    )
    ORDER BY date ASC
    """

    try:
        with _connect() as conn:
            df = pd.read_sql_query(
                query,
                conn,
                params=(symbol.upper(), limit),
                parse_dates=["date"]
            )
    except sqlite3.Error as e:
        raise DataFetcherError(f"Failed to fetch OHLCV for {symbol}: {e}")

    if df.empty:
        raise DataFetcherError(f"No OHLCV data found for {symbol}")

    df.set_index("date", inplace=True)
    return df

def fetch_fundamental(symbol: str) -> pd.Series:
    """Fetch the latest fundamental data."""
    query = """
    SELECT *
    FROM fundamental_data
    WHERE symbol = ?
    LIMIT 1
    """

    try:
        with _connect() as conn:
            df = pd.read_sql_query(query, conn, params=(symbol.upper(),))
    except sqlite3.Error as e:
        raise DataFetcherError(f"Failed to fetch fundamental data for {symbol}: {e}")

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

    try:
        with _connect() as conn:
            df = pd.read_sql_query(query, conn, params=(symbol.upper(),))
    except sqlite3.Error as e:
        raise DataFetcherError(f"Failed to fetch IPO data for {symbol}: {e}")

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

    try:
        with _connect() as conn:
            df = pd.read_sql_query(query, conn, params=(symbol.upper(),))
    except sqlite3.Error as e:
        raise DataFetcherError(f"Failed to fetch stock master data for {symbol}: {e}")

    if df.empty:
        raise DataFetcherError(f"No stock master data found for {symbol}")

    return df.iloc[0]
