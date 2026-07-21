"""
GREEN BULL RIDER V6
Module: backend/repository/stock_repository.py

Repository Layer

Responsibilities
----------------
- Read Database
- Write Database
- No Business Logic
- No Provider Logic
- No AI Logic

Python 3.13 Compatible
"""

from __future__ import annotations

import logging

import pandas as pd

from backend.db.connection import db

logger = logging.getLogger(__name__)


class StockRepository:

    # =====================================================================
    # Stock Master
    # =====================================================================

    def get_active_symbols(self) -> list[dict]:

        rows = db.fetchall(
            """
            SELECT
                symbol,
                company_name,
                exchange
            FROM stock_master
            WHERE UPPER(status)='ACTIVE'
            ORDER BY symbol
            """
        )

        return [dict(row) for row in rows]

    def get_symbol(
        self,
        symbol: str,
    ) -> dict | None:

        row = db.fetchone(
            """
            SELECT *
            FROM stock_master
            WHERE symbol=?
            """,
            (
                symbol.upper(),
            ),
        )

        if row is None:
            return None

        return dict(row)

    def symbol_exists(
        self,
        symbol: str,
    ) -> bool:

        row = db.fetchone(
            """
            SELECT 1
            FROM stock_master
            WHERE symbol=?
            LIMIT 1
            """,
            (
                symbol.upper(),
            ),
        )

        return row is not None

    # =====================================================================
    # Historical Data
    # =====================================================================

    def get_last_history_date(
        self,
        symbol: str,
    ):

        row = db.fetchone(
            """
            SELECT
                MAX(date) AS last_date
            FROM historical_data
            WHERE symbol=?
            """,
            (
                symbol.upper(),
            ),
        )

        if row is None:
            return None

        return row["last_date"]

    def save_history(
        self,
        dataframe: pd.DataFrame,
    ) -> None:

        if dataframe.empty:
            return

        df = dataframe.copy()

        # Fix: Reset index if date is not in columns
        if "date" not in [c.lower() for c in df.columns]:
            df = df.reset_index()
            
        # Fix: Normalize column names to lowercase
        df.columns = [c.lower() for c in df.columns]

        # Fix: Ensure date format
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.strftime('%Y-%m-%d %H:%M:%S')

        rows = df.to_dict("records")

        self.bulk_insert(
            "historical_data",
            rows,
        )

    # =====================================================================
    # Company Profile
    # =====================================================================

    def get_company_profile(
        self,
        symbol: str,
    ) -> dict | None:

        row = db.fetchone(
            """
            SELECT *
            FROM company_profile
            WHERE symbol=?
            """,
            (
                symbol.upper(),
            ),
        )

        if row is None:
            return None

        return dict(row)

    def save_company_profile(
        self,
        data: dict,
    ) -> None:

        columns = ",".join(data.keys())

        placeholders = ",".join(
            "?"
            for _ in data
        )

        db.execute(
            f"""
            INSERT OR REPLACE
            INTO company_profile
            ({columns})
            VALUES
            ({placeholders})
            """,
            tuple(data.values()),
        )

    # =====================================================================
    # Fundamental
    # =====================================================================

    def get_fundamental(
        self,
        symbol: str,
    ) -> dict | None:

        row = db.fetchone(
            """
            SELECT *
            FROM fundamental_data
            WHERE symbol=?
            """,
            (
                symbol.upper(),
            ),
        )

        if row is None:
            return None

        return dict(row)

    def save_fundamental(
        self,
        data: dict,
    ) -> None:

        allowed = {
            "symbol",
            "market_cap",
            "pe",
            "pb",
            "roe",
            "roce",
            "debt_equity",
            "sales_growth",
            "profit_growth",
            "promoter_holding",
            "institutional_holding",
            "fii_holding",
            "dii_holding",
            "updated_at",
            "dividend_yield",
            "sector",
            "industry",
            "eps",
            "book_value",
            "current_ratio",
            "quick_ratio",
            "operating_margin",
            "net_margin",
            "cash",
            "free_cash_flow",
            "enterprise_value",
            "beta",
            "week52_high",
            "week52_low",
            "target_price",
            "recommendation",
            "shares_outstanding",
        }

        data = {k: v for k, v in data.items() if k in allowed}

        columns = ",".join(data.keys())
        placeholders = ",".join("?" for _ in data)

        db.execute(
            f"""
            INSERT OR REPLACE
            INTO fundamental_data
            ({columns})
            VALUES
            ({placeholders})
            """,
            tuple(data.values()),
        )

    # =====================================================================
    # Generic Bulk
    # =====================================================================

    def bulk_insert(
        self,
        table: str,
        rows: list[dict],
    ) -> None:

        if not rows:
            return

        columns = list(rows[0].keys())

        sql = f"""
        INSERT OR REPLACE
        INTO {table}
        ({",".join(columns)})
        VALUES
        ({",".join(["?"] * len(columns))})
        """

        values = [
            tuple(
                row.get(col)
                for col in columns
            )
            for row in rows
        ]

        db.executemany(
            sql,
            values,
        )



    # =====================================================================
    # Financial Data
    # =====================================================================

    def save_financials(self, rows: list[dict]) -> None:
        self.bulk_insert("financial_data", rows)

    # =====================================================================
    # Corporate Actions
    # =====================================================================

    def save_corporate_actions(self, rows: list[dict]) -> None:
        self.bulk_insert("corporate_actions", rows)

    # =====================================================================
    # Shareholding
    # =====================================================================

    def save_shareholding(self, rows) -> None:
        if rows is None:
            return

        if isinstance(rows, dict):
            rows = [rows]

        if not isinstance(rows, list):
            rows = list(rows)

        if not rows:
            return

        self.bulk_insert("shareholding_data", rows)

    # =====================================================================
    # Earnings
    # =====================================================================

    def save_earnings(self, rows: list[dict]) -> None:
        self.bulk_insert("earnings_history", rows)

    # =====================================================================
    # Analyst Data
    # =====================================================================

    def save_analyst_data(self, data: dict) -> None:

        columns = ",".join(data.keys())
        placeholders = ",".join("?" for _ in data)

        db.execute(
            f"""
            INSERT OR REPLACE
            INTO analyst_data
            ({columns})
            VALUES
            ({placeholders})
            """,
            tuple(data.values()),
        )

repository = StockRepository()
