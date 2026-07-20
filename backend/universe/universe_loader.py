"""
GREEN BULL RIDER V6
Module: backend/universe/universe_loader.py

Synchronizes NSE Universe -> stock_master
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, UTC

import pandas as pd

from backend.providers.nse_provider import NSEProvider


class UniverseLoader:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.provider = NSEProvider()

    def refresh(self) -> int:
        df = self.provider.get_universe()

        if df.empty:
            return 0

        now = datetime.now(UTC).isoformat(timespec="seconds")

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        try:
            # Mark everything inactive first
            cur.execute(
                """
                UPDATE stock_master
                SET status='INACTIVE',
                    updated_at=?
                """,
                (now,),
            )

            # Upsert active universe
            rows = [
                (
                    str(r.symbol).strip(),
                    str(r.company_name).strip(),
                    "NSE",
                    now,
                )
                for r in df.itertuples(index=False)
            ]

            cur.executemany(
                """
                INSERT INTO stock_master
                (
                    symbol,
                    company_name,
                    exchange,
                    status,
                    updated_at
                )
                VALUES
                (
                    ?,
                    ?,
                    ?,
                    'ACTIVE',
                    ?
                )
                ON CONFLICT(symbol)
                DO UPDATE SET
                    company_name=excluded.company_name,
                    exchange=excluded.exchange,
                    status='ACTIVE',
                    updated_at=excluded.updated_at;
                """,
                rows,
            )

            conn.commit()
            return len(rows)

        finally:
            conn.close()
