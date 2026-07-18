"""
GREEN BULL RIDER V6
Module: backend/db/connection.py

Enterprise SQLite Connection Layer

Python 3.13 Compatible
"""

from __future__ import annotations

import logging
import sqlite3

from contextlib import contextmanager
from typing import Any
from typing import Iterator
from typing import Sequence

from backend.config.settings import settings

logger = logging.getLogger(__name__)


class Database:
    """
    Enterprise SQLite Database Layer.
    """

    def __init__(self) -> None:

        self._db_path = settings.database_path

    # =====================================================================
    # Connection
    # =====================================================================

    def connect(self) -> sqlite3.Connection:

        conn = sqlite3.connect(
            self._db_path,
            timeout=settings.database.busy_timeout_ms / 1000,
            check_same_thread=False,
        )

        conn.row_factory = sqlite3.Row

        conn.execute(
            f"PRAGMA journal_mode={settings.database.journal_mode.value};"
        )

        conn.execute(
            f"PRAGMA synchronous={settings.database.synchronous_mode.name};"
        )

        conn.execute(
            f"PRAGMA busy_timeout={settings.database.busy_timeout_ms};"
        )

        conn.execute(
            f"PRAGMA mmap_size={settings.database.mmap_size};"
        )

        conn.execute(
            f"PRAGMA foreign_keys={'ON' if settings.database.foreign_keys else 'OFF'};"
        )

        conn.execute(
            f"PRAGMA temp_store={'MEMORY' if settings.database.temp_store_memory else 'DEFAULT'};"
        )

        return conn

    @contextmanager
    def session(self) -> Iterator[sqlite3.Connection]:

        conn = self.connect()

        try:

            yield conn

            conn.commit()

        except Exception:

            conn.rollback()

            logger.exception(
                "Database transaction rolled back."
            )

            raise

        finally:

            conn.close()

    # =====================================================================
    # Execute
    # =====================================================================

    def execute(
        self,
        query: str,
        params: Sequence[Any] = (),
    ) -> None:

        with self.session() as conn:

            conn.execute(query, params)

    def executemany(
        self,
        query: str,
        rows: Sequence[Sequence[Any]],
    ) -> None:

        with self.session() as conn:

            conn.executemany(query, rows)

    # =====================================================================
    # Fetch
    # =====================================================================

    def fetchone(
        self,
        query: str,
        params: Sequence[Any] = (),
    ) -> sqlite3.Row | None:

        with self.session() as conn:

            return conn.execute(
                query,
                params,
            ).fetchone()

    def fetchall(
        self,
        query: str,
        params: Sequence[Any] = (),
    ) -> list[sqlite3.Row]:

        with self.session() as conn:

            return conn.execute(
                query,
                params,
            ).fetchall()

    # =====================================================================
    # Manual Transaction
    # =====================================================================

    def begin(self) -> sqlite3.Connection:

        conn = self.connect()

        conn.execute("BEGIN")

        return conn

    @staticmethod
    def commit(
        conn: sqlite3.Connection,
    ) -> None:

        conn.commit()

        conn.close()

    @staticmethod
    def rollback(
        conn: sqlite3.Connection,
    ) -> None:

        conn.rollback()

        conn.close()

    # =====================================================================
    # Utility
    # =====================================================================

    def table_exists(
        self,
        table_name: str,
    ) -> bool:

        row = self.fetchone(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            AND name=?
            """,
            (table_name,),
        )

        return row is not None

    def health_check(self) -> bool:

        try:

            row = self.fetchone(
                "PRAGMA integrity_check;"
            )

            return (
                row is not None
                and row[0] == "ok"
            )

        except Exception:

            logger.exception(
                "Database integrity check failed."
            )

            return False


db = Database()

