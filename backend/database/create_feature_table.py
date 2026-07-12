"""
GREEN BULL AI

Auto Feature Table Builder
Creates feature_history from indicator engine columns.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from backend.indicators.indicator_engine import run

DB_PATH = Path.home() / "Green-Bull-Data-Engine" / "database" / "market.db"


def sqlite_type(dtype) -> str:

    s = str(dtype).lower()

    if "int" in s:
        return "INTEGER"

    if "float" in s:
        return "REAL"

    if "bool" in s:
        return "INTEGER"

    if "datetime" in s:
        return "TEXT"

    return "TEXT"


def main():

    print("=" * 80)
    print("Generating Feature Table Schema...")
    print("=" * 80)

    df = run("RELIANCE")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS feature_history")

    cols = []

    cols.append("symbol TEXT NOT NULL")
    cols.append("date TEXT NOT NULL")

    for c in df.columns:

        if c in ("symbol", "date"):
            continue

        cols.append(
            f'"{c}" {sqlite_type(df[c].dtype)}'
        )

    cols.append("PRIMARY KEY(symbol,date)")

    sql = (
        "CREATE TABLE feature_history (\n"
        + ",\n".join(cols)
        + "\n);"
    )

    cur.execute(sql)

    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_feature_symbol
        ON feature_history(symbol)
        """
    )

    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_feature_date
        ON feature_history(date)
        """
    )

    conn.commit()

    total = cur.execute(
        """
        PRAGMA table_info(feature_history)
        """
    ).fetchall()

    print()
    print("TOTAL COLUMNS :", len(total))
    print("TABLE CREATED SUCCESSFULLY")

    conn.close()


if __name__ == "__main__":
    main()
