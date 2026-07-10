import sqlite3
from pathlib import Path

import pandas as pd

from backend.indicators.indicator_engine import run

DB_PATH = Path.home() / "Green-Bull-Data-Engine" / "database" / "market.db"


def main():

    print("=" * 100)
    print("RUNNING MOVING AVERAGE PIPELINE")
    print("=" * 100)

    features = run("RELIANCE")

    print(features.tail())

    conn = sqlite3.connect(DB_PATH)

    features.to_sql(
        "feature_history",
        conn,
        if_exists="append",
        index=False,
    )

    conn.commit()

    total = conn.execute(
        """
        SELECT COUNT(*)
        FROM feature_history
        WHERE symbol='RELIANCE'
        """
    ).fetchone()[0]

    print()

    print("=" * 100)
    print("TOTAL ROWS :", total)
    print("=" * 100)

    verify = pd.read_sql_query(
        """
        SELECT *
        FROM feature_history
        WHERE symbol='RELIANCE'
        ORDER BY date DESC
        LIMIT 5
        """,
        conn,
    )

    print()

    print(verify)

    conn.close()


if __name__ == "__main__":

    main()

