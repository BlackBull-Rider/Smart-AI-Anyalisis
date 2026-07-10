"""
Indicator Feature Writer
"""

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path.home() / "Green-Bull-Data-Engine" / "database" / "market.db"


class DataWriterError(Exception):
    pass


def write_features(
    table: str,
    df: pd.DataFrame,
) -> None:

    if df.empty:
        raise DataWriterError("Nothing to write.")

    conn = sqlite3.connect(DB_PATH)

    try:

        df.to_sql(
            table,
            conn,
            if_exists="append",
            index=False,
        )

        conn.commit()

        print(
            f"[OK] {len(df)} rows written -> {table}"
        )

    except Exception:

        conn.rollback()

        raise

    finally:

        conn.close()


if __name__ == "__main__":

    from backend.indicators.indicator_engine import run

    features = run("RELIANCE")

    write_features(
        "feature_history",
        features,
    )

