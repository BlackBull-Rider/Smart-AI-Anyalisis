"""
GREEN BULL AI
Indicator Pipeline

Reads historical_data from GBDE
Calculates indicators
Writes only missing rows into feature_history
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

import pandas as pd

from backend.indicators.indicator_engine import run as indicator_engine

logger = logging.getLogger(__name__)

DB_PATH = (
    Path.home()
    / "Green-Bull-Data-Engine"
    / "database"
    / "market.db"
)


class IndicatorPipeline:

    TABLE = "feature_history"

    def __init__(self):

        self.stats = {
            "processed": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "rows": 0,
            "started": 0.0,
            "finished": 0.0,
        }

        self.conn = sqlite3.connect(DB_PATH)

    # ==========================================================
    # SYMBOLS
    # ==========================================================

    def symbols(self) -> list[str]:

        df = pd.read_sql_query(
            """
            SELECT DISTINCT symbol
            FROM historical_data
            ORDER BY symbol
            """,
            self.conn,
        )

        return df["symbol"].tolist()

    # ==========================================================
    # LAST DATE
    # ==========================================================

    def last_date(
        self,
        symbol: str,
    ):

        row = self.conn.execute(
            f"""
            SELECT MAX(date)
            FROM {self.TABLE}
            WHERE symbol=?
            """,
            (symbol,),
        ).fetchone()

        if row is None:
            return None

        return row[0]

    # ==========================================================
    # FILTER
    # ==========================================================

    def filter_missing(
        self,
        df: pd.DataFrame,
        last_date,
    ) -> pd.DataFrame:

        if last_date is None:
            return df

        return df[
            pd.to_datetime(df["date"])
            > pd.to_datetime(last_date)
        ]

    # ==========================================================
    # SAVE
    # ==========================================================

    def save(
        self,
        df: pd.DataFrame,
    ) -> int:

        if df.empty:
            return 0

        df.to_sql(
            self.TABLE,
            self.conn,
            if_exists="append",
            index=False,
        )

        return len(df)

    # ==========================================================
    # PROGRESS
    # ==========================================================

    def progress(
        self,
        index: int,
        total: int,
        symbol: str,
        rows: int,
        elapsed: float,
    ) -> None:

        print(
            f"[{index:04d}/{total}] "
            f"{symbol:<15} "
            f"+{rows:4d} rows "
            f"{elapsed:.2f}s",
            flush=True,
        )

    # ==========================================================
    # PROCESS SYMBOL
    # ==========================================================

    def process_symbol(
        self,
        symbol: str,
    ) -> int:

        features = indicator_engine(symbol)

        if features is None:
            return 0

        if features.empty:
            return 0

        last = self.last_date(symbol)

        features = self.filter_missing(
            features,
            last,
        )

        if features.empty:
            return 0

        features = features.replace(
            [float("inf"), float("-inf")],
            pd.NA,
        )

        features = features.fillna(pd.NA)

        rows = self.save(features)

        last_date = "-"
        if "date" in features.columns and not features.empty:
            last_date = str(features["date"].iloc[-1])[:10]

        print(
            f"    Symbol  : {symbol}\n"
            f"    Date    : {last_date}\n"
            f"    Features: {len(features.columns)}\n"
            f"    Candles : {len(features)}",
            flush=True,
        )

        return rows


    # ==========================================================
    # RUN
    # ==========================================================

    def run(self) -> None:

        logger.info("=" * 80)
        logger.info("INDICATOR PIPELINE STARTED")
        logger.info("=" * 80)

        self.stats["started"] = time.perf_counter()

        symbols = self.symbols()

        total = len(symbols)

        for index, symbol in enumerate(
            symbols,
            start=1,
        ):

            started = time.perf_counter()

            try:

                rows = self.process_symbol(symbol)

                self.stats["processed"] += 1

                if rows == 0:

                    self.stats["skipped"] += 1

                else:

                    self.stats["success"] += 1
                    self.stats["rows"] += rows

            except Exception:

                logger.exception(
                    "Indicator Pipeline Failed : %s",
                    symbol,
                )

                self.stats["processed"] += 1
                self.stats["failed"] += 1

                rows = 0

            self.progress(
                index,
                total,
                symbol,
                rows,
                time.perf_counter() - started,
            )

        self.conn.commit()

        self.stats["finished"] = time.perf_counter()

        elapsed = (
            self.stats["finished"]
            - self.stats["started"]
        )

        logger.info("=" * 80)
        logger.info("INDICATOR PIPELINE SUMMARY")
        logger.info("=" * 80)
        logger.info("Processed : %d", self.stats["processed"])
        logger.info("Success   : %d", self.stats["success"])
        logger.info("Skipped   : %d", self.stats["skipped"])
        logger.info("Failed    : %d", self.stats["failed"])
        logger.info("Rows      : %d", self.stats["rows"])
        logger.info("Elapsed   : %.2fs", elapsed)
        logger.info("=" * 80)

        self.conn.close()


pipeline = IndicatorPipeline()


def run() -> None:

    pipeline.run()


if __name__ == "__main__":

    run()

