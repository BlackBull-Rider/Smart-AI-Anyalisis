"""
GREEN BULL AI
Institutional Indicator Pipeline V2

Part 1
"""

from __future__ import annotations

import logging
import math
import sqlite3
import time

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
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


class IndicatorPipelineV2:

    FEATURE_TABLE = "feature_history"
    HISTORY_TABLE = "historical_data"

    LOOKBACK = 400
    BATCH_SIZE = 100
    MAX_WORKERS = 8

    def __init__(self):

        self.conn = sqlite3.connect(
            DB_PATH,
            check_same_thread=False,
        )

        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA temp_store=MEMORY")

        self.stats = {
            "processed": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "rows": 0,
            "started": 0.0,
            "finished": 0.0,
        }

    # =====================================================
    # SYMBOL LIST
    # =====================================================

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

    # =====================================================
    # LAST FEATURE DATE CACHE
    # =====================================================

    def build_feature_cache(self) -> dict[str, str]:

        df = pd.read_sql_query(
            f"""
            SELECT
                symbol,
                MAX(date) AS last_date
            FROM {self.FEATURE_TABLE}
            GROUP BY symbol
            """,
            self.conn,
        )

        cache = {}

        for _, row in df.iterrows():
            cache[row["symbol"]] = row["last_date"]

        return cache

    # =====================================================
    # MISSING CANDLE COUNT
    # =====================================================

    def missing_count(
        self,
        symbol: str,
        last_date,
    ) -> int:

        if last_date is None:
            return self.LOOKBACK

        row = self.conn.execute(
            """
            SELECT COUNT(*)
            FROM historical_data
            WHERE symbol=?
            AND date>?
            """,
            (
                symbol,
                last_date,
            ),
        ).fetchone()

        return int(row[0])

    # =====================================================
    # LOAD OHLCV
    # =====================================================

    def load_ohlcv(
        self,
        symbol: str,
        limit: int,
    ) -> pd.DataFrame:

        df = pd.read_sql_query(
            f"""
            SELECT
                date,
                open,
                high,
                low,
                close,
                volume
            FROM {self.HISTORY_TABLE}
            WHERE symbol=?
            ORDER BY date DESC
            LIMIT ?
            """,
            self.conn,
            params=(
                symbol,
                limit,
            ),
            parse_dates=[
                "date",
            ],
        )

        if df.empty:
            return df

        df = df.sort_values("date")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date", drop=False)

        return df

    # =====================================================
    # BUILD JOB
    # =====================================================

    def build_job(
        self,
        symbol: str,
        cache: dict,
    ):

        last_date = cache.get(symbol)

        missing = self.missing_count(
            symbol,
            last_date,
        )

        if missing == 0:
            return None

        limit = max(
            self.LOOKBACK,
            self.LOOKBACK - 1 + missing,
        )

        df = self.load_ohlcv(
            symbol,
            limit,
        )

        if df.empty:
            return None

        return {
            "symbol": symbol,
            "last_date": last_date,
            "missing": missing,
            "df": df,
        }

    # =====================================================
    # BUILD BATCH
    # =====================================================

    def build_batch(
        self,
        symbols,
        cache,
    ):

        jobs = []

        for symbol in symbols:

            job = self.build_job(
                symbol,
                cache,
            )

            if job is not None:
                jobs.append(job)

        return jobs

    # =====================================================
    # SPLIT BATCH
    # =====================================================

    def split_batches(
        self,
        symbols,
    ):

        total = len(symbols)

        for start in range(
            0,
            total,
            self.BATCH_SIZE,
        ):

            yield symbols[
                start:
                start + self.BATCH_SIZE
            ]


    # =====================================================
    # WORKER
    # =====================================================

    def worker(
        self,
        job,
    ):

        symbol = job["symbol"]
        df = job["df"]
        missing = job["missing"]

        started = time.perf_counter()

        try:

            features = indicator_engine(
                symbol,
                df,
            )

            if features is None:
                return None

            if features.empty:
                return None

            if missing < len(features):
                features = features.tail(missing)

            features = (
                features
                .replace(
                    [float("inf"), float("-inf")],
                    pd.NA,
                )
                .fillna(pd.NA)
            )

            valid = int(features.count().sum())
            nulls = int(features.isna().sum().sum())

            last_date = "-"

            if (
                "date" in features.columns
                and not features.empty
            ):
                last_date = str(
                    features["date"].iloc[-1]
                )[:10]

            return {
                "symbol": symbol,
                "rows": len(features),
                "date": last_date,
                "features": features,
                "feature_count": len(features.columns),
                "candles": len(features),
                "values": valid,
                "nulls": nulls,
                "elapsed": (
                    time.perf_counter()
                    - started
                ),
            }

        except Exception:

            logger.exception(
                "Worker Failed : %s",
                symbol,
            )

            return {
                "symbol": symbol,
                "rows": 0,
                "failed": True,
            }

    # =====================================================
    # EXECUTE BATCH
    # =====================================================

    def execute_batch(
        self,
        jobs,
    ):

        results = []

        with ProcessPoolExecutor(
            max_workers=self.MAX_WORKERS,
        ) as executor:

            futures = {
                executor.submit(
                    self.worker,
                    job,
                ): job
                for job in jobs
            }

            for future in as_completed(
                futures,
            ):

                result = future.result()

                if result is None:
                    continue

                results.append(
                    result,
                )

        return results

    # =====================================================
    # SAVE
    # =====================================================

    def save_batch(
        self,
        results,
    ):

        frames = []

        for item in results:

            if item.get("rows", 0) == 0:
                continue

            frames.append(
                item["features"]
            )

        if not frames:
            return 0

        df = pd.concat(
            frames,
            ignore_index=True,
        )

        df.to_sql(
            self.FEATURE_TABLE,
            self.conn,
            if_exists="append",
            index=False,
        )

        self.conn.commit()

        return len(df)


    # =====================================================
    # PROGRESS
    # =====================================================

    def progress(
        self,
        batch_no: int,
        total_batch: int,
        results,
        rows: int,
        elapsed: float,
    ):

        print()
        print("=" * 80)
        print(
            f"BATCH {batch_no}/{total_batch}"
        )
        print("=" * 80)

        for item in sorted(
            results,
            key=lambda x: x["symbol"],
        ):

            if item.get("failed"):

                print(
                    f"{item['symbol']:<15} FAILED"
                )
                continue

            print(
                f"{item['symbol']:<15} "
                f"{item['rows']:>3} "
                f"rows "
                f"{item['elapsed']:.2f}s"
            )

            print(
                f"    Date     : {item['date']}"
            )

            print(
                f"    Features : {item['feature_count']}"
            )

            print(
                f"    Candles  : {item['candles']}"
            )

            print(
                f"    Values   : {item['values']}"
            )

            print(
                f"    Null     : {item['nulls']}"
            )

        print("-" * 80)

        print(
            f"Rows Saved : {rows}"
        )

        print(
            f"Batch Time : {elapsed:.2f}s"
        )

        print("=" * 80)

    # =====================================================
    # RUN
    # =====================================================

    def run(self):

        logger.info(
            "=" * 80
        )

        logger.info(
            "INDICATOR PIPELINE V2"
        )

        logger.info(
            "=" * 80
        )

        self.stats["started"] = (
            time.perf_counter()
        )

        symbols = self.symbols()

        cache = (
            self.build_feature_cache()
        )

        batches = list(
            self.split_batches(
                symbols,
            )
        )

        total_batch = len(
            batches
        )

        for batch_no, batch_symbols in enumerate(
            batches,
            start=1,
        ):

            started = (
                time.perf_counter()
            )

            jobs = self.build_batch(
                batch_symbols,
                cache,
            )

            if not jobs:

                continue

            results = self.execute_batch(
                jobs,
            )

            rows = self.save_batch(
                results,
            )

            self.stats["rows"] += rows

            self.stats["processed"] += len(
                batch_symbols
            )

            self.stats["success"] += len(
                [
                    r
                    for r in results
                    if not r.get(
                        "failed",
                        False,
                    )
                ]
            )

            self.stats["failed"] += len(
                [
                    r
                    for r in results
                    if r.get(
                        "failed",
                        False,
                    )
                ]
            )

            self.progress(
                batch_no,
                total_batch,
                results,
                rows,
                time.perf_counter()
                - started,
            )

        self.stats["finished"] = (
            time.perf_counter()
        )

        logger.info(
            "=" * 80
        )

        logger.info(
            "SUMMARY"
        )

        logger.info(
            "Processed : %d",
            self.stats["processed"],
        )

        logger.info(
            "Success   : %d",
            self.stats["success"],
        )

        logger.info(
            "Failed    : %d",
            self.stats["failed"],
        )

        logger.info(
            "Rows      : %d",
            self.stats["rows"],
        )

        logger.info(
            "Elapsed   : %.2fs",
            self.stats["finished"]
            - self.stats["started"],
        )

        logger.info(
            "=" * 80
        )

        self.conn.close()


pipeline = IndicatorPipelineV2()


def run():

    pipeline.run()


if __name__ == "__main__":

    run()

