"""
GREEN BULL AI
Indicator Pipeline V4 (Final Production Version)

Institutional Grade Multiprocessing Indicator Architecture
Optimized for 5000+ Stocks, Memory Efficiency, and SQLite WAL Concurrency
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Load the external indicator engine
from backend.indicators.indicator_engine import run as indicator_engine


@dataclass
class PipelineConfig:
    """Configuration for the Indicator Pipeline."""
    db_path: str = str(Path.home() / "Green-Bull-Data-Engine" / "database" / "market.db")
    history_table: str = "historical_data"
    feature_table: str = "feature_history"
    checkpoint_file: str = "indicator_checkpoint.txt"
    
    lookback: int = 400
    max_workers: int = os.cpu_count() or 4
    commit_size: int = 15000  # Strict buffer limit to prevent memory explosion
    retry_count: int = 3
    
    # DB Optimization Parameters
    pragma_journal_mode: str = "WAL"
    pragma_synchronous: str = "NORMAL"
    pragma_cache_size: int = -200000
    pragma_temp_store: str = "MEMORY"
    pragma_mmap_size: int = 268435456  # 256MB default for broad compatibility
    pragma_foreign_keys: str = "ON"
    pragma_busy_timeout: int = 60000


@dataclass
class Job:
    """Lightweight job definition to prevent pickling bottlenecks."""
    symbol: str
    missing_count: int
    limit: int
    db_path: str
    history_table: str
    retry_count: int


@dataclass
class JobResult:
    """Worker output returning standard Python primitives, no DataFrames."""
    symbol: str
    success: bool
    rows: List[Tuple] = field(default_factory=list)
    columns: Tuple[str, ...] = field(default_factory=tuple)
    error: str = ""
    elapsed: float = 0.0
    values_count: int = 0
    nulls_count: int = 0


# =============================================================================
# WORKER PROCESS
# =============================================================================

def _process_worker(job: Job) -> JobResult:
    """
    Isolated worker function.
    Workers establish their own read-only DB connection to avoid main-thread locking,
    loading only the strictly required OHLCV data.
    """
    started = time.perf_counter()
    
    for attempt in range(1, job.retry_count + 1):
        try:
            # 1. Open Read-Only Database Connection
            db_uri = f"file:{job.db_path}?mode=ro"
            with sqlite3.connect(db_uri, uri=True, timeout=30.0) as conn:
                query = f"""
                    SELECT * FROM (
                        SELECT date, open, high, low, close, volume 
                        FROM {job.history_table} 
                        WHERE symbol=? 
                        ORDER BY date DESC 
                        LIMIT ?
                    ) ORDER BY date ASC
                """
                df = pd.read_sql_query(
                    query, 
                    conn, 
                    params=(job.symbol, job.limit), 
                    parse_dates=["date"]
                )
            
            if df.empty or len(df) < 5:
                return JobResult(symbol=job.symbol, success=True, elapsed=time.perf_counter() - started)
                
            df = df.set_index("date", drop=False)
            
            # 2. Execute Engine
            features = indicator_engine(job.symbol, df)
            
            if features is None or features.empty:
                return JobResult(symbol=job.symbol, success=True, elapsed=time.perf_counter() - started)
                
            # 3. Filter to missing rows only
            if job.missing_count < len(features):
                features = features.tail(job.missing_count).copy()
                
            # 4. Normalize & Clean
            if "symbol" not in features.columns:
                features.insert(0, "symbol", job.symbol)
                
            if "date" not in features.columns:
                features = features.reset_index()
                if "date" not in features.columns and "index" in features.columns:
                    features.rename(columns={"index": "date"}, inplace=True)
                    
            if "date" not in features.columns:
                return JobResult(
                    symbol=job.symbol, 
                    success=False, 
                    error="Indicator engine dropped the date column.", 
                    elapsed=time.perf_counter() - started
                )
                
            features["date"] = features["date"].astype(str).str[:10]
            
            # Convert NaN/Inf safely for SQLite without turning everything into Object arrays
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.where(pd.notnull(features), None)
            
            values_count = features.count().sum()
            nulls_count = features.isna().sum().sum()
            
            # 5. Extract Native Tuples (Faster and safer than to_numpy for mixed types)
            columns = tuple(features.columns)
            rows = list(features.itertuples(index=False, name=None))
            
            return JobResult(
                symbol=job.symbol,
                success=True,
                rows=rows,
                columns=columns,
                elapsed=time.perf_counter() - started,
                values_count=int(values_count),
                nulls_count=int(nulls_count)
            )
            
        except Exception as e:
            if attempt == job.retry_count:
                error_trace = traceback.format_exc()
                return JobResult(
                    symbol=job.symbol,
                    success=False,
                    error=f"{str(e)}\n{error_trace}",
                    elapsed=time.perf_counter() - started
                )
            time.sleep(1.0 * attempt)
            
    return JobResult(symbol=job.symbol, success=False, elapsed=time.perf_counter() - started)


# =============================================================================
# TERMINAL DISPLAY ENGINE
# =============================================================================

class ProgressDisplay:
    """Robust, cross-platform terminal display using carriage returns."""
    
    def __init__(self, total_symbols: int):
        self.total = total_symbols
        self.start_time = time.perf_counter()
        self.processed = 0
        self.success = 0
        self.failed = 0
        self.rows_inserted = 0
        self.last_print = 0.0
        self.update_interval = 0.5
        
        self.logger = logging.getLogger("IndicatorPipelineV4")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler("indicator_pipeline.log", mode="a")
            formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
        print("=" * 80)
        print(" GREEN BULL AI - INSTITUTIONAL INDICATOR PIPELINE V4")
        print("=" * 80)

    def log_error(self, symbol: str, error: str):
        self.logger.error(f"Failed [{symbol}]:\n{error}")

    def update(self, symbol: str, rows_added: int, success: bool):
        self.processed += 1
        if success:
            self.success += 1
        else:
            self.failed += 1
            
        self.rows_inserted += rows_added
        now = time.perf_counter()
        
        if now - self.last_print >= self.update_interval or self.processed == self.total:
            self.last_print = now
            elapsed = now - self.start_time
            rate = self.processed / elapsed if elapsed > 0 else 0
            eta = (self.total - self.processed) / rate if rate > 0 else 0
            pct = (self.processed / max(1, self.total) * 100)
            
            sys.stdout.write(
                f"\r\033[K[PROCESSING] "
                f"Symbols: {self.processed}/{self.total} ({pct:.1f}%) | "
                f"Rows: {self.rows_inserted:,} | "
                f"Speed: {rate:.1f}/s | "
                f"ETA: {int(eta)}s | "
                f"Failed: {self.failed} | "
                f"Current: {symbol:8s}"
            )
            sys.stdout.flush()

    def finish(self):
        elapsed = time.perf_counter() - self.start_time
        sys.stdout.write("\n")
        print("=" * 80)
        print(" PIPELINE COMPLETED")
        print(f" Total Symbols : {self.total}")
        print(f" Successful    : {self.success}")
        print(f" Failed        : {self.failed}")
        print(f" Rows Saved    : {self.rows_inserted:,}")
        print(f" Total Time    : {elapsed:.2f} seconds")
        print("=" * 80)
        
        summary = {
            "total_symbols": self.total,
            "successful": self.success,
            "failed": self.failed,
            "rows_saved": self.rows_inserted,
            "elapsed_seconds": round(elapsed, 2),
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            with open("pipeline_summary.json", "w") as f:
                json.dump(summary, f, indent=4)
        except Exception as e:
            self.logger.error(f"Failed to write summary JSON: {e}")


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================

class IndicatorPipelineV4:
    """
    Main orchestration class.
    Handles central writer DB connection, worker dispatch, buffering, and prepared statements.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        db_path_obj = Path(self.config.db_path)
        db_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Single writer connection
        self.conn = sqlite3.connect(
            self.config.db_path,
            isolation_level=None,
            timeout=self.config.pragma_busy_timeout / 1000.0
        )
        self._apply_pragmas()
        
        # State Management
        self.buffer: Dict[Tuple[str, ...], List[Tuple]] = defaultdict(list)
        self.buffer_size = 0
        self.pending_checkpoints: List[str] = []
        self._sql_cache: Dict[Tuple[str, ...], str] = {}
        self._feature_cache: Dict[str, str] = {}

    def _apply_pragmas(self):
        pragmas = [
            f"PRAGMA journal_mode={self.config.pragma_journal_mode};",
            f"PRAGMA synchronous={self.config.pragma_synchronous};",
            f"PRAGMA cache_size={self.config.pragma_cache_size};",
            f"PRAGMA temp_store={self.config.pragma_temp_store};",
            f"PRAGMA mmap_size={self.config.pragma_mmap_size};",
            f"PRAGMA foreign_keys={self.config.pragma_foreign_keys};",
            f"PRAGMA busy_timeout={self.config.pragma_busy_timeout};"
        ]
        for p in pragmas:
            self.conn.execute(p)

    def _get_symbols(self) -> List[str]:
        query = f"SELECT DISTINCT symbol FROM {self.config.history_table} ORDER BY symbol"
        return [row[0] for row in self.conn.execute(query).fetchall()]

    def _load_checkpoint(self) -> set:
        if os.path.exists(self.config.checkpoint_file):
            try:
                with open(self.config.checkpoint_file, "r") as f:
                    return set(line.strip() for line in f if line.strip())
            except Exception:
                return set()
        return set()

    def _load_feature_cache(self):
        """Loads the maximum date computed per symbol to determine missing candles."""
        try:
            query = f"SELECT symbol, MAX(date) FROM {self.config.feature_table} GROUP BY symbol"
            self._feature_cache = {row[0]: row[1] for row in self.conn.execute(query).fetchall()}
        except sqlite3.OperationalError:
            self._feature_cache = {}

    def _get_missing_count(self, symbol: str) -> int:
        last_date = self._feature_cache.get(symbol)
        if not last_date:
            return self.config.lookback
        query = f"SELECT COUNT(*) FROM {self.config.history_table} WHERE symbol=? AND date>?"
        row = self.conn.execute(query, (symbol, last_date)).fetchone()
        return int(row[0]) if row else 0

    def _get_upsert_query(self, columns: Tuple[str, ...]) -> str:
        """Prepared statement cache for robust dynamic UPSERTs."""
        if columns not in self._sql_cache:
            cols_csv = ", ".join(columns)
            placeholders = ", ".join(["?"] * len(columns))
            updates = [f"{col}=excluded.{col}" for col in columns if col not in ("symbol", "date")]
            update_csv = ", ".join(updates)
            
            self._sql_cache[columns] = f"""
                INSERT INTO {self.config.feature_table} ({cols_csv})
                VALUES ({placeholders})
                ON CONFLICT(symbol, date)
                DO UPDATE SET {update_csv}
            """
        return self._sql_cache[columns]

    def _flush_buffer(self) -> int:
        """Atomic batch execution using strict transactions and prepared statements."""
        if self.buffer_size == 0 and not self.pending_checkpoints:
            return 0
            
        total_inserted = 0
        
        for attempt in range(1, self.config.retry_count + 1):
            try:
                if self.buffer_size > 0:
                    self.conn.execute("BEGIN IMMEDIATE")
                    for columns, rows in self.buffer.items():
                        query = self._get_upsert_query(columns)
                        self.conn.executemany(query, rows)
                        total_inserted += len(rows)
                    self.conn.execute("COMMIT")
                    self.buffer.clear()
                    self.buffer_size = 0
                
                # Persist checkpoints only after successful DB commit
                if self.pending_checkpoints:
                    with open(self.config.checkpoint_file, "a") as f:
                        for sym in self.pending_checkpoints:
                            f.write(f"{sym}\n")
                    self.pending_checkpoints.clear()
                    
                return total_inserted
                
            except Exception as e:
                if self.conn.in_transaction:
                    self.conn.execute("ROLLBACK")
                    
                if attempt == self.config.retry_count:
                    logging.getLogger("IndicatorPipelineV4").error(f"Fatal DB flush failure: {e}")
                    self.buffer.clear()
                    self.buffer_size = 0
                    self.pending_checkpoints.clear()
                    return 0
                time.sleep(1.0 * attempt)
        return 0

    def run(self):
        """Pipeline orchestration loop utilizing ProcessPoolExecutor with proper wait mapping."""
        symbols = self._get_symbols()
        if not symbols:
            print("No symbols found in historical data. Exiting.")
            return
            
        completed_symbols = self._load_checkpoint()
        self._load_feature_cache()
        display = ProgressDisplay(len(symbols))
        
        max_in_flight = self.config.max_workers * 2
        in_flight_futures = {}
        symbol_iter = iter(symbols)
        total_rows_saved = 0
        
        try:
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                while True:
                    # Queue Refill
                    while len(in_flight_futures) < max_in_flight:
                        try:
                            symbol = next(symbol_iter)
                            
                            # Fast resume using checkpoint file
                            if symbol in completed_symbols:
                                display.update(symbol, 0, success=True)
                                continue
                                
                            missing = self._get_missing_count(symbol)
                            
                            if missing == 0:
                                display.update(symbol, 0, success=True)
                                self.pending_checkpoints.append(symbol)
                                continue
                                
                            limit = max(self.config.lookback, self.config.lookback - 1 + missing)
                            
                            job = Job(
                                symbol=symbol,
                                missing_count=missing,
                                limit=limit,
                                db_path=self.config.db_path,
                                history_table=self.config.history_table,
                                retry_count=self.config.retry_count
                            )
                            
                            future = executor.submit(_process_worker, job)
                            in_flight_futures[future] = symbol
                        except StopIteration:
                            break
                            
                    if not in_flight_futures:
                        break
                        
                    # Await completion using standard robust wait logic with timeout
                    done, _ = wait(in_flight_futures.keys(), timeout=0.5, return_when=FIRST_COMPLETED)
                    
                    for future in done:
                        symbol = in_flight_futures.pop(future)
                        try:
                            result: JobResult = future.result()
                            
                            if not result.success:
                                display.log_error(symbol, result.error)
                                display.update(symbol, 0, success=False)
                                continue
                                
                            if result.rows:
                                self.buffer[result.columns].extend(result.rows)
                                self.buffer_size += len(result.rows)
                                
                            display.update(symbol, len(result.rows), success=True)
                            self.pending_checkpoints.append(symbol)
                            
                            # Flush memory threshold check
                            if self.buffer_size >= self.config.commit_size or len(self.pending_checkpoints) >= 500:
                                saved = self._flush_buffer()
                                total_rows_saved += saved
                                
                        except Exception as e:
                            display.log_error(symbol, f"Future exception: {str(e)}")
                            display.update(symbol, 0, success=False)
                            
                # Final flush
                if self.buffer_size > 0 or self.pending_checkpoints:
                    saved = self._flush_buffer()
                    total_rows_saved += saved

        except KeyboardInterrupt:
            print("\nPipeline aborted by user. Cancelling workers and flushing buffered data...")
            for future in in_flight_futures.keys():
                future.cancel()
            self._flush_buffer()
            sys.exit(1)
        except Exception as e:
            print(f"\nFatal pipeline failure: {e}")
            self._flush_buffer()
        finally:
            self.conn.close()
            display.rows_inserted = total_rows_saved
            display.finish()


def execute():
    config = PipelineConfig()
    pipeline = IndicatorPipelineV4(config)
    pipeline.run()


if __name__ == "__main__":
    execute()
