"""
GREEN BULL RIDER V6 - Institutional-grade AI Stock Analysis Platform
Module: backend/data/market_writer.py
Description: Enterprise Central Persistence Layer. 
             Exclusive database write engine guaranteeing ACID transactions, 
             atomic pipeline commits per symbol, deadlock recovery, background 
             queued writing, and high-performance UPSERTs for all AI pipeline outputs.
             Includes Symbol Locking, Payload Compression, Checksum generation,
             and strictly deterministic nested transaction management.
             Supports SQLite (WAL) and PostgreSQL seamlessly.
             Python 3.13 Compatible. Compile-Safe. Runtime-Safe. Production Locked.
"""

import os
import sys
import io
import json
import csv
import time
import uuid
import zlib
import hashlib
import logging
import sqlite3
import datetime
import threading
import traceback
import contextlib
import tracemalloc
import functools
import queue
import dataclasses
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
from pathlib import Path
from contextvars import ContextVar
from typing import (
    Any, Callable, Dict, Iterable, Iterator, List, Optional, 
    Set, Tuple, Type, Union, Generator, cast
)

import pandas as pd
import numpy as np

# Optional High-Performance Imports
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    ARROW_AVAILABLE = True
except ImportError:
    ARROW_AVAILABLE = False

try:
    import psycopg2
    from psycopg2.extras import execute_batch
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

# =========================================================================
# ENTERPRISE CONTEXT VARIABLES
# =========================================================================
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="SYSTEM")
trace_id_ctx: ContextVar[str] = ContextVar("trace_id", default="")
span_id_ctx: ContextVar[str] = ContextVar("span_id", default="")

# =========================================================================
# EXCEPTIONS
# =========================================================================
class DatabaseError(Exception): """Base exception for all DB ops."""
class WriteError(DatabaseError): """Generic write failure."""
class TransactionError(DatabaseError): """ACID failure or rollback trigger."""
class IntegrityError(DatabaseError): """Constraint violation."""
class ConflictError(DatabaseError): """Key conflict error."""
class SchemaError(DatabaseError): """Missing tables or columns."""
class RetryError(DatabaseError): """Max retries exceeded."""
class ConnectionError(DatabaseError): """DB Connection unreachable."""
class TimeoutError(DatabaseError): """Query timeout."""
class ValidationError(DatabaseError): """Invalid payload shape."""
class ConfigurationError(DatabaseError): """Invalid settings."""

# =========================================================================
# ENUMS
# =========================================================================
class Dialect(str, Enum):
    SQLITE = "SQLITE"
    POSTGRES = "POSTGRES"

class WriteMode(str, Enum):
    INSERT = "INSERT"
    UPSERT = "UPSERT"
    REPLACE = "REPLACE"

class SpanKind(Enum):
    CLIENT = "CLIENT"
    INTERNAL = "INTERNAL"
    PRODUCER = "PRODUCER"

class AuditAction(Enum):
    WRITE = "WRITE"
    DELETE = "DELETE"
    TRUNCATE = "TRUNCATE"
    SYSTEM = "SYSTEM"

class AuditSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

# =========================================================================
# CONFIGURATION
# =========================================================================
@dataclass(frozen=True, slots=True, kw_only=True)
class WriterConfig:
    dialect: Dialect = Dialect.SQLITE
    db_url: str = "sqlite:///database/universe.db"
    max_connections: int = 20
    pool_timeout_sec: float = 30.0
    statement_timeout_sec: float = 60.0
    chunk_size: int = 5000
    retry_count: int = 7
    retry_delay_sec: float = 0.5
    backoff_multiplier: float = 2.0
    enable_wal: bool = True
    enable_auto_vacuum: bool = True
    enable_telemetry: bool = True
    enable_compression: bool = True
    enable_checksums: bool = True

# =========================================================================
# TELEMETRY & OBSERVABILITY ENGINE
# =========================================================================
class SafeMetrics:
    _lock = threading.Lock()
    _stats: Dict[str, float] = defaultdict(float)

    @staticmethod
    def increment(name: str, namespace: str = "writer", amount: int = 1) -> None:
        with SafeMetrics._lock: SafeMetrics._stats[f"{namespace}.{name}"] += amount
        try:
            from backend.core.metrics import metrics_engine
            if hasattr(metrics_engine, 'increment'): metrics_engine.increment(name, namespace=namespace, amount=amount)
        except Exception: pass

    @staticmethod
    def record_latency(name: str, namespace: str, duration: float) -> None:
        with SafeMetrics._lock:
            k = f"{namespace}.{name}_rolling"
            if k not in SafeMetrics._stats: SafeMetrics._stats[k] = duration
            else: SafeMetrics._stats[k] = (SafeMetrics._stats[k] * 0.9) + (duration * 0.1)
        try:
            from backend.core.metrics import metrics_engine
            if hasattr(metrics_engine, 'record_latency'): metrics_engine.record_latency(name, namespace, duration)
        except Exception: pass

    @staticmethod
    def get_internal_stats() -> Dict[str, float]:
        with SafeMetrics._lock: return dict(SafeMetrics._stats)

def trace_span(operation: str, component: str, kind: SpanKind):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            new_span = uuid.uuid4().hex[:8]
            t_id = trace_id_ctx.get() or uuid.uuid4().hex
            t_trace = trace_id_ctx.set(t_id)
            t_span = span_id_ctx.set(new_span)
            start_time = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000
                SafeMetrics.record_latency(f"{component}_{operation}_ms", "writer", duration_ms)
                trace_id_ctx.reset(t_trace)
                span_id_ctx.reset(t_span)
        return wrapper
    return decorator

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False

    def _log(self, level: int, msg: str, **kwargs):
        payload = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "level": logging.getLevelName(level),
            "trace_id": trace_id_ctx.get(),
            "span_id": span_id_ctx.get(),
            "request_id": request_id_ctx.get(),
            "component": "market_writer",
            "thread_id": threading.get_ident(),
            "message": msg
        }
        payload.update(kwargs)
        self.logger.log(level, json.dumps(payload, default=str))

    def debug(self, msg: str, **kwargs): self._log(logging.DEBUG, msg, **kwargs)
    def info(self, msg: str, **kwargs): self._log(logging.INFO, msg, **kwargs)
    def warning(self, msg: str, **kwargs): self._log(logging.WARNING, msg, **kwargs)
    def error(self, msg: str, **kwargs): self._log(logging.ERROR, msg, **kwargs)
    def critical(self, msg: str, **kwargs): self._log(logging.CRITICAL, msg, **kwargs)

_logger = StructuredLogger("MarketWriter")

class AuditEngine:
    @staticmethod
    def record_event(operation: str, action: AuditAction, severity: AuditSeverity, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        try:
            from backend.core.audit import AuditEngine as CoreAudit
            if hasattr(CoreAudit, 'record_event'):
                CoreAudit.record_event(operation=operation, action=action, severity=severity, message=message, metadata=metadata)
        except Exception: pass

# =========================================================================
# RETRY & DEADLOCK RECOVERY DECORATOR
# =========================================================================
def retry_on_deadlock(func: Callable) -> Callable:
    """Instance-bound decorator to catch database locks and retry safely."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs) -> Any:
        retries = 0
        last_err = None
        while retries <= self.config.retry_count:
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                is_transient = any(x in err_str for x in ["locked", "deadlock", "busy", "timeout", "operationalerror", "database is locked"])
                if not is_transient:
                    raise e
                
                retries += 1
                if retries <= self.config.retry_count:
                    delay = self.config.retry_delay_sec * (self.config.backoff_multiplier ** (retries - 1))
                    delay += (uuid.uuid4().int % 100) / 1000.0  # Jitter
                    _logger.warning("Database Lock Detected. Retrying.", operation=func.__name__, retry=retries, delay=delay)
                    SafeMetrics.increment("retry_total")
                    time.sleep(delay)
        
        _logger.error("Max retries exceeded due to deadlock.", operation=func.__name__, error=str(last_err))
        raise RetryError(f"Operation failed after {retries} retries due to locking: {last_err}") from last_err
    return wrapper

# =========================================================================
# DATA NORMALIZER & METADATA INJECTOR
# =========================================================================
class DataNormalizer:
    @staticmethod
    def _convert_dataframe(df: pd.DataFrame) -> List[Dict[str, Any]]:
        if df.empty: return []
        df_clean = df.copy()
        for col in df_clean.columns:
            if pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].dt.strftime('%Y-%m-%d %H:%M:%S.%f%z').replace("NaT", None)
            elif pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf, np.nan], None)
            elif pd.api.types.is_object_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].where(pd.notnull(df_clean[col]), None)
        return df_clean.to_dict(orient='records')

    @classmethod
    def to_records(cls, data: Any) -> List[Dict[str, Any]]:
        if isinstance(data, pd.DataFrame): return cls._convert_dataframe(data)
        if dataclasses.is_dataclass(data): return [dataclasses.asdict(data)]
        if isinstance(data, tuple) and hasattr(data, '_asdict'): return [data._asdict()]
        if isinstance(data, dict): return [data]
        if isinstance(data, str):
            try:
                parsed = json.loads(data)
                return parsed if isinstance(parsed, list) else [parsed]
            except json.JSONDecodeError:
                raise ValidationError("Failed to parse JSON string to records.")
        if isinstance(data, list):
            if not data: return []
            first = data[0]
            if isinstance(first, dict): return data
            if dataclasses.is_dataclass(first): return [dataclasses.asdict(x) for x in data]
            if isinstance(first, tuple) and hasattr(first, '_asdict'): return [x._asdict() for x in data]
            if hasattr(first, '__dict__'): return [vars(x) for x in data]
            
        raise ValidationError(f"Unsupported data format provided to normalizer: {type(data)}")

# =========================================================================
# DATABASE ADAPTERS (SQLITE + POSTGRESQL)
# =========================================================================
class DatabaseAdapter:
    def get_connection(self) -> Any: raise NotImplementedError()
    def execute(self, query: str, params: tuple = ()) -> Any: raise NotImplementedError()
    def executemany(self, query: str, data: List[tuple]) -> Any: raise NotImplementedError()
    def fetchall(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]: raise NotImplementedError()
    @contextlib.contextmanager
    def transaction(self) -> Generator[None, None, None]: raise NotImplementedError()
    def get_columns(self, table_name: str) -> List[str]: raise NotImplementedError()
    def get_primary_keys(self, table_name: str) -> List[str]: raise NotImplementedError()
    def build_upsert_query(self, table: str, columns: List[str], pks: List[str]) -> str: raise NotImplementedError()
    def close(self) -> None: raise NotImplementedError()

class SQLiteAdapter(DatabaseAdapter):
    def __init__(self, config: WriterConfig):
        self.config = config
        self.db_path = config.db_url.replace("sqlite:///", "") if config.db_url.startswith("sqlite:///") else config.db_url
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._schema_cache: Dict[str, Dict[str, List[str]]] = {}

    def get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(self.db_path, timeout=self.config.pool_timeout_sec, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            if self.config.enable_wal:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.execute("PRAGMA temp_store=MEMORY;")
            conn.execute("PRAGMA busy_timeout=15000;")
            self._local.conn = conn
        return self._local.conn

    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        return self.get_connection().execute(query, params)

    def executemany(self, query: str, data: List[tuple]) -> sqlite3.Cursor:
        return self.get_connection().executemany(query, data)

    def fetchall(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        cur = self.execute(query, params)
        return [dict(row) for row in cur.fetchall()]

    @contextlib.contextmanager
    def transaction(self) -> Generator[None, None, None]:
        conn = self.get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise TransactionError(f"Transaction aborted: {e}") from e

    def _introspect_table(self, table_name: str):
        if table_name in self._schema_cache: return
        res = self.fetchall(f"PRAGMA table_info({table_name})")
        if not res: raise SchemaError(f"Table '{table_name}' does not exist.")
        cols = [r['name'] for r in res]
        pks = [r['name'] for r in res if r['pk'] > 0]
        self._schema_cache[table_name] = {"columns": cols, "pks": pks}

    def get_columns(self, table_name: str) -> List[str]:
        self._introspect_table(table_name)
        return self._schema_cache[table_name]["columns"]

    def get_primary_keys(self, table_name: str) -> List[str]:
        self._introspect_table(table_name)
        return self._schema_cache[table_name]["pks"]

    def build_upsert_query(self, table: str, columns: List[str], pks: List[str]) -> str:
        cols_csv = ", ".join(columns)
        vals_csv = ", ".join(["?"] * len(columns))
        if not pks: return f"INSERT INTO {table} ({cols_csv}) VALUES ({vals_csv})"
        if table == "market_data":
            pks = ["symbol", "timestamp", "timeframe"]
        elif table == "live_market_data":
            pks = ["symbol", "timestamp", "timeframe"]
        elif table == "historical_data":
            pks = ["symbol", "timestamp", "timeframe"]
        elif table == "fundamental_data":
            pks = ["symbol", "report_date"]

        conflict_cols = ", ".join(pks)
        updates = ", ".join([f"{col}=excluded.{col}" for col in columns if col not in pks])
        query = f"INSERT INTO {table} ({cols_csv}) VALUES ({vals_csv}) ON CONFLICT({conflict_cols}) "
        query += f"DO UPDATE SET {updates}" if updates else "DO NOTHING"
        return query

    def close(self):
        if hasattr(self._local, "conn"):
            try: self._local.conn.close()
            except Exception: pass
            del self._local.conn

class PostgresAdapter(DatabaseAdapter):
    def __init__(self, config: WriterConfig):
        self.config = config
        self._local = threading.local()
        self._schema_cache: Dict[str, Dict[str, List[str]]] = {}
        if not POSTGRES_AVAILABLE:
            raise ConfigurationError("psycopg2 is required for PostgresAdapter.")

    def get_connection(self) -> Any:
        if not hasattr(self._local, "conn"):
            self._local.conn = psycopg2.connect(self.config.db_url)
        return self._local.conn

    def execute(self, query: str, params: tuple = ()) -> Any:
        conn = self.get_connection()
        cur = conn.cursor()
        cur.execute(query, params)
        return cur

    def executemany(self, query: str, data: List[tuple]) -> Any:
        conn = self.get_connection()
        cur = conn.cursor()
        execute_batch(cur, query, data, page_size=self.config.chunk_size)
        return cur

    def fetchall(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        cur = self.execute(query, params)
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    @contextlib.contextmanager
    def transaction(self) -> Generator[None, None, None]:
        conn = self.get_connection()
        try:
            yield
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise TransactionError(f"Transaction aborted: {e}") from e

    def _introspect_table(self, table_name: str):
        if table_name in self._schema_cache: return
        col_query = "SELECT column_name FROM information_schema.columns WHERE table_name = %s"
        pk_query = """
            SELECT a.attname FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = %s::regclass AND i.indisprimary;
        """
        cols = [r['column_name'] for r in self.fetchall(col_query, (table_name,))]
        if not cols: raise SchemaError(f"Table '{table_name}' does not exist.")
        try: pks = [r['attname'] for r in self.fetchall(pk_query, (table_name,))]
        except Exception: pks = []
        self._schema_cache[table_name] = {"columns": cols, "pks": pks}

    def get_columns(self, table_name: str) -> List[str]:
        self._introspect_table(table_name)
        return self._schema_cache[table_name]["columns"]

    def get_primary_keys(self, table_name: str) -> List[str]:
        self._introspect_table(table_name)
        return self._schema_cache[table_name]["pks"]

    def build_upsert_query(self, table: str, columns: List[str], pks: List[str]) -> str:
        cols_csv = ", ".join(columns)
        vals_csv = ", ".join(["%s"] * len(columns))
        if not pks: return f"INSERT INTO {table} ({cols_csv}) VALUES ({vals_csv})"
        conflict_cols = ", ".join(pks)
        updates = ", ".join([f"{col}=EXCLUDED.{col}" for col in columns if col not in pks])
        query = f"INSERT INTO {table} ({cols_csv}) VALUES ({vals_csv}) ON CONFLICT({conflict_cols}) "
        query += f"DO UPDATE SET {updates}" if updates else "DO NOTHING"
        return query

    def close(self):
        if hasattr(self._local, "conn"):
            try: self._local.conn.close()
            except Exception: pass
            del self._local.conn

class AdapterFactory:
    @staticmethod
    def get_adapter(config: WriterConfig) -> DatabaseAdapter:
        if config.dialect == Dialect.SQLITE: return SQLiteAdapter(config)
        elif config.dialect == Dialect.POSTGRES: return PostgresAdapter(config)
        raise ConfigurationError(f"Unsupported Dialect: {config.dialect}")

# =========================================================================
# ENTERPRISE MARKET WRITER ENGINE
# =========================================================================
class MarketWriter:
    """
    Central Persistence Layer.
    Guarantees Atomic Commit, Thread Safety, and Deduplication via UPSERT.
    Provides Domain-Specific methods mapping output from the entire AI Pipeline.
    Manages deterministic nested transactions and asynchronous queues.
    """
    
    # Strictly defined deterministic order for atomic pipeline commits
    PIPELINE_TABLE_ORDER = [
        "market_data", "fundamental_data", "indicator_data", "feature_data",
        "analysis_data", "score_data", "decision_data", "master_ai_decision",
        "portfolio_data", "dashboard_cache"
    ]
    
    def __init__(self, config: WriterConfig = WriterConfig()):
        self.config = config
        self.adapter = AdapterFactory.get_adapter(config)
        
        self._local = threading.local()
        self._transaction_lock = threading.RLock()
        
        self._symbol_locks_mutex = threading.Lock()
        self._symbol_locks: Dict[str, threading.RLock] = defaultdict(threading.RLock)
        
        # Async Background Writer Queue
        self._async_queue: queue.Queue = queue.Queue()
        self._async_worker = threading.Thread(target=self._background_writer_loop, daemon=True, name="WriterQueueThread")
        self._async_worker.start()
        
        _logger.info("MarketWriter initialized", dialect=self.config.dialect.value, db_url=self.config.db_url)
        AuditEngine.record_event("writer.init", AuditAction.SYSTEM, AuditSeverity.INFO, "MarketWriter Engine started.")

    @contextlib.contextmanager
    def _transaction_context(self) -> Generator[None, None, None]:
        """Safely handles nested transaction boundaries avoiding SQLite errors."""
        with self._transaction_lock:
            if getattr(self._local, 'in_transaction', False):
                yield
            else:
                self._local.in_transaction = True
                try:
                    with self.adapter.transaction():
                        yield
                finally:
                    self._local.in_transaction = False

    @contextlib.contextmanager
    def symbol_lock(self, symbol: str) -> Generator[None, None, None]:
        """Prevents race conditions when multiple threads process the same symbol."""
        with self._symbol_locks_mutex:
            lock = self._symbol_locks[symbol]
        with lock:
            yield

    def _background_writer_loop(self):
        while True:
            task = self._async_queue.get()
            if task is None: break
            try:
                func, args, kwargs = task
                func(*args, **kwargs)
            except Exception as e:
                _logger.error("Background write failed in queue", error=str(e))
            finally:
                self._async_queue.task_done()

    def enqueue_write(self, table_name: str, data: Any, mode: WriteMode = WriteMode.UPSERT) -> None:
        """Schedules a non-blocking asynchronous write operation."""
        self._async_queue.put((self.bulk_write, (table_name, data, mode), {}))

    def _inject_metadata(self, table_name: str, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Automatically injects Checksums, Version Tracking, and Payload Compression."""
        version_id = int(time.time() * 1000)
        
        for r in records:
            if "id" not in r or not r.get("id"):
                r["id"] = str(uuid.uuid4())

            if table_name in ("market_data", "historical_data", "live_market_data"):
                if not r.get("timestamp"):
                    r["timestamp"] = (
                        r.get("date")
                        or r.get("datetime")
                        or r.get("time")
                    )

                ts = r.get("timestamp")
                if ts is not None:
                    try:
                        import pandas as pd
                        if isinstance(ts, pd.Timestamp):
                            r["timestamp"] = ts.isoformat(sep=" ")
                    except Exception:
                        pass

                r.setdefault("timeframe", "1D")

            if table_name == "master_ai_decision" and "version" not in r:
                r["version"] = version_id
                
            if self.config.enable_checksums:
                raw_str = json.dumps(r, sort_keys=True, default=str).encode('utf-8')
                r["record_checksum"] = hashlib.sha256(raw_str).hexdigest()
                
            if self.config.enable_compression and table_name in ("analysis_data", "master_ai_decision"):
                for heavy_field in ("reasoning", "summary", "explanation"):
                    if heavy_field in r and isinstance(r[heavy_field], str):
                        r[f"{heavy_field}_compressed"] = zlib.compress(r[heavy_field].encode('utf-8'))
        return records

    def _execute_batch(self, table_name: str, records: List[Dict[str, Any]], mode: WriteMode) -> int:
        if not records: return 0
        
        db_cols = self.adapter.get_columns(table_name)
        db_pks = self.adapter.get_primary_keys(table_name)
        
        payload_keys = set(records[0].keys())
        target_cols = [c for c in db_cols if c in payload_keys]
        
        if not target_cols:
            raise SchemaError(f"Zero matching columns for {table_name}. Expected: {db_cols}")
            
        if mode == WriteMode.UPSERT:
            print("=" * 80)
            print("UPSERT DEBUG")
            print("table      :", table_name)
            print("db_pks     :", db_pks)
            print("target_cols:", target_cols)
            query = self.adapter.build_upsert_query(table_name, target_cols, db_pks)
            print(query)
            print("=" * 80)
        elif mode == WriteMode.INSERT:
            v_ph = "?" if self.config.dialect == Dialect.SQLITE else "%s"
            query = f"INSERT INTO {table_name} ({', '.join(target_cols)}) VALUES ({', '.join([v_ph]*len(target_cols))})"
        elif mode == WriteMode.REPLACE:
            v_ph = "?" if self.config.dialect == Dialect.SQLITE else "%s"
            query = f"REPLACE INTO {table_name} ({', '.join(target_cols)}) VALUES ({', '.join([v_ph]*len(target_cols))})"
        else:
            raise ValueError(f"Invalid mode: {mode}")

        tuples_data = [tuple(r.get(c) for c in target_cols) for r in records]
        total_written = 0
        
        for i in range(0, len(tuples_data), self.config.chunk_size):
            chunk = tuples_data[i:i + self.config.chunk_size]
            t0 = time.perf_counter()
            self.adapter.executemany(query, chunk)
            dur = (time.perf_counter() - t0) * 1000
            total_written += len(chunk)
            SafeMetrics.increment("rows_written", amount=len(chunk))
            SafeMetrics.record_latency("write_chunk_ms", "writer", dur)
            
        _logger.debug(f"Wrote {total_written} records to {table_name}", mode=mode.value)
        return total_written

    # =========================================================================
    # PIPELINE ATOMIC COMMIT GUARANTEE
    # =========================================================================
    
    @trace_span(operation="writer.commit_pipeline", component="writer", kind=SpanKind.INTERNAL)
    @retry_on_deadlock
    def commit_pipeline(self, symbol: str, pipeline_result: Dict[str, Any]) -> int:
        """
        ATOMIC PIPELINE COMMIT GUARANTEE.
        Writes all output stages for a specific symbol inside a single ACID transaction.
        Executes tables in a strict deterministic order.
        """
        _logger.info(f"Committing AI Pipeline for symbol: {symbol}")
        total_written = 0
        
        extra_tables = [t for t in pipeline_result.keys() if t not in self.PIPELINE_TABLE_ORDER]
        execution_order = self.PIPELINE_TABLE_ORDER + extra_tables
        
        try:
            with self.symbol_lock(symbol):
                with self._transaction_context():
                    for table_name in execution_order:
                        if table_name not in pipeline_result or pipeline_result[table_name] is None:
                            continue
                        
                        records = DataNormalizer.to_records(pipeline_result[table_name])
                        if not records: continue
                        
                        records = self._inject_metadata(table_name, records)
                        total_written += self._execute_batch(table_name, records, WriteMode.UPSERT)
            
            _logger.info(f"Successfully committed pipeline for {symbol}", records=total_written)
            AuditEngine.record_event("writer.commit_pipeline", AuditAction.WRITE, AuditSeverity.INFO, f"Committed pipeline for {symbol}", {"records": total_written})
            return total_written
        except Exception as e:
            SafeMetrics.increment("pipeline_commit_failures")
            _logger.critical(f"Pipeline commit failed for {symbol}. Entire transaction rolled back.", error=str(e))
            raise TransactionError(f"Pipeline atomic commit failed for {symbol}: {e}") from e

    @trace_span(operation="writer.commit_many_symbols", component="writer", kind=SpanKind.INTERNAL)
    def commit_many_symbols(self, pipeline_results_by_symbol: Dict[str, Dict[str, Any]]) -> int:
        total = 0
        for sym, result in pipeline_results_by_symbol.items():
            total += self.commit_pipeline(sym, result)
        return total

    def write_pipeline_result(self, symbol: str, pipeline_result: Dict[str, Any]) -> int:
        return self.commit_pipeline(symbol, pipeline_result)

    def rollback_pipeline(self) -> None:
        _logger.warning("Pipeline rollback triggered by upstream signal.")
        pass # Automatically handled by transaction context exceptions

    # =========================================================================
    # CORE GENERIC APIS
    # =========================================================================
    
    @trace_span(operation="writer.bulk_write", component="writer", kind=SpanKind.INTERNAL)
    @retry_on_deadlock
    def bulk_write(self, table_name: str, data: Any, mode: WriteMode = WriteMode.UPSERT) -> int:
        try:
            records = DataNormalizer.to_records(data)
            records = self._inject_metadata(table_name, records)
            with self._transaction_context():
                written = self._execute_batch(table_name, records, mode)
            AuditEngine.record_event("writer.bulk_write", AuditAction.WRITE, AuditSeverity.INFO, f"Wrote {written} records to {table_name}")
            return written
        except Exception as e:
            SafeMetrics.increment("write_failures")
            _logger.error("Bulk write failed", table=table_name, error=str(e), traceback=traceback.format_exc())
            raise WriteError(f"Bulk write failed for table {table_name}: {e}") from e

    @trace_span(operation="writer.stream_write", component="writer", kind=SpanKind.INTERNAL)
    def stream_write(self, table_name: str, iterator: Iterable[Any], mode: WriteMode = WriteMode.UPSERT) -> int:
        total = 0
        chunk = []
        for item in iterator:
            chunk.append(item)
            if len(chunk) >= self.config.chunk_size:
                total += self.bulk_write(table_name, chunk, mode)
                chunk = []
        if chunk:
            total += self.bulk_write(table_name, chunk, mode)
        return total

    def upsert(self, table_name: str, data: Any) -> int: return self.bulk_write(table_name, data, WriteMode.UPSERT)
    def insert(self, table_name: str, data: Any) -> int: return self.bulk_write(table_name, data, WriteMode.INSERT)
    def replace(self, table_name: str, data: Any) -> int: return self.bulk_write(table_name, data, WriteMode.REPLACE)

    @trace_span(operation="writer.update", component="writer", kind=SpanKind.INTERNAL)
    @retry_on_deadlock
    def update(self, table_name: str, set_clause: str, condition: str, params: tuple = ()) -> int:
        query = f"UPDATE {table_name} SET {set_clause} WHERE {condition}"
        with self._transaction_context():
            return self.adapter.execute(query, params).rowcount

    @trace_span(operation="writer.delete", component="writer", kind=SpanKind.INTERNAL)
    @retry_on_deadlock
    def delete(self, table_name: str, condition: str, params: tuple = ()) -> int:
        query = f"DELETE FROM {table_name} WHERE {condition}"
        with self._transaction_context():
            del_count = self.adapter.execute(query, params).rowcount
        AuditEngine.record_event("writer.delete", AuditAction.DELETE, AuditSeverity.WARNING, f"Deleted {del_count} rows from {table_name}")
        return del_count

    @trace_span(operation="writer.truncate", component="writer", kind=SpanKind.INTERNAL)
    @retry_on_deadlock
    def truncate(self, table_name: str) -> None:
        query = f"DELETE FROM {table_name}" if self.config.dialect == Dialect.SQLITE else f"TRUNCATE TABLE {table_name} CASCADE"
        with self._transaction_context():
            self.adapter.execute(query)
        AuditEngine.record_event("writer.truncate", AuditAction.DELETE, AuditSeverity.CRITICAL, f"Truncated table {table_name}")

    # =========================================================================
    # FILE / DATAFRAME APIS
    # =========================================================================

    def write_dataframe(self, table_name: str, df: pd.DataFrame, mode: WriteMode = WriteMode.UPSERT) -> int: return self.bulk_write(table_name, df, mode)
    def write_records(self, table_name: str, records: List[Dict[str, Any]], mode: WriteMode = WriteMode.UPSERT) -> int: return self.bulk_write(table_name, records, mode)
    def write_dict(self, table_name: str, data: dict, mode: WriteMode = WriteMode.UPSERT) -> int: return self.bulk_write(table_name, data, mode)

    def write_json(self, table_name: str, path_or_json: str, mode: WriteMode = WriteMode.UPSERT) -> int:
        try:
            if os.path.exists(path_or_json):
                with open(path_or_json, 'r', encoding='utf-8') as f: data = json.load(f)
            else:
                data = json.loads(path_or_json)
            return self.bulk_write(table_name, data, mode)
        except Exception as e:
            raise ValidationError(f"Failed to process JSON input: {e}")

    def write_csv(self, table_name: str, path: str, mode: WriteMode = WriteMode.UPSERT) -> int:
        try:
            df = pd.read_csv(path)
            return self.bulk_write(table_name, df, mode)
        except Exception as e:
            raise ValidationError(f"Failed to process CSV file {path}: {e}")

    def write_parquet(self, table_name: str, path: str, mode: WriteMode = WriteMode.UPSERT) -> int:
        if not ARROW_AVAILABLE: raise ConfigurationError("PyArrow is required for Parquet processing.")
        try:
            table = pq.read_table(path)
            df = table.to_pandas()
            return self.bulk_write(table_name, df, mode)
        except Exception as e:
            raise ValidationError(f"Failed to process Parquet file {path}: {e}")

    # =========================================================================
    # DOMAIN SPECIFIC WRITERS (EXPLICIT PLATFORM API)
    # =========================================================================
    
    def write_market_data(self, data: Any) -> int: return self.upsert("market_data", data)
    def write_live_data(self, data: Any) -> int: return self.upsert("live_market_data", data)
    def write_historical(self, data: Any) -> int: return self.upsert("historical_data", data)
    def write_fundamental_data(self, data: Any) -> int: return self.upsert("fundamental_data", data)
    def write_institutional_data(self, data: Any) -> int: return self.upsert("institutional_data", data)
    def write_ipo_data(self, data: Any) -> int: return self.upsert("ipo_data", data)
    def write_bulk_deals(self, data: Any) -> int: return self.upsert("bulk_deals", data)
    def write_block_deals(self, data: Any) -> int: return self.upsert("block_deals", data)
    def write_corporate_actions(self, data: Any) -> int: return self.upsert("corporate_actions", data)
    def write_fno_data(self, data: Any) -> int: return self.upsert("fno_data", data)
    def write_options_chain(self, data: Any) -> int: return self.upsert("options_chain", data)
    def write_indices(self, data: Any) -> int: return self.upsert("indices", data)
    def write_sector_data(self, data: Any) -> int: return self.upsert("sector_data", data)
    def write_industry_data(self, data: Any) -> int: return self.upsert("industry_data", data)
    def write_master(self, data: Any) -> int: return self.upsert("stock_master", data)
    
    def write_indicator_data(self, data: Any) -> int: return self.upsert("indicator_data", data)
    def write_indicator(self, data: Any) -> int: return self.upsert("indicator_data", data)
    def write_indicator_batch(self, data: Any) -> int: return self.upsert("indicator_data", data)
    def write_feature_data(self, data: Any) -> int: return self.upsert("feature_data", data)
    def write_feature(self, data: Any) -> int: return self.upsert("feature_data", data)
    def write_analysis_data(self, data: Any) -> int: return self.upsert("analysis_data", data)
    def write_analyzer(self, data: Any) -> int: return self.upsert("analysis_data", data)
    def write_engine(self, data: Any) -> int: return self.upsert("engine_output", data)
    def write_score_data(self, data: Any) -> int: return self.upsert("score_data", data)
    def write_score(self, data: Any) -> int: return self.upsert("score_data", data)
    def write_decision_data(self, data: Any) -> int: return self.upsert("decision_data", data)
    def write_decision(self, data: Any) -> int: return self.upsert("decision_data", data)
    def write_master_ai(self, data: Any) -> int: return self.upsert("master_ai_decision", data)
    def write_ai_output(self, data: Any) -> int: return self.upsert("ai_output", data)
    
    def write_portfolio(self, data: Any) -> int: return self.upsert("portfolio_data", data)
    def write_dashboard(self, data: Any) -> int: return self.upsert("dashboard_cache", data)
    def write_watchlist(self, data: Any) -> int: return self.upsert("watchlist", data)
    def write_alert(self, data: Any) -> int: return self.upsert("alerts", data)
    def write_scanner_result(self, data: Any) -> int: return self.upsert("scanner_results", data)
    def write_registry(self, data: Any) -> int: return self.upsert("registry", data)
    def write_trace(self, data: Any) -> int: return self.upsert("trace_log", data)
    def write_log(self, data: Any) -> int: return self.upsert("system_log", data)
    def write_validation_log(self, data: Any) -> int: return self.upsert("validation_log", data)
    def write_sync_log(self, data: Any) -> int: return self.upsert("sync_log", data)
    def write_system_log(self, data: Any) -> int: return self.upsert("system_log", data)


    # =========================================================================
    # MAINTENANCE & OPTIMIZATIONS
    # =========================================================================

    def flush(self) -> None:
        self.checkpoint()

    @trace_span(operation="writer.checkpoint", component="writer", kind=SpanKind.INTERNAL)
    def checkpoint(self) -> None:
        if self.config.dialect == Dialect.SQLITE:
            try: self.adapter.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            except Exception as e: _logger.warning("WAL Checkpoint failed", error=str(e))

    @trace_span(operation="writer.vacuum", component="writer", kind=SpanKind.INTERNAL)
    def vacuum(self) -> None:
        try: self.adapter.execute("VACUUM;")
        except Exception as e: _logger.warning("VACUUM failed", error=str(e))

    @trace_span(operation="writer.analyze", component="writer", kind=SpanKind.INTERNAL)
    def analyze(self) -> None:
        try: self.adapter.execute("ANALYZE;")
        except Exception as e: _logger.warning("ANALYZE failed", error=str(e))

    def optimize(self) -> None:
        self.checkpoint()
        self.analyze()
        if self.config.enable_auto_vacuum:
            self.vacuum()


    # =========================================================================
    # SYSTEM METRICS & LIFECYCLE
    # =========================================================================

    def health_check(self) -> Dict[str, Any]:
        status = "HEALTHY"
        try:
            res = self.adapter.fetchall("SELECT 1 as val")
            if not res or res[0].get('val') != 1: raise ConnectionError("Diagnostic query failed.")
        except Exception as e:
            status = "UNHEALTHY"
            _logger.error("Health check failed", error=str(e))
            
        return {
            "status": status,
            "dialect": self.config.dialect.value,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

    def database_statistics(self) -> Dict[str, Any]:
        stats = {"dialect": self.config.dialect.value}
        if self.config.dialect == Dialect.SQLITE:
            try:
                page_count = self.adapter.fetchall("PRAGMA page_count;")[0]["page_count"]
                page_size = self.adapter.fetchall("PRAGMA page_size;")[0]["page_size"]
                stats["database_size_bytes"] = page_count * page_size
            except Exception: pass
        return stats

    def writer_statistics(self) -> Dict[str, Any]:
        mem_curr, mem_peak = tracemalloc.get_traced_memory() if tracemalloc.is_tracing() else (0, 0)
        return {
            "telemetry": SafeMetrics.get_internal_stats(),
            "memory_usage_mb": mem_curr / 10**6,
            "peak_memory_mb": mem_peak / 10**6,
            "active_threads": threading.active_count(),
            "pending_background_tasks": self._async_queue.qsize()
        }

    def close(self) -> None:
        _logger.info("Closing MarketWriter connections gracefully.")
        self._async_queue.put(None)
        self._async_worker.join(timeout=5.0)
        self.adapter.close()


# =========================================================================
# MODULE EXPORTS
# =========================================================================
__all__ = [
    "MarketWriter",
    "WriterConfig",
    "DatabaseAdapter",
    "SQLiteAdapter",
    "PostgresAdapter",
    "AdapterFactory",
    "WriteMode",
    "Dialect",
    "DataNormalizer",
    "DatabaseError",
    "WriteError",
    "TransactionError",
    "IntegrityError",
    "ConflictError",
    "SchemaError",
    "RetryError",
    "ConnectionError",
    "TimeoutError",
    "ValidationError",
    "ConfigurationError"
]
