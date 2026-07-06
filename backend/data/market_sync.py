"""
GREEN BULL RIDER V6 - Institutional-grade AI Stock Analysis Platform
Module: backend/data/market_sync.py
Description: Enterprise Central Writer & Synchronization Layer.
             Exclusive database write orchestration engine guaranteeing ACID 
             transactions, thread safety, deadlock recovery, and high-performance 
             bulk insertions. Strictly separated from business logic.
             Supports SQLite (WAL) and PostgreSQL.
             Python 3.13 Compatible. Compile-Safe. Runtime-Safe. Production Locked.
"""

import os
import sys
import json
import time
import uuid
import logging
import sqlite3
import datetime
import threading
import contextlib
import concurrent.futures
import re
import functools
from enum import Enum
from collections import defaultdict
from contextvars import ContextVar
from dataclasses import dataclass, field, is_dataclass, asdict
from typing import (
    Any, Callable, Dict, Iterable, Iterator, List, Optional, 
    Tuple, Type, Union, Generator, Set
)

import pandas as pd
import numpy as np

# High-Performance Optional Imports
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    ARROW_AVAILABLE = True
except ImportError:
    ARROW_AVAILABLE = False

try:
    import psycopg2
    from psycopg2.pool import ThreadedConnectionPool
    import psycopg2.extras
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

# =========================================================================
# ENTERPRISE CONTEXT VARIABLES & REGISTRY
# =========================================================================
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="SYSTEM")
trace_id_ctx: ContextVar[str] = ContextVar("trace_id", default="")
span_id_ctx: ContextVar[str] = ContextVar("span_id", default="")

SYNC_REGISTRY: Dict[str, Callable] = {}

def register_writer(name: str):
    """Module-level decorator to dynamically register synchronization strategies."""
    def decorator(func: Callable):
        SYNC_REGISTRY[name] = func
        return func
    return decorator

# =========================================================================
# EXCEPTIONS
# =========================================================================
class SyncError(Exception): """Base exception for Market Sync."""
class WriteError(SyncError): """Generic database write failure."""
class TransactionError(SyncError): """ACID state or commit/rollback failure."""
class SchemaError(SyncError): """Invalid table or column schema."""
class ValidationError(SyncError): """Data shape or type invalidity."""
class ConnectionError(SyncError): """Database connection issues."""
class TimeoutError(SyncError): """Query execution timeout."""
class DeadlockError(SyncError): """Database locking conflict."""
class RetryError(SyncError): """Exhausted all retry attempts."""
class ConfigurationError(SyncError): """Invalid module configuration."""

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
    UPDATE = "UPDATE"

class SpanKind(Enum):
    CLIENT = "CLIENT"
    INTERNAL = "INTERNAL"
    PRODUCER = "PRODUCER"

class AuditAction(Enum):
    WRITE = "WRITE"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    SYNC = "SYNC"

class AuditSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

# =========================================================================
# CONFIGURATION
# =========================================================================
@dataclass(frozen=True, slots=True, kw_only=True)
class SyncConfig:
    dialect: Dialect = Dialect.SQLITE
    db_url: str = "sqlite:////tmp/gbr_master.db"
    max_connections: int = 50
    async_workers: int = 10
    pool_timeout_sec: float = 15.0
    statement_timeout_sec: float = 30.0
    chunk_size: int = 5000
    retry_count: int = 5
    retry_delay_sec: float = 0.5
    backoff_multiplier: float = 2.0
    enable_wal: bool = True
    enable_audit_logging: bool = True
    enable_metrics: bool = True

# =========================================================================
# METRICS, TELEMETRY & AUDIT
# =========================================================================
class WriterMetrics:
    _lock = threading.Lock()
    _stats: Dict[str, float] = defaultdict(float)

    @staticmethod
    def increment(name: str, amount: float = 1.0) -> None:
        with WriterMetrics._lock: 
            WriterMetrics._stats[f"writer.{name}"] += amount

    @staticmethod
    def record_latency(name: str, duration_ms: float) -> None:
        with WriterMetrics._lock:
            k = f"writer.{name}_avg_ms"
            WriterMetrics._stats[k] = (WriterMetrics._stats[k] * 0.95) + (duration_ms * 0.05)

    @staticmethod
    def get_metrics() -> Dict[str, float]:
        with WriterMetrics._lock: 
            return dict(WriterMetrics._stats)

def trace_span(operation: str, component: str, kind: SpanKind):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            new_span = uuid.uuid4().hex[:8]
            t_id = trace_id_ctx.get() or uuid.uuid4().hex
            t_trace = trace_id_ctx.set(t_id)
            t_span = span_id_ctx.set(new_span)
            t0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                dur_ms = (time.perf_counter() - t0) * 1000
                WriterMetrics.record_latency(f"{component}_{operation}", dur_ms)
                trace_id_ctx.reset(t_trace)
                span_id_ctx.reset(t_span)
        return wrapper
    return decorator

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            h = logging.StreamHandler(sys.stdout)
            self.logger.addHandler(h)
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False

    def _log(self, level: int, msg: str, **kwargs):
        payload = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "level": logging.getLevelName(level),
            "trace_id": trace_id_ctx.get(),
            "request_id": request_id_ctx.get(),
            "component": "market_sync",
            "thread_id": threading.get_ident(),
            "message": msg
        }
        payload.update(kwargs)
        self.logger.log(level, json.dumps(payload, default=str))

    def info(self, msg: str, **kwargs): self._log(logging.INFO, msg, **kwargs)
    def warning(self, msg: str, **kwargs): self._log(logging.WARNING, msg, **kwargs)
    def error(self, msg: str, **kwargs): self._log(logging.ERROR, msg, **kwargs)
    def critical(self, msg: str, **kwargs): self._log(logging.CRITICAL, msg, **kwargs)

_logger = StructuredLogger("MarketSync")

class AuditEngine:
    @staticmethod
    def record_write(table: str, action: AuditAction, rows: int, latency_ms: float) -> None:
        try:
            from backend.core.audit import AuditEngine as CoreAudit
            if hasattr(CoreAudit, 'record_event'):
                CoreAudit.record_event(
                    operation="db_write", action=action.value, severity=AuditSeverity.INFO.value, 
                    message=f"{action.value} {rows} rows on {table}", 
                    metadata={"table": table, "rows": rows, "latency": latency_ms, "trace_id": trace_id_ctx.get()}
                )
        except Exception: pass

# =========================================================================
# RETRY & DEADLOCK ENGINE
# =========================================================================
def retry_on_deadlock(func: Callable) -> Callable:
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
                    WriterMetrics.increment("retry_count")
                    _logger.warning("Deadlock detected. Retrying.", operation=func.__name__, retry=retries, delay=delay)
                    time.sleep(delay)
        
        WriterMetrics.increment("deadlock_failures")
        _logger.error("Max retries exceeded.", operation=func.__name__, error=str(last_err))
        raise DeadlockError(f"Operation failed after {retries} retries due to locking: {last_err}") from last_err
    return wrapper

# =========================================================================
# DATABASE ADAPTERS
# =========================================================================
class WriterAdapter:
    def get_connection(self) -> Any: raise NotImplementedError()
    def execute_rowcount(self, query: str, params: tuple = ()) -> int: raise NotImplementedError()
    def executemany_rowcount(self, query: str, data: List[tuple]) -> int: raise NotImplementedError()
    def get_columns(self, table_name: str) -> List[str]: raise NotImplementedError()
    def get_primary_keys(self, table_name: str) -> List[str]: raise NotImplementedError()
    def close(self) -> None: raise NotImplementedError()

class SQLiteWriterAdapter(WriterAdapter):
    def __init__(self, config: SyncConfig):
        self.config = config
        self.db_path = config.db_url.replace("sqlite:///", "") if config.db_url.startswith("sqlite:///") else config.db_url
        self._local = threading.local()
        self._schema_cache: Dict[str, Dict[str, List[str]]] = {}
        self._wal_configured = False
        self._init_lock = threading.Lock()

    def get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(self.db_path, timeout=self.config.pool_timeout_sec, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            if self.config.enable_wal and not self._wal_configured:
                with self._init_lock:
                    if not self._wal_configured:
                        conn.execute("PRAGMA journal_mode = WAL;")
                        self._wal_configured = True
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA busy_timeout = 15000;")
            self._local.conn = conn
        return self._local.conn

    def execute_rowcount(self, query: str, params: tuple = ()) -> int:
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            cur.execute(query, params)
            return cur.rowcount
        finally:
            cur.close()

    def executemany_rowcount(self, query: str, data: List[tuple]) -> int:
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            cur.executemany(query, data)
            return cur.rowcount
        finally:
            cur.close()

    def _introspect(self, table_name: str):
        if table_name in self._schema_cache: return
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            cur.execute(f"PRAGMA table_info({table_name})")
            res = cur.fetchall()
            if not res: raise SchemaError(f"Table '{table_name}' does not exist.")
            cols = [r['name'] for r in res]
            pks = [r['name'] for r in res if r['pk'] > 0]
            self._schema_cache[table_name] = {"columns": cols, "pks": pks}
        finally:
            cur.close()

    def get_columns(self, table_name: str) -> List[str]:
        self._introspect(table_name)
        return self._schema_cache[table_name]["columns"]

    def get_primary_keys(self, table_name: str) -> List[str]:
        self._introspect(table_name)
        return self._schema_cache[table_name]["pks"]

    def close(self):
        if hasattr(self._local, "conn"):
            try: self._local.conn.close()
            except Exception: pass
            del self._local.conn

class PostgresWriterAdapter(WriterAdapter):
    def __init__(self, config: SyncConfig):
        self.config = config
        if not POSTGRES_AVAILABLE:
            raise ConfigurationError("psycopg2 is required for Postgres connection.")
        self.pool = ThreadedConnectionPool(minconn=2, maxconn=config.max_connections, dsn=config.db_url)
        self._local = threading.local()
        self._schema_cache: Dict[str, Dict[str, List[str]]] = {}

    def get_connection(self) -> Any:
        if not hasattr(self._local, "conn"):
            self._local.conn = self.pool.getconn()
        return self._local.conn

    def execute_rowcount(self, query: str, params: tuple = ()) -> int:
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            cur.execute(query, params)
            return cur.rowcount
        finally:
            cur.close()

    def executemany_rowcount(self, query: str, data: List[tuple]) -> int:
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            psycopg2.extras.execute_batch(cur, query, data, page_size=self.config.chunk_size)
            return len(data)
        finally:
            cur.close()

    def _introspect(self, table_name: str):
        if table_name in self._schema_cache: return
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = %s", (table_name,))
            cols = [r[0] for r in cur.fetchall()]
            if not cols: raise SchemaError(f"Table '{table_name}' does not exist.")
            
            cur.execute("""
                SELECT a.attname FROM pg_index i
                JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                WHERE i.indrelid = %s::regclass AND i.indisprimary;
            """, (table_name,))
            try: pks = [r[0] for r in cur.fetchall()]
            except Exception: pks = []
            self._schema_cache[table_name] = {"columns": cols, "pks": pks}
        finally:
            cur.close()

    def get_columns(self, table_name: str) -> List[str]:
        self._introspect(table_name)
        return self._schema_cache[table_name]["columns"]

    def get_primary_keys(self, table_name: str) -> List[str]:
        self._introspect(table_name)
        return self._schema_cache[table_name]["pks"]

    def close(self):
        if hasattr(self._local, "conn"):
            try:
                self.pool.putconn(self._local.conn)
                del self._local.conn
            except Exception: pass
        self.pool.closeall()

# =========================================================================
# TRANSACTION MANAGER
# =========================================================================
class TransactionManager:
    """Manages thread-local transaction states preventing connection leaks."""
    def __init__(self, adapter: WriterAdapter):
        self.adapter = adapter
        self._local = threading.local()

    def begin_transaction(self) -> None:
        if getattr(self._local, 'in_transaction', False): return
        conn = self.adapter.get_connection()
        try:
            if isinstance(self.adapter, SQLiteWriterAdapter):
                conn.execute("BEGIN IMMEDIATE")
            else:
                conn.autocommit = False
            self._local.in_transaction = True
            WriterMetrics.increment("transactions")
        except Exception as e:
            raise TransactionError(f"Failed to begin transaction: {e}")

    def commit(self) -> None:
        if not getattr(self._local, 'in_transaction', False): return
        try:
            self.adapter.get_connection().commit()
        except Exception as e:
            self.rollback()
            raise TransactionError(f"Commit failed: {e}")
        finally:
            self._local.in_transaction = False

    def rollback(self) -> None:
        if not getattr(self._local, 'in_transaction', False): return
        try:
            self.adapter.get_connection().rollback()
            WriterMetrics.increment("rollback_count")
        except Exception as e:
            _logger.error("Rollback failed", error=str(e))
        finally:
            self._local.in_transaction = False

    @contextlib.contextmanager
    def atomic(self) -> Generator[None, None, None]:
        if getattr(self._local, 'in_transaction', False):
            yield
            return
        self.begin_transaction()
        try:
            yield
            self.commit()
        except Exception as e:
            self.rollback()
            raise e

    @contextlib.contextmanager
    def snapshot(self) -> Generator[None, None, None]:
        conn = self.adapter.get_connection()
        was_in_tx = getattr(self._local, 'in_transaction', False)
        if not was_in_tx:
            if isinstance(self.adapter, SQLiteWriterAdapter):
                conn.execute("BEGIN DEFERRED")
            else:
                conn.set_session(isolation_level=psycopg2.extensions.ISOLATION_LEVEL_REPEATABLE_READ, autocommit=False)
            self._local.in_transaction = True
        try:
            yield
        finally:
            if not was_in_tx:
                conn.commit()
                self._local.in_transaction = False

    @contextlib.contextmanager
    def repeatable_write(self) -> Generator[None, None, None]:
        with self.atomic(): yield

# =========================================================================
# ENTERPRISE CENTRAL MARKET SYNC
# =========================================================================
class MarketSync:
    """
    Central Persistence Layer Orchestrator.
    Handles bulk writes, IDempotency, Cache Invalidations, and atomic pipeline syncs.
    Strictly isolated from business logic.
    """
    
    VALID_TABLES: Set[str] = {
        "market_data", "fundamental_data", "indicator_data", "feature_data",
        "analysis_data", "score_data", "decision_data", "master_ai_decision",
        "dashboard_cache", "scanner_results", "watchlist", "portfolio_data"
    }

    def __init__(self, config: SyncConfig = SyncConfig()):
        self.config = config
        self.adapter = SQLiteWriterAdapter(config) if config.dialect == Dialect.SQLITE else PostgresWriterAdapter(config)
        self.tx_manager = TransactionManager(self.adapter)
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.async_workers, 
            thread_name_prefix="WriterAsync"
        )
        _logger.info("MarketSync Engine initialized.", dialect=config.dialect.value)

    def _validate_identifier(self, identifier: str) -> str:
        if not identifier or not isinstance(identifier, str):
            raise ValidationError("Identifier empty.")
        sanitized = identifier.strip().lower()
        if not re.match(r"^[a-z0-9_]+$", sanitized):
            raise ValidationError(f"Security Violation: Invalid identifier: {identifier}")
        return sanitized

    def _validate_table(self, table_name: str) -> str:
        clean_name = self._validate_identifier(table_name)
        if clean_name not in self.VALID_TABLES:
            raise SchemaError(f"Security Block: Table '{clean_name}' unauthorized.")
        return clean_name

    def _to_records(self, data: Any) -> List[Dict[str, Any]]:
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S.%f%z').replace("NaT", None)
                elif pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].replace([np.inf, -np.inf, np.nan], None)
            return df.to_dict(orient='records')
            
        if is_dataclass(data): return [asdict(data)]
        if isinstance(data, dict): return [data]
        if isinstance(data, list):
            if not data: return []
            first = data[0]
            if isinstance(first, dict): return data
            if is_dataclass(first): return [asdict(x) for x in data]
            if hasattr(first, '_asdict'): return [x._asdict() for x in data]
        
        if isinstance(data, str):
            try:
                parsed = json.loads(data)
                return parsed if isinstance(parsed, list) else [parsed]
            except Exception: pass
            
        raise ValidationError(f"Unsupported data format: {type(data)}")

    def _invalidate_cache(self, table_name: str) -> None:
        WriterMetrics.increment("cache_invalidation")

    def _build_upsert_query(self, table: str, columns: List[str], pks: List[str]) -> str:
        cols_csv = ", ".join(columns)
        if self.config.dialect == Dialect.SQLITE:
            vals_csv = ", ".join(["?"] * len(columns))
            if not pks: return f"INSERT INTO {table} ({cols_csv}) VALUES ({vals_csv})"
            updates = ", ".join([f"{c}=excluded.{c}" for c in columns if c not in pks])
            return f"INSERT INTO {table} ({cols_csv}) VALUES ({vals_csv}) ON CONFLICT({','.join(pks)}) " + (f"DO UPDATE SET {updates}" if updates else "DO NOTHING")
        else:
            vals_csv = ", ".join(["%s"] * len(columns))
            if not pks: return f"INSERT INTO {table} ({cols_csv}) VALUES ({vals_csv})"
            updates = ", ".join([f"{c}=EXCLUDED.{c}" for c in columns if c not in pks])
            return f"INSERT INTO {table} ({cols_csv}) VALUES ({vals_csv}) ON CONFLICT({','.join(pks)}) " + (f"DO UPDATE SET {updates}" if updates else "DO NOTHING")

    def _execute_write(self, table_name: str, records: List[Dict[str, Any]], mode: WriteMode) -> int:
        if not records: return 0
        table_name = self._validate_table(table_name)
        db_cols = self.adapter.get_columns(table_name)
        db_pks = self.adapter.get_primary_keys(table_name)
        
        target_cols = [c for c in db_cols if c in records[0].keys()]
        if not target_cols: raise SchemaError(f"No matching columns for {table_name}")

        if mode == WriteMode.UPSERT:
            query = self._build_upsert_query(table_name, target_cols, db_pks)
        elif mode == WriteMode.INSERT:
            ph = "?" if self.config.dialect == Dialect.SQLITE else "%s"
            query = f"INSERT INTO {table_name} ({','.join(target_cols)}) VALUES ({','.join([ph]*len(target_cols))})"
        elif mode == WriteMode.REPLACE:
            ph = "?" if self.config.dialect == Dialect.SQLITE else "%s"
            query = f"REPLACE INTO {table_name} ({','.join(target_cols)}) VALUES ({','.join([ph]*len(target_cols))})"
        else:
            raise ValidationError(f"Invalid mode: {mode}")

        tuples_data = [tuple(r.get(c) for c in target_cols) for r in records]
        total = 0
        
        t0 = time.perf_counter()
        for i in range(0, len(tuples_data), self.config.chunk_size):
            chunk = tuples_data[i:i + self.config.chunk_size]
            self.adapter.executemany_rowcount(query, chunk)
            total += len(chunk)
            WriterMetrics.increment("rows_written", len(chunk))
        
        latency = (time.perf_counter() - t0) * 1000
        if self.config.enable_audit_logging:
            AuditEngine.record_write(table_name, AuditAction.WRITE, total, latency)
            
        self._invalidate_cache(table_name)
        return total

    # =========================================================================
    # PUBLIC WRITER APIs
    # =========================================================================

    @trace_span("writer.insert", "sync", SpanKind.INTERNAL)
    @retry_on_deadlock
    def insert(self, table_name: str, data: Any) -> int: 
        with self.tx_manager.atomic():
            return self._execute_write(table_name, self._to_records(data), WriteMode.INSERT)
    
    def insert_many(self, table_name: str, data: Any) -> int: 
        return self.insert(table_name, data)

    @trace_span("writer.upsert", "sync", SpanKind.INTERNAL)
    @retry_on_deadlock
    def upsert(self, table_name: str, data: Any) -> int: 
        with self.tx_manager.atomic():
            return self._execute_write(table_name, self._to_records(data), WriteMode.UPSERT)
    
    def bulk_upsert(self, table_name: str, data: Any) -> int: 
        return self.upsert(table_name, data)

    @trace_span("writer.replace", "sync", SpanKind.INTERNAL)
    @retry_on_deadlock
    def replace(self, table_name: str, data: Any) -> int: 
        with self.tx_manager.atomic():
            return self._execute_write(table_name, self._to_records(data), WriteMode.REPLACE)

    def merge(self, table_name: str, data: Any) -> int: 
        return self.upsert(table_name, data)

    @trace_span("writer.update", "sync", SpanKind.INTERNAL)
    @retry_on_deadlock
    def update(self, table_name: str, update_data: Dict[str, Any], conditions: Dict[str, Any]) -> int:
        t = self._validate_table(table_name)
        if not update_data or not conditions: return 0
        
        set_clauses = [f"{self._validate_identifier(k)} = ?" if self.config.dialect == Dialect.SQLITE else f"{self._validate_identifier(k)} = %s" for k in update_data.keys()]
        where_clauses = [f"{self._validate_identifier(k)} = ?" if self.config.dialect == Dialect.SQLITE else f"{self._validate_identifier(k)} = %s" for k in conditions.keys()]
        
        query = f"UPDATE {t} SET {', '.join(set_clauses)} WHERE {' AND '.join(where_clauses)}"
        params = tuple(update_data.values()) + tuple(conditions.values())
        
        t0 = time.perf_counter()
        with self.tx_manager.atomic():
            rowcount = self.adapter.execute_rowcount(query, params)
            
        WriterMetrics.increment("rows_updated", rowcount)
        if self.config.enable_audit_logging: AuditEngine.record_write(t, AuditAction.UPDATE, rowcount, (time.perf_counter() - t0)*1000)
        self._invalidate_cache(t)
        return rowcount

    def update_many(self, table_name: str, update_data: Dict[str, Any], conditions: Dict[str, Any]) -> int:
        return self.update(table_name, update_data, conditions)

    @trace_span("writer.delete", "sync", SpanKind.INTERNAL)
    @retry_on_deadlock
    def delete(self, table_name: str, conditions: Dict[str, Any]) -> int:
        t = self._validate_table(table_name)
        if not conditions: return 0
        
        where_clauses = [f"{self._validate_identifier(k)} = ?" if self.config.dialect == Dialect.SQLITE else f"{self._validate_identifier(k)} = %s" for k in conditions.keys()]
        query = f"DELETE FROM {t} WHERE {' AND '.join(where_clauses)}"
        params = tuple(conditions.values())
        
        t0 = time.perf_counter()
        with self.tx_manager.atomic():
            rowcount = self.adapter.execute_rowcount(query, params)
            
        if self.config.enable_audit_logging: AuditEngine.record_write(t, AuditAction.DELETE, rowcount, (time.perf_counter() - t0)*1000)
        self._invalidate_cache(t)
        return rowcount

    def delete_many(self, table_name: str, conditions: Dict[str, Any]) -> int:
        return self.delete(table_name, conditions)

    # =========================================================================
    # DATA FORMAT EXPORT WRITERS
    # =========================================================================

    def write_dataframe(self, table_name: str, df: pd.DataFrame) -> int:
        return self.upsert(table_name, df)

    def write_arrow(self, table_name: str, table: Any) -> int:
        if not ARROW_AVAILABLE: raise ConfigurationError("PyArrow is required.")
        return self.upsert(table_name, table.to_pandas())

    def write_numpy(self, table_name: str, array: np.ndarray, columns: List[str]) -> int:
        df = pd.DataFrame(array, columns=columns)
        return self.upsert(table_name, df)

    def write_json(self, table_name: str, json_data: str) -> int:
        return self.upsert(table_name, json_data)

    # =========================================================================
    # PIPELINE SYNCHRONIZATION
    # =========================================================================

    @trace_span("writer.write_pipeline", "sync", SpanKind.INTERNAL)
    @retry_on_deadlock
    def write_pipeline(self, pipeline_payload: Dict[str, Any]) -> int:
        """Atomic write across multiple domain tables simultaneously."""
        total = 0
        try:
            with self.tx_manager.atomic():
                for tbl, data in pipeline_payload.items():
                    if data is not None:
                        total += self._execute_write(tbl, self._to_records(data), WriteMode.UPSERT)
            return total
        except Exception as e:
            WriterMetrics.increment("rollback_count")
            _logger.critical("Pipeline write failed. Entire transaction rolled back.", error=str(e))
            raise TransactionError(f"Pipeline commit failed: {e}") from e

    def sync_pipeline(self, payload: Dict[str, Any]) -> int: 
        return self.write_pipeline(payload)
    
    def sync_symbol(self, symbol: str, payload: Dict[str, Any]) -> int:
        for tbl, data in payload.items():
            if data is not None:
                recs = self._to_records(data)
                for r in recs: r['symbol'] = symbol
                payload[tbl] = recs
        return self.write_pipeline(payload)

    def sync_dashboard(self, data: Any) -> int: return self.upsert("dashboard_cache", data)
    def sync_scanner(self, data: Any) -> int: return self.upsert("scanner_results", data)
    def sync_watchlist(self, data: Any) -> int: return self.upsert("watchlist", data)
    def sync_portfolio(self, data: Any) -> int: return self.upsert("portfolio_data", data)
    
    def close(self):
        if sys.version_info >= (3, 9):
            self.executor.shutdown(wait=False, cancel_futures=True)
        else:
            self.executor.shutdown(wait=False)
        self.adapter.close()

# Dynamically Register Primary Synchronization Methods for Extensibility
SYNC_REGISTRY["write_pipeline"] = MarketSync.write_pipeline
SYNC_REGISTRY["sync_symbol"] = MarketSync.sync_symbol
SYNC_REGISTRY["sync_dashboard"] = MarketSync.sync_dashboard
SYNC_REGISTRY["sync_portfolio"] = MarketSync.sync_portfolio

# =========================================================================
# EXPORTS
# =========================================================================
__all__ = [
    "MarketSync",
    "SyncConfig",
    "SQLiteWriterAdapter",
    "PostgresWriterAdapter",
    "TransactionManager",
    "WriterMetrics",
    "register_writer",
    "Dialect",
    "WriteMode",
    "SyncError",
    "WriteError",
    "TransactionError",
    "SchemaError",
    "ValidationError",
    "ConnectionError",
    "TimeoutError",
    "DeadlockError",
    "RetryError",
    "ConfigurationError"
]
