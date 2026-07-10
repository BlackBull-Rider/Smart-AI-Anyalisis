"""
GREEN BULL RIDER V6 - Institutional-grade AI Stock Analysis Platform
Module: backend/data/market_sync.py
Description: Enterprise Central Writer & Synchronization Layer with Auto-Schema Evolution.
"""

import os, sys, json, time, uuid, logging, sqlite3, datetime, threading, contextlib, concurrent.futures, re, functools, requests, io
from enum import Enum
from collections import defaultdict
from contextvars import ContextVar
from dataclasses import dataclass, field, is_dataclass, asdict
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Type, Union, Generator, Set
import pandas as pd
import numpy as np

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

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="SYSTEM")
trace_id_ctx: ContextVar[str] = ContextVar("trace_id", default="")
span_id_ctx: ContextVar[str] = ContextVar("span_id", default="")
SYNC_REGISTRY: Dict[str, Callable] = {}

def register_writer(name: str):
    def decorator(func: Callable):
        SYNC_REGISTRY[name] = func
        return func
    return decorator

class SyncError(Exception): pass
class WriteError(SyncError): pass
class TransactionError(SyncError): pass
class SchemaError(SyncError): pass
class ValidationError(SyncError): pass
class ConnectionError(SyncError): pass
class TimeoutError(SyncError): pass
class DeadlockError(SyncError): pass
class RetryError(SyncError): pass
class ConfigurationError(SyncError): pass

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

@dataclass(frozen=True, slots=True, kw_only=True)
class SyncConfig:
    dialect: Dialect = Dialect.SQLITE
    db_url: str = "sqlite:////data/data/com.termux/files/home/Smart-AI-Anyalisis/database/universe.db"
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

class WriterMetrics:
    _lock = threading.Lock()
    _stats: Dict[str, float] = defaultdict(float)
    @staticmethod
    def increment(name: str, amount: float = 1.0):
        with WriterMetrics._lock: WriterMetrics._stats[f"writer.{name}"] += amount
    @staticmethod
    def record_latency(name: str, duration_ms: float):
        with WriterMetrics._lock:
            k = f"writer.{name}_avg_ms"
            WriterMetrics._stats[k] = (WriterMetrics._stats[k] * 0.95) + (duration_ms * 0.05)
    @staticmethod
    def get_metrics():
        with WriterMetrics._lock: return dict(WriterMetrics._stats)

def trace_span(operation: str, component: str, kind: SpanKind):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            new_span = uuid.uuid4().hex[:8]
            t_id = trace_id_ctx.get() or uuid.uuid4().hex
            t_trace = trace_id_ctx.set(t_id)
            t_span = span_id_ctx.set(new_span)
            t0 = time.perf_counter()
            try: return func(*args, **kwargs)
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
    def _log(self, level, msg, **kwargs):
        payload = {"timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), "level": logging.getLevelName(level), "message": msg}
        payload.update(kwargs)
        self.logger.log(level, json.dumps(payload, default=str))
    def info(self, msg, **kwargs): self._log(logging.INFO, msg, **kwargs)
    def warning(self, msg, **kwargs): self._log(logging.WARNING, msg, **kwargs)
    def error(self, msg, **kwargs): self._log(logging.ERROR, msg, **kwargs)
    def critical(self, msg, **kwargs): self._log(logging.CRITICAL, msg, **kwargs)

_logger = StructuredLogger("MarketSync")

class AuditEngine:
    @staticmethod
    def record_write(table: str, action: AuditAction, rows: int, latency_ms: float): pass

def retry_on_deadlock(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs) -> Any:
        retries, last_err = 0, None
        while retries <= self.config.retry_count:
            try: return func(self, *args, **kwargs)
            except Exception as e:
                last_err = e
                if not any(x in str(e).lower() for x in ["locked", "deadlock", "busy", "timeout"]): raise e
                retries += 1
                if retries <= self.config.retry_count: time.sleep(self.config.retry_delay_sec * (self.config.backoff_multiplier ** (retries - 1)))
        raise DeadlockError(f"Operation failed: {last_err}") from last_err
    return wrapper

class WriterAdapter:
    def get_connection(self): raise NotImplementedError()
    def execute_rowcount(self, query, params=()): raise NotImplementedError()
    def executemany_rowcount(self, query, data): raise NotImplementedError()
    def get_columns(self, table_name): raise NotImplementedError()
    def get_primary_keys(self, table_name): raise NotImplementedError()
    def close(self): raise NotImplementedError()

class SQLiteWriterAdapter(WriterAdapter):
    def __init__(self, config: SyncConfig):
        self.config = config
        self.db_path = config.db_url.replace("sqlite:///", "")
        self._local = threading.local()
        self._schema_cache, self._wal_configured, self._init_lock = {}, False, threading.Lock()
    def get_connection(self):
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
            self._local.conn = conn
        return self._local.conn
    def execute_rowcount(self, query, params=()):
        conn = self.get_connection()
        cur = conn.cursor()
        try: cur.execute(query, params); return cur.rowcount
        finally: cur.close()
    def executemany_rowcount(self, query, data):
        conn = self.get_connection()
        cur = conn.cursor()
        try: cur.executemany(query, data); return cur.rowcount
        finally: cur.close()
    def _introspect(self, table_name):
        if table_name in self._schema_cache: return
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            cur.execute(f"PRAGMA table_info({table_name})")
            res = cur.fetchall()
            if not res:
                # টেবিল না থাকলে অটো ক্রিয়েট হবে
                cur.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (symbol TEXT, trade_date TEXT, PRIMARY KEY(symbol, trade_date))")
                cur.execute(f"PRAGMA table_info({table_name})")
                res = cur.fetchall()
            cols = [r['name'] for r in res]
            pks = [r['name'] for r in res if r['pk'] > 0]
            self._schema_cache[table_name] = {"columns": cols, "pks": pks}
        finally: cur.close()
    def get_columns(self, table_name):
        self._introspect(table_name); return self._schema_cache[table_name]["columns"]
    def get_primary_keys(self, table_name):
        self._introspect(table_name); return self._schema_cache[table_name]["pks"]
    def close(self):
        if hasattr(self._local, "conn"):
            try: self._local.conn.close()
            except: pass

class TransactionManager:
    def __init__(self, adapter): self.adapter, self._local = adapter, threading.local()
    def begin_transaction(self):
        if getattr(self._local, 'in_transaction', False): return
        self.adapter.get_connection().execute("BEGIN IMMEDIATE")
        self._local.in_transaction = True
    def commit(self):
        if not getattr(self._local, 'in_transaction', False): return
        try: self.adapter.get_connection().commit()
        except Exception as e: self.rollback(); raise
        finally: self._local.in_transaction = False
    def rollback(self):
        if not getattr(self._local, 'in_transaction', False): return
        try: self.adapter.get_connection().rollback()
        except: pass
        finally: self._local.in_transaction = False
    @contextlib.contextmanager
    def atomic(self):
        if getattr(self._local, 'in_transaction', False): yield; return
        self.begin_transaction()
        try: yield; self.commit()
        except Exception as e: self.rollback(); raise e

class MarketSync:
    VALID_TABLES = {"market_data", "fundamental_data", "indicator_data", "feature_data", "analysis_data", "score_data", "decision_data", "master_ai_decision", "dashboard_cache", "scanner_results", "watchlist", "portfolio_data", "equity_master"}

    def __init__(self, config: SyncConfig = SyncConfig()):
        self.config = config
        self.adapter = SQLiteWriterAdapter(config)
        self.tx_manager = TransactionManager(self.adapter)

    def sync_universe(self) -> int:
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        headers = {"User-Agent": "Mozilla/5.0"}
        _logger.info("Starting Enterprise Universe Sync...")
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code != 200: raise ConnectionError(f"Archive Unreachable: {response.status_code}")
            df = pd.read_csv(io.StringIO(response.text))
            df = df[['SYMBOL', 'NAME OF COMPANY']].rename(columns={
                'SYMBOL': 'symbol',
                'NAME OF COMPANY': 'company_name'
            })

            df["exchange"] = "NSE"
            df["is_active"] = 1

            with self.tx_manager.atomic():
                row_count = self._execute_write(
                    "stock_master",
                    self._to_records(df),
                    WriteMode.UPSERT
                )
            _logger.info(f"Universe Sync Completed. {row_count} symbols processed.")
            return row_count
        except Exception as e:
            _logger.critical(f"Universe Sync Failed: {e}")
            return 0

    def _validate_table(self, table_name: str) -> str: return table_name.strip().lower()

    def _to_records(self, data: Any) -> List[Dict[str, Any]]:
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]): df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S').replace("NaT", None)
                elif pd.api.types.is_numeric_dtype(df[col]): df[col] = df[col].replace([np.inf, -np.inf, np.nan], None)
            return df.to_dict(orient='records')
        if isinstance(data, list): return data
        return [data]

    def _execute_write(self, table_name: str, records: List[Dict[str, Any]], mode: WriteMode) -> int:
        if not records: return 0
        table_name = self._validate_table(table_name)
        db_cols = self.adapter.get_columns(table_name)
        db_pks = self.adapter.get_primary_keys(table_name)

        # 🚀 AUTO SCHEMA EVOLUTION (Missing Column Generator)
        incoming_cols = list(records[0].keys())
        missing_cols = [c for c in incoming_cols if c not in db_cols]
        
        if missing_cols:
            conn = self.adapter.get_connection()
            cur = conn.cursor()
            for col in missing_cols:
                val = records[0][col]
                col_type = "TEXT"
                if isinstance(val, int): col_type = "INTEGER"
                elif isinstance(val, float): col_type = "REAL"
                elif isinstance(val, bool): col_type = "BOOLEAN"
                try:
                    cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} {col_type};")
                    _logger.info(f"⚙️ Auto-created missing column: '{col}' ({col_type}) in '{table_name}'")
                except Exception as e:
                    _logger.warning(f"Failed to auto-add column '{col}': {e}")
            if hasattr(cur, 'close'): cur.close()
            if table_name in self.adapter._schema_cache: del self.adapter._schema_cache[table_name]
            db_cols = self.adapter.get_columns(table_name)
        # =======================================================

        target_cols = [c for c in db_cols if c in records[0].keys()]
        if mode == WriteMode.UPSERT:
            cols_csv = ", ".join(target_cols)
            vals_csv = ", ".join(["?"] * len(target_cols))
            if not db_pks: query = f"INSERT INTO {table_name} ({cols_csv}) VALUES ({vals_csv})"
            else:
                updates = ", ".join([f"{c}=excluded.{c}" for c in target_cols if c not in db_pks])
                query = f"INSERT INTO {table_name} ({cols_csv}) VALUES ({vals_csv}) ON CONFLICT({','.join(db_pks)}) " + (f"DO UPDATE SET {updates}" if updates else "DO NOTHING")
        else:
            query = f"REPLACE INTO {table_name} ({','.join(target_cols)}) VALUES ({','.join(['?']*len(target_cols))})"

        tuples_data = [tuple(r.get(c) for c in target_cols) for r in records]
        total = 0
        for i in range(0, len(tuples_data), self.config.chunk_size):
            chunk = tuples_data[i:i + self.config.chunk_size]
            self.adapter.executemany_rowcount(query, chunk)
            total += len(chunk)
        return total

    @retry_on_deadlock
    def insert(self, table_name: str, data: Any) -> int: 
        with self.tx_manager.atomic(): return self._execute_write(table_name, self._to_records(data), WriteMode.INSERT)
    
    @retry_on_deadlock
    def upsert(self, table_name: str, data: Any) -> int: 
        with self.tx_manager.atomic(): return self._execute_write(table_name, self._to_records(data), WriteMode.UPSERT)
    
    @retry_on_deadlock
    def write_pipeline(self, pipeline_payload: Dict[str, Any]) -> int:
        total = 0
        with self.tx_manager.atomic():
            for tbl, data in pipeline_payload.items():
                if data is not None: total += self._execute_write(tbl, self._to_records(data), WriteMode.UPSERT)
        return total

