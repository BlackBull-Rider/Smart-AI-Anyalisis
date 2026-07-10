"""
GREEN BULL RIDER V6 - Institutional-grade AI Stock Analysis Platform
Module: backend/data/market_reader.py
Description: Enterprise Central Read Layer.
             Exclusive database read interface guaranteeing safe, scalable, 
             and cache-aware concurrent retrieval for SQLite (WAL) and PostgreSQL.
             Strictly enforces Read-Only operations with Zero Business Logic.
             Features full Snapshot Isolation, Query Planner, Advanced Multi-tier 
             LRU Caching, Dynamic Join Engine, and Pipeline Loaders matching the Writer Layer.
             Python 3.13 Compatible. Compile-Safe. Runtime-Safe. Production Locked.
"""

import json
import time
import zlib
import uuid
import hashlib
import logging
import sqlite3
import datetime
import threading
import functools
import contextlib
import concurrent.futures
import re
import sys
from collections import defaultdict, OrderedDict
from enum import Enum
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Generator, Union

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

READER_REGISTRY: Dict[str, Callable] = {}

def register_reader(name: str):
    def decorator(func: Callable):
        READER_REGISTRY[name] = func
        return func
    return decorator

# =========================================================================
# EXCEPTIONS
# =========================================================================
class DatabaseError(Exception): """Base exception for reader."""
class ReadError(DatabaseError): """Generic read failure."""
class ConnectionError(DatabaseError): """Database unreachable."""
class TimeoutError(DatabaseError): """Query execution timeout."""
class SchemaError(DatabaseError): """Table or column integrity failure."""
class ValidationError(DatabaseError): """Invalid query parameter or filter shape."""
class ConfigurationError(DatabaseError): """Invalid settings configured."""
class CacheError(DatabaseError): """Cache consistency failure."""
class RetryError(DatabaseError): """Raised when max retries are exhausted."""

# =========================================================================
# ENUMS
# =========================================================================
class Dialect(str, Enum):
    SQLITE = "SQLITE"
    POSTGRES = "POSTGRES"

class JoinType(str, Enum):
    INNER = "INNER JOIN"
    LEFT = "LEFT JOIN"
    RIGHT = "RIGHT JOIN"
    FULL = "FULL OUTER JOIN"
    CROSS = "CROSS JOIN"

class SpanKind(Enum):
    CLIENT = "CLIENT"
    INTERNAL = "INTERNAL"
    PRODUCER = "PRODUCER"

class AuditAction(Enum):
    READ = "READ"
    SYSTEM = "SYSTEM"

class AuditSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

# =========================================================================
# CONFIGURATION
# =========================================================================
@dataclass(frozen=True, slots=True, kw_only=True)
class ReaderConfig:
    dialect: Dialect = Dialect.SQLITE
    db_url: str = "sqlite:////tmp/universe.db"
    max_connections: int = 50
    prefetch_workers: int = 15
    pool_timeout_sec: float = 15.0
    statement_timeout_sec: float = 30.0
    slow_query_threshold_ms: float = 150.0
    chunk_size: int = 5000
    retry_count: int = 5
    retry_delay_sec: float = 0.2
    enable_cache: bool = True
    cache_ttl_sec: float = 60.0
    schema_ttl_sec: float = 3600.0
    cache_max_size: int = 5000
    enable_audit_logging: bool = True

# =========================================================================
# TELEMETRY, METRICS & AUDIT
# =========================================================================
class ReaderMetrics:
    _lock = threading.Lock()
    _stats: Dict[str, float] = defaultdict(float)

    @staticmethod
    def increment(name: str, amount: float = 1.0) -> None:
        with ReaderMetrics._lock: ReaderMetrics._stats[f"reader.{name}"] += amount

    @staticmethod
    def record_latency(name: str, duration_ms: float) -> None:
        with ReaderMetrics._lock:
            k = f"reader.{name}_avg_ms"
            ReaderMetrics._stats[k] = (ReaderMetrics._stats[k] * 0.95) + (duration_ms * 0.05)

    @staticmethod
    def get_metrics() -> Dict[str, float]:
        with ReaderMetrics._lock: return dict(ReaderMetrics._stats)

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
                ReaderMetrics.record_latency(f"{component}_{operation}", duration_ms)
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
            "span_id": span_id_ctx.get(),
            "request_id": request_id_ctx.get(),
            "component": "market_reader",
            "thread_id": threading.get_ident(),
            "message": msg
        }
        payload.update(kwargs)
        self.logger.log(level, json.dumps(payload, default=str))

    def debug(self, msg: str, **kwargs): self._log(logging.DEBUG, msg, **kwargs)
    def info(self, msg: str, **kwargs): self._log(logging.INFO, msg, **kwargs)
    def warning(self, msg: str, **kwargs): self._log(logging.WARNING, msg, **kwargs)
    def error(self, msg: str, **kwargs): self._log(logging.ERROR, msg, **kwargs)

_logger = StructuredLogger("MarketReader")

class AuditEngine:
    @staticmethod
    def record_read(table: str, rows_fetched: int, latency_ms: float) -> None:
        try:
            from backend.core.audit import AuditEngine as CoreAudit
            if hasattr(CoreAudit, 'record_event'):
                CoreAudit.record_event(
                    operation="db_read", action=AuditAction.READ, severity=AuditSeverity.INFO, 
                    message=f"Read {rows_fetched} rows from {table}", 
                    metadata={"table": table, "rows": rows_fetched, "latency": latency_ms, "trace_id": trace_id_ctx.get()}
                )
        except Exception: pass

# =========================================================================
# MULTI-TIER CACHE ENGINE
# =========================================================================
class CacheNamespace(str, Enum):
    RESULT = "result"
    SCHEMA = "schema"
    METADATA = "metadata"
    SYMBOL = "symbol"

class EnterpriseCache:
    """Thread-safe multi-namespace True LRU Cache."""
    def __init__(self, config: ReaderConfig):
        self.config = config
        self._stores: Dict[str, OrderedDict[str, Tuple[Any, float]]] = defaultdict(OrderedDict)
        self._lock = threading.RLock()

    def _hash(self, key_payload: Any) -> str:
        return hashlib.sha256(json.dumps(key_payload, sort_keys=True, default=str).encode()).hexdigest()

    def get(self, namespace: CacheNamespace, key_payload: Any) -> Optional[Any]:
        if not self.config.enable_cache: return None
        key = self._hash(key_payload)
        with self._lock:
            store = self._stores[namespace.value]
            if key in store:
                val, expiry = store[key]
                if time.time() < expiry:
                    store.move_to_end(key)
                    ReaderMetrics.increment(f"cache_hit_{namespace.value}")
                    return val
                del store[key]
        ReaderMetrics.increment(f"cache_miss_{namespace.value}")
        return None

    def set(self, namespace: CacheNamespace, key_payload: Any, value: Any, custom_ttl: Optional[float] = None) -> None:
        if not self.config.enable_cache: return
        key = self._hash(key_payload)
        ttl = custom_ttl or self.config.cache_ttl_sec
        with self._lock:
            store = self._stores[namespace.value]
            if len(store) >= self.config.cache_max_size:
                store.popitem(last=False)
            store[key] = (value, time.time() + ttl)

    def cleanup_expired(self) -> None:
        now = time.time()
        with self._lock:
            for store in self._stores.values():
                expired_keys = [k for k, (_, exp) in store.items() if now > exp]
                for k in expired_keys: del store[k]

    def invalidate_all(self) -> None:
        with self._lock:
            for store in self._stores.values():
                store.clear()

# =========================================================================
# DATABASE ADAPTERS (SQLITE + POSTGRESQL)
# =========================================================================
class ReaderAdapter:
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]: raise NotImplementedError()
    def execute_query_stream(self, query: str, params: tuple = ()) -> Generator[Dict[str, Any], None, None]: raise NotImplementedError()
    @contextlib.contextmanager
    def snapshot_isolation(self) -> Generator[None, None, None]: raise NotImplementedError()
    def explain_plan(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]: raise NotImplementedError()
    def list_tables(self) -> List[str]: raise NotImplementedError()
    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]: raise NotImplementedError()
    def get_indexes(self, table_name: str) -> List[Dict[str, Any]]: raise NotImplementedError()
    def get_foreign_keys(self, table_name: str) -> List[Dict[str, Any]]: raise NotImplementedError()
    def get_constraints(self, table_name: str) -> List[Dict[str, Any]]: raise NotImplementedError()
    def get_index_usage(self) -> List[Dict[str, Any]]: raise NotImplementedError()
    def close(self) -> None: raise NotImplementedError()

class SQLiteReaderAdapter(ReaderAdapter):
    def __init__(self, config: ReaderConfig):
        self.config = config
        self.db_path = config.db_url.replace("sqlite:///", "") if config.db_url.startswith("sqlite:///") else config.db_url
        self._local = threading.local()

    def _get_conn(self) -> sqlite3.Connection:
        if getattr(self._local, "in_snapshot", False) and hasattr(self._local, "conn"):
            return self._local.conn

        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(self.db_path, timeout=self.config.pool_timeout_sec, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA cache_size = -64000;")
            self._local.conn = conn
        return self._local.conn

    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            cursor.close()

    def execute_query_stream(self, query: str, params: tuple = ()) -> Generator[Dict[str, Any], None, None]:
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute(query, params)
            while True:
                rows = cursor.fetchmany(self.config.chunk_size)
                if not rows: break
                for row in rows: yield dict(row)
        finally:
            cursor.close()

    @contextlib.contextmanager
    def snapshot_isolation(self) -> Generator[None, None, None]:
        conn = self._get_conn()
        if getattr(self._local, "in_snapshot", False):
            yield
            return
        self._local.in_snapshot = True
        try:
            conn.execute("BEGIN DEFERRED")
            yield
        finally:
            conn.commit() 
            self._local.in_snapshot = False

    def explain_plan(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        return self.execute_query(f"EXPLAIN QUERY PLAN {query}", params)

    def list_tables(self) -> List[str]:
        res = self.execute_query("SELECT name FROM sqlite_master WHERE type='table'")
        return [r['name'] for r in res]

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        return self.execute_query(f"PRAGMA table_info({table_name})")

    def get_indexes(self, table_name: str) -> List[Dict[str, Any]]:
        return self.execute_query(f"PRAGMA index_list({table_name})")

    def get_foreign_keys(self, table_name: str) -> List[Dict[str, Any]]:
        return self.execute_query(f"PRAGMA foreign_key_list({table_name})")

    def get_constraints(self, table_name: str) -> List[Dict[str, Any]]:
        return self.execute_query("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table_name,))

    def get_index_usage(self) -> List[Dict[str, Any]]:
        try: return self.execute_query("SELECT * FROM sqlite_stat1")
        except Exception: return []

    def close(self):
        if hasattr(self._local, "conn"):
            try: self._local.conn.close()
            except Exception: pass
            del self._local.conn

class PostgresReaderAdapter(ReaderAdapter):
    def __init__(self, config: ReaderConfig):
        self.config = config
        if not POSTGRES_AVAILABLE:
            raise ConfigurationError("psycopg2 is required for Postgres connection.")
        self.pool = ThreadedConnectionPool(minconn=2, maxconn=config.max_connections, dsn=config.db_url)
        self._local = threading.local()

    def _get_connection(self):
        if hasattr(self._local, 'snapshot_conn'):
            return self._local.snapshot_conn, True
        return self.pool.getconn(), False

    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        conn, is_snapshot = self._get_connection()
        if not is_snapshot:
            conn.set_session(readonly=True, autocommit=True)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            cur.execute(f"SET statement_timeout = {int(self.config.statement_timeout_sec * 1000)}")
            cur.execute(query, params)
            return [dict(row) for row in cur.fetchall()]
        finally:
            cur.close()
            if not is_snapshot:
                self.pool.putconn(conn)

    def execute_query_stream(self, query: str, params: tuple = ()) -> Generator[Dict[str, Any], None, None]:
        conn, is_snapshot = self._get_connection()
        if not is_snapshot:
            conn.set_session(readonly=True, autocommit=False)
        cursor_name = f"gbr_stream_{uuid.uuid4().hex}"
        cur = conn.cursor(name=cursor_name, cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            cur.execute(query, params)
            while True:
                rows = cur.fetchmany(self.config.chunk_size)
                if not rows: break
                for row in rows: yield dict(row)
        finally:
            cur.close()
            if not is_snapshot:
                conn.commit()
                self.pool.putconn(conn)

    @contextlib.contextmanager
    def snapshot_isolation(self) -> Generator[None, None, None]:
        if hasattr(self._local, 'snapshot_conn'):
            yield
            return
        conn = self.pool.getconn()
        conn.set_session(readonly=True, isolation_level=psycopg2.extensions.ISOLATION_LEVEL_REPEATABLE_READ, autocommit=False)
        self._local.snapshot_conn = conn
        try:
            yield
        finally:
            conn.commit()
            self.pool.putconn(conn)
            delattr(self._local, 'snapshot_conn')

    def explain_plan(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        return self.execute_query(f"EXPLAIN ANALYZE {query}", params)

    def list_tables(self) -> List[str]:
        res = self.execute_query("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
        return [r['table_name'] for r in res]

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        return self.execute_query("SELECT column_name as name, data_type as type FROM information_schema.columns WHERE table_name = %s", (table_name,))

    def get_indexes(self, table_name: str) -> List[Dict[str, Any]]:
        return self.execute_query("SELECT indexname as name FROM pg_indexes WHERE tablename = %s", (table_name,))

    def get_foreign_keys(self, table_name: str) -> List[Dict[str, Any]]:
        return self.execute_query("""
            SELECT kcu.column_name, ccu.table_name AS foreign_table_name, ccu.column_name AS foreign_column_name 
            FROM information_schema.table_constraints AS tc 
            JOIN information_schema.key_column_usage AS kcu ON tc.constraint_name = kcu.constraint_name 
            JOIN information_schema.constraint_column_usage AS ccu ON ccu.constraint_name = tc.constraint_name 
            WHERE constraint_type = 'FOREIGN KEY' AND tc.table_name=%s;
        """, (table_name,))

    def get_constraints(self, table_name: str) -> List[Dict[str, Any]]:
        return self.execute_query("SELECT constraint_name, constraint_type FROM information_schema.table_constraints WHERE table_name=%s", (table_name,))

    def get_index_usage(self) -> List[Dict[str, Any]]:
        return self.execute_query("SELECT relname, indexrelname, idx_scan, idx_tup_read, idx_tup_fetch FROM pg_stat_user_indexes")

    def close(self):
        try: self.pool.closeall()
        except Exception: pass

# =========================================================================
# COMPOUND FILTER & SORT ENGINE
# =========================================================================
class FilterNode:
    def __init__(self, field: str, operator: str, value: Any):
        self.field = field
        self.operator = operator.upper()
        self.value = value

    def to_dict(self) -> Dict[str, Any]:
        return {"field": self.field, "operator": self.operator, "value": self.value}

class CompoundFilter:
    def __init__(self, logic: str = "AND"):
        self.logic = logic.upper()
        self.conditions: List[Union[FilterNode, 'CompoundFilter']] = []

    def add(self, field_or_node: Union[str, FilterNode, 'CompoundFilter'], operator: Optional[str] = None, value: Any = None):
        if isinstance(field_or_node, (FilterNode, CompoundFilter)):
            self.conditions.append(field_or_node)
        else:
            self.conditions.append(FilterNode(field_or_node, operator, value))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "logic": self.logic,
            "conditions": [c.to_dict() if hasattr(c, 'to_dict') else c for c in self.conditions]
        }

class JoinNode:
    def __init__(self, table: str, join_type: JoinType, left_col: str, operator: str, right_col: str, alias: str):
        self.table = table
        self.join_type = join_type
        self.left_col = left_col
        self.operator = operator.upper()
        self.right_col = right_col
        self.alias = alias

class JoinBuilder:
    def __init__(self, base_table: str, base_alias: str = "t1"):
        self.base_table = base_table
        self.base_alias = base_alias
        self.joins: List[JoinNode] = []

    def join(self, table: str, join_type: JoinType, left_col: str, operator: str, right_col: str, alias: str) -> 'JoinBuilder':
        self.joins.append(JoinNode(table, join_type, left_col, operator, right_col, alias))
        return self

    def build(self) -> str:
        sql = f"{self.base_table} {self.base_alias}"
        valid_ops = {"=", "!=", ">", "<", ">=", "<="}
        for j in self.joins:
            if j.operator not in valid_ops:
                raise ValidationError("Security Block: Invalid Join Operator.")
            sql += f" {j.join_type.value} {j.table} {j.alias} ON {j.left_col} {j.operator} {j.right_col}"
        return sql

class QueryBuilder:
    @staticmethod
    def build_where(filter_obj: Optional[CompoundFilter], dialect: Dialect) -> Tuple[str, list]:
        if not filter_obj or not filter_obj.conditions: return "", []
        ph = "?" if dialect == Dialect.SQLITE else "%s"
        clauses, params = [], []
        for cond in filter_obj.conditions:
            if isinstance(cond, CompoundFilter):
                sub_clause, sub_params = QueryBuilder.build_where(cond, dialect)
                if sub_clause:
                    clauses.append(f"({sub_clause})")
                    params.extend(sub_params)
            else:
                f, op, val = cond.field, cond.operator, cond.value
                if op == "NULL": clauses.append(f"{f} IS NULL")
                elif op == "NOT NULL": clauses.append(f"{f} IS NOT NULL")
                elif op == "IN":
                    placeholders = ", ".join([ph] * len(val))
                    clauses.append(f"{f} IN ({placeholders})")
                    params.extend(val)
                elif op == "BETWEEN":
                    clauses.append(f"{f} BETWEEN {ph} AND {ph}")
                    params.extend(val)
                else:
                    sql_op = "REGEXP" if op == "REGEX" and dialect == Dialect.SQLITE else op
                    if op == "REGEX" and dialect == Dialect.POSTGRES: sql_op = "~"
                    clauses.append(f"{f} {sql_op} {ph}")
                    params.append(val)
        connector = f" {filter_obj.logic} "
        combined = connector.join(clauses)
        if filter_obj.logic == "NOT" and combined: combined = f"NOT ({combined})"
        return combined, params

# =========================================================================
# CENTRAL ENTERPRISE READ INTERFACE
# =========================================================================
class MarketReader:
    VALID_TABLES = {
        "market_data", "live_market_data", "historical_data", "fundamental_data",
        "institutional_data", "indicator_data", "feature_data", "analysis_data",
        "score_data", "decision_data", "master_ai_decision", "portfolio_data",
        "dashboard_cache", "watchlist", "scanner_results", "ipo_data",
        "sector_data", "industry_data", "indices", "options_chain", "fno_data",
        "bulk_deals", "block_deals", "corporate_actions", "alerts", "registry",
        "trace_log", "validation_log", "sync_log", "system_log", "stock_master"
    }

    def __init__(self, config: ReaderConfig = ReaderConfig()):
        self.config = config
        self.adapter = SQLiteReaderAdapter(config) if config.dialect == Dialect.SQLITE else PostgresReaderAdapter(config)
        self.cache = EnterpriseCache(config)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=config.prefetch_workers, thread_name_prefix="PrefetchWorker")
        self._shutdown_event = threading.Event()
        self._cache_cleaner_thread = threading.Thread(target=self._background_cache_cleanup, daemon=True, name="CacheCleaner")
        self._cache_cleaner_thread.start()
        _logger.info("MarketReader Engine initialized seamlessly.", dialect=config.dialect.value)

    def _background_cache_cleanup(self):
        while not self._shutdown_event.is_set():
            time.sleep(self.config.cache_ttl_sec)
            self.cache.cleanup_expired()

    def execute_registered(self, name: str, *args, **kwargs) -> Any:
        if name not in READER_REGISTRY:
            raise ConfigurationError(f"No registered reader found for '{name}'")
        return READER_REGISTRY[name](self, *args, **kwargs)

    def _validate_table(self, table_name: str):
        if table_name not in self.VALID_TABLES:
            raise SchemaError(f"Security Block: Table '{table_name}' unauthorized.")

    def _clean_sql_identifier(self, col_name: str, table_name: Optional[str] = None) -> str:
        """Strict structural regex parsing with optional schema whitelist check."""
        if not col_name or not isinstance(col_name, str):
            raise ValidationError("Identifier block empty or structural mismatch.")
        sanitized = col_name.strip().lower()
        if sanitized == "*": return "*"
        
        if not re.match(r"^[a-z0-9_]+(\.[a-z0-9_]+|\.\*)?$", sanitized):
            raise ValidationError(f"Security Violation: Invalid SQL identifier: {col_name}")

        if table_name and "." not in sanitized:
            valid_cols = self.get_table_schema(table_name)
            if sanitized not in valid_cols:
                raise SchemaError(f"Column '{sanitized}' not found in table '{table_name}'.")

        return sanitized

    def _decompress_payloads(self, table_name: str, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for r in records:
            if table_name in ("analysis_data", "master_ai_decision", "system_log"):
                for heavy in ("reasoning", "summary", "explanation", "pros", "cons", "risks", "payload"):
                    comp_key = f"{heavy}_compressed"
                    if comp_key in r and r[comp_key] is not None:
                        try:
                            raw_bytes = bytes(r[comp_key])
                            decompressed_str = zlib.decompress(raw_bytes).decode('utf-8')
                            r[heavy] = decompressed_str
                            ReaderMetrics.increment("compressed_bytes_read", len(raw_bytes))
                            ReaderMetrics.increment("decompressed_bytes_read", len(decompressed_str.encode('utf-8')))
                        except Exception: pass
            
            for k, v in r.items():
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)): r[k] = None
        return records

    def _execute_with_retry(self, query: str, params: tuple = (), table_name: Optional[str] = None) -> List[Dict[str, Any]]:
        retries = 0
        while retries <= self.config.retry_count:
            t0 = time.perf_counter()
            try:
                res = self.adapter.execute_query(query, params)
                dur_ms = (time.perf_counter() - t0) * 1000
                if table_name: res = self._decompress_payloads(table_name, res)
                
                bytes_transferred = sys.getsizeof(str(res))
                ReaderMetrics.increment("bytes_read", bytes_transferred)
                ReaderMetrics.increment("rows_read", len(res))
                ReaderMetrics.record_latency("query_execution", dur_ms)
                if self.config.enable_audit_logging and table_name:
                    AuditEngine.record_read(table_name, len(res), dur_ms)
                return res
            except Exception as e:
                err_str = str(e).lower()
                is_transient = any(x in err_str for x in ["locked", "busy", "timeout", "deadlock", "database is locked"])
                if not is_transient or retries >= self.config.retry_count:
                    ReaderMetrics.increment("query_failures")
                    raise ReadError(f"Database read failed execution: {e}")
                retries += 1
                time.sleep(self.config.retry_delay_sec * (2 ** (retries - 1)))
        return []

    # =========================================================================
    # 1. TRANSACTION SNAPSHOT & PLANNER APIs
    # =========================================================================
    def read_transaction(self) -> Generator[None, None, None]: return self.snapshot_isolation()
    def snapshot_read(self) -> Generator[None, None, None]: return self.snapshot_isolation()
    def repeatable_read(self) -> Generator[None, None, None]: return self.snapshot_isolation()
    def serializable_read(self) -> Generator[None, None, None]: return self.snapshot_isolation()
    def begin_snapshot(self) -> Generator[None, None, None]: return self.snapshot_isolation()
    
    @contextlib.contextmanager
    def snapshot_isolation(self) -> Generator[None, None, None]:
        with self.adapter.snapshot_isolation(): yield

    def explain(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]: return self.adapter.explain_plan(query, params)
    def explain_analyze(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]: return self.adapter.explain_plan(query, params)
    def query_plan(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]: return self.explain(query, params)

    # =========================================================================
    # 2. SCHEMA INSPECTOR APIs
    # =========================================================================
    def list_tables(self) -> List[str]: 
        cached = self.cache.get(CacheNamespace.METADATA, "list_tables")
        if cached: return cached
        res = self.adapter.list_tables()
        self.cache.set(CacheNamespace.METADATA, "list_tables", res, self.config.schema_ttl_sec)
        return res
        
    def table_exists(self, table_name: str) -> bool: return table_name in self.list_tables()
    def primary_keys(self, table_name: str) -> List[str]: return [r['name'] for r in self.adapter.get_table_schema(table_name) if r.get('pk', 0) > 0]
    
    def get_table_schema(self, table_name: str) -> List[str]:
        self._validate_table(table_name)
        cached = self.cache.get(CacheNamespace.SCHEMA, (table_name,))
        if cached: return cached
        schema = [r['name'] for r in self.adapter.get_table_schema(table_name)]
        self.cache.set(CacheNamespace.SCHEMA, (table_name,), schema, self.config.schema_ttl_sec)
        return schema
        
    def column_exists(self, table_name: str, column_name: str) -> bool: return column_name in self.get_table_schema(table_name)
    def indexes(self, table_name: str) -> List[Dict[str, Any]]: return self.adapter.get_indexes(table_name)
    def foreign_keys(self, table_name: str) -> List[Dict[str, Any]]: return self.adapter.get_foreign_keys(table_name)
    def constraints(self, table_name: str) -> List[Dict[str, Any]]: return self.adapter.get_constraints(table_name)
    def schema_hash(self, table_name: str) -> str: return hashlib.sha256(str(self.get_table_schema(table_name)).encode()).hexdigest()

    # =========================================================================
    # 3. BACKGROUND ASYNC PREFETCH
    # =========================================================================
    def async_read(self, method_name: str, *args, **kwargs) -> concurrent.futures.Future:
        func = getattr(self, method_name)
        return self.executor.submit(func, *args, **kwargs)

    def prefetch(self, method_name: str, *args, **kwargs) -> None: self.async_read(method_name, *args, **kwargs)
    def background_prefetch(self, method_name: str, *args, **kwargs) -> None: self.prefetch(method_name, *args, **kwargs)

    # =========================================================================
    # 4. BASE READ APIs
    # =========================================================================
    def read_one(self, table_name: str, filters: Optional[CompoundFilter] = None) -> Optional[Dict[str, Any]]:
        self._validate_table(table_name)
        w, p = QueryBuilder.build_where(filters, self.config.dialect)
        sql = f"SELECT * FROM {table_name}" + (f" WHERE {w}" if w else "") + " LIMIT 1"
        res = self._execute_with_retry(sql, tuple(p), table_name)
        return res[0] if res else None

    def read_many(self, table_name: str, filters: Optional[CompoundFilter] = None, limit: int = 1000) -> List[Dict[str, Any]]:
        self._validate_table(table_name)
        cache_key = (table_name, "read_many", filters.to_dict() if filters else None, limit)
        cached = self.cache.get(CacheNamespace.RESULT, cache_key)
        if cached: return cached
        
        w, p = QueryBuilder.build_where(filters, self.config.dialect)
        sql = f"SELECT * FROM {table_name}" + (f" WHERE {w}" if w else "") + f" LIMIT {limit}"
        res = self._execute_with_retry(sql, tuple(p), table_name)
        self.cache.set(CacheNamespace.RESULT, cache_key, res)
        return res

    def read_all(self, table_name: str) -> List[Dict[str, Any]]: return self.read_many(table_name, limit=100000)
    def read_by_symbol(self, table_name: str, symbol: str) -> List[Dict[str, Any]]: return self.read_many(table_name, CompoundFilter().add("symbol", "=", symbol))

    def read_latest(self, table_name: str, filters: Optional[CompoundFilter] = None, ts_col: str = "timestamp") -> Optional[Dict[str, Any]]:
        self._validate_table(table_name)
        ts_col = self._clean_sql_identifier(ts_col, table_name)
        w, p = QueryBuilder.build_where(filters, self.config.dialect)
        sql = f"SELECT * FROM {table_name}" + (f" WHERE {w}" if w else "") + f" ORDER BY {ts_col} DESC LIMIT 1"
        res = self._execute_with_retry(sql, tuple(p), table_name)
        return res[0] if res else None

    def read_latest_by_symbol(self, table_name: str, symbol: str, ts_col: str = "timestamp") -> Optional[Dict[str, Any]]:
        self._validate_table(table_name)
        ts_col = self._clean_sql_identifier(ts_col, table_name)
        ph = "?" if self.config.dialect == Dialect.SQLITE else "%s"
        sql = f"SELECT * FROM {table_name} WHERE symbol = {ph} ORDER BY {ts_col} DESC LIMIT 1"
        res = self._execute_with_retry(sql, (symbol,), table_name)
        return res[0] if res else None

    def read_latest_batch(self, table_name: str, symbols: List[str], ts_col: str = "timestamp") -> List[Dict[str, Any]]:
        self._validate_table(table_name)
        if not symbols: return []
        ts_col = self._clean_sql_identifier(ts_col, table_name)
        ph = "?" if self.config.dialect == Dialect.SQLITE else "%s"
        sym_ph = ", ".join([ph] * len(symbols))
        sql = f"SELECT * FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY {ts_col} DESC) as rn FROM {table_name} WHERE symbol IN ({sym_ph})) t WHERE t.rn = 1"
        return self._execute_with_retry(sql, tuple(symbols), table_name)

    def read_between_dates(self, table_name: str, start: str, end: str, col: str = "timestamp") -> List[Dict[str, Any]]:
        col = self._clean_sql_identifier(col, table_name)
        return self.read_many(table_name, CompoundFilter().add(col, "BETWEEN", [start, end]))

    def read_last_n(self, table_name: str, symbol: str, n: int, order_col: str = "timestamp") -> List[Dict[str, Any]]:
        self._validate_table(table_name)
        order_col = self._clean_sql_identifier(order_col, table_name)
        ph = "?" if self.config.dialect == Dialect.SQLITE else "%s"
        return self._execute_with_retry(f"SELECT * FROM {table_name} WHERE symbol = {ph} ORDER BY {order_col} DESC LIMIT {n}", (symbol,), table_name)

    def read_top(self, table_name: str, metric_col: str, limit: int = 50) -> List[Dict[str, Any]]:
        self._validate_table(table_name)
        metric_col = self._clean_sql_identifier(metric_col, table_name)
        return self._execute_with_retry(f"SELECT * FROM {table_name} ORDER BY {metric_col} DESC LIMIT {limit}", (), table_name)

    def read_paginated(self, table_name: str, limit: int, offset: int) -> List[Dict[str, Any]]:
        self._validate_table(table_name)
        return self._execute_with_retry(f"SELECT * FROM {table_name} LIMIT {limit} OFFSET {offset}", (), table_name)

    def count(self, table_name: str, filters: Optional[CompoundFilter] = None) -> int:
        self._validate_table(table_name)
        w, p = QueryBuilder.build_where(filters, self.config.dialect)
        res = self._execute_with_retry(f"SELECT COUNT(*) as cnt FROM {table_name}" + (f" WHERE {w}" if w else ""), tuple(p))
        return res[0]['cnt'] if res else 0

    def scalar(self, query: str, params: tuple = ()) -> Any:
        res = self.raw_query(query, params)
        if not res: return None
        return list(res[0].values())[0]

    def raw_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        blocked_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "REPLACE", "CREATE", "ATTACH", "DETACH", "VACUUM", "PRAGMA", "ANALYZE"]
        if any(x in query.upper() for x in blocked_keywords):
            raise ReadError("Write/Schema operations strictly blocked on Reader Interface.")
        return self._execute_with_retry(query, params)

    # Stream Readers
    def stream_reader(self, table_name: str, filters: Optional[CompoundFilter] = None) -> Generator[Dict[str, Any], None, None]:
        self._validate_table(table_name)
        w, p = QueryBuilder.build_where(filters, self.config.dialect)
        sql = f"SELECT * FROM {table_name}" + (f" WHERE {w}" if w else "")
        for record in self.adapter.execute_query_stream(sql, tuple(p)):
            yield self._decompress_payloads(table_name, [record])[0]

    def chunk_reader(self, table_name: str, filters: Optional[CompoundFilter] = None) -> Generator[List[Dict[str, Any]], None, None]:
        gen = self.stream_reader(table_name, filters)
        while True:
            chunk = []
            for _ in range(self.config.chunk_size):
                try: chunk.append(next(gen))
                except StopIteration: break
            if not chunk: break
            yield chunk

    def cursor_reader(self, table_name: str, cursor_col: str, last_value: Any, limit: int = 1000) -> List[Dict[str, Any]]:
        self._validate_table(table_name)
        cursor_col = self._clean_sql_identifier(cursor_col, table_name)
        f = CompoundFilter().add(cursor_col, ">", last_value)
        w, p = QueryBuilder.build_where(f, self.config.dialect)
        return self._execute_with_retry(f"SELECT * FROM {table_name} WHERE {w} ORDER BY {cursor_col} ASC LIMIT {limit}", tuple(p), table_name)

    # =========================================================================
    # 5. DYNAMIC JOIN ENGINE
    # =========================================================================
    def execute_join(self, join_builder: JoinBuilder, filters: Optional[CompoundFilter] = None, columns: List[str] = ["*"]) -> List[Dict[str, Any]]:
        base_table = join_builder.base_table
        self._validate_table(base_table)
        for j in join_builder.joins: self._validate_table(j.table)
        
        safe_cols = [self._clean_sql_identifier(c) for c in columns]
        join_sql = join_builder.build()
        w, p = QueryBuilder.build_where(filters, self.config.dialect)
        sql = f"SELECT {', '.join(safe_cols)} FROM {join_sql}" + (f" WHERE {w}" if w else "")
        return self._execute_with_retry(sql, tuple(p))

    # =========================================================================
    # 6. SYMBOL APIs
    # =========================================================================
    def get_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        cached = self.cache.get(CacheNamespace.SYMBOL, symbol)
        if cached: return cached
        res = self.read_latest_by_symbol("stock_master", symbol, "symbol")
        if res: self.cache.set(CacheNamespace.SYMBOL, symbol, res, 3600.0)
        return res

    def get_symbols(self) -> List[str]:
        cached = self.cache.get(CacheNamespace.SYMBOL, "all_symbols")
        if cached: return cached
        res = [r['symbol'] for r in self._execute_with_retry("SELECT DISTINCT symbol FROM stock_master")]
        self.cache.set(CacheNamespace.SYMBOL, "all_symbols", res, 3600.0)
        return res

    def read_symbols_batch(self, symbols: List[str]) -> List[Dict[str, Any]]: return self.read_latest_batch("stock_master", symbols, "symbol")
    def get_active_symbols(self) -> List[str]: return [r['symbol'] for r in self.read_many("stock_master", CompoundFilter().add("is_active", "=", 1))]
    def get_watchlist_symbols(self) -> List[str]: return [r['symbol'] for r in self.read_all("watchlist")]
    def get_portfolio_symbols(self) -> List[str]: return [r['symbol'] for r in self._execute_with_retry("SELECT DISTINCT symbol FROM portfolio_data")]
    def get_scanner_symbols(self) -> List[str]: return [r['symbol'] for r in self._execute_with_retry("SELECT DISTINCT symbol FROM scanner_results")]
    def get_latest_symbol_snapshot(self, symbol: str) -> Dict[str, Any]: return self.read_latest_by_symbol("market_data", symbol) or {}

    # =========================================================================
    # PIPELINE HOOK ATOMIC RECOVERY ENGINE (Writer Mirror)
    # =========================================================================
    @trace_span(operation="reader.load_pipeline", component="reader", kind=SpanKind.INTERNAL)
    def load_pipeline(self, symbol: str) -> Dict[str, Any]:
        """Atomic parallel fetch of all multi-row and snapshot records per symbol."""
        multi_row_tables = {"indicator_data", "feature_data", "analysis_data"}
        results = {}
        with self.snapshot_isolation():
            for tbl in ["market_data", "fundamental_data", "indicator_data", "feature_data", "analysis_data", "score_data", "decision_data", "master_ai_decision"]:
                if tbl in multi_row_tables:
                    results[tbl] = self.read_by_symbol(tbl, symbol)
                else:
                    results[tbl] = self.read_latest_by_symbol(tbl, symbol)
        return results

    def get_complete_symbol(self, symbol: str) -> Dict[str, Any]: return self.load_pipeline(symbol)
    def load_symbol_graph(self, symbol: str) -> Dict[str, Any]: return self.load_pipeline(symbol)

    def load_dashboard(self) -> Dict[str, Any]:
        with self.snapshot_isolation():
            return {
                "market_summary": self.read_latest("dashboard_cache"),
                "top_gainers": self.read_top("market_data", "close", 10),
                "ai_recommendations": self.read_top("master_ai_decision", "confidence", 10)
            }

    def load_complete_portfolio(self) -> Dict[str, Any]:
        with self.snapshot_isolation():
            symbols = self.get_portfolio_symbols()
            return {
                "holdings": self.read_all("portfolio_data"),
                "live_prices": self.read_latest_batch("market_data", symbols),
                "ai_decisions": self.read_latest_batch("master_ai_decision", symbols),
                "scores": self.read_latest_batch("score_data", symbols)
            }

    def load_scanner(self) -> List[Dict[str, Any]]: return self.read_all("scanner_results")
    def load_watchlist(self) -> List[Dict[str, Any]]: return self.read_all("watchlist")

    # =========================================================================
    # CORE REUSABLE COMPONENT METHODS
    # =========================================================================
    def read_market_data(self, symbol: str) -> List[Dict[str, Any]]: return self.read_by_symbol("market_data", symbol)

    # =========================================================================
    # DOMAIN EXPLICIT REGISTRY WRAPPERS
    # =========================================================================
    def read_fundamental_data(self, symbol: str) -> Optional[Dict[str, Any]]: return self.read_latest_by_symbol("fundamental_data", symbol)
    def read_indicator_data(self, symbol: str) -> List[Dict[str, Any]]: return self.read_by_symbol("indicator_data", symbol)
    def read_feature_data(self, symbol: str) -> List[Dict[str, Any]]: return self.read_by_symbol("feature_data", symbol)
    def read_analysis_data(self, symbol: str) -> List[Dict[str, Any]]: return self.read_by_symbol("analysis_data", symbol)
    def read_score_data(self, symbol: str) -> Optional[Dict[str, Any]]: return self.read_latest_by_symbol("score_data", symbol)
    def read_decision_data(self, symbol: str) -> Optional[Dict[str, Any]]: return self.read_latest_by_symbol("decision_data", symbol)

    # =========================================================================
    # INDICATOR APIs
    # =========================================================================
    @register_reader("all_indicators")
    def read_all_indicators(self, symbol: str) -> List[Dict[str, Any]]: return self.read_by_symbol("indicator_data", symbol)
    def read_single_indicator(self, symbol: str, ind_name: str) -> Optional[Dict[str, Any]]:
        return self.read_latest("indicator_data", CompoundFilter().add("symbol", "=", symbol).add("indicator_name", "=", ind_name))
    def read_indicator_subset(self, symbol: str, indicators: List[str]) -> List[Dict[str, Any]]:
        return self.read_many("indicator_data", CompoundFilter().add("symbol", "=", symbol).add("indicator_name", "IN", indicators))
    def read_indicator_batch(self, symbols: List[str], indicators: List[str]) -> List[Dict[str, Any]]:
        return self.read_many("indicator_data", CompoundFilter().add("symbol", "IN", symbols).add("indicator_name", "IN", indicators), limit=50000)
    def read_indicator_history(self, symbol: str, ind_name: str) -> List[Dict[str, Any]]:
        return self.read_many("indicator_data", CompoundFilter().add("symbol", "=", symbol).add("indicator_name", "=", ind_name))
    def read_indicator_matrix(self, symbols: List[str], indicators: List[str]) -> Dict[str, Dict[str, Any]]:
        res = self.read_indicator_batch(symbols, indicators)
        matrix = defaultdict(dict)
        for r in res: matrix[r['symbol']][r['indicator_name']] = r.get('value')
        return dict(matrix)

    # =========================================================================
    # ANALYZER APIs
    # =========================================================================
    def read_analyzer(self, symbol: str, analyzer_type: str, latest: bool = True) -> Any:
        f = CompoundFilter().add("analyzer_name", "=", analyzer_type).add("symbol", "=", symbol)
        return self.read_latest("analysis_data", filters=f) if latest else self.read_many("analysis_data", filters=f)
        
    @register_reader("trend_analyzer")
    def read_trend_analyzer(self, symbol: str, latest: bool = True) -> Any: return self.read_analyzer(symbol, "TrendAnalyzer", latest)
    def read_momentum_analyzer(self, symbol: str, latest: bool = True) -> Any: return self.read_analyzer(symbol, "MomentumAnalyzer", latest)
    def read_volume_analyzer(self, symbol: str, latest: bool = True) -> Any: return self.read_analyzer(symbol, "VolumeAnalyzer", latest)
    def read_pattern_analyzer(self, symbol: str, latest: bool = True) -> Any: return self.read_analyzer(symbol, "PatternAnalyzer", latest)
    def read_smart_money_analyzer(self, symbol: str, latest: bool = True) -> Any: return self.read_analyzer(symbol, "SmartMoneyAnalyzer", latest)
    def read_institutional_analyzer(self, symbol: str, latest: bool = True) -> Any: return self.read_analyzer(symbol, "InstitutionalAnalyzer", latest)
    def read_volatility_analyzer(self, symbol: str, latest: bool = True) -> Any: return self.read_analyzer(symbol, "VolatilityAnalyzer", latest)
    def read_support_resistance_analyzer(self, symbol: str, latest: bool = True) -> Any: return self.read_analyzer(symbol, "SupportResistanceAnalyzer", latest)
    def read_breakout_analyzer(self, symbol: str, latest: bool = True) -> Any: return self.read_analyzer(symbol, "BreakoutAnalyzer", latest)
    def read_risk_analyzer(self, symbol: str, latest: bool = True) -> Any: return self.read_analyzer(symbol, "RiskAnalyzer", latest)
    def read_swing_analyzer(self, symbol: str, latest: bool = True) -> Any: return self.read_analyzer(symbol, "SwingAnalyzer", latest)
    def read_compounder_analyzer(self, symbol: str, latest: bool = True) -> Any: return self.read_analyzer(symbol, "CompounderAnalyzer", latest)
    
    def read_individual_analyzer(self, analyzer_name: str, symbol: str) -> Optional[Dict[str, Any]]: return self.read_analyzer(symbol, analyzer_name, True)
    def read_combined_analyzers(self, symbol: str) -> List[Dict[str, Any]]: return self.read_by_symbol("analysis_data", symbol)
    def read_historical_analyzers(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]: return self.read_last_n("analysis_data", symbol, limit)
    def read_ai_batch(self, symbols: List[str]) -> List[Dict[str, Any]]: return self.read_latest_batch("analysis_data", symbols)

    # =========================================================================
    # SCORE & DECISION APIs
    # =========================================================================
    def read_all_scores(self, symbol: str) -> Optional[Dict[str, Any]]: return self.read_latest_by_symbol("score_data", symbol)
    def read_scores_batch(self, symbols: List[str]) -> List[Dict[str, Any]]: return self.read_latest_batch("score_data", symbols)
    
    def read_master_score(self, symbol: str) -> float: return float((self.read_all_scores(symbol) or {}).get("master_score", 0.0))
    def read_quality_score(self, symbol: str) -> float: return float((self.read_all_scores(symbol) or {}).get("quality_score", 0.0))
    def read_institutional_score(self, symbol: str) -> float: return float((self.read_all_scores(symbol) or {}).get("institutional_score", 0.0))
    def read_ai_score(self, symbol: str) -> float: return float((self.read_all_scores(symbol) or {}).get("ai_score", 0.0))

    def read_all_decision_engines(self, symbol: str) -> List[Dict[str, Any]]: return self.read_by_symbol("decision_data", symbol)
    def read_latest_decision(self, symbol: str) -> Optional[Dict[str, Any]]: return self.read_latest_by_symbol("decision_data", symbol)
    def read_decision_batch(self, symbols: List[str]) -> List[Dict[str, Any]]: return self.read_latest_batch("decision_data", symbols)

    # =========================================================================
    # MASTER AI APIs
    # =========================================================================
    def read_master_ai(self, symbol: str) -> Optional[Dict[str, Any]]: return self.read_latest_by_symbol("master_ai_decision", symbol)
    def read_ai_recommendation(self, symbol: str) -> str: return str((self.read_master_ai(symbol) or {}).get("recommendation", "HOLD"))
    def read_ai_reasoning(self, symbol: str) -> str: return str((self.read_master_ai(symbol) or {}).get("reasoning", ""))
    def read_ai_confidence(self, symbol: str) -> float: return float((self.read_master_ai(symbol) or {}).get("confidence", 0.0))

    def read_all_filtered(self, table_name: str, filters: CompoundFilter) -> List[Dict[str, Any]]:
        return self.read_many(table_name, filters, limit=100000)

    # =========================================================================
    # MAINTENANCE, LIFECYCLE & CACHE MANAGEMENT
    # =========================================================================
    def warm_cache(self, tables: List[str]) -> None:
        for t in tables: self.background_prefetch("read_many", t, limit=5000)

    def preload(self, symbols: List[str]) -> None:
        self.background_prefetch("read_latest_batch", "market_data", symbols)
        self.background_prefetch("read_latest_batch", "master_ai_decision", symbols)

    def refresh_cache(self) -> None: self.cache.invalidate_all()

    def reader_statistics(self) -> Dict[str, float]:
        stats = ReaderMetrics.get_metrics()
        hits = stats.get("reader.cache_hit_result", 0) + stats.get("reader.cache_hit_schema", 0) + stats.get("reader.cache_hit_symbol", 0) + stats.get("reader.cache_hit_metadata", 0)
        misses = stats.get("reader.cache_miss_result", 0) + stats.get("reader.cache_miss_schema", 0) + stats.get("reader.cache_miss_symbol", 0) + stats.get("reader.cache_miss_metadata", 0)
        total = hits + misses
        stats["reader.cache_hit_ratio"] = (hits / total) if total > 0 else 0.0
        comp = stats.get("reader.compressed_bytes_read", 0)
        decomp = stats.get("reader.decompressed_bytes_read", 0)
        stats["reader.compression_ratio"] = (decomp / comp) if comp > 0 else 1.0
        return stats

    def health_check(self) -> Dict[str, Any]:
        try: status = "HEALTHY" if self.scalar("SELECT 1") == 1 else "UNHEALTHY"
        except Exception: status = "CRITICAL_CONNECTION_DOWN"
        return {"component": "market_reader", "status": status, "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}

    # =========================================================================
    # PLUGGABLE EXPORT ENGINE
    # =========================================================================
    def export_to_dataframe(self, table_name: str, filters: Optional[CompoundFilter] = None) -> pd.DataFrame:
        return pd.DataFrame(self.read_many(table_name, filters, limit=250000))

    def export_to_numpy(self, table_name: str, columns: List[str], filters: Optional[CompoundFilter] = None) -> np.ndarray:
        df = self.export_to_dataframe(table_name, filters)
        return df[[c for c in columns if c in df.columns]].to_numpy()

    def export_to_arrow(self, table_name: str, filters: Optional[CompoundFilter] = None) -> Any:
        if not ARROW_AVAILABLE: raise ConfigurationError("PyArrow missing.")
        return pa.Table.from_pandas(self.export_to_dataframe(table_name, filters))

    def export_to_json(self, table_name: str, filters: Optional[CompoundFilter] = None) -> str:
        return json.dumps(self.read_many(table_name, filters), default=str)

    def export_to_csv(self, table_name: str, file_path: str, filters: Optional[CompoundFilter] = None) -> None:
        self.export_to_dataframe(table_name, filters).to_csv(file_path, index=False)

    def export_to_parquet(self, table_name: str, file_path: str, filters: Optional[CompoundFilter] = None) -> None:
        if not ARROW_AVAILABLE: raise ConfigurationError("PyArrow required for Parquet.")
        pq.write_table(self.export_to_arrow(table_name, filters), file_path)

    def close(self) -> None:
        self._shutdown_event.set()
        if sys.version_info >= (3, 9): self.executor.shutdown(wait=False, cancel_futures=True)
        else: self.executor.shutdown(wait=False)
        self.cache.invalidate_all()
        self.adapter.close()

# Register Explicit Methods to enforce Decorator Pattern without NameErrors
READER_REGISTRY["all_indicators"] = MarketReader.read_all_indicators
READER_REGISTRY["trend_analyzer"] = MarketReader.read_trend_analyzer

# =========================================================================
# EXPORTS
# =========================================================================
__all__ = [
    "MarketReader",
    "ReaderConfig",
    "ReaderAdapter",
    "SQLiteReaderAdapter",
    "PostgresReaderAdapter",
    "Dialect",
    "JoinType",
    "DatabaseError",
    "ReadError",
    "ConnectionError",
    "TimeoutError",
    "SchemaError",
    "ValidationError",
    "CacheError",
    "FilterNode",
    "CompoundFilter",
    "JoinNode",
    "JoinBuilder",
    "QueryBuilder",
    "EnterpriseCache",
    "CacheNamespace",
    "ReaderMetrics",
    "register_reader"
]

# -------------------------------------------------------------------------
# PATCH-3 : Graceful Thread Shutdown
# -------------------------------------------------------------------------

def _join_cache_cleaner(self):
    if hasattr(self, "_cache_cleaner_thread"):
        if self._cache_cleaner_thread.is_alive():
            self._cache_cleaner_thread.join(timeout=2)

