"""
GREEN BULL RIDER V6 - Institutional-grade AI Stock Analysis Platform
Module: backend/database/connection.py
Description: Centralized, Thread-Local Database Connection Manager.
             Acts as the absolute single source of truth for all database access.
             Provides enterprise-grade pooling abstraction, nested savepoint transactions,
             leak-proof Data Extractors (fetch_all), automated Retries, and rich Observability.
             Fully decoupled from business logic and SQL definitions.
"""

import sqlite3
import threading
import time
from enum import Enum
from contextlib import closing
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Final, Self

from backend.config.settings import settings
from backend.core.logger import AppLogger
from backend.core.exceptions import (
    DatabaseError,
    DatabaseConnectionError,
    DatabaseTimeoutError,
    DatabaseWriteError,
    DatabaseTransactionError,
    IntegrityError
)
from backend.core.trace import TraceEngine, SpanKind, TraceLevel
from backend.core.audit import AuditEngine, AuditAction, AuditResult, AuditSeverity
from backend.core.retry import retry, RetryStrategy

# -------------------------------------------------------------------------
# TYPE HINTS & CONSTANTS
# -------------------------------------------------------------------------
_logger = AppLogger("DatabaseConnection")


# -------------------------------------------------------------------------
# ENUMS
# -------------------------------------------------------------------------

class ConnectionState(str, Enum):
    """Lifecycle states for a database connection."""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    ERROR = "ERROR"
    CLOSED = "CLOSED"


class IsolationLevel(str, Enum):
    """Standard isolation levels for transaction control."""
    DEFERRED = "DEFERRED"
    IMMEDIATE = "IMMEDIATE"
    EXCLUSIVE = "EXCLUSIVE"


# -------------------------------------------------------------------------
# DATACLASSES
# -------------------------------------------------------------------------

@dataclass(slots=True)
class ConnectionStatistics:
    """Mutable telemetry tracking for a thread-local database connection."""
    query_count: int = 0
    transaction_count: int = 0
    error_count: int = 0
    total_execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_count": self.query_count,
            "transaction_count": self.transaction_count,
            "error_count": self.error_count,
            "total_execution_time_ms": round(self.total_execution_time_ms, 3)
        }


# -------------------------------------------------------------------------
# CURSOR LEAK PREVENTION
# -------------------------------------------------------------------------

class ManagedCursor:
    """
    Enterprise safeguard against cursor memory leaks.
    Acts as a proxy to sqlite3.Cursor. Ensures automatic closure upon context exit
    or safe garbage collection during interpreter shutdown.
    """
    __slots__ = ('_cursor', '_closed')

    def __init__(self, cursor: sqlite3.Cursor) -> None:
        self._cursor = cursor
        self._closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            try:
                if hasattr(self, '_cursor') and self._cursor is not None:
                    self._cursor.close()
            except Exception:
                pass
            finally:
                self._closed = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)

    def __iter__(self) -> Any:
        return iter(self._cursor)

    def __del__(self) -> None:
        """Exception-safe fallback destructor for neglected cursors."""
        try:
            self.close()
        except Exception:
            pass


# -------------------------------------------------------------------------
# POOL ABSTRACTION
# -------------------------------------------------------------------------

class SQLiteConnectionPool:
    """
    Abstracts thread-local storage and active connection registries.
    Uses threading.Thread objects as keys to bypass OS thread_id reuse vulnerabilities.
    """
    def __init__(self) -> None:
        self._local = threading.local()
        # Key: Thread object directly, Value: sqlite3.Connection
        self._registry: Dict[threading.Thread, sqlite3.Connection] = {}
        self._lock = threading.Lock()

    def get_local_state(self) -> Any:
        """Safely initializes and retrieves thread-local connection states."""
        if not hasattr(self._local, "state"):
            self._local.state = ConnectionState.DISCONNECTED
            self._local.conn = None
            self._local.stats = ConnectionStatistics()
            self._local.transaction_depth = 0
            self._local.in_transaction = False
        return self._local

    def register(self, conn: sqlite3.Connection) -> None:
        """Registers a connection securely binding it to the current Thread object."""
        with self._lock:
            self._registry[threading.current_thread()] = conn

    def unregister(self) -> None:
        """Removes the current thread's connection from the registry."""
        with self._lock:
            self._registry.pop(threading.current_thread(), None)

    def cleanup_dead_threads(self) -> None:
        """Actively identifies and purges connections tied to dead threads safely."""
        with self._lock:
            dead_threads = [t for t in self._registry.keys() if not t.is_alive()]
            for t in dead_threads:
                conn = self._registry.pop(t, None)
                if conn:
                    try:
                        conn.close()
                    except Exception:
                        pass

    def active_count(self) -> int:
        self.cleanup_dead_threads()
        with self._lock:
            return len(self._registry)

    def close_all(self) -> None:
        """Forcefully closes all active connections globally (e.g., shutdown sequence)."""
        with self._lock:
            for conn in self._registry.values():
                try:
                    conn.close()
                except Exception:
                    pass
            self._registry.clear()


# -------------------------------------------------------------------------
# DATABASE CONNECTION MANAGER (SINGLETON)
# -------------------------------------------------------------------------

class DatabaseConnection:
    """
    Enterprise-grade Database Connection Manager.
    Enforces Strict Thread-Local isolation for SQLite.
    Integrates automatically with Trace, Audit, Retry, and Logging pipelines.
    Supports Nested Transactions via Savepoints, Statement Caching, and Helper APIs.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DatabaseConnection, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        self._pool = SQLiteConnectionPool()

    def _map_exception(self, e: Exception, operation: str) -> DatabaseError:
        """Translates raw driver exceptions into standard platform exceptions."""
        err_msg = str(e).lower()
        
        if isinstance(e, sqlite3.IntegrityError):
            return IntegrityError(f"Integrity violation during {operation}.", cause=e)
            
        if isinstance(e, sqlite3.OperationalError):
            if "timeout" in err_msg or "locked" in err_msg:
                return DatabaseTimeoutError(f"Database lock timeout during {operation}.", cause=e)
            if "readonly" in err_msg:
                return DatabaseWriteError(f"Attempted write on read-only database during {operation}.", cause=e)
            return DatabaseConnectionError(f"Operational failure during {operation}.", cause=e)
            
        if isinstance(e, sqlite3.ProgrammingError):
            return DatabaseWriteError(f"Programming constraint violated during {operation}.", cause=e)

        return DatabaseError(f"Unexpected database error during {operation}.", cause=e)

    def _handle_failure(self, e: Exception, operation: str) -> DatabaseError:
        """Centralized failure handler emitting full traces, exception stacks, and audits."""
        local = self._pool.get_local_state()
        local.stats.error_count += 1
        
        mapped_error = self._map_exception(e, operation)
        metadata = {
            "db.operation": operation,
            "db.error_type": mapped_error.__class__.__name__,
            "db.thread_name": threading.current_thread().name
        }

        # 1. Trace Integration (Full Stack Trace + Tags + Metadata)
        TraceEngine.record_exception(mapped_error)
        TraceEngine.attach_tags({"db.status": "ERROR", "db.module": "connection"})
        for key, val in metadata.items():
            TraceEngine.attach_metadata(key, val)

        # 2. Audit Integration
        AuditEngine.record(
            operation=operation,
            action=AuditAction.SYSTEM,
            result=AuditResult.FAILURE,
            severity=AuditSeverity.CRITICAL,
            message=f"Database failure: {mapped_error.__class__.__name__}",
            metadata=metadata
        )
        
        # 3. Logger Integration
        _logger.error(f"Database execution failure during {operation}", exc_info=mapped_error, metadata=metadata)
        
        return mapped_error

    def connect(self) -> sqlite3.Connection:
        """Establishes a thread-local SQLite connection with optimized caching."""
        local = self._pool.get_local_state()
        
        if local.state == ConnectionState.CONNECTED and local.conn is not None:
            return local.conn

        local.state = ConnectionState.CONNECTING
        
        db_dir = settings.app.base_dir / settings.database.sqlite_db_folder
        db_dir.mkdir(parents=True, exist_ok=True)
        db_path = (db_dir / settings.database.sqlite_db_name).as_posix()
        
        try:
            with TraceEngine.nested_span(operation="db.connect", component="database", kind=SpanKind.CLIENT) as span:
                uri = f"file:{db_path}"
                if settings.security.read_only_mode:
                    uri += "?mode=ro"
                    
                TraceEngine.attach_metadata("db_uri", uri)
                TraceEngine.attach_tags({"db.engine": "sqlite", "db.thread_name": threading.current_thread().name})
                
                # cached_statements=1000 ensures native O(1) statement parsing cache
                conn = sqlite3.connect(
                    uri, 
                    uri=True, 
                    check_same_thread=True,
                    isolation_level=None,
                    timeout=settings.database.busy_timeout_ms / 1000.0,
                    cached_statements=1000
                )
                
                conn.row_factory = sqlite3.Row
                self._apply_pragmas(conn)
                
                local.conn = conn
                local.state = ConnectionState.CONNECTED
                local.transaction_depth = 0
                local.in_transaction = False
                
                self._pool.register(conn)

                _logger.debug("Database connection established successfully.")
                AuditEngine.record_success("db.connect", AuditAction.SYSTEM, "New connection established.")
                
                return conn

        except Exception as e:
            local.state = ConnectionState.ERROR
            raise self._handle_failure(e, "db.connect")


    def get_db_path(self) -> str:
        """
        Returns the active SQLite database path.
        """
        db_dir = settings.app.base_dir / settings.database.sqlite_db_folder
        return (db_dir / settings.database.sqlite_db_name).as_posix()

    def _apply_pragmas(self, conn: sqlite3.Connection) -> None:
        """Applies configured PRAGMA statements strictly matching Global Settings."""
        with closing(conn.cursor()) as cursor:
            cursor.execute(f"PRAGMA journal_mode={settings.database.journal_mode.value};")
            cursor.execute(f"PRAGMA synchronous={settings.database.synchronous_mode.value};")
            cursor.execute(f"PRAGMA busy_timeout={settings.database.busy_timeout_ms};")
            cursor.execute(f"PRAGMA mmap_size={settings.database.mmap_size};")
            temp_store = "MEMORY" if settings.database.temp_store_memory else "DEFAULT"
            cursor.execute(f"PRAGMA temp_store={temp_store};")
            fk_enabled = "ON" if settings.database.enable_foreign_keys else "OFF"
            cursor.execute(f"PRAGMA foreign_keys={fk_enabled};")

    def disconnect(self) -> None:
        """Safely closes the thread-local database connection and purges state."""
        local = self._pool.get_local_state()
        
        if local.conn is not None:
            try:
                with TraceEngine.nested_span(operation="db.disconnect", component="database"):
                    self._pool.unregister()
                    local.conn.close()
                    _logger.debug("Database connection closed gracefully.")
            except Exception as e:
                _logger.warning(f"Error during database disconnect: {e}")
            finally:
                local.conn = None
                local.state = ConnectionState.DISCONNECTED
                local.transaction_depth = 0
                local.in_transaction = False

    @retry(operation_name="db.reconnect", max_attempts=3, strategy=RetryStrategy.EXPONENTIAL_JITTER)
    def reconnect(self) -> sqlite3.Connection:
        """
        Forces a clean disconnect and validates the subsequent connection establishment.
        Automatically managed by the RetryEngine for resiliency.
        """
        self.disconnect()
        conn = self.connect()
        if not self.health_check():
            raise DatabaseConnectionError("Health check failed post-reconnect.")
        _logger.info("Database reconnected successfully.")
        return conn

    def is_connected(self) -> bool:
        local = self._pool.get_local_state()
        return local.state == ConnectionState.CONNECTED and local.conn is not None

    def health_check(self) -> bool:
        """Performs a lightweight query to validate connection integrity."""
        if not self.is_connected():
            return False
        try:
            with TraceEngine.nested_span(operation="db.health_check", component="database"):
                with closing(self.get_connection().cursor()) as cursor:
                    cursor.execute("SELECT 1;")
                    return cursor.fetchone() is not None
        except Exception:
            return False

    def get_connection(self) -> sqlite3.Connection:
        if not self.is_connected():
            return self.connect()
        return self._pool.get_local_state().conn

    def cursor(self) -> ManagedCursor:
        """Generates a leak-proof ManagedCursor."""
        return ManagedCursor(self.get_connection().cursor())

    def begin(self, isolation: IsolationLevel = IsolationLevel.DEFERRED) -> None:
        """Initiates a transaction boundary. Supports nested SAVEPOINTs."""
        local = self._pool.get_local_state()
        depth = local.transaction_depth
        
        try:
            with TraceEngine.nested_span(operation="db.begin", component="database"):
                TraceEngine.attach_metadata("transaction_depth", depth)
                TraceEngine.attach_tags({"db.isolation": isolation.value})
                
                with closing(self.get_connection().cursor()) as cursor:
                    if depth == 0:
                        cursor.execute(f"BEGIN {isolation.value};")
                        local.in_transaction = True
                    else:
                        cursor.execute(f"SAVEPOINT sp_{depth};")
                        
                    local.transaction_depth += 1
                    local.stats.transaction_count += 1
                    
        except Exception as e:
            raise self._handle_failure(e, "db.begin")

    def commit(self) -> None:
        """Commits the active transaction boundary. Releases savepoints safely."""
        local = self._pool.get_local_state()
        if local.transaction_depth == 0:
            raise DatabaseTransactionError("No active transaction to commit.")
            
        depth = local.transaction_depth - 1
        try:
            with TraceEngine.nested_span(operation="db.commit", component="database"):
                TraceEngine.attach_metadata("transaction_depth", depth)
                
                with closing(self.get_connection().cursor()) as cursor:
                    if depth == 0:
                        cursor.execute("COMMIT;")
                        local.in_transaction = False
                    else:
                        cursor.execute(f"RELEASE SAVEPOINT sp_{depth};")
                        
                    local.transaction_depth = depth
                    
                AuditEngine.record_success(operation="db.commit", action=AuditAction.UPDATE, message="Transaction committed.")
                
        except Exception as e:
            raise self._handle_failure(e, "db.commit")

    def rollback(self) -> None:
        """Rolls back the active transaction boundary. Strictly cleans up Savepoints."""
        local = self._pool.get_local_state()
        if local.transaction_depth == 0:
            _logger.warning("Rollback requested, but no active transaction was found.")
            return
            
        depth = local.transaction_depth - 1
        try:
            with TraceEngine.nested_span(operation="db.rollback", component="database"):
                TraceEngine.attach_metadata("transaction_depth", depth)
                
                with closing(self.get_connection().cursor()) as cursor:
                    if depth == 0:
                        cursor.execute("ROLLBACK;")
                        local.in_transaction = False
                    else:
                        cursor.execute(f"ROLLBACK TO SAVEPOINT sp_{depth};")
                        cursor.execute(f"RELEASE SAVEPOINT sp_{depth};")
                        
                    local.transaction_depth = depth
                    
                AuditEngine.record_success(operation="db.rollback", action=AuditAction.UPDATE, message="Transaction rolled back.")
                
        except Exception as e:
            raise self._handle_failure(e, "db.rollback")

    def execute(self, sql: str, parameters: Tuple = ()) -> ManagedCursor:
        """Executes a single SQL statement securely and returns a leak-proof ManagedCursor."""
        conn = self.get_connection()
        local = self._pool.get_local_state()
        start_time = time.perf_counter()
        
        try:
            with TraceEngine.nested_span(operation="db.execute", component="database", kind=SpanKind.CLIENT):
                op_type = sql.lstrip().split(" ")[0].upper()[:15]
                TraceEngine.attach_tags({"db.operation_type": op_type, "db.thread_name": threading.current_thread().name})
                TraceEngine.attach_metadata("sql_length", len(sql))
                
                cursor = conn.cursor()
                cursor.execute(sql, parameters)
                
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                local.stats.query_count += 1
                local.stats.total_execution_time_ms += duration_ms
                
                TraceEngine.attach_metadata("affected_rows", cursor.rowcount)
                TraceEngine.attach_metadata("execution_time_ms", round(duration_ms, 3))
                
                return ManagedCursor(cursor)
                
        except Exception as e:
            print("\n========== SQL ERROR ==========")
            print("SQL :", sql)
            print("PARAMS :", parameters)
            print("ERROR :", repr(e))
            print("===============================\n")
            raise self._handle_failure(e, "db.execute")

    def executemany(self, sql: str, parameters: List[Tuple]) -> ManagedCursor:
        """Executes a parameterized SQL statement across a bulk sequence of values."""
        conn = self.get_connection()
        local = self._pool.get_local_state()
        start_time = time.perf_counter()
        
        try:
            with TraceEngine.nested_span(operation="db.executemany", component="database", kind=SpanKind.CLIENT):
                op_type = sql.lstrip().split(" ")[0].upper()[:15]
                TraceEngine.attach_tags({"db.operation_type": op_type, "db.thread_name": threading.current_thread().name})
                TraceEngine.attach_metadata("batch_size", len(parameters))
                
                cursor = conn.cursor()
                cursor.executemany(sql, parameters)
                
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                local.stats.query_count += 1
                local.stats.total_execution_time_ms += duration_ms
                
                TraceEngine.attach_metadata("affected_rows", cursor.rowcount)
                TraceEngine.attach_metadata("execution_time_ms", round(duration_ms, 3))
                
                return ManagedCursor(cursor)
                
        except Exception as e:
            raise self._handle_failure(e, "db.executemany")

    def executescript(self, script: str) -> None:
        """Executes a block of raw SQL statements (DDL/Migrations). Prevents cursor leaks."""
        conn = self.get_connection()
        local = self._pool.get_local_state()
        start_time = time.perf_counter()
        
        try:
            with TraceEngine.nested_span(operation="db.executescript", component="database", kind=SpanKind.CLIENT):
                TraceEngine.attach_tags({"db.operation_type": "SCRIPT", "db.thread_name": threading.current_thread().name})
                TraceEngine.attach_metadata("script_length", len(script))
                
                with closing(conn.cursor()) as cursor:
                    cursor.executescript(script)
                
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                local.stats.query_count += 1
                local.stats.total_execution_time_ms += duration_ms
                TraceEngine.attach_metadata("execution_time_ms", round(duration_ms, 3))
                
        except Exception as e:
            raise self._handle_failure(e, "db.executescript")

    # -------------------------------------------------------------------------
    # HELPER APIS (Leak-Proof Data Extractors)
    # -------------------------------------------------------------------------

    def fetch_all(self, sql: str, parameters: Tuple = ()) -> List[sqlite3.Row]:
        """Executes query and retrieves all rows securely, automatically closing the cursor."""
        with self.execute(sql, parameters) as cursor:
            return cursor.fetchall()

    def fetch_one(self, sql: str, parameters: Tuple = ()) -> Optional[sqlite3.Row]:
        """Executes query and retrieves a single row securely, automatically closing the cursor."""
        with self.execute(sql, parameters) as cursor:
            return cursor.fetchone()

    def fetch_scalar(self, sql: str, parameters: Tuple = ()) -> Any:
        """Executes query and retrieves the first column of the first row securely."""
        with self.execute(sql, parameters) as cursor:
            result = cursor.fetchone()
            return result[0] if result else None

    # -------------------------------------------------------------------------
    # INFRASTRUCTURE MAINTENANCE
    # -------------------------------------------------------------------------

    def close_all(self) -> None:
        """Global maintenance function closing all tracked active connections."""
        active = self._pool.active_count()
        self._pool.close_all()
        if active > 0:
            _logger.info(f"Closed {active} active database connections globally via Pool Abstraction.")

    def database_info(self) -> Dict[str, Any]:
        """Exposes static database infrastructure configuration details."""
        return {
            "engine": settings.database.engine.value,
            "journal_mode": settings.database.journal_mode.value,
            "synchronous_mode": settings.database.synchronous_mode.value,
            "foreign_keys_enabled": settings.database.enable_foreign_keys,
            "read_only_mode": settings.security.read_only_mode,
            "active_connections": self._pool.active_count()
        }

    def connection_statistics(self) -> ConnectionStatistics:
        """Returns the current telemetry snapshot for the thread-local connection."""
        return self._pool.get_local_state().stats


# -------------------------------------------------------------------------
# CONTEXT MANAGER (SESSION)
# -------------------------------------------------------------------------

class DatabaseSession:
    """
    Pythonic Context Manager enforcing strict Transactional Boundaries.
    Supports Nested Savepoints transparently through the DB Connection.
    """
    def __init__(self, isolation: IsolationLevel = IsolationLevel.DEFERRED):
        self.isolation = isolation
        self.db = db_manager

    def __enter__(self) -> Self:
        self.db.begin(self.isolation)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            self.db.rollback()
        else:
            self.db.commit()


# -------------------------------------------------------------------------
# Repository Compatibility Layer
# -------------------------------------------------------------------------

# Preserve original execute()
DatabaseConnection._execute_original = DatabaseConnection.execute

def _execute_compat(self, sql: str, params=None, parameters=None):
    """
    Compatibility wrapper:
    Repository -> execute(sql, params=...)
    Native     -> execute(sql, parameters=...)
    """
    if parameters is None:
        parameters = params if params is not None else ()
    return self._execute_original(sql=sql, parameters=parameters)

DatabaseConnection.execute = _execute_compat


def execute_many(self, sql: str, params_list):
    conn = self.connect()
    with conn:
        conn.executemany(sql, params_list)

DatabaseConnection.execute_many = execute_many


def fetch_all_compat(self, sql: str, params=None, parameters=None):
    if parameters is None:
        parameters = params if params is not None else ()
    return self._fetch_all_original(sql, parameters)

DatabaseConnection._fetch_all_original = DatabaseConnection.fetch_all
DatabaseConnection.fetch_all = fetch_all_compat



# -------------------------------------------------------------------------
# EXPORT GLOBAL INSTANCE
# -------------------------------------------------------------------------

db_manager: Final[DatabaseConnection] = DatabaseConnection()

__all__ = [
    "ConnectionState",
    "IsolationLevel",
    "ConnectionStatistics",
    "ManagedCursor",
    "DatabaseConnection",
    "DatabaseSession",
    "db_manager"
]


def fetch_one_compat(self, sql:str, params=None, parameters=None):
    if parameters is None:
        parameters=params if params is not None else ()
    return self._fetch_one_original(sql, parameters)

DatabaseConnection._fetch_one_original = DatabaseConnection.fetch_one
DatabaseConnection.fetch_one = fetch_one_compat
