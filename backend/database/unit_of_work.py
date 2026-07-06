"""
GREEN BULL RIDER V6 - Institutional-grade AI Stock Analysis Platform
Module: backend/database/unit_of_work.py
Description: Enterprise Production-Locked Transaction Engine.
             Provides pure, thread-safe, and async-safe transaction management,
             savepoints for nested transactions, and Identity Map lifecycle mapping.
             Integrates fully with institutional telemetry (Trace, Metrics, Audit).
             Implements Propagation, Isolation Levels, Read-Only TX, Hooks,
             and strictly resolves ContextVar Async Thread detachment.
             Python 3.13 Compatible. Compile-Safe.
"""

import time
import uuid
import asyncio
import threading
from enum import Enum
from functools import wraps
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, TypeVar, Literal
)

from backend.database.connection import db_manager
from backend.core.logger import AppLogger
from backend.core.trace import trace_span, SpanKind
from backend.core.metrics import metrics_engine
from backend.core.audit import AuditEngine, AuditAction, AuditSeverity
from backend.core.exceptions import DatabaseError, GreenBullError
from backend.config.settings import settings

_logger = AppLogger("TransactionEngine")

T = TypeVar('T')

# =========================================================================
# UTILITIES
# =========================================================================

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _get_task_id() -> Optional[int]:
    try:
        return id(asyncio.current_task())
    except RuntimeError:
        return None

class SafeMetrics:
    """Safely wraps metrics engine to prevent crashes on missing methods."""
    @staticmethod
    def increment(name: str, namespace: str = "database") -> None:
        try:
            if hasattr(metrics_engine, 'increment'):
                metrics_engine.increment(name, namespace=namespace)
        except Exception: pass

    @staticmethod
    def decrement(name: str, namespace: str = "database") -> None:
        try:
            if hasattr(metrics_engine, 'decrement'):
                metrics_engine.decrement(name, namespace=namespace)
        except Exception: pass

    @staticmethod
    def record_latency(name: str, namespace: str, duration: float) -> None:
        try:
            if hasattr(metrics_engine, 'record_latency'):
                metrics_engine.record_latency(name, namespace, duration)
        except Exception: pass


# =========================================================================
# EXCEPTIONS
# =========================================================================

class TransactionError(DatabaseError):
    """Base exception for all transaction-related failures."""
    error_code: str = "GBR-TX-001"

class NestedTransactionError(TransactionError):
    """Raised when nested transaction validation boundaries are violated."""
    error_code: str = "GBR-TX-002"

class SavepointError(TransactionError):
    """Raised when savepoint creation or release explicitly fails."""
    error_code: str = "GBR-TX-003"

class DeadlockError(TransactionError):
    """Raised when max retries are exceeded due to DB locks/deadlocks."""
    error_code: str = "GBR-TX-004"

class TransactionTimeoutError(TransactionError):
    """Raised when an active transaction exceeds its maximum allowed execution time."""
    error_code: str = "GBR-TX-005"


# =========================================================================
# ENUMS
# =========================================================================

class IsolationLevel(str, Enum):
    READ_UNCOMMITTED = "READ UNCOMMITTED"
    READ_COMMITTED = "READ COMMITTED"
    REPEATABLE_READ = "REPEATABLE READ"
    SERIALIZABLE = "SERIALIZABLE"

class Propagation(str, Enum):
    REQUIRED = "REQUIRED"
    REQUIRES_NEW = "REQUIRES_NEW"
    MANDATORY = "MANDATORY"
    SUPPORTS = "SUPPORTS"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    NEVER = "NEVER"


# =========================================================================
# RESILIENCE
# =========================================================================

class RetryPolicy:
    """Exponential backoff retry wrapper to handle transient database locks."""
    @staticmethod
    def execute(func: Callable, *args: Any, max_retries: int = 3, base_delay: float = 0.1, **kwargs: Any) -> Any:
        retries = 0
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                err_str = str(e).lower()
                if "locked" in err_str or "deadlock" in err_str or "timeout" in err_str or "busy" in err_str:
                    retries += 1
                    if retries > max_retries:
                        raise DeadlockError(f"Max retries ({max_retries}) exceeded for database lock resolution.") from e
                    time.sleep(base_delay * (2 ** (retries - 1)))
                else:
                    raise e


# =========================================================================
# IDENTITY MAP
# =========================================================================

class IdentityMap:
    """
    Enterprise scoped in-memory cache mapping unique identifiers to objects.
    Guarantees structural singularity preventing dirty-read regressions.
    """
    def __init__(self) -> None:
        self._map: Dict[Tuple[str, str], Any] = {}
        self._lock = threading.RLock()

    @trace_span(operation="identity_map.get", component="database", kind=SpanKind.INTERNAL)
    def get(self, table_name: str, pk: str) -> Optional[Any]:
        with self._lock: return self._map.get((table_name, str(pk)))

    @trace_span(operation="identity_map.put", component="database", kind=SpanKind.INTERNAL)
    def put(self, table_name: str, pk: str, entity: Any) -> None:
        with self._lock: self._map[(table_name, str(pk))] = entity

    @trace_span(operation="identity_map.remove", component="database", kind=SpanKind.INTERNAL)
    def remove(self, table_name: str, pk: str) -> None:
        with self._lock: self._map.pop((table_name, str(pk)), None)

    @trace_span(operation="identity_map.clear", component="database", kind=SpanKind.INTERNAL)
    def clear(self) -> None:
        with self._lock: self._map.clear()

    @trace_span(operation="identity_map.contains", component="database", kind=SpanKind.INTERNAL)
    def contains(self, table_name: str, pk: str) -> bool:
        with self._lock: return (table_name, str(pk)) in self._map

    def count(self) -> int:
        with self._lock: return len(self._map)

    def keys(self) -> Tuple[Tuple[str, str], ...]:
        with self._lock: return tuple(self._map.keys())

    def values(self) -> Tuple[Any, ...]:
        with self._lock: return tuple(self._map.values())

    def items(self) -> Tuple[Tuple[Tuple[str, str], Any], ...]:
        with self._lock: return tuple(self._map.items())


# =========================================================================
# STATE TRACKING
# =========================================================================

@dataclass
class TransactionState:
    """Enterprise structure tracking the lifecycle of an active transaction."""
    tx_id: str
    parent_tx: Optional[str] = None
    start_time: float = field(default_factory=time.perf_counter)
    is_root: bool = False
    read_only: bool = False
    isolation_level: Optional[IsolationLevel] = None
    timeout_sec: float = 60.0
    thread_id: int = field(default_factory=threading.get_ident)
    task_id: Optional[int] = field(default_factory=_get_task_id)

    hooks_before_begin: List[Callable] = field(default_factory=list)
    hooks_after_begin: List[Callable] = field(default_factory=list)
    hooks_before_commit: List[Callable] = field(default_factory=list)
    hooks_after_commit: List[Callable] = field(default_factory=list)
    hooks_before_rollback: List[Callable] = field(default_factory=list)
    hooks_after_rollback: List[Callable] = field(default_factory=list)

@dataclass
class SuspendedContext:
    tx_stack: Optional[List[str]]
    registry: Optional[Dict[str, TransactionState]]
    identity_map: Optional[IdentityMap]

# ContextVars ensuring thread/async safe boundaries
_tx_stack: ContextVar[Optional[List[str]]] = ContextVar("transaction_stack", default=None)
_tx_registry: ContextVar[Optional[Dict[str, TransactionState]]] = ContextVar("transaction_registry", default=None)
_identity_map_ctx: ContextVar[Optional[IdentityMap]] = ContextVar("identity_map", default=None)


# =========================================================================
# TRANSACTION MANAGER
# =========================================================================

class TransactionManager:
    """
    Core orchestrator governing transactional boundaries, commit directives, 
    savepoint rollbacks, and lifecycle event hooks mapped through ContextVars.
    """

    @classmethod
    def _get_stack(cls) -> List[str]:
        stack = _tx_stack.get()
        if stack is None:
            stack = []
            _tx_stack.set(stack)
        return stack

    @classmethod
    def _get_registry(cls) -> Dict[str, TransactionState]:
        reg = _tx_registry.get()
        if reg is None:
            reg = {}
            _tx_registry.set(reg)
        return reg

    @classmethod
    def get_identity_map(cls) -> Optional[IdentityMap]:
        return _identity_map_ctx.get()

    @classmethod
    def set_identity_map(cls, imap: Optional[IdentityMap]) -> None:
        _identity_map_ctx.set(imap)

    @classmethod
    def suspend(cls) -> SuspendedContext:
        """Suspends current transaction contexts for REQUIRES_NEW / NOT_SUPPORTED."""
        ctx = SuspendedContext(
            tx_stack=_tx_stack.get(),
            registry=_tx_registry.get(),
            identity_map=_identity_map_ctx.get()
        )
        _tx_stack.set(None)
        _tx_registry.set(None)
        _identity_map_ctx.set(None)
        return ctx

    @classmethod
    def resume(cls, ctx: SuspendedContext) -> None:
        """Restores previously suspended transaction contexts."""
        _tx_stack.set(ctx.tx_stack)
        _tx_registry.set(ctx.registry)
        _identity_map_ctx.set(ctx.identity_map)

    @classmethod
    def add_hook(
        cls, 
        hook_type: Literal['before_begin', 'after_begin', 'before_commit', 'after_commit', 'before_rollback', 'after_rollback'], 
        callback: Callable
    ) -> None:
        tx_id = cls.current_transaction()
        if not tx_id: raise TransactionError("Cannot attach hook without an active transaction context.")
        state = cls._get_registry().get(tx_id)
        if state:
            getattr(state, f"hooks_{hook_type}").append(callback)

    @classmethod
    def check_timeout(cls) -> None:
        tx_id = cls.current_transaction()
        if not tx_id: return
        state = cls._get_registry().get(tx_id)
        if state and (time.perf_counter() - state.start_time) > state.timeout_sec:
            raise TransactionTimeoutError(f"Transaction {tx_id} exceeded timeout threshold of {state.timeout_sec}s.")

    @classmethod
    def enforce_write_access(cls) -> None:
        tx_id = cls.current_transaction()
        if not tx_id: return
        state = cls._get_registry().get(tx_id)
        if state and state.read_only:
            raise TransactionError(f"Write operation rejected. Transaction {tx_id} is strictly READ-ONLY.")

    # ------------------ SYNC EXECUTION ------------------

    @classmethod
    @trace_span(operation="tx.begin", component="database", kind=SpanKind.INTERNAL)
    def begin(cls, read_only: bool = False, timeout_sec: float = 60.0, isolation_level: Optional[IsolationLevel] = None) -> str:
        stack = cls._get_stack()
        reg = cls._get_registry()
        tx_id = f"tx_{uuid.uuid4().hex[:8]}"
        is_root = len(stack) == 0
        parent_tx = stack[-1] if not is_root else None

        # Prepare state
        state = TransactionState(
            tx_id=tx_id, parent_tx=parent_tx, is_root=is_root, 
            read_only=read_only, isolation_level=isolation_level, timeout_sec=timeout_sec
        )
        
        for hook in state.hooks_before_begin: hook()

        try:
            if is_root:
                sql = "BEGIN TRANSACTION;"
                if isolation_level:
                    if "SQLITE" in str(getattr(settings.database, "dialect", "")).upper():
                        # SQLite doesn't dynamically set isolation via standard BEGIN syntax usually, but BEGIN IMMEDIATE isolates writes.
                        sql = "BEGIN IMMEDIATE TRANSACTION;"
                    else:
                        sql = f"BEGIN TRANSACTION ISOLATION LEVEL {isolation_level.value};"
                
                conn = db_manager.get_connection()

                if not getattr(conn, "in_transaction", False):
                    RetryPolicy.execute(db_manager.execute, sql=sql)
                _logger.info(f"BEGIN TRANSACTION [{tx_id}] | Depth: 1 | ReadOnly: {read_only}")
                SafeMetrics.increment("transactions_started")
                SafeMetrics.increment("active_transactions")
                AuditEngine.record_success("transaction.begin", AuditAction.SYSTEM, f"Root transaction {tx_id} established.")
            else:
                sql = f"SAVEPOINT {tx_id};"
                RetryPolicy.execute(db_manager.execute, sql=sql)
                _logger.info(f"SAVEPOINT CREATED [{tx_id}] | Depth: {len(stack) + 1}")
                SafeMetrics.increment("nested_transactions")
                SafeMetrics.increment("savepoints")
                AuditEngine.record_success("transaction.savepoint", AuditAction.SYSTEM, f"Nested savepoint {tx_id} established.")
            
            stack.append(tx_id)
            reg[tx_id] = state
            
            for hook in state.hooks_after_begin: hook()
            return tx_id
        except Exception as e:
            _logger.error(f"Failed to begin transaction {tx_id}: {e}", exc_info=True)
            raise TransactionError(f"Transaction initialization failed: {e}") from e

    @classmethod
    @trace_span(operation="tx.commit", component="database", kind=SpanKind.INTERNAL)
    def commit(cls, tx_id: str) -> None:
        stack = cls._get_stack()
        if not stack or stack[-1] != tx_id:
            raise NestedTransactionError(f"Transaction mismatch. Expected {stack[-1] if stack else None}, got {tx_id}.")

        cls.check_timeout()
        state = cls._get_registry().get(tx_id)

        try:
            if state:
                for hook in state.hooks_before_commit: hook()
        except Exception as e:
            cls.rollback(tx_id)
            raise TransactionError(f"Before commit hook failed for {tx_id}. Rolled back. Error: {e}") from e

        try:
            stack.pop()
            if not stack:
                conn = db_manager.get_connection()
                if getattr(conn, "in_transaction", False):
                    RetryPolicy.execute(db_manager.execute, sql="COMMIT;")
                _logger.info(f"COMMIT TRANSACTION [{tx_id}] | Depth: 0")
                SafeMetrics.increment("transactions_committed")
                SafeMetrics.decrement("active_transactions")
                
                if state:
                    duration = (time.perf_counter() - state.start_time) * 1000.0
                    SafeMetrics.record_latency("tx_duration_ms", "database", duration)
                
                AuditEngine.record_success("transaction.commit", AuditAction.SYSTEM, f"Root transaction {tx_id} committed.")
            else:
                RetryPolicy.execute(db_manager.execute, sql=f"RELEASE SAVEPOINT {tx_id};")
                _logger.info(f"RELEASE SAVEPOINT [{tx_id}] | Depth: {len(stack)}")
                AuditEngine.record_success("transaction.nested_commit", AuditAction.SYSTEM, f"Savepoint {tx_id} released.")

            if state:
                for hook in state.hooks_after_commit: hook()

        except Exception as e:
            _logger.error(f"Failed to commit sequence {tx_id}: {e}", exc_info=True)
            raise TransactionError(f"Transaction commit fault: {e}") from e
        finally:
            cls._get_registry().pop(tx_id, None)

    @classmethod
    @trace_span(operation="tx.rollback", component="database", kind=SpanKind.INTERNAL)
    def rollback(cls, tx_id: str) -> None:
        stack = cls._get_stack()
        if not stack or tx_id not in stack:
            _logger.warning(f"Rollback ignored for {tx_id}. Target is absent from isolated execution context.")
            return

        state = cls._get_registry().get(tx_id)

        if state:
            for hook in state.hooks_before_rollback: 
                try: hook()
                except Exception as e: _logger.error(f"Error in before_rollback hook for {tx_id}: {e}")

        try:
            idx = stack.index(tx_id)
            del stack[idx:]
            
            if not stack:
                RetryPolicy.execute(db_manager.execute, sql="ROLLBACK;")
                _logger.warning(f"ROLLBACK TRANSACTION [{tx_id}] | Depth: 0")
                SafeMetrics.increment("transactions_rollback")
                SafeMetrics.decrement("active_transactions")
                AuditEngine.record_failure("transaction.rollback", AuditAction.SYSTEM, AuditSeverity.WARNING, f"Root transaction {tx_id} rolled back.")
            else:
                RetryPolicy.execute(db_manager.execute, sql=f"ROLLBACK TO SAVEPOINT {tx_id};")
                RetryPolicy.execute(db_manager.execute, sql=f"RELEASE SAVEPOINT {tx_id};")
                _logger.warning(f"ROLLBACK TO SAVEPOINT [{tx_id}] | Depth: {len(stack)}")
                SafeMetrics.increment("rollback_to_savepoint")
                AuditEngine.record_failure("transaction.nested_rollback", AuditAction.SYSTEM, AuditSeverity.WARNING, f"Savepoint {tx_id} rolled back and released.")

            if state:
                for hook in state.hooks_after_rollback: 
                    try: hook()
                    except Exception as e: _logger.error(f"Error in after_rollback hook for {tx_id}: {e}")

        except Exception as e:
            _logger.error(f"Failed to rollback sequence {tx_id}: {e}", exc_info=True)
            raise SavepointError(f"Transaction rollback fault: {e}") from e
        finally:
            cls._get_registry().pop(tx_id, None)

    # ------------------ ASYNC EXECUTION DELEGATES ------------------

    @classmethod
    async def begin_async(cls, read_only: bool = False, timeout_sec: float = 60.0, isolation_level: Optional[IsolationLevel] = None) -> str:
        stack = cls._get_stack()
        reg = cls._get_registry()
        tx_id = f"tx_{uuid.uuid4().hex[:8]}"
        is_root = len(stack) == 0
        parent_tx = stack[-1] if not is_root else None

        state = TransactionState(
            tx_id=tx_id, parent_tx=parent_tx, is_root=is_root, 
            read_only=read_only, isolation_level=isolation_level, timeout_sec=timeout_sec
        )
        
        for hook in state.hooks_before_begin: hook()

        try:
            if is_root:
                sql = "BEGIN TRANSACTION;"
                if isolation_level:
                    if "SQLITE" in str(getattr(settings.database, "dialect", "")).upper():
                        sql = "BEGIN IMMEDIATE TRANSACTION;"
                    else:
                        sql = f"BEGIN TRANSACTION ISOLATION LEVEL {isolation_level.value};"
                
                await asyncio.to_thread(RetryPolicy.execute, db_manager.execute, sql=sql)
                _logger.info(f"BEGIN TRANSACTION (Async) [{tx_id}] | Depth: 1 | ReadOnly: {read_only}")
                SafeMetrics.increment("transactions_started")
                SafeMetrics.increment("active_transactions")
                AuditEngine.record_success("transaction.begin", AuditAction.SYSTEM, f"Root transaction {tx_id} established.")
            else:
                sql = f"SAVEPOINT {tx_id};"
                await asyncio.to_thread(RetryPolicy.execute, db_manager.execute, sql=sql)
                _logger.info(f"SAVEPOINT CREATED (Async) [{tx_id}] | Depth: {len(stack) + 1}")
                SafeMetrics.increment("nested_transactions")
                SafeMetrics.increment("savepoints")
                AuditEngine.record_success("transaction.savepoint", AuditAction.SYSTEM, f"Nested savepoint {tx_id} established.")
            
            stack.append(tx_id)
            reg[tx_id] = state
            
            for hook in state.hooks_after_begin: hook()
            return tx_id
        except Exception as e:
            raise TransactionError(f"Transaction initialization failed: {e}") from e

    @classmethod
    async def commit_async(cls, tx_id: str) -> None:
        stack = cls._get_stack()
        if not stack or stack[-1] != tx_id:
            raise NestedTransactionError(f"Transaction mismatch. Expected {stack[-1] if stack else None}, got {tx_id}.")

        cls.check_timeout()
        state = cls._get_registry().get(tx_id)

        try:
            if state:
                for hook in state.hooks_before_commit: hook()
        except Exception as e:
            await cls.rollback_async(tx_id)
            raise TransactionError(f"Before commit hook failed for {tx_id}. Rolled back. Error: {e}") from e

        try:
            stack.pop()
            if not stack:
                conn = db_manager.get_connection()
                if getattr(conn, "in_transaction", False):
                    await asyncio.to_thread(
                        RetryPolicy.execute,
                        db_manager.execute,
                        sql="COMMIT;"
                    )
                _logger.info(f"COMMIT TRANSACTION (Async) [{tx_id}] | Depth: 0")
                SafeMetrics.increment("transactions_committed")
                SafeMetrics.decrement("active_transactions")
                
                if state:
                    duration = (time.perf_counter() - state.start_time) * 1000.0
                    SafeMetrics.record_latency("tx_duration_ms", "database", duration)

                AuditEngine.record_success("transaction.commit", AuditAction.SYSTEM, f"Root transaction {tx_id} committed.")
            else:
                await asyncio.to_thread(RetryPolicy.execute, db_manager.execute, sql=f"RELEASE SAVEPOINT {tx_id};")
                _logger.info(f"RELEASE SAVEPOINT (Async) [{tx_id}] | Depth: {len(stack)}")
                AuditEngine.record_success("transaction.nested_commit", AuditAction.SYSTEM, f"Savepoint {tx_id} released.")

            if state:
                for hook in state.hooks_after_commit: hook()

        except Exception as e:
            raise TransactionError(f"Transaction commit fault: {e}") from e
        finally:
            cls._get_registry().pop(tx_id, None)

    @classmethod
    async def rollback_async(cls, tx_id: str) -> None:
        stack = cls._get_stack()
        if not stack or tx_id not in stack: return

        state = cls._get_registry().get(tx_id)

        if state:
            for hook in state.hooks_before_rollback:
                try: hook()
                except Exception as e: _logger.error(f"Error in before_rollback hook for {tx_id}: {e}")

        try:
            idx = stack.index(tx_id)
            del stack[idx:]
            
            if not stack:
                await asyncio.to_thread(RetryPolicy.execute, db_manager.execute, sql="ROLLBACK;")
                _logger.warning(f"ROLLBACK TRANSACTION (Async) [{tx_id}] | Depth: 0")
                SafeMetrics.increment("transactions_rollback")
                SafeMetrics.decrement("active_transactions")
                AuditEngine.record_failure("transaction.rollback", AuditAction.SYSTEM, AuditSeverity.WARNING, f"Root transaction {tx_id} rolled back.")
            else:
                await asyncio.to_thread(RetryPolicy.execute, db_manager.execute, sql=f"ROLLBACK TO SAVEPOINT {tx_id};")
                await asyncio.to_thread(RetryPolicy.execute, db_manager.execute, sql=f"RELEASE SAVEPOINT {tx_id};")
                _logger.warning(f"ROLLBACK TO SAVEPOINT (Async) [{tx_id}] | Depth: {len(stack)}")
                SafeMetrics.increment("rollback_to_savepoint")
                AuditEngine.record_failure("transaction.nested_rollback", AuditAction.SYSTEM, AuditSeverity.WARNING, f"Savepoint {tx_id} rolled back and released.")

            if state:
                for hook in state.hooks_after_rollback:
                    try: hook()
                    except Exception as e: _logger.error(f"Error in after_rollback hook for {tx_id}: {e}")

        except Exception as e:
            raise SavepointError(f"Transaction rollback fault: {e}") from e
        finally:
            cls._get_registry().pop(tx_id, None)

    # ------------------ CONTEXT STATE ------------------

    @classmethod
    def current_transaction(cls) -> Optional[str]:
        stack = _tx_stack.get()
        return stack[-1] if stack else None

    @classmethod
    def transaction_depth(cls) -> int:
        stack = _tx_stack.get()
        return len(stack) if stack else 0

    @classmethod
    def is_inside_transaction(cls) -> bool:
        return cls.transaction_depth() > 0

    @classmethod
    def clear(cls) -> None:
        _tx_stack.set(None)
        _tx_registry.set(None)


# =========================================================================
# CONTEXT MANAGERS: UNIT OF WORK
# =========================================================================

class UnitOfWork:
    """
    Enterprise Data Access Scope Manager.
    Orchestrates execution sequences mapping Transaction boundaries to Identity Maps.
    Provides Propagation overrides for complex boundary management.
    """
    def __init__(self, propagation: Propagation = Propagation.REQUIRED, isolation_level: Optional[IsolationLevel] = None, read_only: bool = False) -> None:
        self.propagation = propagation
        self.isolation_level = isolation_level
        self.read_only = read_only
        self.tx_id: Optional[str] = None
        self.is_root: bool = False
        self.suspended_ctx: Optional[SuspendedContext] = None
        self.active: bool = True

    @property
    def identity_map(self) -> Optional[IdentityMap]:
        return TransactionManager.get_identity_map()

    def __enter__(self) -> 'UnitOfWork':
        in_tx = TransactionManager.is_inside_transaction()

        if self.propagation == Propagation.NEVER:
            if in_tx: raise NestedTransactionError("NEVER propagation forbids an active transaction.")
            self.active = False
            return self

        if self.propagation == Propagation.NOT_SUPPORTED:
            if in_tx: self.suspended_ctx = TransactionManager.suspend()
            self.active = False
            return self

        if self.propagation == Propagation.SUPPORTS:
            if not in_tx:
                self.active = False
                return self
            # else participate

        if self.propagation == Propagation.MANDATORY:
            if not in_tx: raise NestedTransactionError("MANDATORY propagation requires an active transaction.")
            # participate

        if self.propagation == Propagation.REQUIRES_NEW:
            if in_tx: self.suspended_ctx = TransactionManager.suspend()
            # proceed to begin new context

        in_tx_now = TransactionManager.is_inside_transaction()
        if not in_tx_now:
            TransactionManager.set_identity_map(IdentityMap())
            self.is_root = True

        self.tx_id = TransactionManager.begin(read_only=self.read_only, isolation_level=self.isolation_level)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if not self.active:
            if self.suspended_ctx: TransactionManager.resume(self.suspended_ctx)
            return

        if self.tx_id is None:
            raise TransactionError("UnitOfWork exit context executed without initialized transaction.")

        try:
            if exc_type is not None:
                TransactionManager.rollback(self.tx_id)
            else:
                TransactionManager.commit(self.tx_id)
        finally:
            if self.is_root:
                imap = TransactionManager.get_identity_map()
                if imap is not None: imap.clear()
                TransactionManager.set_identity_map(None)
                TransactionManager.clear()
            if self.suspended_ctx:
                TransactionManager.resume(self.suspended_ctx)


class AsyncUnitOfWork:
    """
    Asynchronous Enterprise Data Access Scope Manager.
    Properly handles ContextVar mappings natively, delegating IO logic safely to threads.
    """
    def __init__(self, propagation: Propagation = Propagation.REQUIRED, isolation_level: Optional[IsolationLevel] = None, read_only: bool = False) -> None:
        self.propagation = propagation
        self.isolation_level = isolation_level
        self.read_only = read_only
        self.tx_id: Optional[str] = None
        self.is_root: bool = False
        self.suspended_ctx: Optional[SuspendedContext] = None
        self.active: bool = True

    @property
    def identity_map(self) -> Optional[IdentityMap]:
        return TransactionManager.get_identity_map()

    async def __aenter__(self) -> 'AsyncUnitOfWork':
        in_tx = TransactionManager.is_inside_transaction()

        if self.propagation == Propagation.NEVER:
            if in_tx: raise NestedTransactionError("NEVER propagation forbids an active transaction.")
            self.active = False
            return self

        if self.propagation == Propagation.NOT_SUPPORTED:
            if in_tx: self.suspended_ctx = TransactionManager.suspend()
            self.active = False
            return self

        if self.propagation == Propagation.SUPPORTS:
            if not in_tx:
                self.active = False
                return self

        if self.propagation == Propagation.MANDATORY:
            if not in_tx: raise NestedTransactionError("MANDATORY propagation requires an active transaction.")

        if self.propagation == Propagation.REQUIRES_NEW:
            if in_tx: self.suspended_ctx = TransactionManager.suspend()

        in_tx_now = TransactionManager.is_inside_transaction()
        if not in_tx_now:
            TransactionManager.set_identity_map(IdentityMap())
            self.is_root = True

        self.tx_id = await TransactionManager.begin_async(read_only=self.read_only, isolation_level=self.isolation_level)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if not self.active:
            if self.suspended_ctx: TransactionManager.resume(self.suspended_ctx)
            return

        if self.tx_id is None:
            raise TransactionError("AsyncUnitOfWork exit context executed without initialized transaction.")

        try:
            if exc_type is not None:
                await TransactionManager.rollback_async(self.tx_id)
            else:
                await TransactionManager.commit_async(self.tx_id)
        finally:
            if self.is_root:
                imap = TransactionManager.get_identity_map()
                if imap is not None: imap.clear()
                TransactionManager.set_identity_map(None)
                TransactionManager.clear()
            if self.suspended_ctx:
                TransactionManager.resume(self.suspended_ctx)


# =========================================================================
# CONTEXT MANAGERS: PURE TRANSACTION
# =========================================================================

class TransactionContext:
    """Lightweight sequence orchestrator ignoring Identity Map boundaries."""
    def __init__(self, read_only: bool = False, isolation_level: Optional[IsolationLevel] = None) -> None:
        self.tx_id: Optional[str] = None
        self.read_only = read_only
        self.isolation_level = isolation_level

    def __enter__(self) -> 'TransactionContext':
        self.tx_id = TransactionManager.begin(read_only=self.read_only, isolation_level=self.isolation_level)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.tx_id is None: raise TransactionError("TransactionContext exit triggered without binding.")
        try:
            if exc_type is not None: TransactionManager.rollback(self.tx_id)
            else: TransactionManager.commit(self.tx_id)
        finally:
            if not TransactionManager.is_inside_transaction():
                TransactionManager.clear()


class AsyncTransactionContext:
    """Lightweight async sequence orchestrator. No Identity Map managed here."""
    def __init__(self, read_only: bool = False, isolation_level: Optional[IsolationLevel] = None) -> None:
        self.tx_id: Optional[str] = None
        self.read_only = read_only
        self.isolation_level = isolation_level

    async def __aenter__(self) -> 'AsyncTransactionContext':
        self.tx_id = await TransactionManager.begin_async(read_only=self.read_only, isolation_level=self.isolation_level)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.tx_id is None: raise TransactionError("AsyncTransactionContext exit triggered without binding.")
        try:
            if exc_type is not None: await TransactionManager.rollback_async(self.tx_id)
            else: await TransactionManager.commit_async(self.tx_id)
        finally:
            if not TransactionManager.is_inside_transaction():
                TransactionManager.clear()


# =========================================================================
# EXECUTION DECORATORS
# =========================================================================

def transactional(propagation: Propagation = Propagation.REQUIRED, isolation_level: Optional[IsolationLevel] = None, read_only: bool = False) -> Callable:
    """Decorator executing functions encapsulated strictly within UnitOfWork domains."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with UnitOfWork(propagation=propagation, isolation_level=isolation_level, read_only=read_only):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def async_transactional(propagation: Propagation = Propagation.REQUIRED, isolation_level: Optional[IsolationLevel] = None, read_only: bool = False) -> Callable:
    """Decorator executing async functions encapsulated strictly within AsyncUnitOfWork domains."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with AsyncUnitOfWork(propagation=propagation, isolation_level=isolation_level, read_only=read_only):
                return await func(*args, **kwargs)
        return wrapper
    return decorator


# =========================================================================
# COMPONENT EXPORTS
# =========================================================================

__all__ = [
    "IsolationLevel",
    "Propagation",
    "IdentityMap",
    "TransactionManager",
    "TransactionContext",
    "AsyncTransactionContext",
    "UnitOfWork",
    "AsyncUnitOfWork",
    "transactional",
    "async_transactional",
    "TransactionError",
    "NestedTransactionError",
    "SavepointError",
    "DeadlockError",
    "TransactionTimeoutError"
]
