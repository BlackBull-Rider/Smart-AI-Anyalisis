"""
GREEN BULL RIDER V6 - Institutional-grade AI Stock Analysis Platform
Module: backend/database/repository.py
Description: Enterprise Production-Locked Repository Layer.
             Provides a unified, secure, thread-safe, and asynchronous-safe 
             abstraction over database operations. Implements CQRS (Read/Write Split),
             Advanced Fluent Query Builder (Safe JOIN, GROUP BY, Window Functions),
             Specification Pattern (AND/OR/NOT), Identity Map, Circuit Breaker, 
             Exponential Retry Policy, and Unit of Work. Strictly decoupled from Raw SQL.
             Python 3.13 Compatible. SQLite + PostgreSQL support. Compile-Safe.
"""

import re
import time
import uuid
import asyncio
import threading
import weakref
import dataclasses
from enum import Enum
from contextvars import ContextVar
from collections import OrderedDict
from dataclasses import dataclass, field as dc_field
from typing import (
    Any, AsyncGenerator, Callable, Dict, Generic, Iterable, Iterator, List, 
    Literal, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, Final, cast
)

# Internal Platform Integrations
from backend.config.settings import settings
from backend.core.logger import AppLogger
from backend.core.trace import TraceEngine, SpanKind, trace_span
from backend.core.metrics import metrics_engine
from backend.core.audit import AuditEngine, AuditAction, AuditSeverity
from backend.core.exceptions import GreenBullError, DatabaseError
from backend.database.connection import db_manager

# Models Integration
from backend.database.models import (
    BaseModel, SerializerRegistry, MetaDataRegistry, TableMetadata, Dialect, TimestampHelper
)

_logger = AppLogger("RepositoryEngine")

T = TypeVar('T', bound=BaseModel)


# =========================================================================
# EXCEPTIONS
# =========================================================================

class RepositoryError(GreenBullError): error_code: str = "GBR-REP-001"
class ValidationError(RepositoryError): error_code: str = "GBR-REP-002"
class OptimisticLockError(RepositoryError): error_code: str = "GBR-REP-003"
class DeadlockError(RepositoryError): error_code: str = "GBR-REP-004"
class TransactionError(RepositoryError): error_code: str = "GBR-REP-005"
class ConnectionError(RepositoryError): error_code: str = "GBR-REP-006"
class SerializationError(RepositoryError): error_code: str = "GBR-REP-007"
class SecurityError(RepositoryError): error_code: str = "GBR-REP-008"
class CircuitBreakerError(RepositoryError): error_code: str = "GBR-REP-009"


# =========================================================================
# ENUMS & CONSTANTS
# =========================================================================

class QueryOperator(str, Enum):
    EQ = "="; NEQ = "!="; GT = ">"; LT = "<"; GTE = ">="; LTE = "<="
    IN = "IN"; NOT_IN = "NOT IN"; LIKE = "LIKE"; ILIKE = "ILIKE"
    IS_NULL = "IS NULL"; IS_NOT_NULL = "IS NOT NULL"

class SortDirection(str, Enum):
    ASC = "ASC"; DESC = "DESC"

class JoinType(str, Enum):
    INNER = "INNER JOIN"; LEFT = "LEFT JOIN"; RIGHT = "RIGHT JOIN"; FULL = "FULL OUTER JOIN"

ACTIVE_DIALECT: Dialect = Dialect(getattr(settings.database, "dialect", "SQLITE").upper())

# Extended to allow table.column, COUNT(*), SUM(price), etc.
_SQL_IDENTIFIER_REGEX = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_\.\*]*(?:\([a-zA-Z0-9_\.\*]+\))?$")

def _validate_identifier(identifier: str) -> str:
    """Strictly validates SQL Identifiers preventing structural SQL Injection."""
    if not _SQL_IDENTIFIER_REGEX.match(identifier):
        raise SecurityError(f"Invalid SQL Identifier detected: {identifier}")
    return identifier


# =========================================================================
# RESILIENCE: CIRCUIT BREAKER & RETRY POLICY
# =========================================================================

class CircuitBreaker:
    """State-machine based circuit breaker to prevent cascading DB failures."""
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"
        self._lock = threading.RLock()

    def __call__(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                else:
                    raise CircuitBreakerError("Database circuit breaker is OPEN.")
        try:
            result = func(*args, **kwargs)
            with self._lock:
                self.failures = 0  # Reset on any success
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
            return result
        except Exception as e:
            with self._lock:
                self.failures += 1
                self.last_failure_time = time.time()
                if self.failures >= self.failure_threshold:
                    self.state = "OPEN"
            raise e

db_circuit_breaker = CircuitBreaker()

class RetryPolicy:
    """Exponential backoff retry wrapper specifically targeting Deadlocks and Timeouts."""
    @staticmethod
    def execute(func: Callable, *args: Any, max_retries: int = 3, base_delay: float = 0.1, **kwargs: Any) -> Any:
        retries = 0
        while True:
            try:
                return db_circuit_breaker(func, *args, **kwargs)
            except Exception as e:
                err_str = str(e).lower()
                if "database is locked" in err_str or "deadlock" in err_str or "timeout" in err_str:
                    retries += 1
                    if retries > max_retries:
                        raise DeadlockError(f"Max retries ({max_retries}) exceeded for database operation.") from e
                    time.sleep(base_delay * (2 ** (retries - 1)))
                else:
                    raise e


# =========================================================================
# SPECIFICATION PATTERN
# =========================================================================

@dataclass(kw_only=True)
class FilterCriteria:
    field: str = ""
    operator: QueryOperator = QueryOperator.EQ
    value: Any = None
    is_or_group: bool = False
    children: Optional[List['FilterCriteria']] = None

class Specification(Generic[T]):
    """Enterprise Specification Pattern for decoupled filtering logic."""
    def to_filters(self) -> List[FilterCriteria]: raise NotImplementedError()

class AndSpecification(Specification[T]):
    def __init__(self, *specs: Specification[T]): self.specs = specs
    def to_filters(self) -> List[FilterCriteria]:
        return [f for spec in self.specs for f in spec.to_filters()]

class OrSpecification(Specification[T]):
    """Generates grouped filters for OR condition execution in the SQL Builder."""
    def __init__(self, *specs: Specification[T]): self.specs = specs
    def to_filters(self) -> List[FilterCriteria]:
        children = []
        for spec in self.specs:
            children.extend(spec.to_filters())
        return [FilterCriteria(is_or_group=True, children=children)]

class NotSpecification(Specification[T]):
    def __init__(self, spec: Specification[T]): self.spec = spec
    def to_filters(self) -> List[FilterCriteria]:
        filters = []
        for f in self.spec.to_filters():
            if f.is_or_group:
                raise NotImplementedError("NOT over OR groups is currently unsupported.")
            inverse_op = {
                QueryOperator.EQ: QueryOperator.NEQ, QueryOperator.NEQ: QueryOperator.EQ,
                QueryOperator.GT: QueryOperator.LTE, QueryOperator.LT: QueryOperator.GTE,
                QueryOperator.GTE: QueryOperator.LT, QueryOperator.LTE: QueryOperator.GT,
                QueryOperator.IN: QueryOperator.NOT_IN, QueryOperator.NOT_IN: QueryOperator.IN,
                QueryOperator.IS_NULL: QueryOperator.IS_NOT_NULL, QueryOperator.IS_NOT_NULL: QueryOperator.IS_NULL,
            }.get(f.operator, f.operator)
            filters.append(FilterCriteria(field=f.field, operator=inverse_op, value=f.value))
        return filters


# =========================================================================
# DATA STRUCTURES
# =========================================================================

@dataclass(kw_only=True, slots=True)
class Page(Generic[T]):
    items: List[T]
    total_records: int
    page: int
    page_size: int
    total_pages: int

    @classmethod
    def create(cls, items: List[T], total: int, page: int, size: int) -> 'Page[T]':
        total_pages = (total + size - 1) // size if size > 0 else 1
        return cls(items=items, total_records=total, page=page, page_size=size, total_pages=total_pages)

@dataclass(kw_only=True, slots=True)
class SortCriteria:
    field: str
    direction: SortDirection = SortDirection.ASC


# =========================================================================
# ADVANCED SQL QUERY BUILDER
# =========================================================================

class QueryContext:
    __slots__ = ('sql', 'parameters')
    def __init__(self, sql: str, parameters: Tuple[Any, ...]):
        self.sql = sql; self.parameters = parameters

class SQLQueryBuilder:
    """Fluent API for advanced SQL generation preventing injection vulnerabilities."""
    def __init__(self, table_name: str, table_meta: TableMetadata, dialect: Dialect = ACTIVE_DIALECT):
        self.table = _validate_identifier(table_name)
        self.table_meta = table_meta
        self.dialect = dialect
        self._projections: List[str] = []
        self._joins: List[str] = []
        self._where: List[str] = []
        self._group_by: List[str] = []
        self._having: List[str] = []
        self._order_by: List[str] = []
        self._window: List[str] = []
        self._params: List[Any] = []
        self._param_idx = 1
        self._limit: Optional[int] = None
        self._offset: Optional[int] = None
        self._for_update: bool = False

    def _token(self) -> str:
        if self.dialect == Dialect.SQLITE: return "?"
        t = f"%s"
        self._param_idx += 1
        return t

    def select(self, *fields: str, alias: Optional[str] = None) -> 'SQLQueryBuilder':
        for f in fields:
            clean = _validate_identifier(f)
            if alias: self._projections.append(f"{clean} AS {_validate_identifier(alias)}")
            else: self._projections.append(clean)
        return self

    def join(self, table: str, left_col: str, right_col: str, join_type: JoinType = JoinType.INNER) -> 'SQLQueryBuilder':
        t_clean = _validate_identifier(table)
        lc_clean = _validate_identifier(left_col)
        rc_clean = _validate_identifier(right_col)
        self._joins.append(f"{join_type.value} {t_clean} ON {lc_clean} = {rc_clean}")
        return self

    def _build_conditions(self, criteria: List[FilterCriteria]) -> Tuple[List[str], List[Any]]:
        clauses, params = [], []
        for f in criteria:
            if f.is_or_group and f.children:
                sub_c, sub_p = self._build_conditions(f.children)
                if sub_c:
                    clauses.append("(" + " OR ".join(sub_c) + ")")
                    params.extend(sub_p)
                continue

            col = _validate_identifier(f.field)
            col_meta = self.table_meta.columns.get(col)
            val = SerializerRegistry.to_sql(f.value, dataclasses.asdict(col_meta) if col_meta else {}, self.dialect)

            if f.operator in (QueryOperator.IS_NULL, QueryOperator.IS_NOT_NULL):
                clauses.append(f"{col} {f.operator.value}")
            elif f.operator in (QueryOperator.IN, QueryOperator.NOT_IN):
                if not isinstance(val, (list, tuple, set)): raise ValidationError("IN requires iterable")
                if not val:
                    clauses.append("1 = 0" if f.operator == QueryOperator.IN else "1 = 1")
                else:
                    tokens = ",".join([self._token() for _ in val])
                    clauses.append(f"{col} {f.operator.value} ({tokens})")
                    params.extend(val)
            else:
                clauses.append(f"{col} {f.operator.value} {self._token()}")
                params.append(val)
        return clauses, params

    def where(self, criteria: List[FilterCriteria]) -> 'SQLQueryBuilder':
        c, p = self._build_conditions(criteria)
        self._where.extend(c); self._params.extend(p)
        return self

    def group_by(self, *fields: str) -> 'SQLQueryBuilder':
        self._group_by.extend([_validate_identifier(f) for f in fields])
        return self

    def having(self, criteria: List[FilterCriteria]) -> 'SQLQueryBuilder':
        c, p = self._build_conditions(criteria)
        self._having.extend(c); self._params.extend(p)
        return self

    def window(self, name: str, partition_by: str, order_by: str, frame: Optional[str] = None) -> 'SQLQueryBuilder':
        clean_n = _validate_identifier(name)
        clean_p = _validate_identifier(partition_by)
        clean_o = _validate_identifier(order_by)
        clause = f"{clean_n} AS (PARTITION BY {clean_p} ORDER BY {clean_o}"
        if frame:
            safe_frame = re.sub(r"[^a-zA-Z0-9\s_]", "", frame)
            clause += f" {safe_frame}"
        clause += ")"
        self._window.append(clause)
        return self

    def order_by(self, sorts: List[SortCriteria]) -> 'SQLQueryBuilder':
        self._order_by.extend([f"{_validate_identifier(s.field)} {s.direction.value}" for s in sorts])
        return self

    def limit(self, limit: int) -> 'SQLQueryBuilder':
        self._limit = limit; return self

    def offset(self, offset: int) -> 'SQLQueryBuilder':
        self._offset = offset; return self

    def for_update(self) -> 'SQLQueryBuilder':
        self._for_update = True; return self

    def build(self) -> QueryContext:
        proj = ", ".join(self._projections) if self._projections else "*"
        sql = f"SELECT {proj} FROM {self.table}"
        
        if self._joins: sql += " " + " ".join(self._joins)
        if self._where: sql += " WHERE " + " AND ".join(self._where)
        if self._group_by: sql += " GROUP BY " + ", ".join(self._group_by)
        if self._having: sql += " HAVING " + " AND ".join(self._having)
        if self._window: sql += " WINDOW " + ", ".join(self._window)
        if self._order_by: sql += " ORDER BY " + ", ".join(self._order_by)
        if self._limit is not None: sql += f" LIMIT {self._limit}"
        if self._offset is not None: sql += f" OFFSET {self._offset}"
        if self._for_update and self.dialect == Dialect.POSTGRES: sql += " FOR UPDATE"

        return QueryContext(sql, tuple(self._params))
        
    def build_count(self) -> QueryContext:
        self._projections = ["COUNT(*) as count"]
        self._order_by = []; self._limit = None; self._offset = None
        return self.build()
        
    def build_aggregate(self, func: Literal['SUM', 'AVG', 'MIN', 'MAX'], field: str) -> QueryContext:
        self._projections = [f"{func}({_validate_identifier(field)}) as aggr_val"]
        self._order_by = []; self._limit = None; self._offset = None
        return self.build()


class SQLWriteBuilder:
    """Dedicated string builder for Insert/Update/Delete constraints."""
    @staticmethod
    def _token(dialect: Dialect) -> str: return "?" if dialect == Dialect.SQLITE else "%s"

    @classmethod
    def build_insert(cls, table_name: str, columns: List[str], dialect: Dialect = ACTIVE_DIALECT) -> str:
        table = _validate_identifier(table_name)
        safe_cols = [_validate_identifier(c) for c in columns]
        tokens = ", ".join([cls._token(dialect) for _ in columns])
        return f"INSERT INTO {table} ({', '.join(safe_cols)}) VALUES ({tokens})"

    @classmethod
    def build_update(cls, table_name: str, columns: List[str], pk_field: str, dialect: Dialect = ACTIVE_DIALECT, version_field: Optional[str] = None) -> str:
        table = _validate_identifier(table_name)
        safe_pk = _validate_identifier(pk_field)
        safe_cols = [_validate_identifier(c) for c in columns]
        
        set_str = ", ".join([f"{c} = {cls._token(dialect)}" for c in safe_cols])
        where_clause = f"WHERE {safe_pk} = {cls._token(dialect)}"
        
        if version_field:
            where_clause += f" AND {_validate_identifier(version_field)} = {cls._token(dialect)}"
            
        return f"UPDATE {table} SET {set_str} {where_clause}"

    @classmethod
    def build_delete(cls, table_name: str, pk_field: str, dialect: Dialect = ACTIVE_DIALECT) -> str:
        return f"DELETE FROM {_validate_identifier(table_name)} WHERE {_validate_identifier(pk_field)} = {cls._token(dialect)}"

    @classmethod
    def build_upsert(cls, table_name: str, columns: List[str], conflict_keys: List[str], dialect: Dialect = ACTIVE_DIALECT) -> str:
        insert_sql = cls.build_insert(table_name, columns, dialect)
        safe_conflicts = [_validate_identifier(k) for k in conflict_keys]
        update_cols = [c for c in columns if c not in safe_conflicts]
        
        if not update_cols:
            return f"{insert_sql} ON CONFLICT ({', '.join(safe_conflicts)}) DO NOTHING"
            
        prefix = "excluded" if dialect == Dialect.SQLITE else "EXCLUDED"
        updates = ", ".join([f"{_validate_identifier(c)} = {prefix}.{_validate_identifier(c)}" for c in update_cols])
        return f"{insert_sql} ON CONFLICT ({', '.join(safe_conflicts)}) DO UPDATE SET {updates}"


# =========================================================================
# IDENTITY MAP & UNIT OF WORK
# =========================================================================

class IdentityMap:
    """Thread-local identity map guaranteeing object singularity within a transaction."""
    def __init__(self):
        self._map: Dict[Tuple[str, str], BaseModel] = {}

    def get(self, table: str, pk: str) -> Optional[BaseModel]:
        return self._map.get((table, str(pk)))

    def put(self, table: str, pk: str, model: BaseModel) -> None:
        self._map[(table, str(pk))] = model

    def remove(self, table: str, pk: str) -> None:
        self._map.pop((table, str(pk)), None)

    def clear(self) -> None:
        self._map.clear()

_uow_identity_map: ContextVar[Optional[IdentityMap]] = ContextVar("uow_identity_map", default=None)
_tx_stack: ContextVar[Optional[List[str]]] = ContextVar("transaction_stack", default=None)

class TransactionContext:
    def __enter__(self):
        self.tx_id = TransactionManager.begin()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None: TransactionManager.rollback(self.tx_id)
        else: TransactionManager.commit(self.tx_id)

class AsyncTransactionContext:
    async def __aenter__(self):
        self.tx_id = await asyncio.to_thread(TransactionManager.begin)
        return self
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None: await asyncio.to_thread(TransactionManager.rollback, self.tx_id)
        else: await asyncio.to_thread(TransactionManager.commit, self.tx_id)

class TransactionManager:
    """Thread-safe & Async-safe Nested Transaction & Savepoint Engine."""
    
    @classmethod
    def _get_stack(cls) -> List[str]:
        stack = _tx_stack.get()
        if stack is None:
            stack = []
            _tx_stack.set(stack)
        return stack

    @classmethod
    def begin(cls) -> str:
        stack = cls._get_stack()
        tx_id = f"tx_{uuid.uuid4().hex[:8]}"
        try:
            if not stack:
                RetryPolicy.execute(db_manager.execute, sql="BEGIN TRANSACTION;")
                _logger.debug(f"Transaction BEGIN [{tx_id}]")
            else:
                RetryPolicy.execute(db_manager.execute, sql=f"SAVEPOINT {tx_id};")
                _logger.debug(f"Savepoint CREATED [{tx_id}]")
            stack.append(tx_id)
            return tx_id
        except Exception as e:
            raise TransactionError("Failed to begin transaction") from e

    @classmethod
    def commit(cls, tx_id: str) -> None:
        stack = cls._get_stack()
        if not stack or stack[-1] != tx_id: raise TransactionError(f"Transaction mismatch: expected {stack[-1] if stack else None}, got {tx_id}")
        try:
            stack.pop()
            if not stack:
                RetryPolicy.execute(db_manager.execute, sql="COMMIT;")
                _logger.debug(f"Transaction COMMIT [{tx_id}]")
            else:
                RetryPolicy.execute(db_manager.execute, sql=f"RELEASE SAVEPOINT {tx_id};")
                _logger.debug(f"Savepoint RELEASED [{tx_id}]")
        except Exception as e:
            raise TransactionError("Failed to commit transaction") from e

    @classmethod
    def rollback(cls, tx_id: str) -> None:
        stack = cls._get_stack()
        if not stack or tx_id not in stack: return
        try:
            idx = stack.index(tx_id)
            del stack[idx:]
            if not stack:
                RetryPolicy.execute(db_manager.execute, sql="ROLLBACK;")
                _logger.debug(f"Transaction ROLLBACK [{tx_id}]")
            else:
                RetryPolicy.execute(db_manager.execute, sql=f"ROLLBACK TO SAVEPOINT {tx_id};")
                _logger.debug(f"Savepoint ROLLBACK [{tx_id}]")
        except Exception as e:
            raise TransactionError("Failed to rollback transaction") from e

class UnitOfWork:
    def __init__(self):
        self.tx_id = None
        self.is_root = False

    def __enter__(self):
        stack = _tx_stack.get()
        if stack is None:
            stack = []
            _tx_stack.set(stack)
            _uow_identity_map.set(IdentityMap())
            self.is_root = True

        self.tx_id = TransactionManager.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None: TransactionManager.rollback(self.tx_id)
        else: TransactionManager.commit(self.tx_id)
        
        if self.is_root:
            imap = _uow_identity_map.get()
            if imap: imap.clear()
            _uow_identity_map.set(None)
            _tx_stack.set(None)

class AsyncUnitOfWork:
    def __init__(self):
        self._sync_uow = UnitOfWork()

    async def __aenter__(self):
        await asyncio.to_thread(self._sync_uow.__enter__)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await asyncio.to_thread(self._sync_uow.__exit__, exc_type, exc_val, exc_tb)

def transactional(func: Callable):
    def wrapper(*args, **kwargs):
        with UnitOfWork(): return func(*args, **kwargs)
    return wrapper

def async_transactional(func: Callable):
    async def wrapper(*args, **kwargs):
        async with AsyncUnitOfWork(): return await func(*args, **kwargs)
    return wrapper


# =========================================================================
# REPOSITORY CACHE
# =========================================================================

class RepositoryCache(Generic[T]):
    def __init__(self, max_size: int = 2000, ttl_sec: float = 60.0):
        self._cache: OrderedDict[Any, Tuple[float, T]] = OrderedDict()
        self._weakrefs: weakref.WeakValueDictionary[Any, T] = weakref.WeakValueDictionary()
        self.max_size = max_size
        self.ttl_sec = ttl_sec
        self._lock = threading.RLock()

    def get(self, pk: Any) -> Optional[T]:
        with self._lock:
            if pk in self._cache:
                timestamp, model = self._cache[pk]
                if (time.time() - timestamp) > self.ttl_sec:
                    del self._cache[pk]
                    metrics_engine.increment("repo_cache_expire", namespace="repository")
                    return None
                self._cache.move_to_end(pk)
                metrics_engine.increment("repo_cache_hit", namespace="repository")
                return model
            
            model = self._weakrefs.get(pk)
            if model: return model
            return None

    def put(self, pk: Any, model: T) -> None:
        with self._lock:
            self._cache[pk] = (time.time(), model)
            self._weakrefs[pk] = model
            self._cache.move_to_end(pk)
            if len(self._cache) > self.max_size: self._cache.popitem(last=False)

    def invalidate(self, pk: Any) -> None:
        with self._lock: self._cache.pop(pk, None)


# =========================================================================
# CQRS: READ REPOSITORY
# =========================================================================

class ReadRepository(Generic[T]):
    def __init__(self, model_class: Type[T], dialect: Dialect = ACTIVE_DIALECT):
        self.model_cls = model_class
        self.dialect = dialect
        self.table_name = model_class.table_name()
        self.pk_field = model_class.primary_key_field()
        
        tables = MetaDataRegistry.reflect()
        if self.table_name not in tables:
            raise RepositoryError(f"Table metadata for {self.table_name} missing.")
        self.table_meta = tables[self.table_name]

        self.has_soft_delete = any(f.name == "is_deleted" for f in dataclasses.fields(self.model_cls))
        self.cache = RepositoryCache[T]()

    def _map_row_to_model(self, row: Dict[str, Any]) -> T:
        pk = dict(row).get(self.pk_field)
        imap = _uow_identity_map.get()
        if imap and pk:
            existing = imap.get(self.table_name, pk)
            if existing: return cast(T, existing)

        model = self.model_cls.from_dict(row)
        self.cache.put(model.primary_key(), model)
        if imap and pk: imap.put(self.table_name, pk, model)
        return model

    @trace_span(operation="repo.find_by_id", component="repository", kind=SpanKind.INTERNAL)
    def find_by_id(self, pk: Any, include_deleted: bool = False) -> Optional[T]:
        imap = _uow_identity_map.get()
        if imap:
            im_model = imap.get(self.table_name, pk)
            if im_model: return cast(T, im_model)

        cached = self.cache.get(pk)
        if cached:
            if not include_deleted and self.has_soft_delete and getattr(cached, 'is_deleted', False): return None
            return cached

        filters = [FilterCriteria(field=self.pk_field, operator=QueryOperator.EQ, value=pk)]
        if self.has_soft_delete and not include_deleted:
            filters.append(FilterCriteria(field="is_deleted", operator=QueryOperator.EQ, value=False))
            
        ctx = SQLQueryBuilder(self.table_name, self.table_meta, self.dialect).where(filters).limit(1).build()
        row = RetryPolicy.execute(db_manager.fetch_one, sql=ctx.sql, params=ctx.parameters)
        
        if row: return self._map_row_to_model(row)
        return None

    @trace_span(operation="repo.find_one", component="repository", kind=SpanKind.INTERNAL)
    def find_one(self, *filters: FilterCriteria, include_deleted: bool = False) -> Optional[T]:
        flist = list(filters)
        if self.has_soft_delete and not include_deleted:
            flist.append(FilterCriteria(field="is_deleted", operator=QueryOperator.EQ, value=False))

        ctx = SQLQueryBuilder(self.table_name, self.table_meta, self.dialect).where(flist).limit(1).build()
        row = RetryPolicy.execute(db_manager.fetch_one, sql=ctx.sql, params=ctx.parameters)
        
        if row: return self._map_row_to_model(row)
        return None

    @trace_span(operation="repo.find", component="repository", kind=SpanKind.INTERNAL)
    def find(self, spec: Specification[T], include_deleted: bool = False) -> List[T]:
        return self.find_many(*spec.to_filters(), include_deleted=include_deleted)

    @trace_span(operation="repo.find_many", component="repository", kind=SpanKind.INTERNAL)
    def find_many(
        self, *filters: FilterCriteria, sorts: List[SortCriteria] = None,
        limit: Optional[int] = None, offset: Optional[int] = None,
        include_deleted: bool = False, for_update: bool = False
    ) -> List[T]:
        flist = list(filters)
        if self.has_soft_delete and not include_deleted:
            flist.append(FilterCriteria(field="is_deleted", operator=QueryOperator.EQ, value=False))

        builder = SQLQueryBuilder(self.table_name, self.table_meta, self.dialect).where(flist)
        if sorts: builder.order_by(sorts)
        if limit: builder.limit(limit)
        if offset: builder.offset(offset)
        if for_update: builder.for_update()

        ctx = builder.build()
        rows = RetryPolicy.execute(db_manager.fetch_all, sql=ctx.sql, params=ctx.parameters)
        return [self._map_row_to_model(r) for r in rows]

    @trace_span(operation="repo.count", component="repository", kind=SpanKind.INTERNAL)
    def count(self, *filters: FilterCriteria, include_deleted: bool = False) -> int:
        flist = list(filters)
        if self.has_soft_delete and not include_deleted:
            flist.append(FilterCriteria(field="is_deleted", operator=QueryOperator.EQ, value=False))

        ctx = SQLQueryBuilder(self.table_name, self.table_meta, self.dialect).where(flist).build_count()
        row = RetryPolicy.execute(db_manager.fetch_one, sql=ctx.sql, params=ctx.parameters)
        return row['count'] if row else 0

    def exists(self, *filters: FilterCriteria, include_deleted: bool = False) -> bool:
        return self.count(*filters, include_deleted=include_deleted) > 0

    @trace_span(operation="repo.aggregate", component="repository", kind=SpanKind.INTERNAL)
    def aggregate(self, aggr_func: Literal['SUM', 'AVG', 'MIN', 'MAX'], field: str, *filters: FilterCriteria, include_deleted: bool = False) -> Any:
        flist = list(filters)
        if self.has_soft_delete and not include_deleted:
            flist.append(FilterCriteria(field="is_deleted", operator=QueryOperator.EQ, value=False))

        ctx = SQLQueryBuilder(self.table_name, self.table_meta, self.dialect).where(flist).build_aggregate(aggr_func, field)
        row = RetryPolicy.execute(db_manager.fetch_one, sql=ctx.sql, params=ctx.parameters)
        return row['aggr_val'] if row else None

    def stream(self, *filters: FilterCriteria, batch_size: int = 1000, sorts: List[SortCriteria] = None, include_deleted: bool = False) -> Iterator[T]:
        offset = 0
        while True:
            batch = self.find_many(*filters, sorts=sorts, limit=batch_size, offset=offset, include_deleted=include_deleted)
            if not batch: break
            for item in batch: yield item
            offset += batch_size

    @trace_span(operation="repo.paginate", component="repository", kind=SpanKind.INTERNAL)
    def paginate(self, page: int = 1, page_size: int = 100, filters: List[FilterCriteria] = None, sorts: List[SortCriteria] = None, include_deleted: bool = False) -> Page[T]:
        f = filters or []
        total = self.count(*f, include_deleted=include_deleted)
        offset = (page - 1) * page_size
        items = self.find_many(*f, sorts=sorts, limit=page_size, offset=offset, include_deleted=include_deleted)
        return Page.create(items, total, page, page_size)


# =========================================================================
# CQRS: WRITE REPOSITORY
# =========================================================================

class WriteRepository(Generic[T]):
    def __init__(self, model_class: Type[T], read_repo: ReadRepository[T], dialect: Dialect = ACTIVE_DIALECT):
        self.model_cls = model_class
        self.dialect = dialect
        self.table_name = model_class.table_name()
        self.pk_field = model_class.primary_key_field()
        self.table_meta = MetaDataRegistry.reflect()[self.table_name]
        self.has_soft_delete = any(f.name == "is_deleted" for f in dataclasses.fields(self.model_cls))
        self.has_optimistic_lock = any(f.name == "version_id" for f in dataclasses.fields(self.model_cls))
        self.cache = read_repo.cache
        self._read_repo = read_repo

    @trace_span(operation="repo.save", component="repository", kind=SpanKind.INTERNAL)
    def save(self, model: T) -> T:
        existing = self._read_repo.find_by_id(model.primary_key())
        if existing: return self.update(model)
        return self.insert(model)

    @trace_span(operation="repo.insert", component="repository", kind=SpanKind.INTERNAL)
    def insert(self, model: T) -> T:
        model.validate()
        model._run_hooks('before_insert')

        cols = model.insert_columns()
        sql = SQLWriteBuilder.build_insert(self.table_name, cols, self.dialect)
        params = model.insert_values()

        RetryPolicy.execute(db_manager.execute, sql=sql, params=params)
        model._run_hooks('after_insert')
        
        self.cache.put(model.primary_key(), model)
        imap = _uow_identity_map.get()
        if imap: imap.put(self.table_name, model.primary_key(), model)
            
        model.reset_dirty_state()
        return model

    @trace_span(operation="repo.update", component="repository", kind=SpanKind.INTERNAL)
    def update(self, model: T) -> T:
        if not model.is_dirty(): return model

        model.validate()
        model._run_hooks('before_update')

        cols = model.update_columns()
        if not cols: return model

        params = []
        for col in cols:
            col_meta = self.table_meta.columns.get(col)
            val = getattr(model, col)
            params.append(SerializerRegistry.to_sql(val, dataclasses.asdict(col_meta) if col_meta else {}, self.dialect))

        pk_val = model.primary_key()
        params.append(pk_val)
        
        version_field = "version_id" if self.has_optimistic_lock else None
        if self.has_optimistic_lock:
            original_version = getattr(model, '_original_state', {}).get('version_id', 1)
            params.append(original_version)

        sql = SQLWriteBuilder.build_update(self.table_name, cols, self.pk_field, self.dialect, version_field=version_field)
        res = RetryPolicy.execute(db_manager.execute, sql=sql, params=tuple(params))
        
        if self.has_optimistic_lock and getattr(res, 'rowcount', -1) == 0:
            raise OptimisticLockError(f"Concurrent modification detected on {self.table_name} [{pk_val}]")

        model._run_hooks('after_update')
        
        self.cache.put(pk_val, model)
        model.reset_dirty_state()
        return model

    @trace_span(operation="repo.delete", component="repository", kind=SpanKind.INTERNAL)
    def delete(self, model: T, hard: bool = False) -> None:
        pk_val = model.primary_key()
        if self.has_soft_delete and not hard:
            model._run_hooks('before_delete')
            if hasattr(model, 'soft_delete'): model.soft_delete()
            else:
                setattr(model, 'is_deleted', True)
                setattr(model, 'deleted_at', TimestampHelper.now_utc())
            self.update(model)
        else:
            model._run_hooks('before_delete')
            sql = SQLWriteBuilder.build_delete(self.table_name, self.pk_field, self.dialect)
            RetryPolicy.execute(db_manager.execute, sql=sql, params=(pk_val,))
            model._run_hooks('after_delete')

        self.cache.invalidate(pk_val)
        imap = _uow_identity_map.get()
        if imap: imap.remove(self.table_name, pk_val)

    @trace_span(operation="repo.restore", component="repository", kind=SpanKind.INTERNAL)
    def restore(self, model: T) -> T:
        if not self.has_soft_delete: raise ValidationError("Model does not support Soft Delete.")
        setattr(model, 'is_deleted', False)
        setattr(model, 'deleted_at', None)
        return self.update(model)

    @trace_span(operation="repo.bulk_insert", component="repository", kind=SpanKind.INTERNAL)
    def bulk_insert(self, models: List[T]) -> None:
        if not models: return
        for m in models:
            m.validate()
            m._run_hooks('before_insert')

        cols = models[0].insert_columns()
        sql = SQLWriteBuilder.build_insert(self.table_name, cols, self.dialect)
        params_list = [m.insert_values() for m in models]
        
        RetryPolicy.execute(db_manager.execute_many, sql=sql, params_list=params_list)
        
        imap = _uow_identity_map.get()
        for m in models:
            m._run_hooks('after_insert')
            self.cache.put(m.primary_key(), m)
            if imap: imap.put(self.table_name, m.primary_key(), m)
            m.reset_dirty_state()
            
        AuditEngine.record_success(f"repo.bulk_insert.{self.table_name}", AuditAction.CREATE, f"Inserted {len(models)} rows.")

    @trace_span(operation="repo.bulk_update", component="repository", kind=SpanKind.INTERNAL)
    def bulk_update(self, models: List[T]) -> None:
        if not models: return
        with UnitOfWork():
            for m in models: self.update(m)

    @trace_span(operation="repo.bulk_delete", component="repository", kind=SpanKind.INTERNAL)
    def bulk_delete(self, models: List[T], hard: bool = False) -> None:
        if not models: return
        with UnitOfWork():
            for m in models: self.delete(m, hard=hard)

    @trace_span(operation="repo.upsert", component="repository", kind=SpanKind.INTERNAL)
    def upsert(self, model: T, conflict_keys: List[str]) -> T:
        model.validate()
        model._run_hooks('before_update') 
        
        cols = model.insert_columns()
        sql = SQLWriteBuilder.build_upsert(self.table_name, cols, conflict_keys, self.dialect)
        params = model.insert_values()
        
        RetryPolicy.execute(db_manager.execute, sql=sql, params=params)
        self.cache.invalidate(model.primary_key())
        
        res = self._read_repo.find_by_id(model.primary_key())
        if res is None: raise DatabaseError("Upsert resolution failed.")
        return res

    @trace_span(operation="repo.batch_upsert", component="repository", kind=SpanKind.INTERNAL)
    def batch_upsert(self, models: List[T], conflict_keys: List[str]) -> None:
        if not models: return
        for m in models:
            m.validate()
            m._run_hooks('before_update')
            
        cols = models[0].insert_columns()
        sql = SQLWriteBuilder.build_upsert(self.table_name, cols, conflict_keys, self.dialect)
        params_list = [m.insert_values() for m in models]
        
        RetryPolicy.execute(db_manager.execute_many, sql=sql, params_list=params_list)
        for m in models: self.cache.invalidate(m.primary_key())


# =========================================================================
# GENERIC REPOSITORY (CQRS COMPOSITION)
# =========================================================================

class GenericRepository(ReadRepository[T], WriteRepository[T]):
    def __init__(self, model_class: Type[T], dialect: Dialect = ACTIVE_DIALECT):
        ReadRepository.__init__(self, model_class, dialect)
        WriteRepository.__init__(self, model_class, self, dialect)


# =========================================================================
# ASYNC GENERIC REPOSITORY
# =========================================================================

class AsyncGenericRepository(Generic[T]):
    """Enterprise Async Data Access Layer utilizing asyncio.to_thread delegation."""
    def __init__(self, model_class: Type[T], dialect: Dialect = ACTIVE_DIALECT):
        self._sync_repo = GenericRepository[T](model_class, dialect)

    async def find_by_id(self, pk: Any, include_deleted: bool = False) -> Optional[T]:
        return await asyncio.to_thread(self._sync_repo.find_by_id, pk, include_deleted)

    async def find(self, spec: Specification[T], include_deleted: bool = False) -> List[T]:
        return await asyncio.to_thread(self._sync_repo.find, spec, include_deleted)

    async def find_one(self, *filters: FilterCriteria, include_deleted: bool = False) -> Optional[T]:
        return await asyncio.to_thread(self._sync_repo.find_one, *filters, include_deleted=include_deleted)

    async def find_many(self, *filters: FilterCriteria, sorts: List[SortCriteria] = None, limit: Optional[int] = None, offset: Optional[int] = None, include_deleted: bool = False, for_update: bool = False) -> List[T]:
        return await asyncio.to_thread(self._sync_repo.find_many, *filters, sorts=sorts, limit=limit, offset=offset, include_deleted=include_deleted, for_update=for_update)

    async def count(self, *filters: FilterCriteria, include_deleted: bool = False) -> int:
        return await asyncio.to_thread(self._sync_repo.count, *filters, include_deleted=include_deleted)

    async def exists(self, *filters: FilterCriteria, include_deleted: bool = False) -> bool:
        return await asyncio.to_thread(self._sync_repo.exists, *filters, include_deleted=include_deleted)

    async def aggregate(self, aggr_func: Literal['SUM', 'AVG', 'MIN', 'MAX'], field: str, *filters: FilterCriteria, include_deleted: bool = False) -> Any:
        return await asyncio.to_thread(self._sync_repo.aggregate, aggr_func, field, *filters, include_deleted=include_deleted)

    async def paginate(self, page: int = 1, page_size: int = 100, filters: List[FilterCriteria] = None, sorts: List[SortCriteria] = None, include_deleted: bool = False) -> Page[T]:
        return await asyncio.to_thread(self._sync_repo.paginate, page=page, page_size=page_size, filters=filters, sorts=sorts, include_deleted=include_deleted)

    async def stream(self, *filters: FilterCriteria, batch_size: int = 1000, sorts: List[SortCriteria] = None, include_deleted: bool = False) -> AsyncGenerator[T, None]:
        offset = 0
        while True:
            batch = await self.find_many(*filters, sorts=sorts, limit=batch_size, offset=offset, include_deleted=include_deleted)
            if not batch: break
            for item in batch: yield item
            offset += batch_size

    async def save(self, model: T) -> T:
        return await asyncio.to_thread(self._sync_repo.save, model)

    async def insert(self, model: T) -> T:
        return await asyncio.to_thread(self._sync_repo.insert, model)

    async def update(self, model: T) -> T:
        return await asyncio.to_thread(self._sync_repo.update, model)

    async def delete(self, model: T, hard: bool = False) -> None:
        await asyncio.to_thread(self._sync_repo.delete, model, hard)

    async def restore(self, model: T) -> T:
        return await asyncio.to_thread(self._sync_repo.restore, model)

    async def bulk_insert(self, models: List[T]) -> None:
        await asyncio.to_thread(self._sync_repo.bulk_insert, models)

    async def bulk_update(self, models: List[T]) -> None:
        await asyncio.to_thread(self._sync_repo.bulk_update, models)

    async def bulk_delete(self, models: List[T], hard: bool = False) -> None:
        await asyncio.to_thread(self._sync_repo.bulk_delete, models, hard)

    async def upsert(self, model: T, conflict_keys: List[str]) -> T:
        return await asyncio.to_thread(self._sync_repo.upsert, model, conflict_keys)

    async def batch_upsert(self, models: List[T], conflict_keys: List[str]) -> None:
        await asyncio.to_thread(self._sync_repo.batch_upsert, models, conflict_keys)


# =========================================================================
# REPOSITORY REGISTRY & FACTORY
# =========================================================================

class RepositoryRegistry:
    """Enterprise DI Container for Repository Interfaces."""
    _repos: Dict[Type[BaseModel], GenericRepository] = {}
    _async_repos: Dict[Type[BaseModel], AsyncGenericRepository] = {}
    _lock = threading.Lock()

    @classmethod
    def get_sync(cls, model_class: Type[T]) -> GenericRepository[T]:
        with cls._lock:
            if model_class not in cls._repos:
                cls._repos[model_class] = GenericRepository[T](model_class)
            return cls._repos[model_class]

    @classmethod
    def get_async(cls, model_class: Type[T]) -> AsyncGenericRepository[T]:
        with cls._lock:
            if model_class not in cls._async_repos:
                cls._async_repos[model_class] = AsyncGenericRepository[T](model_class)
            return cls._async_repos[model_class]

class RepositoryFactory:
    @staticmethod
    def create(model_class: Type[T]) -> GenericRepository[T]:
        return RepositoryRegistry.get_sync(model_class)
        
    @staticmethod
    def create_async(model_class: Type[T]) -> AsyncGenericRepository[T]:
        return RepositoryRegistry.get_async(model_class)


# =========================================================================
# EXPORTS
# =========================================================================

__all__ = [
    # Exceptions
    "RepositoryError", "ValidationError", "OptimisticLockError", "DeadlockError",
    "TransactionError", "ConnectionError", "SerializationError", "SecurityError", "CircuitBreakerError",

    # Enums & Configurations
    "QueryOperator", "SortDirection", "JoinType", "ACTIVE_DIALECT",
    
    # Primitives & Resiliency
    "CircuitBreaker", "db_circuit_breaker", "RetryPolicy",
    
    # Specification Pattern
    "FilterCriteria", "Specification", "AndSpecification", "OrSpecification", "NotSpecification",
    
    # Query Builders & Types
    "Page", "SortCriteria", "QueryContext", "SQLQueryBuilder", "SQLWriteBuilder",
    
    # UOW & Transactions
    "IdentityMap", "UnitOfWork", "AsyncUnitOfWork", "TransactionContext", "AsyncTransactionContext", 
    "transactional", "async_transactional",
    
    # Cache
    "RepositoryCache",
    
    # Repositories
    "ReadRepository", "WriteRepository", "GenericRepository", "AsyncGenericRepository",
    
    # DI Factory
    "RepositoryRegistry", "RepositoryFactory"
]
