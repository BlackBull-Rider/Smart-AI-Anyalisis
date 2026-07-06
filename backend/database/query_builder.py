"""
GREEN BULL RIDER V6 - Institutional-grade AI Stock Analysis Platform
Module: backend/database/query_builder.py
Description: Enterprise Production-Locked Fluent SQL Query Builder.
             Provides a strictly immutable, highly secure, and optimized 
             SQL generation engine. Prevents all structural SQL injection 
             vectors through rigorous regex validation and enforced parameterization.
             Supports advanced SQL semantics (Recursive CTEs, Window Functions, 
             Upserts, Safe Joins, UNION, EXISTS, CASE WHEN, CAST) for SQLite and PostgreSQL. 
             Fully integrated with institutional telemetry (Trace, Metrics, Audit).
             Python 3.13 Compatible.
"""

import re
import copy
import hashlib
from enum import Enum
from dataclasses import dataclass, field as dc_field
from collections import OrderedDict
from typing import (
    Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, Final, cast
)

# Internal Platform Integrations
from backend.core.logger import AppLogger
from backend.core.trace import TraceEngine, SpanKind, trace_span
from backend.core.metrics import metrics_engine
from backend.core.audit import AuditEngine, AuditAction, AuditSeverity
from backend.core.exceptions import GreenBullError

_logger = AppLogger("QueryBuilder")

T = TypeVar('T')


# =========================================================================
# EXCEPTIONS & TELEMETRY HELPER
# =========================================================================

class QueryBuilderError(GreenBullError):
    """Base exception for all query builder failures."""
    error_code: str = "GBR-QB-001"

class ValidationError(QueryBuilderError):
    """Raised when query constraints or typings are violated."""
    error_code: str = "GBR-QB-002"

class SecurityError(QueryBuilderError):
    """Raised when an active SQL Injection attempt or invalid identifier is detected."""
    error_code: str = "GBR-QB-003"

class QuerySyntaxError(QueryBuilderError):
    """Raised when conflicting query clauses are constructed."""
    error_code: str = "GBR-QB-004"

class UnsupportedDialectError(QueryBuilderError):
    """Raised when a query feature is unsupported by the active dialect."""
    error_code: str = "GBR-QB-005"

class WindowFunctionError(QueryBuilderError):
    """Raised for malformed window function definitions."""
    error_code: str = "GBR-QB-006"


def _raise_error(err_class: Type[Exception], msg: str, action: AuditAction = AuditAction.SYSTEM) -> None:
    """Centralized error handling ensuring 100% telemetry coverage."""
    metrics_engine.increment("query_builder_errors", namespace="database")
    if err_class == SecurityError:
        metrics_engine.increment("query_security_rejected", namespace="database")
    
    AuditEngine.record_failure("query_builder", action, AuditSeverity.CRITICAL, msg)
    _logger.error(f"QueryBuilder Error [{err_class.__name__}]: {msg}")
    
    e = err_class(msg)
    TraceEngine.record_exception(e)
    raise e


# =========================================================================
# ENUMS
# =========================================================================

class Dialect(str, Enum):
    SQLITE = "SQLITE"
    POSTGRES = "POSTGRES"

class QueryOperator(str, Enum):
    EQ = "="; NEQ = "!="; GT = ">"; LT = "<"; GTE = ">="; LTE = "<="
    IN = "IN"; NOT_IN = "NOT IN"; LIKE = "LIKE"; ILIKE = "ILIKE"
    BETWEEN = "BETWEEN"; IS_NULL = "IS NULL"; IS_NOT_NULL = "IS NOT NULL"

class JoinType(str, Enum):
    INNER = "INNER JOIN"; LEFT = "LEFT JOIN"; RIGHT = "RIGHT JOIN"
    FULL = "FULL OUTER JOIN"; CROSS = "CROSS JOIN"

class SortDirection(str, Enum):
    ASC = "ASC"; DESC = "DESC"

class AggregateFunction(str, Enum):
    COUNT = "COUNT"; SUM = "SUM"; AVG = "AVG"; MIN = "MIN"; MAX = "MAX"

class WindowFunction(str, Enum):
    ROW_NUMBER = "ROW_NUMBER"; RANK = "RANK"; DENSE_RANK = "DENSE_RANK"
    LAG = "LAG"; LEAD = "LEAD"; FIRST_VALUE = "FIRST_VALUE"
    LAST_VALUE = "LAST_VALUE"; NTILE = "NTILE"

class StatementType(str, Enum):
    SELECT = "SELECT"; INSERT = "INSERT"; UPDATE = "UPDATE"
    DELETE = "DELETE"; UPSERT = "UPSERT"; MERGE = "MERGE"


# =========================================================================
# SECURITY & VALIDATION ENGINE
# =========================================================================

class SecurityValidator:
    """
    Enterprise structural validation engine.
    Forbids raw SQL composition, ensuring identifiers match highly restricted,
    precompiled regular expressions to neutralize structural injection vectors.
    """
    
    _FORBIDDEN_KEYWORDS = re.compile(
        r"(?i)\b(UNION|SELECT|DROP|ALTER|INSERT|DELETE|UPDATE|EXEC|PRAGMA|COPY|TRUNCATE|GRANT|REVOKE)\b"
    )
    _FORBIDDEN_CHARS = re.compile(r"[;'\"]|--|\/\*|\*\/")
    
    _IDENTIFIER = r"[a-zA-Z_][a-zA-Z0-9_]*"
    _COL_REF = fr"(?:{_IDENTIFIER}\.)?{_IDENTIFIER}|\*"
    
    _ALLOWED_FUNCTIONS = r"(COUNT|SUM|AVG|MIN|MAX|UPPER|LOWER|TRIM|ABS|ROUND|COALESCE|NULLIF)"
    _FUNC_CALL = fr"^{_ALLOWED_FUNCTIONS}\({_COL_REF}\)$"
    
    _EXACT_IDENTIFIER = re.compile(fr"^{_COL_REF}$")
    _EXACT_FUNCTION = re.compile(_FUNC_CALL, re.IGNORECASE)

    @classmethod
    def sanitize(cls, identifier: str) -> str:
        if not isinstance(identifier, str):
            _raise_error(SecurityError, f"Identifier must be a string, got {type(identifier)}")
            
        if cls._FORBIDDEN_CHARS.search(identifier):
            _raise_error(SecurityError, f"Forbidden character sequence in identifier: {identifier}")
            
        if cls._FORBIDDEN_KEYWORDS.search(identifier):
            _raise_error(SecurityError, f"Forbidden SQL keyword in identifier: {identifier}")

        if cls._EXACT_IDENTIFIER.match(identifier) or cls._EXACT_FUNCTION.match(identifier):
            return identifier

        _raise_error(SecurityError, f"Structural SQL Injection prevented. Invalid identifier: {identifier}")
        return "" # Unreachable, satisfies type checker

    @classmethod
    def sanitize_alias(cls, alias: str) -> str:
        if not re.match(fr"^{cls._IDENTIFIER}$", alias):
            _raise_error(SecurityError, f"Invalid alias identifier: {alias}")
        return alias


# =========================================================================
# CORE QUERY CONTEXT
# =========================================================================

@dataclass(frozen=True, slots=True, kw_only=True)
class QueryContext:
    sql: str
    parameters: Tuple[Any, ...]
    statement_type: StatementType
    estimated_complexity: int
    execution_metadata: Dict[str, Any]
    
    @property
    def query_hash(self) -> str:
        return hashlib.sha256(self.sql.encode('utf-8')).hexdigest()

    def is_select(self) -> bool: return self.statement_type == StatementType.SELECT
    def is_write(self) -> bool: return self.statement_type in (StatementType.INSERT, StatementType.UPDATE, StatementType.DELETE, StatementType.UPSERT, StatementType.MERGE)
    def uses_join(self) -> bool: return self.execution_metadata.get('has_joins', False)
    def uses_window(self) -> bool: return self.execution_metadata.get('has_windows', False)
    def uses_group_by(self) -> bool: return self.execution_metadata.get('has_groups', False)


class ParameterManager:
    __slots__ = ('dialect', 'params', '_counter')

    def __init__(self, dialect: Dialect):
        self.dialect = dialect
        self.params: List[Any] = []
        self._counter = 1

    def add(self, value: Any) -> str:
        self.params.append(value)
        return "%s" if self.dialect == Dialect.POSTGRES else "?"

    def export(self) -> Tuple[Any, ...]:
        return tuple(self.params)


# =========================================================================
# ABSTRACT SYNTAX TREE (AST) NODES
# =========================================================================

@dataclass(frozen=True, slots=True, kw_only=True)
class ProjectionNode:
    field: str
    alias: Optional[str] = None
    
    def render(self) -> str:
        safe_field = SecurityValidator.sanitize(self.field)
        if self.alias:
            return f"{safe_field} AS {SecurityValidator.sanitize_alias(self.alias)}"
        return safe_field

@dataclass(frozen=True, slots=True, kw_only=True)
class ConditionNode:
    field: str
    operator: QueryOperator
    value: Any
    
    def render(self, pm: ParameterManager) -> str:
        safe_field = SecurityValidator.sanitize(self.field)
        op = self.operator.value
        
        if self.operator in (QueryOperator.IS_NULL, QueryOperator.IS_NOT_NULL):
            return f"{safe_field} {op}"
            
        if self.operator in (QueryOperator.IN, QueryOperator.NOT_IN):
            if not isinstance(self.value, (list, tuple, set)):
                _raise_error(ValidationError, f"Operator {op} requires iterable value.")
            if not self.value:
                return "1=0" if self.operator == QueryOperator.IN else "1=1"
            tokens = ", ".join([pm.add(v) for v in self.value])
            return f"{safe_field} {op} ({tokens})"
            
        if self.operator == QueryOperator.BETWEEN:
            if not isinstance(self.value, (list, tuple)) or len(self.value) != 2:
                _raise_error(ValidationError, "BETWEEN operator requires a 2-tuple value.")
            return f"{safe_field} BETWEEN {pm.add(self.value[0])} AND {pm.add(self.value[1])}"

        if hasattr(self.value, '_build_sql'):
            sub_sql = self.value._build_sql(pm)
            return f"{safe_field} {op} ({sub_sql})"

        return f"{safe_field} {op} {pm.add(self.value)}"

@dataclass(frozen=True, slots=True, kw_only=True)
class ExistsNode:
    subquery: Any
    not_exists: bool = False

    def render(self, pm: ParameterManager) -> str:
        if not hasattr(self.subquery, '_build_sql'):
            _raise_error(ValidationError, "EXISTS requires a valid QueryBuilder instance.")
        op = "NOT EXISTS" if self.not_exists else "EXISTS"
        return f"{op} ({self.subquery._build_sql(pm)})"

@dataclass(frozen=True, slots=True, kw_only=True)
class JoinConditionNode:
    left_field: str
    operator: QueryOperator
    right_field: str

    def render(self) -> str:
        lf = SecurityValidator.sanitize(self.left_field)
        rf = SecurityValidator.sanitize(self.right_field)
        return f"{lf} {self.operator.value} {rf}"

@dataclass(frozen=True, slots=True, kw_only=True)
class JoinNode:
    table: str
    condition: JoinConditionNode
    type: JoinType
    
    def render(self) -> str:
        safe_table = SecurityValidator.sanitize(self.table)
        return f"{self.type.value} {safe_table} ON {self.condition.render()}"

@dataclass(frozen=True, slots=True, kw_only=True)
class SortNode:
    field: str
    direction: SortDirection
    
    def render(self) -> str:
        return f"{SecurityValidator.sanitize(self.field)} {self.direction.value}"

@dataclass(frozen=True, slots=True, kw_only=True)
class WindowNode:
    func: WindowFunction
    partition_by: List[str]
    order_by: List[SortNode]
    alias: str
    
    def render(self) -> str:
        safe_alias = SecurityValidator.sanitize_alias(self.alias)
        parts = ", ".join([SecurityValidator.sanitize(p) for p in self.partition_by])
        orders = ", ".join([o.render() for o in self.order_by])
        
        over_clause = []
        if parts: over_clause.append(f"PARTITION BY {parts}")
        if orders: over_clause.append(f"ORDER BY {orders}")
        
        return f"{self.func.value}() OVER ({' '.join(over_clause)}) AS {safe_alias}"

@dataclass(frozen=True, slots=True, kw_only=True)
class CaseWhenNode:
    conditions: List[Tuple[ConditionNode, Any]]
    else_result: Any
    alias: Optional[str] = None

    def render(self, pm: ParameterManager) -> str:
        clauses = []
        for cond, res in self.conditions:
            clauses.append(f"WHEN {cond.render(pm)} THEN {pm.add(res)}")
        else_clause = f" ELSE {pm.add(self.else_result)}" if self.else_result is not None else ""
        
        sql = f"CASE {' '.join(clauses)}{else_clause} END"
        if self.alias:
            sql += f" AS {SecurityValidator.sanitize_alias(self.alias)}"
        return sql

@dataclass(frozen=True, slots=True, kw_only=True)
class CastNode:
    field: str
    data_type: str
    alias: Optional[str] = None

    def render(self) -> str:
        safe_field = SecurityValidator.sanitize(self.field)
        safe_type = SecurityValidator.sanitize_alias(self.data_type) # reusing basic strict rules
        sql = f"CAST({safe_field} AS {safe_type})"
        if self.alias:
            sql += f" AS {SecurityValidator.sanitize_alias(self.alias)}"
        return sql


# =========================================================================
# BASE IMMUTABLE BUILDER
# =========================================================================

class BaseFluentBuilder:
    __slots__ = ('_dialect', '_table')

    def __init__(self, table: str, dialect: Dialect):
        self._table = SecurityValidator.sanitize(table)
        self._dialect = dialect

    def _clone(self) -> 'BaseFluentBuilder':
        cls = self.__class__
        new_instance = cls.__new__(cls)
        for slot in self._get_all_slots():
            val = getattr(self, slot)
            if isinstance(val, (list, dict, set)):
                setattr(new_instance, slot, copy.deepcopy(val))
            else:
                setattr(new_instance, slot, val)
        return new_instance

    def _get_all_slots(self) -> List[str]:
        slots = []
        for cls in type(self).__mro__:
            slots.extend(getattr(cls, '__slots__', []))
        return list(set(slots))


# =========================================================================
# QUERY BUILDERS (SELECT, INSERT, UPDATE, DELETE, UPSERT, AGGREGATE)
# =========================================================================

class SelectBuilder(BaseFluentBuilder):
    """Institutional-grade Immutable Select Builder supporting CTEs, Windows, Unions and Subqueries."""
    __slots__ = (
        '_projections', '_joins', '_wheres', '_group_bys', '_havings', 
        '_order_bys', '_windows', '_cases', '_casts', '_limit', '_offset', 
        '_for_update', '_ctes', '_recursive_ctes', '_unions', '_distinct'
    )

    def __init__(self, table: str, dialect: Dialect):
        super().__init__(table, dialect)
        self._projections: List[ProjectionNode] = []
        self._joins: List[JoinNode] = []
        self._wheres: List[Union[ConditionNode, ExistsNode]] = []
        self._group_bys: List[str] = []
        self._havings: List[ConditionNode] = []
        self._order_bys: List[SortNode] = []
        self._windows: List[WindowNode] = []
        self._cases: List[CaseWhenNode] = []
        self._casts: List[CastNode] = []
        self._limit: Optional[int] = None
        self._offset: Optional[int] = None
        self._for_update: bool = False
        self._distinct: bool = False
        self._ctes: Dict[str, 'SelectBuilder'] = {}
        self._recursive_ctes: Dict[str, Tuple['SelectBuilder', 'SelectBuilder']] = {}
        self._unions: List[Tuple['SelectBuilder', bool]] = []

    @trace_span(operation="qb.distinct", component="database", kind=SpanKind.INTERNAL)
    def distinct(self) -> 'SelectBuilder':
        _logger.debug("Applying DISTINCT")
        b = cast(SelectBuilder, self._clone())
        b._distinct = True
        return b

    @trace_span(operation="qb.select", component="database", kind=SpanKind.INTERNAL)
    def select(self, field: str, alias: Optional[str] = None) -> 'SelectBuilder':
        _logger.debug(f"Adding select projection: {field} AS {alias}")
        b = cast(SelectBuilder, self._clone())
        b._projections.append(ProjectionNode(field=field, alias=alias))
        return b

    @trace_span(operation="qb.case", component="database", kind=SpanKind.INTERNAL)
    def case(self, conditions: List[Tuple[Tuple[str, QueryOperator, Any], Any]], else_result: Any, alias: Optional[str] = None) -> 'SelectBuilder':
        _logger.debug(f"Adding CASE WHEN clause: alias={alias}")
        b = cast(SelectBuilder, self._clone())
        cond_nodes = [(ConditionNode(field=f, operator=o, value=v), res) for (f, o, v), res in conditions]
        b._cases.append(CaseWhenNode(conditions=cond_nodes, else_result=else_result, alias=alias))
        return b

    @trace_span(operation="qb.cast", component="database", kind=SpanKind.INTERNAL)
    def cast(self, field: str, data_type: str, alias: Optional[str] = None) -> 'SelectBuilder':
        _logger.debug(f"Adding CAST({field} AS {data_type})")
        b = cast(SelectBuilder, self._clone())
        b._casts.append(CastNode(field=field, data_type=data_type, alias=alias))
        return b

    @trace_span(operation="qb.with_cte", component="database", kind=SpanKind.INTERNAL)
    def with_cte(self, name: str, builder: 'SelectBuilder') -> 'SelectBuilder':
        _logger.debug(f"Adding CTE: {name}")
        metrics_engine.increment("query_cte_total", namespace="database")
        b = cast(SelectBuilder, self._clone())
        b._ctes[SecurityValidator.sanitize_alias(name)] = builder
        return b

    @trace_span(operation="qb.with_recursive_cte", component="database", kind=SpanKind.INTERNAL)
    def with_recursive_cte(self, name: str, base_query: 'SelectBuilder', recursive_query: 'SelectBuilder') -> 'SelectBuilder':
        _logger.debug(f"Adding Recursive CTE: {name}")
        metrics_engine.increment("query_cte_total", namespace="database")
        b = cast(SelectBuilder, self._clone())
        b._recursive_ctes[SecurityValidator.sanitize_alias(name)] = (base_query, recursive_query)
        return b

    @trace_span(operation="qb.join", component="database", kind=SpanKind.INTERNAL)
    def join(self, table: str, left_col: str, operator: QueryOperator, right_col: str, join_type: JoinType = JoinType.INNER) -> 'SelectBuilder':
        _logger.debug(f"Adding {join_type.value}: {table}")
        metrics_engine.increment("query_join_total", namespace="database")
        b = cast(SelectBuilder, self._clone())
        cond = JoinConditionNode(left_field=left_col, operator=operator, right_field=right_col)
        b._joins.append(JoinNode(table=table, condition=cond, type=join_type))
        return b

    @trace_span(operation="qb.where", component="database", kind=SpanKind.INTERNAL)
    def where(self, field: str, operator: QueryOperator, value: Any) -> 'SelectBuilder':
        _logger.debug(f"Adding WHERE: {field} {operator.value}")
        b = cast(SelectBuilder, self._clone())
        b._wheres.append(ConditionNode(field=field, operator=operator, value=value))
        return b

    @trace_span(operation="qb.where_exists", component="database", kind=SpanKind.INTERNAL)
    def where_exists(self, subquery: 'SelectBuilder', not_exists: bool = False) -> 'SelectBuilder':
        _logger.debug(f"Adding {'NOT ' if not_exists else ''}EXISTS clause")
        metrics_engine.increment("query_subquery_total", namespace="database")
        b = cast(SelectBuilder, self._clone())
        b._wheres.append(ExistsNode(subquery=subquery, not_exists=not_exists))
        return b

    @trace_span(operation="qb.union", component="database", kind=SpanKind.INTERNAL)
    def union(self, builder: 'SelectBuilder', union_all: bool = False) -> 'SelectBuilder':
        _logger.debug(f"Adding UNION{' ALL' if union_all else ''}")
        b = cast(SelectBuilder, self._clone())
        b._unions.append((builder, union_all))
        return b

    @trace_span(operation="qb.group_by", component="database", kind=SpanKind.INTERNAL)
    def group_by(self, field: str) -> 'SelectBuilder':
        _logger.debug(f"Adding GROUP BY: {field}")
        b = cast(SelectBuilder, self._clone())
        b._group_bys.append(SecurityValidator.sanitize(field))
        return b

    @trace_span(operation="qb.having", component="database", kind=SpanKind.INTERNAL)
    def having(self, field: str, operator: QueryOperator, value: Any) -> 'SelectBuilder':
        _logger.debug(f"Adding HAVING: {field} {operator.value}")
        b = cast(SelectBuilder, self._clone())
        b._havings.append(ConditionNode(field=field, operator=operator, value=value))
        return b

    @trace_span(operation="qb.window", component="database", kind=SpanKind.INTERNAL)
    def window(self, func: WindowFunction, partition_by: List[str], order_by: List[Tuple[str, SortDirection]], alias: str) -> 'SelectBuilder':
        _logger.debug(f"Adding WINDOW function: {func.value}")
        metrics_engine.increment("query_window_total", namespace="database")
        b = cast(SelectBuilder, self._clone())
        orders = [SortNode(field=f, direction=d) for f, d in order_by]
        b._windows.append(WindowNode(func=func, partition_by=partition_by, order_by=orders, alias=alias))
        return b

    @trace_span(operation="qb.order_by", component="database", kind=SpanKind.INTERNAL)
    def order_by(self, field: str, direction: SortDirection = SortDirection.ASC) -> 'SelectBuilder':
        _logger.debug(f"Adding ORDER BY: {field} {direction.value}")
        b = cast(SelectBuilder, self._clone())
        b._order_bys.append(SortNode(field=field, direction=direction))
        return b

    @trace_span(operation="qb.limit", component="database", kind=SpanKind.INTERNAL)
    def limit(self, val: int) -> 'SelectBuilder':
        _logger.debug(f"Adding LIMIT: {val}")
        if val < 0: _raise_error(ValidationError, "Limit must be positive.")
        b = cast(SelectBuilder, self._clone())
        b._limit = val
        return b

    @trace_span(operation="qb.offset", component="database", kind=SpanKind.INTERNAL)
    def offset(self, val: int) -> 'SelectBuilder':
        _logger.debug(f"Adding OFFSET: {val}")
        if val < 0: _raise_error(ValidationError, "Offset must be positive.")
        b = cast(SelectBuilder, self._clone())
        b._offset = val
        return b

    @trace_span(operation="qb.for_update", component="database", kind=SpanKind.INTERNAL)
    def for_update(self) -> 'SelectBuilder':
        _logger.debug("Applying FOR UPDATE lock")
        b = cast(SelectBuilder, self._clone())
        b._for_update = True
        return b

    def _build_sql(self, pm: ParameterManager) -> str:
        sql_parts = []
        
        # CTEs
        if self._ctes or self._recursive_ctes:
            cte_clauses = []
            for name, cte_b in self._ctes.items():
                cte_clauses.append(f"{name} AS ({cte_b._build_sql(pm)})")
            for name, (base_b, rec_b) in self._recursive_ctes.items():
                cte_clauses.append(f"{name} AS ({base_b._build_sql(pm)} UNION ALL {rec_b._build_sql(pm)})")
            
            prefix = "WITH RECURSIVE" if self._recursive_ctes and self._dialect == Dialect.POSTGRES else "WITH"
            sql_parts.append(f"{prefix} " + ", ".join(cte_clauses))
            
        select_clause = "SELECT DISTINCT" if self._distinct else "SELECT"
        
        if not self._projections and not self._windows and not self._cases and not self._casts:
            select_clause += " *"
        else:
            projs = [p.render() for p in self._projections]
            projs.extend([w.render() for w in self._windows])
            projs.extend([c.render(pm) for c in self._cases])
            projs.extend([c.render() for c in self._casts])
            select_clause += " " + ", ".join(projs)
            
        sql_parts.append(select_clause)
        sql_parts.append(f"FROM {self._table}")
        
        if self._joins: sql_parts.extend([j.render() for j in self._joins])
        if self._wheres: sql_parts.append("WHERE " + " AND ".join([w.render(pm) for w in self._wheres]))
        if self._group_bys: sql_parts.append("GROUP BY " + ", ".join(self._group_bys))
        if self._havings:
            if not self._group_bys: _raise_error(QuerySyntaxError, "HAVING clause used without GROUP BY.")
            sql_parts.append("HAVING " + " AND ".join([h.render(pm) for h in self._havings]))
        if self._unions:
            for u_builder, is_all in self._unions:
                op = "UNION ALL" if is_all else "UNION"
                sql_parts.append(f"{op} {u_builder._build_sql(pm)}")
        if self._order_bys: sql_parts.append("ORDER BY " + ", ".join([o.render() for o in self._order_bys]))
        if self._limit is not None: sql_parts.append(f"LIMIT {self._limit}")
        if self._offset is not None: sql_parts.append(f"OFFSET {self._offset}")
        if self._for_update:
            if self._dialect == Dialect.SQLITE: _raise_error(UnsupportedDialectError, "FOR UPDATE not supported in SQLite.")
            sql_parts.append("FOR UPDATE")

        return " ".join(sql_parts)

    @trace_span(operation="qb.build", component="database", kind=SpanKind.INTERNAL)
    def build(self) -> QueryContext:
        _logger.debug(f"Building QueryContext for table {self._table}")
        metrics_engine.increment("query_build_total", namespace="database")
        
        pm = ParameterManager(self._dialect)
        sql = self._build_sql(pm)
        
        complexity = len(self._joins)*2 + len(self._wheres) + (3 if self._windows else 0) + (5 if self._ctes else 0)
        metadata = {
            "has_joins": len(self._joins) > 0, "has_windows": len(self._windows) > 0,
            "has_groups": len(self._group_bys) > 0, "is_cte": len(self._ctes) > 0,
            "query_cacheable": complexity < 10 and not self._for_update
        }
        
        ctx = QueryContext(
            sql=sql, parameters=pm.export(), statement_type=StatementType.SELECT,
            estimated_complexity=complexity, execution_metadata=metadata
        )
        
        AuditEngine.record_success("query_builder", AuditAction.SYSTEM, f"Generated safe SELECT query. Hash: {ctx.query_hash}")
        return ctx


class AggregateBuilder(BaseFluentBuilder):
    """Immutable Aggregate Query Builder (COUNT, SUM, AVG, MIN, MAX)."""
    __slots__ = ('_func', '_field', '_wheres')

    def __init__(self, table: str, dialect: Dialect, func: AggregateFunction, field: str = "*"):
        super().__init__(table, dialect)
        self._func = func
        self._field = SecurityValidator.sanitize(field)
        self._wheres: List[ConditionNode] = []

    @trace_span(operation="qb.aggregate_where", component="database", kind=SpanKind.INTERNAL)
    def where(self, field: str, operator: QueryOperator, value: Any) -> 'AggregateBuilder':
        b = cast(AggregateBuilder, self._clone())
        b._wheres.append(ConditionNode(field=field, operator=operator, value=value))
        return b

    @trace_span(operation="qb.build_aggregate", component="database", kind=SpanKind.INTERNAL)
    def build(self) -> QueryContext:
        metrics_engine.increment("query_build_total", namespace="database")
        pm = ParameterManager(self._dialect)
        
        sql = f"SELECT {self._func.value}({self._field}) AS aggr_val FROM {self._table}"
        if self._wheres:
            sql += " WHERE " + " AND ".join([w.render(pm) for w in self._wheres])
            
        return QueryContext(
            sql=sql, parameters=pm.export(), statement_type=StatementType.SELECT,
            estimated_complexity=1, execution_metadata={"is_aggregate": True}
        )


class InsertBuilder(BaseFluentBuilder):
    __slots__ = ('_values', '_returning')

    def __init__(self, table: str, dialect: Dialect):
        super().__init__(table, dialect)
        self._values: Dict[str, Any] = OrderedDict()
        self._returning: List[str] = []

    @trace_span(operation="qb.insert_value", component="database", kind=SpanKind.INTERNAL)
    def value(self, field: str, val: Any) -> 'InsertBuilder':
        b = cast(InsertBuilder, self._clone())
        b._values[SecurityValidator.sanitize(field)] = val
        return b

    @trace_span(operation="qb.insert_returning", component="database", kind=SpanKind.INTERNAL)
    def returning(self, field: str) -> 'InsertBuilder':
        if self._dialect == Dialect.SQLITE: _raise_error(UnsupportedDialectError, "RETURNING requires Postgres.")
        b = cast(InsertBuilder, self._clone())
        b._returning.append(SecurityValidator.sanitize(field))
        return b

    @trace_span(operation="qb.build_insert", component="database", kind=SpanKind.INTERNAL)
    def build(self) -> QueryContext:
        metrics_engine.increment("query_build_total", namespace="database")
        if not self._values: _raise_error(QuerySyntaxError, "Insert requires values.")
            
        pm = ParameterManager(self._dialect)
        cols = list(self._values.keys())
        placeholders = [pm.add(self._values[c]) for c in cols]
        
        sql = f"INSERT INTO {self._table} ({', '.join(cols)}) VALUES ({', '.join(placeholders)})"
        if self._returning: sql += f" RETURNING {', '.join(self._returning)}"
            
        return QueryContext(sql=sql, parameters=pm.export(), statement_type=StatementType.INSERT, estimated_complexity=1, execution_metadata={})


class UpdateBuilder(BaseFluentBuilder):
    __slots__ = ('_values', '_wheres')

    def __init__(self, table: str, dialect: Dialect):
        super().__init__(table, dialect)
        self._values: Dict[str, Any] = OrderedDict()
        self._wheres: List[ConditionNode] = []

    @trace_span(operation="qb.update_set", component="database", kind=SpanKind.INTERNAL)
    def set(self, field: str, val: Any) -> 'UpdateBuilder':
        b = cast(UpdateBuilder, self._clone())
        b._values[SecurityValidator.sanitize(field)] = val
        return b

    @trace_span(operation="qb.update_where", component="database", kind=SpanKind.INTERNAL)
    def where(self, field: str, operator: QueryOperator, value: Any) -> 'UpdateBuilder':
        b = cast(UpdateBuilder, self._clone())
        b._wheres.append(ConditionNode(field=field, operator=operator, value=value))
        return b

    @trace_span(operation="qb.build_update", component="database", kind=SpanKind.INTERNAL)
    def build(self) -> QueryContext:
        metrics_engine.increment("query_build_total", namespace="database")
        if not self._values: _raise_error(QuerySyntaxError, "Update requires set values.")
        if not self._wheres: _raise_error(SecurityError, "Unrestricted UPDATE prevented.")
            
        pm = ParameterManager(self._dialect)
        set_clauses = [f"{c} = {pm.add(v)}" for c, v in self._values.items()]
        where_clauses = [w.render(pm) for w in self._wheres]
        
        sql = f"UPDATE {self._table} SET {', '.join(set_clauses)} WHERE {' AND '.join(where_clauses)}"
        return QueryContext(sql=sql, parameters=pm.export(), statement_type=StatementType.UPDATE, estimated_complexity=2, execution_metadata={})


class DeleteBuilder(BaseFluentBuilder):
    __slots__ = ('_wheres',)

    def __init__(self, table: str, dialect: Dialect):
        super().__init__(table, dialect)
        self._wheres: List[ConditionNode] = []

    @trace_span(operation="qb.delete_where", component="database", kind=SpanKind.INTERNAL)
    def where(self, field: str, operator: QueryOperator, value: Any) -> 'DeleteBuilder':
        b = cast(DeleteBuilder, self._clone())
        b._wheres.append(ConditionNode(field=field, operator=operator, value=value))
        return b

    @trace_span(operation="qb.build_delete", component="database", kind=SpanKind.INTERNAL)
    def build(self) -> QueryContext:
        metrics_engine.increment("query_build_total", namespace="database")
        if not self._wheres: _raise_error(SecurityError, "Unrestricted DELETE prevented.")
            
        pm = ParameterManager(self._dialect)
        where_clauses = [w.render(pm) for w in self._wheres]
        
        sql = f"DELETE FROM {self._table} WHERE {' AND '.join(where_clauses)}"
        return QueryContext(sql=sql, parameters=pm.export(), statement_type=StatementType.DELETE, estimated_complexity=2, execution_metadata={})


class UpsertBuilder(BaseFluentBuilder):
    __slots__ = ('_values', '_conflict_keys')

    def __init__(self, table: str, dialect: Dialect):
        super().__init__(table, dialect)
        self._values: Dict[str, Any] = OrderedDict()
        self._conflict_keys: List[str] = []

    @trace_span(operation="qb.upsert_value", component="database", kind=SpanKind.INTERNAL)
    def value(self, field: str, val: Any) -> 'UpsertBuilder':
        b = cast(UpsertBuilder, self._clone())
        b._values[SecurityValidator.sanitize(field)] = val
        return b

    @trace_span(operation="qb.upsert_on_conflict", component="database", kind=SpanKind.INTERNAL)
    def on_conflict(self, *keys: str) -> 'UpsertBuilder':
        b = cast(UpsertBuilder, self._clone())
        b._conflict_keys = [SecurityValidator.sanitize(k) for k in keys]
        return b

    @trace_span(operation="qb.build_upsert", component="database", kind=SpanKind.INTERNAL)
    def build(self) -> QueryContext:
        metrics_engine.increment("query_build_total", namespace="database")
        if not self._values: _raise_error(QuerySyntaxError, "Upsert requires values.")
        if not self._conflict_keys: _raise_error(QuerySyntaxError, "Upsert requires ON CONFLICT keys.")

        pm = ParameterManager(self._dialect)
        cols = list(self._values.keys())
        placeholders = [pm.add(self._values[c]) for c in cols]
        
        base_insert = f"INSERT INTO {self._table} ({', '.join(cols)}) VALUES ({', '.join(placeholders)})"
        conflict_target = ", ".join(self._conflict_keys)
        
        update_cols = [c for c in cols if c not in self._conflict_keys]
        if not update_cols:
            sql = f"{base_insert} ON CONFLICT ({conflict_target}) DO NOTHING"
        else:
            prefix = "excluded" if self._dialect == Dialect.SQLITE else "EXCLUDED"
            updates = ", ".join([f"{c} = {prefix}.{c}" for c in update_cols])
            sql = f"{base_insert} ON CONFLICT ({conflict_target}) DO UPDATE SET {updates}"

        return QueryContext(sql=sql, parameters=pm.export(), statement_type=StatementType.UPSERT, estimated_complexity=3, execution_metadata={})


# =========================================================================
# FACADE ENTRY POINT
# =========================================================================

class QueryBuilder:
    """Enterprise Central Facade producing specialized Builder components."""
    
    @staticmethod
    def select(table: str, dialect: Dialect = Dialect.POSTGRES) -> SelectBuilder:
        return SelectBuilder(table, dialect)

    @staticmethod
    def aggregate(table: str, func: AggregateFunction, field: str = "*", dialect: Dialect = Dialect.POSTGRES) -> AggregateBuilder:
        return AggregateBuilder(table, dialect, func, field)
        
    @staticmethod
    def insert(table: str, dialect: Dialect = Dialect.POSTGRES) -> InsertBuilder:
        return InsertBuilder(table, dialect)
        
    @staticmethod
    def update(table: str, dialect: Dialect = Dialect.POSTGRES) -> UpdateBuilder:
        return UpdateBuilder(table, dialect)
        
    @staticmethod
    def delete(table: str, dialect: Dialect = Dialect.POSTGRES) -> DeleteBuilder:
        return DeleteBuilder(table, dialect)
        
    @staticmethod
    def upsert(table: str, dialect: Dialect = Dialect.POSTGRES) -> UpsertBuilder:
        return UpsertBuilder(table, dialect)


# =========================================================================
# EXPORTS
# =========================================================================

__all__ = [
    "QueryBuilderError",
    "ValidationError",
    "SecurityError",
    "QuerySyntaxError",
    "UnsupportedDialectError",
    "WindowFunctionError",
    
    "Dialect",
    "QueryOperator",
    "JoinType",
    "SortDirection",
    "AggregateFunction",
    "WindowFunction",
    "StatementType",
    
    "QueryContext",
    "QueryBuilder",
    
    "SelectBuilder",
    "AggregateBuilder",
    "InsertBuilder",
    "UpdateBuilder",
    "DeleteBuilder",
    "UpsertBuilder"
]
