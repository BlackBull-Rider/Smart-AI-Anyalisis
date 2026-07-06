"""
GREEN BULL RIDER V6 - Institutional-grade AI Stock Analysis Platform
Module: backend/core/audit.py
Description: Centralized Immutable Audit Engine.
             Provides enterprise-grade, cryptographically verifiable audit trails 
             for regulatory compliance and security monitoring. Strictly decoupled 
             from business logic and database persistence layers.
"""

import time
import json
import uuid
import hashlib
import contextvars
from enum import Enum
from datetime import datetime, timezone
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast, Tuple, Generator

# Internal Platform Integrations
from backend.config.settings import settings
from backend.core.logger import AppLogger
from backend.core.trace import TraceEngine
from backend.core.exceptions import GreenBullError

# -------------------------------------------------------------------------
# TYPE HINTS & CONSTANTS
# -------------------------------------------------------------------------
F = TypeVar('F', bound=Callable[..., Any])

# Instantiate a dedicated logger for audit emissions
_logger = AppLogger("AuditEngine")

# -------------------------------------------------------------------------
# ENUMS
# -------------------------------------------------------------------------

class AuditSeverity(str, Enum):
    """Severity classification for audit events."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AuditAction(str, Enum):
    """Categorized domain-agnostic operational actions for compliance tracking."""
    READ = "READ"
    CREATE = "CREATE"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    EXECUTE = "EXECUTE"
    LOGIN = "LOGIN"
    LOGOUT = "LOGOUT"
    SYSTEM = "SYSTEM"
    DENY = "DENY"


class AuditResult(str, Enum):
    """Terminal state of an audited operation."""
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    DENIED = "DENIED"
    PENDING = "PENDING"


# -------------------------------------------------------------------------
# IMMUTABLE DATACLASSES
# -------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AuditStatistics:
    """Immutable summary of an audit session's telemetry."""
    actor: str
    session_id: Optional[str]
    total_records: int
    success_count: int
    failure_count: int
    start_time_utc: str
    end_time_utc: str
    duration_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor,
            "session_id": self.session_id,
            "total_records": self.total_records,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "start_time_utc": self.start_time_utc,
            "end_time_utc": self.end_time_utc,
            "duration_ms": round(self.duration_ms, 3)
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """
    Immutable representation of a discrete audited event.
    Features cryptographic checksum verification to guarantee record integrity.
    """
    audit_id: str
    timestamp_utc: str
    trace_id: Optional[str]
    execution_id: Optional[str]
    session_id: Optional[str]
    request_id: Optional[str]
    correlation_id: Optional[str]
    actor: str
    component: str
    module: str
    operation: str
    action: AuditAction
    result: AuditResult
    severity: AuditSeverity
    message: str
    metadata: Dict[str, Any]
    tags: Dict[str, str]
    
    # Generated deterministically post-initialization
    checksum: str = field(default="", init=False)

    def __post_init__(self) -> None:
        """Computes a deterministic SHA256 checksum binding all immutable fields."""
        payload_dict = {
            "audit_id": self.audit_id,
            "timestamp_utc": self.timestamp_utc,
            "trace_id": self.trace_id,
            "execution_id": self.execution_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "actor": self.actor,
            "component": self.component,
            "module": self.module,
            "operation": self.operation,
            "action": self.action.value,
            "result": self.result.value,
            "severity": self.severity.value,
            "message": self.message,
            "metadata": self.metadata,
            "tags": self.tags
        }
        # Deterministic serialization for hashing
        serialized_payload = json.dumps(payload_dict, sort_keys=True, default=str)
        record_hash = hashlib.sha256(serialized_payload.encode('utf-8')).hexdigest()
        
        # Bypass frozen constraint safely during initialization exclusively
        object.__setattr__(self, 'checksum', record_hash)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the complete immutable record into a JSON-compatible structural dictionary."""
        return {
            "audit_id": self.audit_id,
            "timestamp_utc": self.timestamp_utc,
            "trace_id": self.trace_id,
            "execution_id": self.execution_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "actor": self.actor,
            "component": self.component,
            "module": self.module,
            "operation": self.operation,
            "action": self.action.value,
            "result": self.result.value,
            "severity": self.severity.value,
            "message": self.message,
            "metadata": self.metadata,
            "tags": self.tags,
            "checksum": self.checksum
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass(slots=True)
class AuditContext:
    """
    Mutable context container for the current localized audit session.
    Carries default attribution metadata inherited by all spawned records.
    """
    actor: str
    component: str
    module: str
    start_time_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    start_time_perf: float = field(default_factory=time.perf_counter)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    
    _context_token: Optional[contextvars.Token] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor,
            "component": self.component,
            "module": self.module,
            "metadata": self.metadata,
            "tags": self.tags
        }


# -------------------------------------------------------------------------
# CONTEXT ISOLATION (THREAD-SAFE & ASYNC-SAFE)
# -------------------------------------------------------------------------

_CURRENT_AUDIT_CTX: contextvars.ContextVar[Optional[AuditContext]] = contextvars.ContextVar("current_audit_ctx", default=None)

# Isolated session metrics
_AUDIT_TOTAL_RECORDS: contextvars.ContextVar[int] = contextvars.ContextVar("audit_total_records", default=0)
_AUDIT_SUCCESS_COUNT: contextvars.ContextVar[int] = contextvars.ContextVar("audit_success_count", default=0)
_AUDIT_FAILURE_COUNT: contextvars.ContextVar[int] = contextvars.ContextVar("audit_failure_count", default=0)

# Global Sink Registry for decoupled operational persistency (SQLite, Kafka, Elastic, etc.)
_AUDIT_SINKS: List[Callable[[AuditRecord], None]] = []


# -------------------------------------------------------------------------
# AUDIT ENGINE
# -------------------------------------------------------------------------

class AuditEngine:
    """
    Central orchestration interface for distributed audit tracing.
    Ensures deterministic, cryptographically hashed compliance logs seamlessly linked
    with the Platform Logger and Trace Engine. Operations are strictly non-blocking.
    """

    @classmethod
    def register_audit_sink(cls, callback: Callable[[AuditRecord], None]) -> None:
        """Subscribes an external, decoupled engine to the verified audit stream."""
        if callback not in _AUDIT_SINKS:
            _AUDIT_SINKS.append(callback)

    @classmethod
    def unregister_audit_sink(cls, callback: Callable[[AuditRecord], None]) -> None:
        """Detaches an external engine from the verified audit stream."""
        if callback in _AUDIT_SINKS:
            _AUDIT_SINKS.remove(callback)

    @staticmethod
    def _generate_id() -> str:
        """Generates a globally unique identifier for audit records."""
        return uuid.uuid4().hex

    @classmethod
    def start_audit(cls, actor: str, component: str, module: str = "core") -> AuditContext:
        """
        Initializes a thread-safe execution boundary for audit attribution.
        """
        ctx = AuditContext(
            actor=actor,
            component=component,
            module=module
        )
        
        token = _CURRENT_AUDIT_CTX.set(ctx)
        ctx._context_token = token
        
        _AUDIT_TOTAL_RECORDS.set(0)
        _AUDIT_SUCCESS_COUNT.set(0)
        _AUDIT_FAILURE_COUNT.set(0)
        
        return ctx

    @classmethod
    def end_audit(cls) -> Optional[AuditStatistics]:
        """
        Closes the active attribution context and emits finalized session statistics.
        """
        ctx = cls.current_context()
        if not ctx:
            return None
            
        trace_ctx = TraceEngine.current_trace()
        session_id = trace_ctx.session_id if trace_ctx else None
        
        duration_ms = (time.perf_counter() - ctx.start_time_perf) * 1000.0
        
        stats = AuditStatistics(
            actor=ctx.actor,
            session_id=session_id,
            total_records=_AUDIT_TOTAL_RECORDS.get(),
            success_count=_AUDIT_SUCCESS_COUNT.get(),
            failure_count=_AUDIT_FAILURE_COUNT.get(),
            start_time_utc=ctx.start_time_utc,
            end_time_utc=datetime.now(timezone.utc).isoformat(),
            duration_ms=duration_ms
        )
        
        if ctx._context_token:
            _CURRENT_AUDIT_CTX.reset(ctx._context_token)
            
        cls.clear_context()
        return stats

    @staticmethod
    def current_context() -> Optional[AuditContext]:
        """Retrieves the active thread-local AuditContext securely."""
        return _CURRENT_AUDIT_CTX.get()

    @staticmethod
    def clear_context() -> None:
        """Purges contextual bindings from the current operational scope."""
        _CURRENT_AUDIT_CTX.set(None)
        _AUDIT_TOTAL_RECORDS.set(0)
        _AUDIT_SUCCESS_COUNT.set(0)
        _AUDIT_FAILURE_COUNT.set(0)

    @classmethod
    def attach_metadata(cls, key: str, value: Any) -> None:
        """Attaches localized structural metadata spanning all subsequent session records."""
        ctx = cls.current_context()
        if ctx:
            ctx.metadata[key] = value

    @classmethod
    def attach_tags(cls, tags: Dict[str, str]) -> None:
        """Attaches localized indexing tags spanning all subsequent session records."""
        ctx = cls.current_context()
        if ctx:
            ctx.tags.update(tags)

    @classmethod
    def record(
        cls,
        operation: str,
        action: AuditAction,
        result: AuditResult,
        message: str,
        severity: AuditSeverity = AuditSeverity.INFO,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        override_actor: Optional[str] = None
    ) -> AuditRecord:
        """
        Core generation and verified emission pathway for immutable audit records.
        Automatically correlates with the TraceEngine observability framework.
        """
        ctx = cls.current_context()
        
        # Fallback values for detached record emissions
        actor = override_actor or (ctx.actor if ctx else "SYSTEM_UNKNOWN")
        component = ctx.component if ctx else "SYSTEM_CORE"
        module = ctx.module if ctx else "UNBOUND"
        
        combined_meta = dict(ctx.metadata) if ctx else {}
        if metadata:
            combined_meta.update(metadata)
            
        combined_tags = dict(ctx.tags) if ctx else {}
        if tags:
            combined_tags.update(tags)

        # Telemetry structural correlation
        trace_ctx = TraceEngine.current_trace()
        trace_id = trace_ctx.trace_id if trace_ctx else None
        execution_id = trace_ctx.execution_id if trace_ctx else None
        session_id = trace_ctx.session_id if trace_ctx else None
        request_id = trace_ctx.request_id if trace_ctx else None
        correlation_id = trace_ctx.correlation_id if trace_ctx else None

        record = AuditRecord(
            audit_id=cls._generate_id(),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            trace_id=trace_id,
            execution_id=execution_id,
            session_id=session_id,
            request_id=request_id,
            correlation_id=correlation_id,
            actor=actor,
            component=component,
            module=module,
            operation=operation,
            action=action,
            result=result,
            severity=severity,
            message=message,
            metadata=combined_meta,
            tags=combined_tags
        )

        # Operational Statistics
        _AUDIT_TOTAL_RECORDS.set(_AUDIT_TOTAL_RECORDS.get() + 1)
        if result == AuditResult.SUCCESS:
            _AUDIT_SUCCESS_COUNT.set(_AUDIT_SUCCESS_COUNT.get() + 1)
        elif result in (AuditResult.FAILURE, AuditResult.DENIED):
            _AUDIT_FAILURE_COUNT.set(_AUDIT_FAILURE_COUNT.get() + 1)

        # Emission to Platform Logger
        _logger.audit(
            action=record.action.value,
            actor=record.actor,
            status=record.result.value,
            details=record.to_dict()
        )

        # Future-Ready Sinks Execution
        for sink in _AUDIT_SINKS:
            try:
                sink(record)
            except Exception:
                # Sinks must not interrupt core platform telemetry routines
                pass

        return record

    @classmethod
    def record_success(
        cls,
        operation: str,
        action: AuditAction,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditRecord:
        """Syntactic wrapper for successful audit outcome emissions."""
        return cls.record(
            operation=operation,
            action=action,
            result=AuditResult.SUCCESS,
            message=message,
            severity=AuditSeverity.INFO,
            metadata=metadata
        )

    @classmethod
    def record_failure(
        cls,
        operation: str,
        action: AuditAction,
        message: str,
        severity: AuditSeverity = AuditSeverity.WARNING,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditRecord:
        """Syntactic wrapper for failed or denied audit outcome emissions."""
        return cls.record(
            operation=operation,
            action=action,
            result=AuditResult.FAILURE,
            message=message,
            severity=severity,
            metadata=metadata
        )

    @classmethod
    @contextmanager
    def audit_context(
        cls,
        actor: str,
        component: str,
        module: str = "core"
    ) -> Generator[AuditContext, None, None]:
        """
        Context manager spanning an entire unified audit boundary session.
        Guarantees closure and cleanup upon exit.
        """
        ctx = cls.start_audit(actor=actor, component=component, module=module)
        try:
            yield ctx
        finally:
            cls.end_audit()


# -------------------------------------------------------------------------
# DECORATORS
# -------------------------------------------------------------------------

def audit_operation(
    action: AuditAction,
    component: str,
    module: str = "core",
    override_actor: Optional[str] = None
) -> Callable[[F], F]:
    """
    Decorator for tracking the entirety of an actionable function footprint 
    within the immutable Audit Engine. Correlates execution timings natively.
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            operation = func.__qualname__
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                
                AuditEngine.record_success(
                    operation=operation,
                    action=action,
                    message=f"Operation {operation} executed successfully.",
                    metadata={"duration_ms": round(duration_ms, 3)}
                )
                return result
                
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                error_msg = str(e)
                
                # Check custom properties for GreenBull errors
                error_code = getattr(e, "error_code", "UNKNOWN_ERROR")
                
                AuditEngine.record_failure(
                    operation=operation,
                    action=action,
                    message=f"Operation {operation} failed: {error_code} - {error_msg}",
                    severity=AuditSeverity.CRITICAL if isinstance(e, GreenBullError) else AuditSeverity.WARNING,
                    metadata={
                        "error_type": e.__class__.__name__,
                        "error_code": error_code,
                        "duration_ms": round(duration_ms, 3)
                    }
                )
                raise

        return cast(F, wrapper)
    return decorator


# -------------------------------------------------------------------------
# MODULE EXPORTS
# -------------------------------------------------------------------------

__all__ = [
    "AuditSeverity",
    "AuditAction",
    "AuditResult",
    "AuditStatistics",
    "AuditRecord",
    "AuditContext",
    "AuditEngine",
    "audit_operation"
]
