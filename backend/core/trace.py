"""
GREEN BULL RIDER V6 - Institutional-grade AI Stock Analysis Platform
Module: backend/core/trace.py
Description: Centralized Distributed Trace Engine.
             Provides OpenTelemetry-compatible execution tracking, nested span 
             management (via immutable stack), and deterministic performance telemetry.
             Fully automated, thread-safe, async-safe, and designed for 
             event-driven pipelines with Zero Global Mutable State.
"""

import time
import uuid
import json
import traceback
import contextvars
import random
from enum import Enum
from datetime import datetime, timezone
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import wraps
from typing import Any, Dict, List, Optional, Generator, TypeVar, Callable, cast, Tuple

# Internal Platform Integrations
from backend.config.settings import settings
from backend.core.logger import AppLogger, PlatformContext
from backend.core.exceptions import GreenBullError

# -------------------------------------------------------------------------
# TYPE HINTS & CONSTANTS
# -------------------------------------------------------------------------
F = TypeVar('F', bound=Callable[..., Any])
MAX_EVENTS_PER_SPAN = 500

# Instantiate a dedicated logger for trace emissions
_logger = AppLogger("TraceEngine")

# -------------------------------------------------------------------------
# ENUMS
# -------------------------------------------------------------------------

class TraceStatus(str, Enum):
    """Represents the terminal state of a trace or span."""
    OK = "OK"
    ERROR = "ERROR"
    IN_PROGRESS = "IN_PROGRESS"
    CANCELLED = "CANCELLED"


class SpanKind(str, Enum):
    """OpenTelemetry compatible Span Kinds."""
    INTERNAL = "INTERNAL"
    SERVER = "SERVER"
    CLIENT = "CLIENT"
    PRODUCER = "PRODUCER"
    CONSUMER = "CONSUMER"


class TraceLevel(str, Enum):
    """Severity levels for discrete trace timeline events."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# -------------------------------------------------------------------------
# IMMUTABLE DATACLASSES
# -------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TraceEvent:
    """Immutable representation of a point-in-time event within a span."""
    name: str
    severity: TraceLevel
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass(frozen=True, slots=True)
class TraceStatistics:
    """Immutable summary of a completed distributed trace."""
    trace_id: str
    execution_id: str
    span_count: int
    event_count: int
    error_count: int
    warning_count: int
    total_duration_ms: float
    is_sampled: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "execution_id": self.execution_id,
            "span_count": self.span_count,
            "event_count": self.event_count,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "total_duration_ms": round(self.total_duration_ms, 3),
            "is_sampled": self.is_sampled
        }
        
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass(frozen=True, slots=True)
class TraceContext:
    """
    Immutable root context for a distributed execution trace.
    Carries OpenTelemetry compatible global identifiers propagated across all child spans.
    """
    trace_id: str
    execution_id: str
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    # OpenTelemetry compatibility and Trace Sampling fields
    trace_flags: int = 1
    trace_state: str = ""
    resource: Dict[str, Any] = field(default_factory=dict)
    sample_rate: float = 1.0
    is_sampled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "execution_id": self.execution_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "trace_flags": self.trace_flags,
            "trace_state": self.trace_state,
            "is_sampled": self.is_sampled
        }
        
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass(slots=True)
class TraceSpan:
    """
    State container for an active execution span.
    Maintains O(1) duration arithmetic using hardware performance counters.
    Designed with slots to minimize memory footprint.
    Duration and end state are frozen upon completion.
    """
    span_id: str
    trace_id: str
    operation: str
    module: str
    component: str
    kind: SpanKind
    parent_span_id: Optional[str] = None
    
    start_time_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    start_time_perf: float = field(default_factory=time.perf_counter)
    
    end_time_utc: Optional[str] = None
    end_time_perf: Optional[float] = None
    duration_ms: float = 0.0
    
    status: TraceStatus = TraceStatus.IN_PROGRESS
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    events: List[TraceEvent] = field(default_factory=list)
    exception: Optional[Dict[str, Any]] = None
    
    _is_finished: bool = field(default=False, repr=False, compare=False)

    def finalize(self, status: TraceStatus = TraceStatus.OK) -> None:
        """Freezes the span duration and applies terminal states."""
        if self._is_finished:
            return
            
        self.end_time_perf = time.perf_counter()
        self.end_time_utc = datetime.now(timezone.utc).isoformat()
        self.duration_ms = (self.end_time_perf - self.start_time_perf) * 1000.0
        
        # Do not override ERROR status if it was set via exception capture
        if self.status != TraceStatus.ERROR:
            self.status = status
            
        self._is_finished = True

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the span into a structural JSON-ready dictionary."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "operation": self.operation,
            "module": self.module,
            "component": self.component,
            "kind": self.kind.value,
            "start_time_utc": self.start_time_utc,
            "end_time_utc": self.end_time_utc,
            "duration_ms": round(self.duration_ms, 3),
            "status": self.status.value,
            "metadata": self.metadata,
            "tags": self.tags,
            "event_count": len(self.events),
            "events": [evt.to_dict() for evt in self.events],
            "exception": self.exception
        }
        
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# -------------------------------------------------------------------------
# CONTEXT ISOLATION (THREAD-SAFE & ASYNC-SAFE STACK)
# -------------------------------------------------------------------------

_CURRENT_TRACE: contextvars.ContextVar[Optional[TraceContext]] = contextvars.ContextVar("current_trace", default=None)

# Immutable Tuple Stack ensures perfect async isolation and hierarchy tracking
_SPAN_STACK: contextvars.ContextVar[Tuple[TraceSpan, ...]] = contextvars.ContextVar("span_stack", default=())

# Isolated trace statistics counters
_TRACE_SPAN_COUNT: contextvars.ContextVar[int] = contextvars.ContextVar("trace_span_count", default=0)
_TRACE_EVENT_COUNT: contextvars.ContextVar[int] = contextvars.ContextVar("trace_event_count", default=0)
_TRACE_ERROR_COUNT: contextvars.ContextVar[int] = contextvars.ContextVar("trace_error_count", default=0)
_TRACE_WARNING_COUNT: contextvars.ContextVar[int] = contextvars.ContextVar("trace_warning_count", default=0)
_TRACE_START_PERF: contextvars.ContextVar[float] = contextvars.ContextVar("trace_start_perf", default=0.0)


# -------------------------------------------------------------------------
# TRACE ENGINE
# -------------------------------------------------------------------------

class TraceEngine:
    """
    Central orchestration engine for distributed execution tracing.
    Provides automated context management, telemetry correlation, and OpenTelemetry compatibility.
    All operations are strictly thread-safe and async-safe via ContextVars.
    """

    @staticmethod
    def _generate_id() -> str:
        """Generates a globally unique, high-entropy 32-character hexadecimal identifier."""
        return uuid.uuid4().hex

    @classmethod
    def start_trace(
        cls,
        execution_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        sample_rate: float = 1.0,
        resource: Optional[Dict[str, Any]] = None
    ) -> TraceContext:
        """Initializes a new distributed execution trace."""
        trace_id = cls._generate_id()
        active_exec = execution_id or cls._generate_id()
        active_corr = correlation_id or cls._generate_id()
        is_sampled = random.random() <= sample_rate

        ctx = TraceContext(
            trace_id=trace_id,
            execution_id=active_exec,
            session_id=session_id,
            request_id=request_id,
            correlation_id=active_corr,
            sample_rate=sample_rate,
            is_sampled=is_sampled,
            resource=resource or {}
        )
        
        _CURRENT_TRACE.set(ctx)
        _SPAN_STACK.set(())
        
        _TRACE_SPAN_COUNT.set(0)
        _TRACE_EVENT_COUNT.set(0)
        _TRACE_ERROR_COUNT.set(0)
        _TRACE_WARNING_COUNT.set(0)
        _TRACE_START_PERF.set(time.perf_counter())

        PlatformContext.set_context(
            trace_id=trace_id,
            correlation_id=active_corr,
            execution_id=active_exec,
            request_id=request_id,
            session_id=session_id
        )

        if is_sampled:
            _logger.info(f"Trace Initialized: [{trace_id}]", metadata=ctx.to_dict())
            
        return ctx

    @classmethod
    def end_trace(cls) -> Optional[TraceStatistics]:
        """Finalizes the active trace, validates unclosed spans, and releases context bindings."""
        context = cls.current_trace()
        if not context:
            return None
            
        # Trace End Validation: Check for unclosed orphaned spans
        open_spans = _SPAN_STACK.get()
        if open_spans:
            _logger.warning(
                f"Trace [{context.trace_id}] ended with {len(open_spans)} unclosed spans. "
                f"Check missing span closure for: {[s.operation for s in open_spans]}"
            )

        duration_ms = (time.perf_counter() - _TRACE_START_PERF.get()) * 1000.0
        
        stats = TraceStatistics(
            trace_id=context.trace_id,
            execution_id=context.execution_id,
            span_count=_TRACE_SPAN_COUNT.get(),
            event_count=_TRACE_EVENT_COUNT.get(),
            error_count=_TRACE_ERROR_COUNT.get(),
            warning_count=_TRACE_WARNING_COUNT.get(),
            total_duration_ms=duration_ms,
            is_sampled=context.is_sampled
        )

        if context.is_sampled:
            _logger.info(f"Trace Finalized: [{context.trace_id}] ({stats.total_duration_ms:.2f}ms)", metadata=stats.to_dict())
            
        cls.clear_trace()
        return stats

    @staticmethod
    def current_trace() -> Optional[TraceContext]:
        """Retrieves the active TraceContext securely."""
        return _CURRENT_TRACE.get()

    @classmethod
    def current_span(cls) -> Optional[TraceSpan]:
        """Retrieves the active TraceSpan from the top of the immutable stack."""
        stack = _SPAN_STACK.get()
        return stack[-1] if stack else None

    @classmethod
    def start_span(
        cls,
        operation: str,
        module: str = "core",
        component: str = "engine",
        kind: SpanKind = SpanKind.INTERNAL
    ) -> TraceSpan:
        """
        Initializes a new execution span. Pushes it onto the ContextVar tuple stack 
        to guarantee perfect hierarchical Parent-Child alignment.
        """
        trace = cls.current_trace()
        if not trace:
            trace = cls.start_trace()
            
        parent_span = cls.current_span()
        parent_span_id = parent_span.span_id if parent_span else None
        
        span = TraceSpan(
            span_id=cls._generate_id(),
            trace_id=trace.trace_id,
            parent_span_id=parent_span_id,
            operation=operation,
            module=module,
            component=component,
            kind=kind
        )
        
        # Push to immutable stack
        current_stack = _SPAN_STACK.get()
        _SPAN_STACK.set(current_stack + (span,))
        
        _TRACE_SPAN_COUNT.set(_TRACE_SPAN_COUNT.get() + 1)
        
        if trace.is_sampled:
            _logger.debug(
                f"Span Started: [{operation}]", 
                metadata={"span_id": span.span_id, "trace_id": span.trace_id, "parent_span_id": parent_span_id}
            )
            
        return span

    @classmethod
    def end_span(cls, span: TraceSpan, status: TraceStatus = TraceStatus.OK) -> None:
        """
        Finalizes an execution span, freezes duration, and pops it securely from the stack.
        """
        span.finalize(status=status)

        if span.status == TraceStatus.ERROR:
            _TRACE_ERROR_COUNT.set(_TRACE_ERROR_COUNT.get() + 1)

        # Pop from immutable stack safely
        current_stack = _SPAN_STACK.get()
        if current_stack and current_stack[-1] == span:
            _SPAN_STACK.set(current_stack[:-1])
        else:
            # Fallback if spans were closed out of order, rebuild stack safely
            new_stack = tuple(s for s in current_stack if s != span)
            _SPAN_STACK.set(new_stack)

        trace = cls.current_trace()
        if trace and trace.is_sampled:
            _logger.info(
                f"Span Ended: [{span.operation}] ({span.duration_ms:.2f}ms)", 
                metadata=span.to_dict()
            )

    @classmethod
    def record_event(
        cls, 
        name: str, 
        severity: TraceLevel = TraceLevel.INFO, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Records a point-in-time event with Memory Protection (MAX_EVENTS limit)."""
        span = cls.current_span()
        if not span:
            return
            
        _TRACE_EVENT_COUNT.set(_TRACE_EVENT_COUNT.get() + 1)
        if severity == TraceLevel.WARNING:
            _TRACE_WARNING_COUNT.set(_TRACE_WARNING_COUNT.get() + 1)
            
        # Memory Protection: Prevent endless event inflation
        if len(span.events) >= MAX_EVENTS_PER_SPAN:
            if not any(e.name == "TRUNCATED_EVENTS" for e in span.events):
                span.events.append(TraceEvent("TRUNCATED_EVENTS", TraceLevel.WARNING, metadata={"dropped": True}))
        else:
            event = TraceEvent(name=name, severity=severity, metadata=metadata or {})
            span.events.append(event)
        
        trace = cls.current_trace()
        if trace and trace.is_sampled:
            log_meta = {"span_id": span.span_id, "trace_id": span.trace_id, "event": event.to_dict()}
            if severity in (TraceLevel.ERROR, TraceLevel.CRITICAL):
                _logger.error(f"Trace Event: {name}", metadata=log_meta)
            elif severity == TraceLevel.WARNING:
                _logger.warning(f"Trace Event: {name}", metadata=log_meta)
            else:
                _logger.debug(f"Trace Event: {name}", metadata=log_meta)

    @classmethod
    def record_exception(cls, exception: Exception) -> None:
        """
        Records an exception against the active span and flags it as ERROR.
        """
        span = cls.current_span()
        if not span:
            return
            
        span.status = TraceStatus.ERROR
        
        error_payload: Dict[str, Any] = {
            "type": exception.__class__.__name__,
            "message": str(exception),
            "stacktrace": "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        }
        
        if isinstance(exception, GreenBullError):
            error_payload.update({
                "error_code": exception.error_code,
                "retryable": exception.retryable,
                "recoverable": exception.recoverable,
                "platform_context": exception.context,
                "platform_details": exception.details
            })
            
        span.exception = error_payload
        
        trace = cls.current_trace()
        if trace and trace.is_sampled:
            _logger.error(
                f"Trace Exception: {error_payload['type']}", 
                metadata={"span_id": span.span_id, "trace_id": span.trace_id, "exception": error_payload}
            )

    @classmethod
    def attach_metadata(cls, key: str, value: Any) -> None:
        """Attaches arbitrary JSON-serializable metadata to the active span."""
        span = cls.current_span()
        if span:
            span.metadata[key] = value

    @classmethod
    def attach_tags(cls, tags: Dict[str, str]) -> None:
        """Attaches indexed key-value string tags to the active span."""
        span = cls.current_span()
        if span:
            span.tags.update(tags)

    @staticmethod
    def clear_trace() -> None:
        """Cleanses all trace telemetry from the current execution context entirely."""
        _CURRENT_TRACE.set(None)
        _SPAN_STACK.set(())
        _TRACE_SPAN_COUNT.set(0)
        _TRACE_EVENT_COUNT.set(0)
        _TRACE_ERROR_COUNT.set(0)
        _TRACE_WARNING_COUNT.set(0)
        _TRACE_START_PERF.set(0.0)
        PlatformContext.clear()

    @classmethod
    @contextmanager
    def trace_context(
        cls,
        execution_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        sample_rate: float = 1.0
    ) -> Generator[TraceContext, None, None]:
        """
        Context manager for safely bounding an entire distributed trace lifecycle.
        Guarantees finalization, unclosed span validation, and cleanup.
        """
        cls.start_trace(execution_id, session_id, request_id, correlation_id, sample_rate)
        try:
            yield cls.current_trace()  # type: ignore
        finally:
            cls.end_trace()

    @classmethod
    @contextmanager
    def nested_span(
        cls,
        operation: str,
        module: str = "core",
        component: str = "engine",
        kind: SpanKind = SpanKind.INTERNAL
    ) -> Generator[TraceSpan, None, None]:
        """
        Context manager for safely bounding an isolated execution span using Stack Hierarchy.
        Automatically records raised exceptions and finalizes performance timing.
        """
        span = cls.start_span(operation=operation, module=module, component=component, kind=kind)
        try:
            yield span
            cls.end_span(span, status=TraceStatus.OK)
        except Exception as e:
            cls.record_exception(e)
            cls.end_span(span, status=TraceStatus.ERROR)
            raise


# -------------------------------------------------------------------------
# DECORATORS
# -------------------------------------------------------------------------

def trace_span(
    operation: Optional[str] = None, 
    module: str = "core", 
    component: str = "engine", 
    kind: SpanKind = SpanKind.INTERNAL
) -> Callable[[F], F]:
    """
    Decorator to automatically wrap a function execution within an OTEL-compatible Trace Span.
    If operation name is omitted, the function's fully qualified name is utilized.
    
    Usage:
        @trace_span(operation="fetch_market_data", component="provider")
        def fetch_data(symbol):
            pass
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            op_name = operation or f"{func.__module__}.{func.__qualname__}"
            
            with TraceEngine.nested_span(operation=op_name, module=module, component=component, kind=kind):
                return func(*args, **kwargs)
                
        return cast(F, wrapper)
    return decorator


# -------------------------------------------------------------------------
# MODULE EXPORTS
# -------------------------------------------------------------------------

__all__ = [
    "TraceStatus",
    "SpanKind",
    "TraceLevel",
    "TraceEvent",
    "TraceStatistics",
    "TraceContext",
    "TraceSpan",
    "TraceEngine",
    "trace_span"
]
