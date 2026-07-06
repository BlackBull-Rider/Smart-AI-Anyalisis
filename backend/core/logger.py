"""
GREEN BULL RIDER V6 - Institutional-grade AI Stock Analysis Platform
Module: backend/core/logger.py
Description: Production-grade centralized logging engine. Serves as the single 
             logging entry point for the entire platform. Provides high-throughput,
             non-blocking, thread-safe structured JSON and human-readable logging
             integrated natively with modern context-tracking telemetry.
Production Locked.
"""

import logging
import logging.handlers
import json
import os
import sys
import time
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from queue import Queue
from typing import Any, Callable, Generator, Optional

# -------------------------------------------------------------------------
# OBSERVABILITY CONTEXT MANAGEMENT (THREAD-SAFE & ASYNC-READY)
# -------------------------------------------------------------------------

_TRACE_ID: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
_CORRELATION_ID: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)
_EXECUTION_ID: ContextVar[Optional[str]] = ContextVar("execution_id", default=None)
_REQUEST_ID: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_SESSION_ID: ContextVar[Optional[str]] = ContextVar("session_id", default=None)

class PlatformContext:
    """
    Unified manager for platform execution contexts.
    Leverages ContextVars to guarantee absolute thread isolation and async safety.
    """
    
    @staticmethod
    def set_context(
        trace_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        execution_id: Optional[str] = None,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> None:
        """Sets the tracking IDs for the current context."""
        if trace_id is not None:
            _TRACE_ID.set(trace_id)
        if correlation_id is not None:
            _CORRELATION_ID.set(correlation_id)
        if execution_id is not None:
            _EXECUTION_ID.set(execution_id)
        if request_id is not None:
            _REQUEST_ID.set(request_id)
        if session_id is not None:
            _SESSION_ID.set(session_id)

    @staticmethod
    def clear() -> None:
        """Clears all telemetry contexts for the current execution thread/task."""
        _TRACE_ID.set(None)
        _CORRELATION_ID.set(None)
        _EXECUTION_ID.set(None)
        _REQUEST_ID.set(None)
        _SESSION_ID.set(None)

    @staticmethod
    def get_context_dict() -> dict[str, Optional[str]]:
        """Retrieves a snapshots of all execution tracking IDs."""
        return {
            "trace_id": _TRACE_ID.get(),
            "correlation_id": _CORRELATION_ID.get(),
            "execution_id": _EXECUTION_ID.get(),
            "request_id": _REQUEST_ID.get(),
            "session_id": _SESSION_ID.get(),
        }


# -------------------------------------------------------------------------
# AUTOMATION EXTENSIBILITY HOOKS
# -------------------------------------------------------------------------

# Global registry for third-party automation components (Trace, Audit, Monitoring, etc.)
# Allows plug-and-play expansion without modifying this module.
AUTOMATION_LOG_HOOKS: list[Callable[[dict[str, Any]], None]] = []
AUDIT_LOG_HOOKS: list[Callable[[dict[str, Any]], None]] = []

_AUDIT_LOGGING_ENABLED: bool = True

def register_automation_hook(callback: Callable[[dict[str, Any]], None]) -> None:
    """Registers an external engine interceptor (e.g., Trace Engine, Monitoring)."""
    AUTOMATION_LOG_HOOKS.append(callback)

def register_audit_hook(callback: Callable[[dict[str, Any]], None]) -> None:
    """Registers an external audit record writer (e.g., Audit Engine)."""
    AUDIT_LOG_HOOKS.append(callback)

def set_audit_logging_enabled(enabled: bool) -> None:
    """Dynamically activates or deactivates audit event interceptors."""
    global _AUDIT_LOGGING_ENABLED
    _AUDIT_LOGGING_ENABLED = enabled


# -------------------------------------------------------------------------
# STRUCTURED FORMATTERS
# -------------------------------------------------------------------------

class StructuredJSONFormatter(logging.Formatter):
    """
    Deterministic JSON formatter optimized for high-performance ingestion engines,
    Event Buses, Time-Series telemetry layers, and distributed trace monitors.
    """
    def format(self, record: logging.LogRecord) -> str:
        ctx = PlatformContext.get_context_dict()
        
        # Base telemetry payload
        log_data: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger_name": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "file_path": record.pathname,
            "line_number": record.lineno,
            "function_name": record.funcName,
            "process_id": record.process,
            "thread_id": record.thread,
            "thread_name": record.threadName,
            **ctx
        }

        # Extract structured metrics if attached via extra
        if hasattr(record, "metrics"):
            log_data["metrics"] = getattr(record, "metrics")

        # Extract extra structured metadata context
        if hasattr(record, "metadata"):
            log_data["metadata"] = getattr(record, "metadata")

        # Handle exception serialization natively and safely
        if record.exc_info:
            exc_type, exc_val, exc_tb = record.exc_info
            error_payload: dict[str, Any] = {
                "type": exc_type.__name__ if exc_type else "UnknownException",
                "message": str(exc_val),
            }
            # Handle standard custom platform exception serialization safely via duck typing
            if exc_val and hasattr(exc_val, "to_dict") and callable(getattr(exc_val, "to_dict")):
                try:
                    error_payload["platform_details"] = exc_val.to_dict()
                except Exception:
                    pass
            
            if record.exc_text:
                error_payload["stack_trace"] = record.exc_text
            else:
                error_payload["stack_trace"] = self.formatException(record.exc_info)
                
            log_data["exception"] = error_payload

        # Trigger registered automation layer processing callbacks
        for hook in AUTOMATION_LOG_HOOKS:
            try:
                hook(log_data)
            except Exception:
                pass  # Core Logger isolation requirement: ignore side-effects of external engines

        # If flagged specifically as an audit payload, pipe into Audit Hooks
        if getattr(record, "is_audit_event", False) and _AUDIT_LOGGING_ENABLED:
            for audit_hook in AUDIT_LOG_HOOKS:
                try:
                    audit_hook(log_data)
                except Exception:
                    pass

        return json.dumps(log_data, default=str)


class HumanReadableConsoleFormatter(logging.Formatter):
    """
    High-visibility console formatter designed explicitly for localized human monitoring,
    local debugging sessions, and synchronous container system output streams.
    """
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        ctx = PlatformContext.get_context_dict()
        trace_str = f" | Trace: {ctx['trace_id']}" if ctx["trace_id"] else ""
        
        base_format = f"[{timestamp}] [{record.levelname:<8}] ({record.name}) {record.getMessage()}{trace_str}"
        
        if hasattr(record, "metrics") and record.metrics:
            base_format += f" | Metrics: {record.metrics}"
            
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            base_format += f"\n{record.exc_text}"
            
        return base_format


# -------------------------------------------------------------------------
# CORE PLATFORM LOGGER SYSTEM
# -------------------------------------------------------------------------

class PlatformLoggerManager:
    """
    Manages initialization, configuration, and non-blocking asynchronous structural 
    dispatch mechanics for the centralized Green Bull Rider V6 logging framework.
    """
    _lock = threading.Lock()
    _initialized = False
    _queue_listener: Optional[logging.handlers.QueueListener] = None

    @classmethod
    def initialize(
        cls, 
        log_level: int = logging.INFO, 
        log_directory: str = "logs", 
        log_filename: str = "green_bull_platform.log",
        backup_count: int = 30
    ) -> None:
        """
        Sets up non-blocking asynchronous decoupled queue listeners for both
        file (Rotating JSON) and stdout (Console) operational streams.
        """
        with cls._lock:
            if cls._initialized:
                return

            os.makedirs(log_directory, exist_ok=True)
            log_filepath = os.path.join(log_directory, log_filename)

            # Asynchronous Internal Log Queue Setup
            log_queue: Queue = Queue(-1)
            queue_handler = logging.handlers.QueueHandler(log_queue)

            # Route root logger metrics safely into non-blocking queue
            root_logger = logging.getLogger()
            root_logger.setLevel(log_level)
            root_logger.addHandler(queue_handler)

            # Stream Handlers Configuration
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(HumanReadableConsoleFormatter())

            file_handler = logging.handlers.TimedRotatingFileHandler(
                filename=log_filepath,
                when="D",
                interval=1,
                backupCount=backup_count,
                encoding="utf-8"
            )
            file_handler.setFormatter(StructuredJSONFormatter())

            # Start Background Log Processing Listener Thread
            cls._queue_listener = logging.handlers.QueueListener(
                log_queue, 
                console_handler, 
                file_handler, 
                respect_handler_level=True
            )
            cls._queue_listener.start()
            cls._initialized = True

            # Register execution lifecycle boundary event
            logging.getLogger("PlatformInfrastructure").info(
                "Green Bull Rider V6 Central Logging System initialized dynamically.",
                extra={"metadata": {"status": "STARTUP_SUCCESS"}}
            )

    @classmethod
    def shutdown(cls) -> None:
        """Gracefully flushes remaining log records and terminates execution listener loops."""
        with cls._lock:
            if cls._queue_listener:
                logging.getLogger("PlatformInfrastructure").info(
                    "Green Bull Rider V6 Central Logging System terminating operational bounds.",
                    extra={"metadata": {"status": "SHUTDOWN_SEQUENCE"}}
                )
                cls._queue_listener.stop()
                cls._queue_listener = None
                cls._initialized = False


# -------------------------------------------------------------------------
# PERFORMANCE ENGINE INSTRUMENTATION
# -------------------------------------------------------------------------

@contextmanager
def performance_span(
    operation: str, 
    category: str = "pipeline", 
    logger_name: str = "PerformanceEngine"
) -> Generator[None, None, None]:
    """
    High-precision performance measurement context manager.
    Tracks internal delta metrics across CPU, wall-time execution paths, and memory baselines.
    """
    target_logger = logging.getLogger(logger_name)
    start_time = time.perf_counter()
    
    # Cross-platform thread usage calculation
    start_cpu = time.process_time()
    
    # Memory profiling allocations
    start_memory = 0
    if sys.platform != "win32":
        import resource
        start_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    try:
        yield
    finally:
        end_time = time.perf_counter()
        end_cpu = time.process_time()
        
        duration_ms = (end_time - start_time) * 1000.0
        cpu_ms = (end_cpu - start_cpu) * 1000.0
        
        end_memory = 0
        if sys.platform != "win32":
            import resource
            end_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        
        memory_delta_kb = end_memory - start_memory

        metrics_payload = {
            "operation": operation,
            "category": category,
            "execution_duration_ms": round(duration_ms, 3),
            "cpu_time_ms": round(cpu_ms, 3),
            "memory_delta_kb": memory_delta_kb
        }
        
        target_logger.info(
            f"Performance span tracking finished for context path: {operation}",
            extra={"metrics": metrics_payload}
        )


# -------------------------------------------------------------------------
# STRUCTURAL PLATFORM LOGGING INTERFACES
# -------------------------------------------------------------------------

class AppLogger:
    """
    Unified operational telemetry interface proxy.
    Wraps standard logging layers with direct structured schema injection mechanics.
    """
    def __init__(self, domain_name: str) -> None:
        self._logger = logging.getLogger(domain_name)

    def debug(self, msg: str, metadata: Optional[dict[str, Any]] = None) -> None:
        self._logger.debug(msg, extra={"metadata": metadata} if metadata else None)

    def info(self, msg: str, metadata: Optional[dict[str, Any]] = None) -> None:
        self._logger.info(msg, extra={"metadata": metadata} if metadata else None)

    def warning(self, msg: str, metadata: Optional[dict[str, Any]] = None) -> None:
        self._logger.warning(msg, extra={"metadata": metadata} if metadata else None)

    def error(self, msg: str, exc_info: Any = None, metadata: Optional[dict[str, Any]] = None) -> None:
        self._logger.error(msg, exc_info=exc_info, extra={"metadata": metadata} if metadata else None)

    def critical(self, msg: str, exc_info: Any = None, metadata: Optional[dict[str, Any]] = None) -> None:
        self._logger.critical(msg, exc_info=exc_info, extra={"metadata": metadata} if metadata else None)

    def audit(self, action: str, actor: str, status: str, details: Optional[dict[str, Any]] = None) -> None:
        """Emits an immutable structural audit tracking signature."""
        payload = {
            "action": action,
            "actor": actor,
            "status": status,
            "audit_details": details or {}
        }
        self._logger.info(
            f"AUDIT EVENT: System Action [{action}] executed by [{actor}] with status [{status}].",
            extra={"is_audit_event": True, "metadata": payload}
        )

    def log_retry(self, attempt: int, max_attempts: int, delay: float, reason: str, operation: str) -> None:
        """Automated tracking wrapper for platform pipeline retry cycles."""
        metadata = {
            "retry_attempt": attempt,
            "max_attempts": max_attempts,
            "backoff_delay_sec": delay,
            "failure_reason": reason,
            "operation": operation
        }
        self._logger.warning(f"Retry execution loop engaged for operation: {operation}. Attempt {attempt}/{max_attempts}.", extra={"metadata": metadata})

    def log_scheduler_job(self, job_id: str, trigger_event: str, status: str, next_run: Optional[str] = None) -> None:
        """Automated structure tracking parser for core execution schedulers."""
        metadata = {"job_id": job_id, "trigger_event": trigger_event, "status": status, "next_scheduled_run": next_run}
        self._logger.info(f"Scheduler context boundary loop evaluated for Job [{job_id}] -> Status: {status}.", extra={"metadata": metadata})

    def log_database_query(self, sql_op: str, execution_time_ms: float, rows_affected: int, correlation_sig: str) -> None:
        """Automated schema tracer mapping internal data queries."""
        metrics = {"database_timing_ms": execution_time_ms, "rows_affected": rows_affected, "sql_operation": sql_op}
        metadata = {"query_signature": correlation_sig}
        self._logger.debug(f"Database query operation tracking complete. Duration: {execution_time_ms}ms.", extra={"metrics": metrics, "metadata": metadata})

    def log_provider_request(self, provider_name: str, endpoint: str, status_code: int, timing_ms: float) -> None:
        """Automated tracking mapper intercepting external provider data requests."""
        metrics = {"provider_timing_ms": timing_ms}
        metadata = {"provider": provider_name, "endpoint": endpoint, "http_status": status_code}
        self._logger.info(f"Data ingestion provider request context finished for [{provider_name}] -> Status: {status_code}.", extra={"metrics": metrics, "metadata": metadata})

    def log_sync_operation(self, pipeline_engine: str, phase: str, sync_status: str, payload_size: int) -> None:
        """Automated sync tracking engine parser for data pipeline engines."""
        metadata = {"pipeline_engine": pipeline_engine, "sync_phase": phase, "status": sync_status, "record_count": payload_size}
        self._logger.info(f"Sync Engine tracking update inside pipeline [{pipeline_engine}] -> Phase: {phase} | Status: {sync_status}.", extra={"metadata": metadata})

    def log_ai_execution(self, model_signature: str, evaluation_logic: str, raw_tokens: int, processing_duration_ms: float) -> None:
        """Automated tracing matrix binding model analytical evaluations."""
        metrics = {"pipeline_timing_ms": processing_duration_ms, "ai_tokens_consumed": raw_tokens}
        metadata = {"model_signature": model_signature, "evaluation_logic": evaluation_logic}
        self._logger.info(f"AI Decision Engine computational inference path evaluated using model: [{model_signature}].", extra={"metrics": metrics, "metadata": metadata})

    def log_cache_event(self, action_type: str, lookup_key: str, cache_hit: bool) -> None:
        """Automated cache tracking wrapper for state persistence layers."""
        metadata = {"cache_action": action_type, "lookup_key": lookup_key, "is_cache_hit": cache_hit}
        self._logger.debug(f"Cache abstraction event parsed for key [{lookup_key}] -> Hit: {cache_hit}.", extra={"metadata": metadata})

    def log_lifecycle_event(self, stage: str, signal_code: str, status: str) -> None:
        """Lifecycle parser capturing global framework system execution parameters."""
        metadata = {"lifecycle_stage": stage, "signal_code": signal_code, "status": status}
        self._logger.info(f"Framework boundaries monitoring component shifted state context to [{stage}] -> Status: {status}.", extra={"metadata": metadata})


# -------------------------------------------------------------------------
# MODULE EXPORTS
# -------------------------------------------------------------------------

__all__ = (
    "PlatformContext",
    "PlatformLoggerManager",
    "AppLogger",
    "StructuredJSONFormatter",
    "HumanReadableConsoleFormatter",
    "performance_span",
    "register_automation_hook",
    "register_audit_hook",
    "set_audit_logging_enabled",
)
