"""
GREEN BULL RIDER V6 - Institutional-grade AI Stock Analysis Platform
Module: backend/core/retry.py
Description: Enterprise-grade centralized Retry Engine.
             Provides resilient execution wrappers with configurable backoff strategies,
             jitter integration, cancellation support, and deterministic execution boundaries.
             Strictly decoupled from business logic and providers.
             Fully integrated with Trace, Audit, and Logging pipelines.
"""

import time
import random
import threading
from enum import Enum
from functools import wraps
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, Optional, Tuple, Type, TypeVar, cast
)

# Internal Platform Integrations
from backend.core.logger import AppLogger
from backend.core.trace import TraceEngine, SpanKind, TraceLevel
from backend.core.audit import AuditEngine, AuditAction, AuditResult, AuditSeverity
from backend.core.exceptions import GreenBullError

# -------------------------------------------------------------------------
# TYPE HINTS & CONSTANTS
# -------------------------------------------------------------------------
F = TypeVar('F', bound=Callable[..., Any])
ExceptionTuple = Tuple[Type[Exception], ...]

_logger = AppLogger("RetryEngine")


# -------------------------------------------------------------------------
# EXCEPTIONS
# -------------------------------------------------------------------------

class RetryExceededError(GreenBullError):
    """Raised when an operation fails permanently after exhausting all retry attempts."""
    error_code: str = "GBR-RETRY-001"


# -------------------------------------------------------------------------
# ENUMS
# -------------------------------------------------------------------------

class RetryStrategy(str, Enum):
    """Supported mathematical strategies for delay calculation."""
    FIXED = "FIXED"
    LINEAR = "LINEAR"
    EXPONENTIAL = "EXPONENTIAL"
    EXPONENTIAL_JITTER = "EXPONENTIAL_JITTER"
    IMMEDIATE = "IMMEDIATE"


# -------------------------------------------------------------------------
# DATACLASSES
# -------------------------------------------------------------------------

@dataclass(slots=True)
class RetryContext:
    """
    Mutable state tracker bound to the current execution thread.
    Monitors attempts, accumulates performance metrics, and tracks exceptions.
    """
    operation_name: str
    attempt_number: int = 1
    start_time_perf: float = field(default_factory=time.perf_counter)
    elapsed_time: float = 0.0
    last_exception: Optional[Exception] = None
    next_delay: float = 0.0

    def update_elapsed_time(self) -> None:
        """Updates the total elapsed time since the retry context was created."""
        self.elapsed_time = time.perf_counter() - self.start_time_perf


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """
    Immutable blueprint dictating the structural rules of a retry execution.
    """
    max_attempts: int = 3
    initial_delay: float = 1.0
    maximum_delay: float = 60.0
    multiplier: float = 2.0
    jitter: float = 0.5
    retryable_exceptions: ExceptionTuple = (Exception,)
    operation_name: Optional[str] = None
    timeout: Optional[float] = None
    is_cancelled: Optional[Callable[[], bool]] = None
    
    on_retry: Optional[Callable[[RetryContext], None]] = None
    on_success: Optional[Callable[[RetryContext], None]] = None
    on_failure: Optional[Callable[[RetryContext], None]] = None
    
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_JITTER


@dataclass(frozen=True, slots=True)
class RetryStatistics:
    """Immutable final telemetry data for the Retry Engine's global state."""
    total_retries: int
    successful_retries: int
    failed_retries: int
    average_retry_count: float
    average_execution_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_retries": self.total_retries,
            "successful_retries": self.successful_retries,
            "failed_retries": self.failed_retries,
            "average_retry_count": round(self.average_retry_count, 2),
            "average_execution_time_ms": round(self.average_execution_time_ms, 3)
        }


# -------------------------------------------------------------------------
# RETRY ENGINE (SINGLETON)
# -------------------------------------------------------------------------

class RetryEngine:
    """
    Central Orchestrator for all retry boundaries.
    Fully integrated with Observability stack. Thread-safe execution and statistics.
    """
    _instance: Optional['RetryEngine'] = None
    _init_lock = threading.Lock()

    def __new__(cls) -> 'RetryEngine':
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super(RetryEngine, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initializes thread-safe global statistics tracking."""
        self._stats_lock = threading.Lock()
        self._total_retries = 0
        self._successful_retries = 0
        self._failed_retries = 0
        self._total_retry_count = 0
        self._total_execution_time_perf = 0.0

    # -------------------------------------------------------------------------
    # STATISTICS MANAGEMENT
    # -------------------------------------------------------------------------

    def get_statistics(self) -> RetryStatistics:
        """Retrieves a thread-safe snapshot of global retry telemetry."""
        with self._stats_lock:
            total_resolved = self._successful_retries + self._failed_retries
            avg_retry = (self._total_retry_count / total_resolved) if total_resolved > 0 else 0.0
            avg_time = (self._total_execution_time_perf / total_resolved) * 1000.0 if total_resolved > 0 else 0.0
            
            return RetryStatistics(
                total_retries=self._total_retries,
                successful_retries=self._successful_retries,
                failed_retries=self._failed_retries,
                average_retry_count=avg_retry,
                average_execution_time_ms=avg_time
            )

    def reset_statistics(self) -> None:
        """Purges global retry statistics cleanly."""
        with self._stats_lock:
            self._total_retries = 0
            self._successful_retries = 0
            self._failed_retries = 0
            self._total_retry_count = 0
            self._total_execution_time_perf = 0.0

    def _update_success_stats(self, context: RetryContext) -> None:
        """Safely updates global metrics upon a successful execution."""
        with self._stats_lock:
            self._successful_retries += 1
            self._total_retry_count += (context.attempt_number - 1)
            self._total_execution_time_perf += context.elapsed_time

    def _update_failure_stats(self, context: RetryContext) -> None:
        """Safely updates global metrics upon terminal execution failure."""
        with self._stats_lock:
            self._failed_retries += 1
            self._total_retry_count += (context.attempt_number - 1)
            self._total_execution_time_perf += context.elapsed_time

    def _record_retry_attempt(self) -> None:
        """Safely increments the global discrete retry attempt counter."""
        with self._stats_lock:
            self._total_retries += 1

    # -------------------------------------------------------------------------
    # DELAY CALCULATION
    # -------------------------------------------------------------------------

    @staticmethod
    def calculate_delay(policy: RetryPolicy, attempt: int) -> float:
        """
        Computes the exact sleep duration based on the selected mathematical strategy.
        Guarantees delay never exceeds the maximum_delay bounds.
        """
        if policy.strategy == RetryStrategy.IMMEDIATE:
            return 0.0

        if policy.strategy == RetryStrategy.FIXED:
            delay = policy.initial_delay

        elif policy.strategy == RetryStrategy.LINEAR:
            delay = policy.initial_delay * attempt

        elif policy.strategy == RetryStrategy.EXPONENTIAL:
            delay = policy.initial_delay * (policy.multiplier ** (attempt - 1))

        elif policy.strategy == RetryStrategy.EXPONENTIAL_JITTER:
            exp_delay = policy.initial_delay * (policy.multiplier ** (attempt - 1))
            jitter_amount = exp_delay * policy.jitter
            delay = random.uniform(max(0.0, exp_delay - jitter_amount), exp_delay + jitter_amount)

        else:
            delay = policy.initial_delay

        return min(delay, policy.maximum_delay)

    # -------------------------------------------------------------------------
    # CORE EXECUTION ENGINE
    # -------------------------------------------------------------------------

    def execute(self, func: Callable[..., Any], policy: RetryPolicy, *args: Any, **kwargs: Any) -> Any:
        """
        Executes a synchronous callable with robust retry boundaries.
        Utilizes monotonic timers to avoid busy waiting and system clock drifts.
        Fully instrumented with Audit, Trace, and Logger bindings.
        """
        op_name = policy.operation_name or func.__qualname__
        context = RetryContext(operation_name=op_name)

        AuditEngine.record(
            operation=op_name,
            action=AuditAction.EXECUTE,
            result=AuditResult.PENDING,
            severity=AuditSeverity.INFO,
            message=f"Starting execution of {op_name} with {policy.max_attempts} max attempts."
        )

        while True:
            context.update_elapsed_time()

            # 1. Cancellation Check
            if policy.is_cancelled and policy.is_cancelled():
                self._update_failure_stats(context)
                msg = f"Operation {op_name} cancelled explicitly before attempt {context.attempt_number}."
                _logger.warning(msg)
                AuditEngine.record_failure(op_name, AuditAction.EXECUTE, msg, AuditSeverity.WARNING)
                raise RetryExceededError(message=msg, operation=op_name)

            # 2. Global Timeout Check
            if policy.timeout is not None and context.elapsed_time >= policy.timeout:
                self._update_failure_stats(context)
                msg = f"Operation {op_name} exceeded global timeout of {policy.timeout}s."
                _logger.error(msg)
                AuditEngine.record_failure(op_name, AuditAction.EXECUTE, msg, AuditSeverity.CRITICAL)
                raise RetryExceededError(message=msg, operation=op_name)

            # 3. Execution Block
            try:
                with TraceEngine.nested_span(operation=op_name, component="retry", kind=SpanKind.INTERNAL):
                    TraceEngine.attach_metadata("attempt", context.attempt_number)
                    result = func(*args, **kwargs)

                # Execution Success Pipeline
                context.update_elapsed_time()
                self._update_success_stats(context)
                
                if policy.on_success:
                    try:
                        policy.on_success(context)
                    except Exception as cb_err:
                        _logger.warning(f"Success callback failed for {op_name}: {cb_err}")

                if context.attempt_number > 1:
                    _logger.info(f"Operation {op_name} succeeded on attempt {context.attempt_number}.")
                    AuditEngine.record_success(
                        operation=op_name,
                        action=AuditAction.EXECUTE,
                        message=f"Operation recovered and succeeded on attempt {context.attempt_number}."
                    )
                else:
                    _logger.debug(f"Operation {op_name} succeeded on initial attempt.")

                return result

            # 4. Failure Evaluation Block
            except Exception as e:
                context.last_exception = e
                context.update_elapsed_time()
                TraceEngine.record_exception(e)

                is_retryable = isinstance(e, policy.retryable_exceptions)
                max_exhausted = context.attempt_number >= policy.max_attempts

                if not is_retryable or max_exhausted:
                    # Terminal Failure Pipeline
                    self._update_failure_stats(context)
                    
                    if policy.on_failure:
                        try:
                            policy.on_failure(context)
                        except Exception as cb_err:
                            _logger.warning(f"Failure callback failed for {op_name}: {cb_err}")

                    reason = "Max attempts exhausted" if max_exhausted else "Non-retryable exception"
                    fail_msg = f"Terminal failure for {op_name}: {reason}. Exception: {e}"
                    
                    _logger.error(fail_msg, exc_info=e)
                    AuditEngine.record_failure(
                        operation=op_name,
                        action=AuditAction.EXECUTE,
                        message=fail_msg,
                        severity=AuditSeverity.CRITICAL,
                        metadata={"attempts": context.attempt_number, "duration_ms": context.elapsed_time * 1000.0}
                    )

                    if max_exhausted:
                        raise RetryExceededError(
                            message=fail_msg,
                            cause=e,
                            operation=op_name,
                            details={"attempts": context.attempt_number, "elapsed_sec": context.elapsed_time}
                        ) from e
                    raise e

                # 5. Pre-Retry Pipeline
                self._record_retry_attempt()
                context.next_delay = self.calculate_delay(policy, context.attempt_number)
                
                retry_msg = (f"Attempt {context.attempt_number}/{policy.max_attempts} failed for {op_name}. "
                             f"Retrying in {context.next_delay:.2f}s. Exception: {e}")
                
                _logger.warning(retry_msg)
                TraceEngine.record_event(
                    name="RETRY_SCHEDULED",
                    severity=TraceLevel.WARNING,
                    metadata={
                        "attempt": context.attempt_number,
                        "delay": context.next_delay,
                        "exception_type": e.__class__.__name__
                    }
                )

                if policy.on_retry:
                    try:
                        policy.on_retry(context)
                    except Exception as cb_err:
                        _logger.warning(f"Retry callback failed for {op_name}: {cb_err}")

                time.sleep(context.next_delay)
                context.attempt_number += 1


# -------------------------------------------------------------------------
# GLOBAL SINGLETON EXPORT
# -------------------------------------------------------------------------

retry_engine = RetryEngine()


# -------------------------------------------------------------------------
# DECORATORS
# -------------------------------------------------------------------------

def retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    maximum_delay: float = 60.0,
    multiplier: float = 2.0,
    jitter: float = 0.5,
    retryable_exceptions: ExceptionTuple = (Exception,),
    operation_name: Optional[str] = None,
    timeout: Optional[float] = None,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_JITTER,
    is_cancelled: Optional[Callable[[], bool]] = None,
    on_retry: Optional[Callable[[RetryContext], None]] = None,
    on_success: Optional[Callable[[RetryContext], None]] = None,
    on_failure: Optional[Callable[[RetryContext], None]] = None
) -> Callable[[F], F]:
    """
    Enterprise-grade decorator orchestrating synchronous execution retry boundaries.
    
    Usage:
        @retry(max_attempts=5, strategy=RetryStrategy.FULL_JITTER)
        def fetch_data(symbol):
            ...
    """
    def decorator(func: F) -> F:
        policy = RetryPolicy(
            max_attempts=max_attempts,
            initial_delay=initial_delay,
            maximum_delay=maximum_delay,
            multiplier=multiplier,
            jitter=jitter,
            retryable_exceptions=retryable_exceptions,
            operation_name=operation_name or func.__qualname__,
            timeout=timeout,
            strategy=strategy,
            is_cancelled=is_cancelled,
            on_retry=on_retry,
            on_success=on_success,
            on_failure=on_failure
        )

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            return retry_engine.execute(func, policy, *args, **kwargs)

        return cast(F, sync_wrapper)
    return decorator


# -------------------------------------------------------------------------
# MODULE EXPORTS
# -------------------------------------------------------------------------

__all__ = [
    "RetryExceededError",
    "RetryStrategy",
    "RetryContext",
    "RetryPolicy",
    "RetryStatistics",
    "RetryEngine",
    "retry_engine",
    "retry"
]
