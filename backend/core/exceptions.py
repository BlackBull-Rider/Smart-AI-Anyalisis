"""
GREEN BULL RIDER V6 - Institutional-grade AI Stock Analysis Platform
Module: backend/core/exceptions.py
Description: Global Exception Layer. Defines the single source of truth for all 
             domain, infrastructure, and operational errors across the entire platform.
             Designed for event-driven telemetry, audit tracing, and fully automated pipelines.
             Production Locked.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


# -------------------------------------------------------------------------
# BASE EXCEPTION
# -------------------------------------------------------------------------

@dataclass(slots=True, eq=False)
class GreenBullError(Exception):
    """
    The root exception for all GREEN BULL RIDER V6 operations.
    Every platform-specific error strictly inherits from this base class.
    """
    message: str
    error_code: str = "GBR-SYS-000"
    details: dict[str, Any] = field(default_factory=dict)
    cause: Optional[Exception] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    module: str = ""
    operation: str = ""
    retryable: bool = False
    recoverable: bool = False
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initializes the built-in Exception mechanics while preserving dataclass features."""
        Exception.__init__(self, self.message)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the exception to a dictionary for REST APIs, Event Buses, and Audit Traces."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "timestamp": self.timestamp,
            "module": self.module,
            "operation": self.operation,
            "retryable": self.retryable,
            "recoverable": self.recoverable,
            "details": self.details,
            "context": self.context,
            "cause": f"{self.cause.__class__.__name__}: {str(self.cause)}" if self.cause else None
        }

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} code={self.error_code} "
            f"retryable={self.retryable} recoverable={self.recoverable} "
            f"message='{self.message}'>"
        )


# -------------------------------------------------------------------------
# CORE EXCEPTIONS
# -------------------------------------------------------------------------

@dataclass(slots=True, eq=False)
class ConfigurationError(GreenBullError):
    error_code: str = "GBR-CORE-001"

@dataclass(slots=True, eq=False)
class ValidationError(GreenBullError):
    error_code: str = "GBR-CORE-002"

@dataclass(slots=True, eq=False)
class InitializationError(GreenBullError):
    error_code: str = "GBR-CORE-003"

@dataclass(slots=True, eq=False)
class DependencyError(GreenBullError):
    error_code: str = "GBR-CORE-004"

@dataclass(slots=True, eq=False)
class PermissionError(GreenBullError):
    error_code: str = "GBR-CORE-005"

@dataclass(slots=True, eq=False)
class AuthenticationError(GreenBullError):
    error_code: str = "GBR-CORE-006"

@dataclass(slots=True, eq=False)
class AuthorizationError(GreenBullError):
    error_code: str = "GBR-CORE-007"


# -------------------------------------------------------------------------
# DATABASE EXCEPTIONS
# -------------------------------------------------------------------------

@dataclass(slots=True, eq=False)
class DatabaseError(GreenBullError):
    error_code: str = "GBR-DB-000"

@dataclass(slots=True, eq=False)
class DatabaseConnectionError(DatabaseError):
    error_code: str = "GBR-DB-001"
    retryable: bool = True

@dataclass(slots=True, eq=False)
class DatabaseTimeoutError(DatabaseError):
    error_code: str = "GBR-DB-002"
    retryable: bool = True

@dataclass(slots=True, eq=False)
class DatabaseReadError(DatabaseError):
    error_code: str = "GBR-DB-003"

@dataclass(slots=True, eq=False)
class DatabaseWriteError(DatabaseError):
    error_code: str = "GBR-DB-004"
    retryable: bool = True

@dataclass(slots=True, eq=False)
class DatabaseTransactionError(DatabaseError):
    error_code: str = "GBR-DB-005"

@dataclass(slots=True, eq=False)
class MigrationError(DatabaseError):
    error_code: str = "GBR-DB-006"

@dataclass(slots=True, eq=False)
class SchemaError(DatabaseError):
    error_code: str = "GBR-DB-007"

@dataclass(slots=True, eq=False)
class IntegrityError(DatabaseError):
    error_code: str = "GBR-DB-008"

@dataclass(slots=True, eq=False)
class DuplicateRecordError(DatabaseError):
    error_code: str = "GBR-DB-009"

@dataclass(slots=True, eq=False)
class RecordNotFoundError(DatabaseError):
    error_code: str = "GBR-DB-010"


# -------------------------------------------------------------------------
# DATA EXCEPTIONS
# -------------------------------------------------------------------------

@dataclass(slots=True, eq=False)
class DataError(GreenBullError):
    error_code: str = "GBR-DATA-000"

@dataclass(slots=True, eq=False)
class DataValidationError(DataError):
    error_code: str = "GBR-DATA-001"

@dataclass(slots=True, eq=False)
class DataCorruptionError(DataError):
    error_code: str = "GBR-DATA-002"

@dataclass(slots=True, eq=False)
class DataIntegrityError(DataError):
    error_code: str = "GBR-DATA-003"

@dataclass(slots=True, eq=False)
class MissingDataError(DataError):
    error_code: str = "GBR-DATA-004"

@dataclass(slots=True, eq=False)
class DataQualityError(DataError):
    error_code: str = "GBR-DATA-005"


# -------------------------------------------------------------------------
# PROVIDER EXCEPTIONS
# -------------------------------------------------------------------------

@dataclass(slots=True, eq=False)
class ProviderError(GreenBullError):
    error_code: str = "GBR-PROV-000"

@dataclass(slots=True, eq=False)
class ProviderConnectionError(ProviderError):
    error_code: str = "GBR-PROV-001"
    retryable: bool = True

@dataclass(slots=True, eq=False)
class ProviderTimeoutError(ProviderError):
    error_code: str = "GBR-PROV-002"
    retryable: bool = True

@dataclass(slots=True, eq=False)
class ProviderRateLimitError(ProviderError):
    error_code: str = "GBR-PROV-003"
    retryable: bool = True

@dataclass(slots=True, eq=False)
class ProviderAuthenticationError(ProviderError):
    error_code: str = "GBR-PROV-004"

@dataclass(slots=True, eq=False)
class ProviderResponseError(ProviderError):
    error_code: str = "GBR-PROV-005"

@dataclass(slots=True, eq=False)
class ProviderDataError(ProviderError):
    error_code: str = "GBR-PROV-006"

@dataclass(slots=True, eq=False)
class SymbolNotFoundError(ProviderError):
    error_code: str = "GBR-PROV-007"


# -------------------------------------------------------------------------
# SYNC EXCEPTIONS
# -------------------------------------------------------------------------

@dataclass(slots=True, eq=False)
class SyncError(GreenBullError):
    error_code: str = "GBR-SYNC-000"

@dataclass(slots=True, eq=False)
class InitialSyncError(SyncError):
    error_code: str = "GBR-SYNC-001"
    retryable: bool = True

@dataclass(slots=True, eq=False)
class IncrementalSyncError(SyncError):
    error_code: str = "GBR-SYNC-002"
    retryable: bool = True

@dataclass(slots=True, eq=False)
class GapRecoveryError(SyncError):
    error_code: str = "GBR-SYNC-003"
    retryable: bool = True

@dataclass(slots=True, eq=False)
class HistoricalSyncError(SyncError):
    error_code: str = "GBR-SYNC-004"
    retryable: bool = True

@dataclass(slots=True, eq=False)
class SchedulerError(SyncError):
    error_code: str = "GBR-SYNC-005"

@dataclass(slots=True, eq=False)
class ResumeError(SyncError):
    error_code: str = "GBR-SYNC-006"

@dataclass(slots=True, eq=False)
class CheckpointError(SyncError):
    error_code: str = "GBR-SYNC-007"


# -------------------------------------------------------------------------
# CACHE EXCEPTIONS
# -------------------------------------------------------------------------

@dataclass(slots=True, eq=False)
class CacheError(GreenBullError):
    error_code: str = "GBR-CACHE-000"

@dataclass(slots=True, eq=False)
class CacheMissError(CacheError):
    error_code: str = "GBR-CACHE-001"
    recoverable: bool = True

@dataclass(slots=True, eq=False)
class CacheWriteError(CacheError):
    error_code: str = "GBR-CACHE-002"

@dataclass(slots=True, eq=False)
class CacheReadError(CacheError):
    error_code: str = "GBR-CACHE-003"


# -------------------------------------------------------------------------
# TRACE EXCEPTIONS
# -------------------------------------------------------------------------

@dataclass(slots=True, eq=False)
class TraceError(GreenBullError):
    error_code: str = "GBR-TRACE-000"

@dataclass(slots=True, eq=False)
class TraceWriteError(TraceError):
    error_code: str = "GBR-TRACE-001"

@dataclass(slots=True, eq=False)
class TraceSerializationError(TraceError):
    error_code: str = "GBR-TRACE-002"


# -------------------------------------------------------------------------
# AUDIT EXCEPTIONS
# -------------------------------------------------------------------------

@dataclass(slots=True, eq=False)
class AuditError(GreenBullError):
    error_code: str = "GBR-AUDIT-000"

@dataclass(slots=True, eq=False)
class AuditWriteError(AuditError):
    error_code: str = "GBR-AUDIT-001"


# -------------------------------------------------------------------------
# INDICATOR EXCEPTIONS
# -------------------------------------------------------------------------

@dataclass(slots=True, eq=False)
class IndicatorError(GreenBullError):
    error_code: str = "GBR-IND-000"

@dataclass(slots=True, eq=False)
class CalculationError(IndicatorError):
    error_code: str = "GBR-IND-001"

@dataclass(slots=True, eq=False)
class InsufficientDataError(IndicatorError):
    error_code: str = "GBR-IND-002"

@dataclass(slots=True, eq=False)
class UnsupportedIndicatorError(IndicatorError):
    error_code: str = "GBR-IND-003"


# -------------------------------------------------------------------------
# ANALYZER EXCEPTIONS
# -------------------------------------------------------------------------

@dataclass(slots=True, eq=False)
class AnalyzerError(GreenBullError):
    error_code: str = "GBR-ANZ-000"

@dataclass(slots=True, eq=False)
class PatternAnalyzerError(AnalyzerError):
    error_code: str = "GBR-ANZ-001"

@dataclass(slots=True, eq=False)
class MomentumAnalyzerError(AnalyzerError):
    error_code: str = "GBR-ANZ-002"

@dataclass(slots=True, eq=False)
class TrendAnalyzerError(AnalyzerError):
    error_code: str = "GBR-ANZ-003"

@dataclass(slots=True, eq=False)
class VolumeAnalyzerError(AnalyzerError):
    error_code: str = "GBR-ANZ-004"

@dataclass(slots=True, eq=False)
class VolatilityAnalyzerError(AnalyzerError):
    error_code: str = "GBR-ANZ-005"

@dataclass(slots=True, eq=False)
class SmartMoneyAnalyzerError(AnalyzerError):
    error_code: str = "GBR-ANZ-006"


# -------------------------------------------------------------------------
# AI EXCEPTIONS
# -------------------------------------------------------------------------

@dataclass(slots=True, eq=False)
class AIError(GreenBullError):
    error_code: str = "GBR-AI-000"

@dataclass(slots=True, eq=False)
class InferenceError(AIError):
    error_code: str = "GBR-AI-001"
    retryable: bool = True

@dataclass(slots=True, eq=False)
class PromptError(AIError):
    error_code: str = "GBR-AI-002"

@dataclass(slots=True, eq=False)
class ModelError(AIError):
    error_code: str = "GBR-AI-003"

@dataclass(slots=True, eq=False)
class ConfidenceError(AIError):
    error_code: str = "GBR-AI-004"


# -------------------------------------------------------------------------
# PORTFOLIO EXCEPTIONS
# -------------------------------------------------------------------------

@dataclass(slots=True, eq=False)
class PortfolioError(GreenBullError):
    error_code: str = "GBR-PORT-000"

@dataclass(slots=True, eq=False)
class PositionError(PortfolioError):
    error_code: str = "GBR-PORT-001"

@dataclass(slots=True, eq=False)
class OrderError(PortfolioError):
    error_code: str = "GBR-PORT-002"

@dataclass(slots=True, eq=False)
class RiskError(PortfolioError):
    error_code: str = "GBR-PORT-003"


# -------------------------------------------------------------------------
# API EXCEPTIONS
# -------------------------------------------------------------------------

@dataclass(slots=True, eq=False)
class APIError(GreenBullError):
    error_code: str = "GBR-API-000"

@dataclass(slots=True, eq=False)
class RequestError(APIError):
    error_code: str = "GBR-API-001"

@dataclass(slots=True, eq=False)
class ResponseError(APIError):
    error_code: str = "GBR-API-002"

@dataclass(slots=True, eq=False)
class SerializationError(APIError):
    error_code: str = "GBR-API-003"


# -------------------------------------------------------------------------
# UTILITY EXCEPTIONS
# -------------------------------------------------------------------------

@dataclass(slots=True, eq=False)
class TimeoutError(GreenBullError):
    error_code: str = "GBR-UTIL-001"
    retryable: bool = True

@dataclass(slots=True, eq=False)
class RetryExceededError(GreenBullError):
    error_code: str = "GBR-UTIL-002"

@dataclass(slots=True, eq=False)
class ConcurrencyError(GreenBullError):
    error_code: str = "GBR-UTIL-003"
    retryable: bool = True

@dataclass(slots=True, eq=False)
class ThreadError(GreenBullError):
    error_code: str = "GBR-UTIL-004"

@dataclass(slots=True, eq=False)
class EventBusError(GreenBullError):
    error_code: str = "GBR-UTIL-005"
    retryable: bool = True

@dataclass(slots=True, eq=False)
class NotificationError(GreenBullError):
    error_code: str = "GBR-UTIL-006"


# -------------------------------------------------------------------------
# MODULE EXPORTS
# -------------------------------------------------------------------------

__all__ = (
    "GreenBullError",
    
    "ConfigurationError", "ValidationError", "InitializationError",
    "DependencyError", "PermissionError", "AuthenticationError", "AuthorizationError",
    
    "DatabaseError", "DatabaseConnectionError", "DatabaseTimeoutError",
    "DatabaseReadError", "DatabaseWriteError", "DatabaseTransactionError",
    "MigrationError", "SchemaError", "IntegrityError", "DuplicateRecordError", "RecordNotFoundError",
    
    "DataError", "DataValidationError", "DataCorruptionError",
    "DataIntegrityError", "MissingDataError", "DataQualityError",
    
    "ProviderError", "ProviderConnectionError", "ProviderTimeoutError",
    "ProviderRateLimitError", "ProviderAuthenticationError", "ProviderResponseError",
    "ProviderDataError", "SymbolNotFoundError",
    
    "SyncError", "InitialSyncError", "IncrementalSyncError",
    "GapRecoveryError", "HistoricalSyncError", "SchedulerError",
    "ResumeError", "CheckpointError",
    
    "CacheError", "CacheMissError", "CacheWriteError", "CacheReadError",
    
    "TraceError", "TraceWriteError", "TraceSerializationError",
    
    "AuditError", "AuditWriteError",
    
    "IndicatorError", "CalculationError", "InsufficientDataError", "UnsupportedIndicatorError",
    
    "AnalyzerError", "PatternAnalyzerError", "MomentumAnalyzerError",
    "TrendAnalyzerError", "VolumeAnalyzerError", "VolatilityAnalyzerError", "SmartMoneyAnalyzerError",
    
    "AIError", "InferenceError", "PromptError", "ModelError", "ConfidenceError",
    
    "PortfolioError", "PositionError", "OrderError", "RiskError",
    
    "APIError", "RequestError", "ResponseError", "SerializationError",
    
    "TimeoutError", "RetryExceededError", "ConcurrencyError",
    "ThreadError", "EventBusError", "NotificationError",
)

# ============================================================================
# PIPELINE EXCEPTIONS
# ============================================================================

class PipelineError(GreenBullError):
    """Base exception for all pipeline errors."""
    pass


class RetryableError(PipelineError):
    """Temporary error that may be retried safely."""
    pass


class FatalError(PipelineError):
    """Non-recoverable pipeline error."""
    pass

