"""
GREEN BULL RIDER V6 - Institutional-grade AI Stock Analysis Platform
Module: backend/pipeline/market_pipeline.py
Description: Enterprise Production Pipeline Layer.
             Orchestrates the data flow from Providers to Database Write.
             Strictly handles synchronization, validation, transactions,
             telemetry, audit, and fault tolerance. 
             Provides a robust foundation for Indicator -> AI Pipeline layers.
             Python 3.13 Compatible. Compile-Safe. Runtime-Safe. Production Locked.
"""

import os
import time
import datetime
import uuid
import random
import contextlib
import threading
import zlib
from enum import Enum, auto
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Type, Callable, Generator, Mapping
from contextvars import ContextVar
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------------------------------------------------------
# Internal Dependencies
# -----------------------------------------------------------------------------
from backend.config.settings import settings
from backend.core.logger import AppLogger
from backend.core.trace import TraceEngine, trace_span, SpanKind
from backend.core.audit import AuditEngine as CoreAudit, AuditAction, AuditSeverity

class AuditEngine:
    @staticmethod
    def record_event(operation: str, action: AuditAction, severity: AuditSeverity, message: str, metadata: dict = None) -> None:
        try:
            if severity in (AuditSeverity.CRITICAL, AuditSeverity.WARNING):
                if hasattr(CoreAudit, 'record_failure'):
                    CoreAudit.record_failure(operation=operation, action=action, message=message, severity=severity, metadata=metadata)
            else:
                if hasattr(CoreAudit, 'record_success'):
                    CoreAudit.record_success(operation=operation, action=action, message=message, metadata=metadata)
        except Exception:
            pass

from backend.core.metrics import metrics_engine
from backend.core.scheduler import Scheduler
from backend.core.event_bus import EventBus, Event, Event
from backend.core.exceptions import (
    ValidationError, 
    ProviderError, 
    DatabaseError,
    PipelineError, 
    RetryableError, 
    FatalError
)

from backend.database.connection import DatabaseSession
from backend.database.unit_of_work import UnitOfWork
from backend.data.market_reader import MarketReader
from backend.data.market_writer import MarketWriter
from backend.data.market_validator import MarketValidator
from backend.data.providers.yfinance import YahooFinanceProvider, YahooFinanceConfig
from backend.data.providers.nse import NSEProvider, NSEConfig


# -----------------------------------------------------------------------------
# ENUMS
# -----------------------------------------------------------------------------
class PipelineState(Enum):
    PENDING = auto()
    INITIALIZING = auto()
    RUNNING = auto()
    RETRYING = auto()
    SUCCESS = auto()
    FAILED = auto()
    CANCELLED = auto()

class PipelineStage(Enum):
    BOOTSTRAP = auto()
    INCREMENTAL_CHECK = auto()
    PROVIDER_FETCH = auto()
    VALIDATION = auto()
    COMPRESSION = auto()
    DATABASE_WRITE = auto()
    COMMIT = auto()
    ROLLBACK = auto()


# -----------------------------------------------------------------------------
# DATACLASSES & IMMUTABLE EVENTS
# -----------------------------------------------------------------------------
    symbols: Optional[List[str]] = None
    timeframe: str = "1D"
    exchange: str = "NSE"
    provider_name: str = field(default_factory=lambda: getattr(settings.provider, "default", "nse"))
    batch_size: int = 50
    retry_max_attempts: int = 5
    retry_backoff_factor: float = 2.0


@dataclass
class PipelineConfig:
    start_date: str
    end_date: str
    symbols: Optional[List[str]] = None
    timeframe: str = "1D"
    exchange: str = "NSE"
    provider_name: str = field(
        default_factory=lambda: getattr(settings.provider, "default", "nse")
    )
    batch_size: int = 50
    retry_max_attempts: int = 5
    retry_backoff_factor: float = 2.0

@dataclass
class ValidationResult:
    valid_records: List[Dict[str, Any]]
    invalid_records: List[Dict[str, Any]]
    warnings: List[str]
    duplicates: int

@dataclass
class PipelineStatistics:
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    retry_count: int = 0
    provider_latency: float = 0.0
    reader_latency: float = 0.0
    validator_latency: float = 0.0
    writer_latency: float = 0.0
    db_latency: float = 0.0
    pipeline_latency: float = 0.0
    records_read: int = 0
    records_valid: int = 0
    records_invalid: int = 0
    records_written: int = 0

    def increment(self, metric: str, amount: int = 1) -> None:
        with self._lock:
            current = getattr(self, metric)
            setattr(self, metric, current + amount)

    def add_latency(self, metric: str, amount: float) -> None:
        with self._lock:
            current = getattr(self, metric)
            setattr(self, metric, current + amount)

    def to_dict(self) -> Dict[str, Any]:
        """Safely serializes statistics avoiding lock serialization issues."""
        with self._lock:
            return {
                "execution_count": self.execution_count,
                "success_count": self.success_count,
                "failure_count": self.failure_count,
                "retry_count": self.retry_count,
                "provider_latency": self.provider_latency,
                "reader_latency": self.reader_latency,
                "validator_latency": self.validator_latency,
                "writer_latency": self.writer_latency,
                "db_latency": self.db_latency,
                "pipeline_latency": self.pipeline_latency,
                "records_read": self.records_read,
                "records_valid": self.records_valid,
                "records_invalid": self.records_invalid,
                "records_written": self.records_written
            }

@dataclass
class PipelineSummary:
    pipeline_id: str
    execution_id: str
    total_processed: int
    successful: int
    failed: int
    success_symbols: List[str]
    failed_symbols: List[str]
    retry_symbols: List[str]
    execution_time: float
    statistics: Dict[str, Any]
    timestamp: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))

@dataclass
class PipelineContext:
    pipeline_id: str
    execution_id: str
    trace_id: str
    config: PipelineConfig
    stats: PipelineStatistics
    start_time: float
    state: PipelineState
    success_symbols: List[str] = field(default_factory=list)
    failed_symbols: List[str] = field(default_factory=list)
    retry_symbols: List[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def mark_success(self, symbols: List[str]) -> None:
        with self._lock:
            self.success_symbols.extend(symbols)

    def mark_failed(self, symbols: List[str]) -> None:
        with self._lock:
            self.failed_symbols.extend(symbols)

    def mark_retry(self, symbols: List[str]) -> None:
        with self._lock:
            self.retry_symbols.extend(symbols)

@dataclass
class PipelineResult:
    context: PipelineContext
    success: bool
    errors: List[Exception] = field(default_factory=list)
    summary: Optional[PipelineSummary] = None


# -----------------------------------------------------------------------------
# CONTEXT VARIABLES
# -----------------------------------------------------------------------------
pipeline_context_var: ContextVar[PipelineContext] = ContextVar("pipeline_context")


# -----------------------------------------------------------------------------
# REGISTRIES & SERVICES
# -----------------------------------------------------------------------------
class ProviderRegistry:
    """Registry for dynamic provider instantiation and extension."""
    _providers: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, provider_instance: Any) -> None:
        cls._providers[name] = provider_instance

    @classmethod
    def get(cls, name: str) -> Optional[Any]:
        return cls._providers.get(name)

    @classmethod
    def get_all(cls) -> Dict[str, Any]:
        return cls._providers


class CompressionService:
    """Service to handle automated compression hooks configured via schema mappings."""
    def compress_fields(self, records: List[Dict[str, Any]], fields_to_compress: List[str]) -> List[Dict[str, Any]]:
        for record in records:
            for field_name in fields_to_compress:
                if record.get(field_name):
                    compressed_key = f"{field_name}_compressed"
                    record[compressed_key] = zlib.compress(str(record[field_name]).encode('utf-8'))
        return records


# -----------------------------------------------------------------------------
# PIPELINE OBSERVABILITY & HEALTH
# -----------------------------------------------------------------------------
class PipelineMonitor:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.logger = AppLogger(self.__class__.__name__)

    def _get_labels(self, context: PipelineContext) -> Dict[str, str]:
        return {
            "provider": context.config.provider_name,
            "timeframe": context.config.timeframe,
            "module": "market_pipeline"
        }

    def log_stage(self, context: PipelineContext, stage: PipelineStage) -> None:
        context.state = PipelineState.RUNNING
        self.logger.info(f"Transitioning to Stage: {stage.name} [Exec ID: {context.execution_id}]")
        AuditEngine.record_event(
            operation=stage.name, action=AuditAction.SYSTEM, severity=AuditSeverity.INFO,
            message=f"Pipeline entering stage: {stage.name}",
            metadata={"execution_id": context.execution_id, "trace_id": context.trace_id}
        )
        self.event_bus.publish(Event(
            name="PipelineStageChanged", source="market_pipeline", 
            payload=MappingProxyType({"stage": stage.name, "execution_id": context.execution_id})
        ))

    def log_success(self, context: PipelineContext) -> None:
        context.state = PipelineState.SUCCESS
        labels = self._get_labels(context)
        metrics_engine.increment("pipeline.success_count")
        metrics_engine.record_latency("pipeline.pipeline_latency", "pipeline", context.stats.pipeline_latency)
        
        self.logger.info(f"Pipeline Completed Successfully [Exec ID: {context.execution_id}]")
        AuditEngine.record_event(
            operation="Pipeline Finish", action=AuditAction.SYSTEM, severity=AuditSeverity.INFO,
            message="Pipeline executed successfully",
            metadata={"execution_id": context.execution_id, "records_written": context.stats.records_written}
        )
        self.event_bus.publish(Event(
            name="PipelineCompleted", source="market_pipeline", 
            payload=MappingProxyType({"execution_id": context.execution_id, "metrics": context.stats.to_dict()})
        ))

    def log_failure(self, context: PipelineContext, error: Exception) -> None:
        context.state = PipelineState.FAILED
        labels = self._get_labels(context)
        metrics_engine.increment("pipeline.failure_count")
        
        self.logger.error(f"Pipeline Failed: {str(error)} [Exec ID: {context.execution_id}]")
        AuditEngine.record_event(
            operation="Pipeline Failure", action=AuditAction.SYSTEM, severity=AuditSeverity.CRITICAL,
            message=f"Pipeline failed with fatal error: {str(error)}",
            metadata={"execution_id": context.execution_id, "trace_id": context.trace_id}
        )
        self.event_bus.publish(Event(
            name="PipelineFailed", source="market_pipeline", 
            payload=MappingProxyType({"error": str(error), "execution_id": context.execution_id})
        ))


class PipelineHooks:
    def __init__(self, monitor: PipelineMonitor):
        self.monitor = monitor

    def pre_start(self, context: PipelineContext) -> None:
        self.monitor.log_stage(context, PipelineStage.BOOTSTRAP)

    def post_finish(self, context: PipelineContext) -> None:
        self.monitor.log_success(context)

    def on_error(self, context: PipelineContext, error: Exception) -> None:
        self.monitor.log_failure(context, error)


class PipelineHealth:
    def __init__(self, db_session_cls: Type[DatabaseSession]):
        self.db_session_cls = db_session_cls
        self.logger = AppLogger(self.__class__.__name__)

    def check_components(self) -> bool:
        try:
            with self.db_session_cls() as session:
                session.db.execute("SELECT 1")
        except Exception as e:
            self.logger.error(f"Database Health Check Failed: {e}")
            return False

        for name, provider in ProviderRegistry.get_all().items():
            if hasattr(provider, "supports_health_check") and provider.supports_health_check():
                if hasattr(provider, "health_check") and not provider.health_check():
                    self.logger.error(f"Provider Health Check Failed: {name}")
                    return False

        return True


class PipelineBootstrap:
    def __init__(self, health_checker: PipelineHealth):
        self.health_checker = health_checker
        self.logger = AppLogger(self.__class__.__name__)

    def verify_readiness(self) -> bool:
        self.logger.info("Verifying pipeline component readiness...")
        if not self.health_checker.check_components():
            self.logger.critical("Pipeline Bootstrap Failed. Core components degraded.")
            raise FatalError("Pipeline components failed health check during bootstrap phase.")
        self.logger.info("Pipeline Bootstrap Successful. Components verified.")
        return True


# -----------------------------------------------------------------------------
# ERROR HANDLING & RETRY ENGINE
# -----------------------------------------------------------------------------
class PipelineErrorHandler:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.logger = AppLogger(self.__class__.__name__)

    def handle(self, error: Exception, context: PipelineContext) -> None:
        TraceEngine.record_exception(error)
        
        if isinstance(error, ValidationError):
            context.stats.increment("records_invalid")
            AuditEngine.record_event(
                operation="Validation Error", action=AuditAction.SYSTEM, severity=AuditSeverity.WARNING,
                message=str(error), metadata={"execution_id": context.execution_id}
            )
        elif isinstance(error, DatabaseError):
            context.stats.increment("failure_count")
            AuditEngine.record_event(
                operation="Database Error", action=AuditAction.SYSTEM, severity=AuditSeverity.CRITICAL,
                message=str(error), metadata={"execution_id": context.execution_id}
            )
            self.event_bus.publish(Event(
                "DatabaseRollbackTriggered", 
                MappingProxyType({"execution_id": context.execution_id, "error": str(error)})
            ))
        elif isinstance(error, ProviderError):
            context.stats.increment("failure_count")
            AuditEngine.record_event(
                operation="Provider Error", action=AuditAction.SYSTEM, severity=AuditSeverity.WARNING,
                message=str(error), metadata={"execution_id": context.execution_id}
            )
        else:
            AuditEngine.record_event(
                operation="Unknown Pipeline Error", action=AuditAction.SYSTEM, severity=AuditSeverity.CRITICAL,
                message=str(error), metadata={"execution_id": context.execution_id}
            )
        self.logger.error(f"Pipeline Error Handler invoked for: {type(error).__name__} - {str(error)}")


class PipelineRetryPolicy:
    def __init__(self, monitor: PipelineMonitor):
        self.monitor = monitor
        self.logger = AppLogger(self.__class__.__name__)

    def execute_with_retry(self, action: Callable, context: PipelineContext, batch: List[str], *args: Any, **kwargs: Any) -> Any:
        max_retries = context.config.retry_max_attempts
        backoff_factor = context.config.retry_backoff_factor
        retries = 0
        
        while True:
            try:
                return action(*args, **kwargs)
            except (RetryableError, ProviderError, DatabaseError) as e:
                retries += 1
                if retries > max_retries:
                    self.logger.error(f"Exhausted {max_retries} retries for batch execution_id {context.execution_id}.")
                    context.mark_failed(batch)
                    raise PipelineError(f"Operation failed after {retries} retries.") from e
                
                context.state = PipelineState.RETRYING
                context.stats.increment("retry_count")
                context.mark_retry(batch)
                
                self.logger.warning(f"Retrying (Attempt {retries}) due to: {str(e)}")
                AuditEngine.record_event(
                    operation="Pipeline Retry", action=AuditAction.SYSTEM, severity=AuditSeverity.WARNING,
                    message="Retrying pipeline stage", metadata={"execution_id": context.execution_id, "attempt": retries}
                )
                
                sleep_time = (backoff_factor ** retries) + random.uniform(0, 1)
                time.sleep(sleep_time)


# -----------------------------------------------------------------------------
# LIFECYCLE & EXECUTION ENGINE
# -----------------------------------------------------------------------------
class PipelineLifecycle:
    def __init__(self, hooks: PipelineHooks):
        self.hooks = hooks

    @contextlib.contextmanager
    def manage(self, config: PipelineConfig) -> Generator[PipelineContext, None, None]:
        pipeline_id = str(uuid.uuid4())
        execution_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        
        context = PipelineContext(
            pipeline_id=pipeline_id,
            execution_id=execution_id,
            trace_id=trace_id,
            config=config,
            stats=PipelineStatistics(),
            start_time=time.perf_counter(),
            state=PipelineState.INITIALIZING
        )
        token = pipeline_context_var.set(context)
        
        metrics_engine.increment("pipeline.execution_count")
        AuditEngine.record_event(
            operation="Pipeline Start", action=AuditAction.SYSTEM, severity=AuditSeverity.INFO,
            message="Pipeline context created and executing.", metadata={"execution_id": execution_id}
        )
        
        self.hooks.pre_start(context)
        try:
            yield context
            context.stats.pipeline_latency = time.perf_counter() - context.start_time
            self.hooks.post_finish(context)
        except Exception as e:
            context.stats.pipeline_latency = time.perf_counter() - context.start_time
            self.hooks.on_error(context, e)
            raise
        finally:
            pipeline_context_var.reset(token)


class PipelineExecutor:
    def __init__(self,
                 reader: MarketReader,
                 validator: MarketValidator,
                 writer: MarketWriter,
                 uow: UnitOfWork,
                 monitor: PipelineMonitor,
                 event_bus: EventBus,
                 compression_service: CompressionService):
        self.reader = reader
        self.validator = validator
        self.writer = writer
        self.uow = uow
        self.monitor = monitor
        self.event_bus = event_bus
        self.compression_service = compression_service
        self.logger = AppLogger(self.__class__.__name__)

    @trace_span(operation="executor.process_batch", component="pipeline", kind=SpanKind.INTERNAL)
    def process_batch(self, context: PipelineContext, batch: List[str]) -> None:
        self.logger.info(f"Initiating pipeline execution for batch of {len(batch)} symbols. [Exec ID: {context.execution_id}]")

        # -------------------------------------------------------
        # STAGE 1: Database Read (Incremental Sync Check Map)
        # -------------------------------------------------------
        self.monitor.log_stage(context, PipelineStage.INCREMENTAL_CHECK)
        t_reader = time.perf_counter()

        since_map: Dict[str, str] = {sym: context.config.start_date for sym in batch}
        if hasattr(self.reader, 'read_latest_batch'):
            latest_records = self.reader.read_latest_batch(
                table_name="market_data",
                symbols=batch,
                ts_col="timestamp"
            )
            if latest_records:
                for rec in latest_records:
                    sym = rec.get("symbol")
                    ts = rec.get("timestamp")
                    if sym and ts:
                        since_map[sym] = max(context.config.start_date, str(ts))

        context.stats.add_latency("reader_latency", time.perf_counter() - t_reader)

        # -------------------------------------------------------
        # STAGE 2: Provider Batch Fetch (With Smart Auto-Discovery)
        # -------------------------------------------------------
        self.monitor.log_stage(context, PipelineStage.PROVIDER_FETCH)
        provider = ProviderRegistry.get(context.config.provider_name)
        if not provider:
            raise ProviderError(f"Data provider '{context.config.provider_name}' is not securely registered in ProviderRegistry.")

        t_provider = time.perf_counter()
        raw_data = []

        if hasattr(provider, 'fetch_batch'):
            print("=" * 80)
            print("USING FETCH_BATCH")
            print(provider.__class__.__name__)
            print("=" * 80)
            raw_data = provider.fetch_batch(
                symbols=batch,
                since_map=since_map,
                end_date=context.config.end_date,
                timeframe=context.config.timeframe
            )
        else:
            self.logger.info(f"Provider {context.config.provider_name} does not support fetch_batch. Falling back to sequential smart fetch.")
            import inspect
            
            # Auto-discover the exact fetch method name in the provider class
            method_names = ['fetch_market_data', 'fetch_data', 'get_historical_data', 'get_history', 'equity_history', 'history', 'get_data', 'fetch']
            fetch_func = None
            for name in method_names:
                # Blacklist specific methods that are not for single-symbol fetch
                if name in ['download_equity_bhavcopy', 'download_bhavcopy', 'upload_data']:
                    continue
                if hasattr(provider, name):
                    fetch_func = getattr(provider, name)
                    break
            
            if not fetch_func:
                public_methods = [m for m in dir(provider) if callable(getattr(provider, m)) and not m.startswith('_') and m not in ['ping', 'health_check', 'supports_health_check']]
                fetch_func = getattr(provider, public_methods[0]) if public_methods else None

            for sym in batch:
                start_dt = since_map.get(sym, context.config.start_date)
                try:
                    if fetch_func:
                        print("="*80)
                        print("FETCH METHOD:", fetch_func.__qualname__)
                        print("FETCH MODULE:", fetch_func.__module__)
                        print("="*80)
                        sig = inspect.signature(fetch_func)
                        kwargs = {}
                        if 'symbol' in sig.parameters: kwargs['symbol'] = sym
                        elif 'ticker' in sig.parameters: kwargs['ticker'] = sym

                        if 'start_date' in sig.parameters: kwargs['start_date'] = start_dt
                        elif 'from_date' in sig.parameters: kwargs['from_date'] = start_dt
                        elif 'start' in sig.parameters: kwargs['start'] = start_dt
                        elif 'date_val' in sig.parameters: kwargs['date_val'] = start_dt
                        elif 'date' in sig.parameters: kwargs['date'] = start_dt

                        if 'end_date' in sig.parameters: kwargs['end_date'] = context.config.end_date
                        elif 'to_date' in sig.parameters: kwargs['to_date'] = context.config.end_date
                        elif 'end' in sig.parameters: kwargs['end'] = context.config.end_date

                        data = fetch_func(**kwargs)
                        print("="*80)
                        print("FETCH DEBUG")
                        print("symbol =", sym)
                        print("type   =", type(data))
                        try:
                            print(data.head())
                        except Exception:
                            print(data)
                        print("="*80)
                        if data:
                            if isinstance(data, list):
                                raw_data.extend(data)
                            else:
                                raw_data.append(data)
                    else:
                        self.logger.error(f"No valid fetch method found in {provider.__class__.__name__}")
                except Exception as ex:
                    import traceback
                    print("=" * 80)
                    print("FETCH FAILED")
                    print("symbol =", sym)
                    traceback.print_exc()
                    print("=" * 80)
                    self.logger.error(f"Failed to fetch {sym} from provider: {ex}")

        context.stats.add_latency("provider_latency", time.perf_counter() - t_provider)

        if not raw_data:
            self.logger.warning(f"No new market data retrieved from provider for batch. [Exec ID: {context.execution_id}]")
            context.mark_success(batch)
            return

        # -------------------------------------------------------
        # STAGE 3: Validation
        # -------------------------------------------------------
        self.monitor.log_stage(context, PipelineStage.VALIDATION)
        t_validator = time.perf_counter()

        if hasattr(self.validator, 'validate_batch'):
            validation_result = self.validator.validate_batch(raw_data)
            valid_records = validation_result.valid_records
            invalid_count = len(validation_result.invalid_records)
        else:
            valid_records = self.validator.validate(raw_data)
            invalid_count = len(raw_data) - len(valid_records)

        context.stats.add_latency("validator_latency", time.perf_counter() - t_validator)

        context.stats.increment("records_valid", len(valid_records))
        context.stats.increment("records_invalid", invalid_count)
        
        event_payload = MappingProxyType({"batch_size": len(batch), "valid_count": len(valid_records)})
        if hasattr(self.event_bus, 'publish_async'):
            self.event_bus.publish_async(Event("ValidationCompleted", "market_pipeline", event_payload))
        else:
            self.event_bus.publish(Event("ValidationCompleted", "market_pipeline", event_payload))

        if not valid_records:
            raise ValidationError("Data validation failed entirely for the batch. No valid records advanced.")

        # -------------------------------------------------------
        # STAGE 4: Compression Hook
        # -------------------------------------------------------
        self.monitor.log_stage(context, PipelineStage.COMPRESSION)
        valid_records = self.compression_service.compress_fields(valid_records, fields_to_compress=["summary", "reasoning", "explanation"])

        # -------------------------------------------------------
        # STAGE 5: Database Write (UnitOfWork fully managed here)
        # -------------------------------------------------------
        self.monitor.log_stage(context, PipelineStage.DATABASE_WRITE)
        t_writer = time.perf_counter()

        try:
            with self.uow:
                self.writer.write_market_data(valid_records)
                self.monitor.log_stage(context, PipelineStage.COMMIT)
        except Exception as e:
            self.monitor.log_stage(context, PipelineStage.ROLLBACK)
            raise DatabaseError(f"Database persistence sequence failed for batch. Error: {str(e)}") from e

        db_lat = time.perf_counter() - t_writer
        context.stats.add_latency("writer_latency", db_lat)
        context.stats.add_latency("db_latency", db_lat)
        context.stats.increment("records_written", len(valid_records))

        context.mark_success(batch)
        
        write_payload = MappingProxyType({"records_written": len(valid_records), "execution_id": context.execution_id})
        if hasattr(self.event_bus, 'publish_async'):
            self.event_bus.publish_async(Event("WriteCompleted", "market_pipeline", write_payload))
        else:
            self.event_bus.publish(Event("WriteCompleted", "market_pipeline", write_payload))
            
        self.logger.info(f"Pipeline execution completed flawlessly for batch. [Exec ID: {context.execution_id}]")


class PipelineCoordinator:
    def __init__(self, 
                 executor: PipelineExecutor, 
                 retry_policy: PipelineRetryPolicy, 
                 error_handler: PipelineErrorHandler):
        self.executor = executor
        self.retry_policy = retry_policy
        self.error_handler = error_handler
        self.logger = AppLogger(self.__class__.__name__)

    @trace_span(operation="coordinator.run_parallel", component="pipeline", kind=SpanKind.INTERNAL)
    def run_parallel(self, context: PipelineContext) -> PipelineResult:
        symbols = context.config.symbols
        batch_size = context.config.batch_size
        batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
        
        cpu_count = os.cpu_count() or 4
        optimal_workers = 1  # Forced to 1 for SQLite stability on Termux
        
        self.logger.info(f"Coordinating parallel processing: {len(symbols)} symbols into {len(batches)} batches using {optimal_workers} workers.")
        errors: List[Exception] = []
        
        for batch in batches:
            try:
                self._execute_batch_with_retry(context, batch)
            except Exception as e:
                self.logger.error(f"Batch coordination failed: {e}")
                errors.append(e)
                self.error_handler.handle(e, context)

        success = len(errors) == 0
        summary = PipelineSummary(
            pipeline_id=context.pipeline_id,
            execution_id=context.execution_id,
            total_processed=len(symbols),
            successful=len(context.success_symbols),
            failed=len(context.failed_symbols),
            success_symbols=context.success_symbols,
            failed_symbols=context.failed_symbols,
            retry_symbols=context.retry_symbols,
            execution_time=time.perf_counter() - context.start_time,
            statistics=context.stats.to_dict()
        )

        return PipelineResult(
            context=context,
            success=success,
            errors=errors,
            summary=summary
        )

    def _execute_batch_with_retry(self, context: PipelineContext, batch: List[str]) -> None:
        def _action():
            self.executor.process_batch(context, batch)
            
        self.retry_policy.execute_with_retry(action=_action, context=context, batch=batch)


# -----------------------------------------------------------------------------
# PIPELINE ADVANCED SCHEDULER & RUNNER
# -----------------------------------------------------------------------------
class PipelineRunner:
    def __init__(self,
                 coordinator: PipelineCoordinator,
                 lifecycle: PipelineLifecycle,
                 bootstrap: PipelineBootstrap,
                 scheduler: Type[Scheduler],
                 event_bus: EventBus):
        self.coordinator = coordinator
        self.lifecycle = lifecycle
        self.bootstrap = bootstrap
        self.scheduler = scheduler
        self.event_bus = event_bus
        self.logger = AppLogger(self.__class__.__name__)

    def run(self, config: PipelineConfig) -> PipelineResult:
        self.logger.info("PipelineRunner Execution Sequence Initiated.")
        self.event_bus.publish(Event(name="PipelineStarted", source="market_pipeline", payload= MappingProxyType({"config": config.__dict__})))
        
        self.bootstrap.verify_readiness()

        # -------------------------------------------------------
        # AUTO LOAD ACTIVE NSE UNIVERSE
        # -------------------------------------------------------
        if config.symbols is None:
            with DatabaseSession() as session:
                rows = session.db.fetch_all(
                    """
                    SELECT symbol
                    FROM stock_master
                    WHERE COALESCE(is_active, 1) = 1
                    ORDER BY symbol
                    """,

                )

            config.symbols = [
                row["symbol"] if hasattr(row, "keys") else row[0]
                for row in rows
            ]

            self.logger.info(
                f"Loaded {len(config.symbols)} active symbols from equity_master."
            )

        with self.lifecycle.manage(config) as context:
            result = self.coordinator.run_parallel(context)
            
        self.event_bus.publish(Event(name="PipelineCompleted", source="market_pipeline", payload= MappingProxyType({"status": "SUCCESS" if result.success else "FAILED"})))
        self.logger.info("PipelineRunner Sequence Concluded.")
        return result

    def schedule_market_close(self, config: PipelineConfig) -> None:
        """Triggers pipeline explicitly utilizing strictly mapped configuration or fallbacks."""
        cron_expr = getattr(settings.scheduler, "market_close", "45 15 * * 1-5")
        self.logger.info(f"Registering Market Close Trigger via cron: {cron_expr}")
        self.scheduler.add_job(self.run, args=(config,), cron=cron_expr, skip_holidays=True)

    def schedule_recovery(self, config: PipelineConfig) -> None:
        """Registers a fallback recovery pipeline designed for off-peak hours."""
        cron_expr = getattr(settings.scheduler, "market_recovery", "00 02 * * 2-6")
        self.logger.info(f"Registering Recovery Job via cron: {cron_expr}")
        self.scheduler.add_job(self.run, args=(config,), cron=cron_expr, skip_holidays=False)


# -----------------------------------------------------------------------------
# ENTERPRISE FACTORY
# -----------------------------------------------------------------------------
class PipelineFactory:
    @classmethod
    def create_production_pipeline(cls) -> PipelineRunner:
        import os
        # Force Absolute Path for Termux Compatibility
        db_abs = os.path.abspath("database/universe.db")
        db_url = f"sqlite:///{db_abs}"
        
        try:
            from backend.data.providers.nse import NSEConfig
            nse_cfg = NSEConfig(db_path=db_abs)
        except Exception:
            nse_cfg = None

        try:
            from backend.data.providers.yfinance import YahooFinanceConfig
            yf_cfg = YahooFinanceConfig()
        except Exception:
            yf_cfg = None

        try:
            from backend.data.market_reader import ReaderConfig
            reader_cfg = ReaderConfig(db_url=db_url)
        except Exception:
            reader_cfg = None

        try:
            from backend.data.market_writer import WriterConfig
            writer_cfg = WriterConfig(db_url=db_url)
        except Exception:
            writer_cfg = None

        ProviderRegistry.register("nse", NSEProvider(nse_cfg) if nse_cfg else NSEProvider())
        ProviderRegistry.register("yfinance", YahooFinanceProvider(yf_cfg) if yf_cfg else YahooFinanceProvider())
        
        event_bus = EventBus()
        scheduler = Scheduler()
        
        reader = MarketReader(reader_cfg) if reader_cfg else MarketReader()
        writer = MarketWriter(writer_cfg) if writer_cfg else MarketWriter()
        validator = MarketValidator()
        uow = UnitOfWork()
        compression_service = CompressionService()
        
        health_checker = PipelineHealth(db_session_cls=DatabaseSession)
        bootstrap = PipelineBootstrap(health_checker=health_checker)
        
        monitor = PipelineMonitor(event_bus=event_bus)
        hooks = PipelineHooks(monitor=monitor)
        lifecycle = PipelineLifecycle(hooks=hooks)
        
        error_handler = PipelineErrorHandler(event_bus=event_bus)
        retry_policy = PipelineRetryPolicy(monitor=monitor)
        
        executor = PipelineExecutor(
            reader=reader,
            validator=validator,
            writer=writer,
            uow=uow,
            monitor=monitor,
            event_bus=event_bus,
            compression_service=compression_service
        )
        
        coordinator = PipelineCoordinator(
            executor=executor,
            retry_policy=retry_policy,
            error_handler=error_handler
        )
        
        return PipelineRunner(
            coordinator=coordinator,
            lifecycle=lifecycle,
            bootstrap=bootstrap,
            scheduler=scheduler,
            event_bus=event_bus
        )

# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import datetime
    from backend.data.market_sync import MarketSync

    print("🌐 Auto Model: Syncing Latest NSE Universe...")
    # ১. পাইপলাইন চালুর আগেই সিস্টেম নিজে থেকে লেটেস্ট সিম্বল ডাটাবেসে আপডেট করে নেবে
    try:
        sync_engine = MarketSync()
        sync_engine.sync_universe()
    except Exception as e:
        print(f"⚠️ Universe Sync Error: {e}")

    pipeline = PipelineFactory.create_production_pipeline()

    config = PipelineConfig(
        start_date="2000-01-01",
        end_date=datetime.date.today().isoformat(),
        timeframe="1D",
        exchange="yfinance",  # <-- NSE এর বদলে yfinance (বা YAHOO) দিন, নাহলে Bhavcopy ক্র্যাশ করবে
        symbols=None,         # ডাটাবেস থেকে অটোমেটিক ২৩৮২টা স্টক লোড হবে
        batch_size=50,
        provider_name="yfinance"
    )

    print(f"🚀 Starting Full Sync for all symbols (Since 2000)...")
    
    # OS লেভেলে Thread ব্লক করা (যাতে Termux Segfault না দেয়)
    import os
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'

    result = pipeline.run(config)

    print("=" * 60)
    print("FULL SYNC COMPLETED")
    print("=" * 60)
    print(f"Success : {result.success}")
    if hasattr(result, 'summary') and result.summary:
        print(f"Processed : {result.summary.total_processed}")
