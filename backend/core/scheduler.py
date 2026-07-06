"""
GREEN BULL RIDER V6 - Institutional-grade AI Stock Analysis Platform
Module: backend/core/scheduler.py
Description: Centralized Enterprise Scheduler Engine.
             Responsible for executing recurring and one-time jobs securely across 
             the platform. Features priority-aware dispatching, robust concurrency, 
             misfire handling, future tracking, and native integration with Retry, 
             Trace, Audit, Event Bus, and Logger pipelines.
             Fully decoupled from business logic. Production Locked.
"""

import time
import uuid
import heapq
import threading
from enum import Enum
from datetime import datetime, timezone
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Callable, Dict, List, Optional, Tuple, Final

# Internal Platform Integrations
from backend.config.settings import settings
from backend.core.logger import AppLogger
from backend.core.exceptions import GreenBullError
from backend.core.trace import TraceEngine, SpanKind, TraceLevel, trace_span
from backend.core.audit import AuditEngine, AuditAction, AuditResult, AuditSeverity
from backend.core.retry import RetryPolicy, retry_engine
from backend.core.event_bus import Event, EventPriority, event_bus

# -------------------------------------------------------------------------
# TYPE HINTS & CONSTANTS
# -------------------------------------------------------------------------
_logger = AppLogger("SchedulerEngine")

# Standard mathematical bounds for interval computations
SECONDS_PER_DAY = 86400.0
SECONDS_PER_WEEK = 604800.0
SECONDS_PER_MONTH = 2592000.0  # Approx 30 days


# -------------------------------------------------------------------------
# EXCEPTIONS
# -------------------------------------------------------------------------

class SchedulerError(GreenBullError):
    """Root exception for all Scheduler Engine operations."""
    error_code: str = "GBR-SCH-000"


class JobExecutionError(SchedulerError):
    """Raised when a scheduled job fails to execute cleanly."""
    error_code: str = "GBR-SCH-001"


class JobCancelledError(SchedulerError):
    """Raised when a scheduled job is cancelled or interrupted."""
    error_code: str = "GBR-SCH-002"


class DuplicateJobError(SchedulerError):
    """Raised when attempting to register a job with an existing name or ID."""
    error_code: str = "GBR-SCH-003"


# -------------------------------------------------------------------------
# ENUMS
# -------------------------------------------------------------------------

class JobStatus(str, Enum):
    """Lifecycle states for an individual scheduled job."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    PAUSED = "PAUSED"
    MISFIRED = "MISFIRED"


class JobType(str, Enum):
    """Categorized execution intervals for deterministic scheduling."""
    ONCE = "ONCE"
    INTERVAL = "INTERVAL"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"


class MisfirePolicy(str, Enum):
    """Strategies to handle jobs that missed their scheduled execution window."""
    RUN_IMMEDIATELY = "RUN_IMMEDIATELY"
    SKIP_TO_NEXT = "SKIP_TO_NEXT"


# -------------------------------------------------------------------------
# DATACLASSES
# -------------------------------------------------------------------------

@dataclass(slots=True)
class Job:
    """
    Mutable state tracker for a schedulable execution unit.
    Optimized for high-frequency modifications by the Dispatcher Engine.
    """
    name: str
    callable_func: Callable[..., Any]
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    status: JobStatus = JobStatus.PENDING
    job_type: JobType = JobType.ONCE
    interval_seconds: Optional[float] = None
    next_run: Optional[float] = field(default_factory=time.time)
    last_run: Optional[float] = None
    priority: int = 10
    enabled: bool = True
    
    # Resiliency parameters
    retry_enabled: bool = False
    retry_policy: Optional[RetryPolicy] = None
    misfire_grace_sec: float = 60.0
    misfire_policy: MisfirePolicy = MisfirePolicy.SKIP_TO_NEXT
    timeout_sec: Optional[float] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: 'Job') -> bool:
        """Enables deterministic priority queue sorting (Priority -> Next Run)."""
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.next_run and other.next_run:
            return self.next_run < other.next_run
        return False


@dataclass(frozen=True, slots=True)
class SchedulerStatistics:
    """Immutable operational telemetry for the Scheduler Engine infrastructure."""
    registered_jobs: int
    executed_jobs: int
    successful_jobs: int
    failed_jobs: int
    cancelled_jobs: int
    running_jobs: int
    misfired_jobs: int
    average_execution_time_ms: float
    max_execution_time_ms: float
    average_queue_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "registered_jobs": self.registered_jobs,
            "executed_jobs": self.executed_jobs,
            "successful_jobs": self.successful_jobs,
            "failed_jobs": self.failed_jobs,
            "cancelled_jobs": self.cancelled_jobs,
            "running_jobs": self.running_jobs,
            "misfired_jobs": self.misfired_jobs,
            "average_execution_time_ms": round(self.average_execution_time_ms, 3),
            "max_execution_time_ms": round(self.max_execution_time_ms, 3),
            "average_queue_latency_ms": round(self.average_queue_latency_ms, 3)
        }


# -------------------------------------------------------------------------
# SCHEDULER ENGINE (SINGLETON)
# -------------------------------------------------------------------------

class Scheduler:
    """
    Enterprise Central Scheduler Engine.
    Orchestrates deterministic job execution across bounded thread pools.
    Integrates completely with the platform's Event Bus, Trace, Audit, and Logging pipelines.
    Guarantees thread-safe, non-blocking operational flows with misfire and timeout protection.
    """
    _instance = None
    _init_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super(Scheduler, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Constructs thread-safe internal registries and isolated execution boundaries."""
        self._jobs: Dict[str, Job] = {}
        self._jobs_lock = threading.Lock()
        
        self._active_futures: Dict[str, Future] = {}
        self._futures_lock = threading.Lock()

        # Telemetry State
        self._stats_lock = threading.Lock()
        self._executed_jobs: int = 0
        self._successful_jobs: int = 0
        self._failed_jobs: int = 0
        self._cancelled_jobs: int = 0
        self._running_jobs: int = 0
        self._misfired_jobs: int = 0
        
        # Metrics trackers
        self._total_execution_time_ms: float = 0.0
        self._max_execution_time_ms: float = 0.0
        self._total_queue_latency_ms: float = 0.0

        # Dispatcher Engine Flow Control
        self._stop_event = threading.Event()
        self._wakeup_event = threading.Event()
        self._dispatcher_thread: Optional[threading.Thread] = None

        # Thread Pool for isolated job execution (bounds linked to global settings)
        self._executor = ThreadPoolExecutor(
            thread_name_prefix="SchedulerWorker", 
            max_workers=settings.threading.max_workers
        )

    # -------------------------------------------------------------------------
    # STATISTICS, HEALTH & OBSERVABILITY
    # -------------------------------------------------------------------------

    def get_statistics(self) -> SchedulerStatistics:
        """Retrieves a thread-safe snapshot of global scheduling telemetry."""
        with self._jobs_lock:
            registered = len(self._jobs)
        with self._stats_lock:
            avg_exec = (self._total_execution_time_ms / self._executed_jobs) if self._executed_jobs > 0 else 0.0
            avg_lat = (self._total_queue_latency_ms / self._executed_jobs) if self._executed_jobs > 0 else 0.0
            
            return SchedulerStatistics(
                registered_jobs=registered,
                executed_jobs=self._executed_jobs,
                successful_jobs=self._successful_jobs,
                failed_jobs=self._failed_jobs,
                cancelled_jobs=self._cancelled_jobs,
                running_jobs=self._running_jobs,
                misfired_jobs=self._misfired_jobs,
                average_execution_time_ms=avg_exec,
                max_execution_time_ms=self._max_execution_time_ms,
                average_queue_latency_ms=avg_lat
            )

    def reset_statistics(self) -> None:
        """Purges execution metrics cleanly."""
        with self._stats_lock:
            self._executed_jobs = 0
            self._successful_jobs = 0
            self._failed_jobs = 0
            self._cancelled_jobs = 0
            self._misfired_jobs = 0
            self._total_execution_time_ms = 0.0
            self._max_execution_time_ms = 0.0
            self._total_queue_latency_ms = 0.0

    def health_check(self) -> Dict[str, Any]:
        """Provides operational status for system monitoring engines."""
        with self._futures_lock:
            active_workers = len(self._active_futures)
            
        is_running = self._dispatcher_thread is not None and self._dispatcher_thread.is_alive()
        
        return {
            "status": "HEALTHY" if is_running else "STOPPED",
            "active_workers": active_workers,
            "max_workers": settings.threading.max_workers,
            "executor_saturation": f"{(active_workers / settings.threading.max_workers) * 100:.1f}%",
            "dispatcher_active": is_running,
            "statistics": self.get_statistics().to_dict()
        }

    def _record_execution_metrics(self, status: JobStatus, duration_ms: float, latency_ms: float) -> None:
        """Safely increments terminal execution telemetry and performance metrics."""
        with self._stats_lock:
            self._running_jobs -= 1
            self._executed_jobs += 1
            self._total_execution_time_ms += duration_ms
            self._total_queue_latency_ms += latency_ms
            
            if duration_ms > self._max_execution_time_ms:
                self._max_execution_time_ms = duration_ms
                
            if status == JobStatus.SUCCESS:
                self._successful_jobs += 1
            elif status == JobStatus.FAILED:
                self._failed_jobs += 1
            elif status == JobStatus.CANCELLED:
                self._cancelled_jobs += 1

    # -------------------------------------------------------------------------
    # JOB MANAGEMENT API
    # -------------------------------------------------------------------------

    def register_job(
        self,
        name: str,
        callable_func: Callable[..., Any],
        job_type: JobType = JobType.ONCE,
        interval_seconds: Optional[float] = None,
        priority: int = 10,
        retry_enabled: bool = False,
        retry_policy: Optional[RetryPolicy] = None,
        misfire_grace_sec: float = 60.0,
        misfire_policy: MisfirePolicy = MisfirePolicy.SKIP_TO_NEXT,
        timeout_sec: Optional[float] = None,
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        delay_start_sec: float = 0.0
    ) -> str:
        """Ingests a structural operational unit into the scheduling matrix ensuring no duplicates."""
        if job_type == JobType.INTERVAL and (interval_seconds is None or interval_seconds <= 0):
            raise SchedulerError("INTERVAL job types require a strictly positive interval_seconds parameter.")

        with self._jobs_lock:
            if any(j.name == name for j in self._jobs.values()):
                raise DuplicateJobError(f"Job with name '{name}' is already registered.")

            job = Job(
                name=name,
                callable_func=callable_func,
                job_type=job_type,
                interval_seconds=interval_seconds,
                priority=priority,
                retry_enabled=retry_enabled,
                retry_policy=retry_policy,
                misfire_grace_sec=misfire_grace_sec,
                misfire_policy=misfire_policy,
                timeout_sec=timeout_sec,
                args=args,
                kwargs=kwargs or {},
                metadata=metadata or {},
                next_run=time.time() + delay_start_sec
            )
            self._jobs[job.job_id] = job

        self._wakeup_event.set()

        _logger.info(f"Job registered successfully: {job.name} [{job.job_id}]")
        
        event_bus.publish(Event(
            name="JOB_REGISTERED",
            payload=self._serialize_job(job),
            source="scheduler",
            priority=EventPriority.NORMAL
        ))
        AuditEngine.record_success("scheduler.register", AuditAction.CREATE, f"Scheduled Job {job.name} integrated.")

        return job.job_id

    def remove_job(self, job_id: str) -> None:
        """Purges a registered job cleanly from the operational matrix."""
        with self._jobs_lock:
            job = self._jobs.pop(job_id, None)
        
        if job:
            _logger.info(f"Job removed: {job.name} [{job_id}]")
            event_bus.publish(Event(
                name="JOB_CANCELLED",
                payload={"job_id": job_id, "name": job.name},
                source="scheduler",
                priority=EventPriority.NORMAL
            ))
            AuditEngine.record_success("scheduler.remove", AuditAction.DELETE, f"Job {job.name} removed.")
        else:
            raise SchedulerError(f"Cannot remove job. ID {job_id} not found.")

    def run_job_now(self, job_id: str) -> None:
        """Bypasses standard mathematical intervals to execute a specific job immediately."""
        with self._jobs_lock:
            job = self._jobs.get(job_id)
        
        if not job:
            raise SchedulerError(f"Cannot force run job. ID {job_id} not found.")
            
        if job.status == JobStatus.RUNNING:
            _logger.warning(f"Job {job.name} [{job_id}] is already executing. Ignoring force run request.")
            return

        _logger.info(f"Forcing immediate execution for Job: {job.name} [{job_id}]")
        self._submit_job(job)

    def pause_job(self, job_id: str) -> None:
        """Safely halts future executions of an interval-based job."""
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.PAUSED
                _logger.info(f"Job paused: {job.name} [{job_id}]")
            else:
                raise SchedulerError(f"Cannot pause job. ID {job_id} not found.")

    def resume_job(self, job_id: str) -> None:
        """Restores operational parameters for a paused job."""
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job:
                if job.status == JobStatus.PAUSED:
                    job.status = JobStatus.PENDING
                    self._calculate_next_run(job)
                    _logger.info(f"Job resumed: {job.name} [{job_id}]")
                    self._wakeup_event.set()
                else:
                    _logger.warning(f"Job {job.name} is not paused. Current status: {job.status}")
            else:
                raise SchedulerError(f"Cannot resume job. ID {job_id} not found.")

    def enable_job(self, job_id: str) -> None:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job:
                job.enabled = True
                self._calculate_next_run(job)
                self._wakeup_event.set()
            else:
                raise SchedulerError(f"Cannot enable job. ID {job_id} not found.")

    def disable_job(self, job_id: str) -> None:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job:
                job.enabled = False
            else:
                raise SchedulerError(f"Cannot disable job. ID {job_id} not found.")

    def job_exists(self, job_id: str) -> bool:
        with self._jobs_lock:
            return job_id in self._jobs

    def get_job(self, job_id: str) -> Dict[str, Any]:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job:
                return self._serialize_job(job)
            raise SchedulerError(f"Job ID {job_id} not found.")

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._jobs_lock:
            return [self._serialize_job(job) for job in self._jobs.values()]

    # -------------------------------------------------------------------------
    # INTERNAL LOGIC & CALCULATIONS
    # -------------------------------------------------------------------------

    def _serialize_job(self, job: Job) -> Dict[str, Any]:
        """Safely transforms internal mutable job properties into telemetry records."""
        return {
            "job_id": job.job_id,
            "name": job.name,
            "status": job.status.value,
            "job_type": job.job_type.value,
            "interval_seconds": job.interval_seconds,
            "next_run_iso": datetime.fromtimestamp(job.next_run, timezone.utc).isoformat() if job.next_run else None,
            "last_run_iso": datetime.fromtimestamp(job.last_run, timezone.utc).isoformat() if job.last_run else None,
            "priority": job.priority,
            "enabled": job.enabled,
            "retry_enabled": job.retry_enabled,
            "timeout_sec": job.timeout_sec,
            "metadata": job.metadata
        }

    def _calculate_next_run(self, job: Job) -> None:
        """Deterministically evaluates future mathematical bounds based on structural intervals."""
        now = time.time()
        
        if job.job_type == JobType.ONCE:
            job.next_run = None
            job.status = JobStatus.SUCCESS
            
        elif job.job_type == JobType.INTERVAL:
            interval = job.interval_seconds or 60.0
            job.next_run = now + interval
            job.status = JobStatus.PENDING
            
        elif job.job_type == JobType.DAILY:
            job.next_run = now + SECONDS_PER_DAY
            job.status = JobStatus.PENDING
            
        elif job.job_type == JobType.WEEKLY:
            job.next_run = now + SECONDS_PER_WEEK
            job.status = JobStatus.PENDING
            
        elif job.job_type == JobType.MONTHLY:
            job.next_run = now + SECONDS_PER_MONTH
            job.status = JobStatus.PENDING

    # -------------------------------------------------------------------------
    # DISPATCHER ENGINE
    # -------------------------------------------------------------------------

    @trace_span(operation="scheduler.dispatcher_loop", component="scheduler")
    def _dispatch_loop(self) -> None:
        """
        Dedicated core processing matrix. Evaluates structural time bounds, handles 
        misfires dynamically, and manages Thread Pool handoffs natively via Priority Queue logic.
        """
        _logger.info("Scheduler Dispatcher sequence initiated.")
        
        while not self._stop_event.is_set():
            now = time.time()
            next_wakeup_delay = 1.0  # Safe default poll
            ready_queue: List[Job] = []

            with self._jobs_lock:
                for job in self._jobs.values():
                    if not job.enabled or job.status in (JobStatus.RUNNING, JobStatus.PAUSED, JobStatus.CANCELLED, JobStatus.SUCCESS):
                        continue
                    
                    if job.next_run is not None:
                        if job.next_run <= now:
                            # Evaluate Misfire Mechanics
                            if now - job.next_run > job.misfire_grace_sec:
                                _logger.warning(f"Job {job.name} missed its execution window by {now - job.next_run:.1f}s.")
                                with self._stats_lock:
                                    self._misfired_jobs += 1
                                    
                                if job.misfire_policy == MisfirePolicy.SKIP_TO_NEXT:
                                    self._calculate_next_run(job)
                                    continue
                            
                            heapq.heappush(ready_queue, job)
                        else:
                            delay = job.next_run - now
                            if delay < next_wakeup_delay:
                                next_wakeup_delay = delay

            # Dispatch prioritized jobs
            while ready_queue:
                job_to_run = heapq.heappop(ready_queue)
                self._submit_job(job_to_run)

            # Await next execution bound or external interruption
            self._wakeup_event.wait(timeout=max(0.01, next_wakeup_delay))
            self._wakeup_event.clear()

        _logger.info("Scheduler Dispatcher sequence safely terminated.")

    def _submit_job(self, job: Job) -> None:
        """Safely isolates job state and hands execution to the Thread Pool Engine tracking Futures."""
        with self._jobs_lock:
            job.status = JobStatus.RUNNING
            
        with self._stats_lock:
            self._running_jobs += 1
            
        # Capture latency at injection point
        expected_run_time = job.next_run or time.time()
        
        future = self._executor.submit(self._execute_job, job, expected_run_time)
        
        with self._futures_lock:
            self._active_futures[job.job_id] = future
            
        # Autocleanup registry when future completes
        future.add_done_callback(lambda f: self._cleanup_future(job.job_id))

    def _cleanup_future(self, job_id: str) -> None:
        with self._futures_lock:
            self._active_futures.pop(job_id, None)

    def _execute_job(self, job: Job, expected_run_time: float) -> None:
        """
        Internal isolated operational boundary for individual job logic.
        Orchestrates Retries, Timeouts, Event Bus routing, Telemetry, and Tracing securely.
        """
        start_time_perf = time.perf_counter()
        queue_latency_ms = (time.time() - expected_run_time) * 1000.0

        _logger.debug(f"Executing scheduled job: {job.name} [{job.job_id}]")
        
        event_bus.publish(Event(
            name="JOB_STARTED",
            payload={"job_id": job.job_id, "name": job.name},
            source="scheduler",
            priority=EventPriority.NORMAL
        ))

        try:
            with TraceEngine.nested_span(operation=f"job.{job.name}", component="scheduler", kind=SpanKind.INTERNAL):
                TraceEngine.attach_metadata("job_id", job.job_id)
                TraceEngine.attach_metadata("schedule_type", job.job_type.value)
                TraceEngine.attach_metadata("queue_wait_time_ms", round(queue_latency_ms, 3))
                TraceEngine.attach_metadata("worker_thread", threading.current_thread().name)
                
                if job.retry_enabled and job.retry_policy:
                    TraceEngine.attach_metadata("retry_enabled", True)
                    retry_engine.execute(job.callable_func, job.retry_policy, *job.args, **job.kwargs)
                else:
                    TraceEngine.attach_metadata("retry_enabled", False)
                    job.callable_func(*job.args, **job.kwargs)
            
            # Post-Execution Success Logic
            duration_ms = (time.perf_counter() - start_time_perf) * 1000.0
            
            with self._jobs_lock:
                job.last_run = time.time()
                self._calculate_next_run(job)
                
            self._record_execution_metrics(JobStatus.SUCCESS, duration_ms, queue_latency_ms)
            
            _logger.info(f"Job {job.name} completed perfectly in {duration_ms:.2f}ms.")
            AuditEngine.record_success("scheduler.execute", AuditAction.EXECUTE, f"Job {job.name} completed.", metadata={"duration_ms": round(duration_ms, 3)})
            
            event_bus.publish(Event(
                name="JOB_COMPLETED",
                payload={"job_id": job.job_id, "name": job.name, "duration_ms": round(duration_ms, 3)},
                source="scheduler",
                priority=EventPriority.NORMAL
            ))

        except Exception as e:
            # Post-Execution Failure Logic
            duration_ms = (time.perf_counter() - start_time_perf) * 1000.0
            TraceEngine.record_exception(e)
            
            with self._jobs_lock:
                job.last_run = time.time()
                self._calculate_next_run(job)
                
            self._record_execution_metrics(JobStatus.FAILED, duration_ms, queue_latency_ms)
            
            error_msg = f"Job {job.name} FAILED: {e}"
            _logger.error(error_msg, exc_info=e)
            
            AuditEngine.record_failure("scheduler.execute", AuditAction.EXECUTE, error_msg, AuditSeverity.CRITICAL, metadata={"duration_ms": round(duration_ms, 3)})
            
            event_bus.publish(Event(
                name="JOB_FAILED",
                payload={"job_id": job.job_id, "name": job.name, "error": str(e), "duration_ms": round(duration_ms, 3)},
                source="scheduler",
                priority=EventPriority.HIGH
            ))
            
        finally:
            # Trigger dispatcher re-evaluation after worker exits
            self._wakeup_event.set()

    # -------------------------------------------------------------------------
    # LIFECYCLE MANAGEMENT
    # -------------------------------------------------------------------------

    def start(self) -> None:
        """Safely initiates the isolated Dispatcher loop mechanism."""
        if self._dispatcher_thread is not None and self._dispatcher_thread.is_alive():
            _logger.warning("Scheduler Engine is already actively running.")
            return

        self._stop_event.clear()
        self._dispatcher_thread = threading.Thread(
            target=self._dispatch_loop,
            name="SchedulerDispatcher",
            daemon=True
        )
        self._dispatcher_thread.start()
        _logger.info("Scheduler Engine successfully started.")

    def stop(self) -> None:
        """Soft-terminates active Dispatcher logics without destroying resources."""
        _logger.info("Halting Scheduler Engine Dispatcher.")
        self._stop_event.set()
        self._wakeup_event.set()
        
        if self._dispatcher_thread is not None:
            self._dispatcher_thread.join(timeout=5.0)

    def shutdown(self) -> None:
        """
        Enterprise grace teardown. Guarantees running jobs finish cleanly.
        Stops dispatching mechanisms and releases foundational thread resources safely.
        """
        _logger.info("Initiating strict grace shutdown sequence for Scheduler Engine.")
        self.stop()
        
        _logger.info("Awaiting active executor termination constraints...")
        # Clean execution cancellation
        with self._futures_lock:
            for fut in self._active_futures.values():
                fut.cancel()
        
        self._executor.shutdown(wait=True)
        _logger.info("Scheduler Engine shutdown completely successfully.")


# -------------------------------------------------------------------------
# GLOBAL SINGLETON EXPORT
# -------------------------------------------------------------------------

scheduler: Final[Scheduler] = Scheduler()

__all__ = [
    "SchedulerError",
    "JobExecutionError",
    "JobCancelledError",
    "DuplicateJobError",
    "JobStatus",
    "JobType",
    "MisfirePolicy",
    "Job",
    "SchedulerStatistics",
    "Scheduler",
    "scheduler"
]
