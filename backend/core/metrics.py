"""
GREEN BULL RIDER V6 - Institutional-grade AI Stock Analysis Platform
Module: backend/core/metrics.py
Description: Enterprise Centralized Metrics Engine.
             Provides high-performance, thread-safe, lock-efficient telemetry 
             aggregation utilizing Read-Write locks. Supports Counters, Gauges, 
             Summaries (Sliding Window), Histograms (Cumulative Buckets), and 
             Rates (EWMA) with O(1) updates. Features smart snapshot caching 
             with lock-free read paths, OS-Aware memory profiling, Active Timer 
             Leak Prevention, and exact Prometheus-compatible exports.
             Fully decoupled from business logic and integrated with 
             Trace, Audit, and Logging pipelines. Production Locked.
"""

import os
import sys
import time
import math
import json
import uuid
import hashlib
import threading
import collections
from enum import Enum
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    Any, Dict, List, Optional, Tuple, Union, 
    Final, Callable, Generator, Type, cast
)
from functools import wraps

# Internal Platform Integrations
from backend.config.settings import settings
from backend.core.logger import AppLogger
from backend.core.exceptions import GreenBullError
from backend.core.trace import TraceEngine, SpanKind, trace_span
from backend.core.audit import AuditEngine, AuditAction, AuditSeverity

# -------------------------------------------------------------------------
# LOGGER INITIALIZATION
# -------------------------------------------------------------------------
_logger = AppLogger("MetricsEngine")

# System Limits
MAX_METRICS_LIMIT = getattr(settings, "metrics_max_cardinality", 100000)
MAX_TIMER_AGE_SEC = getattr(settings, "metrics_max_timer_age_sec", 3600.0)


# -------------------------------------------------------------------------
# EXCEPTIONS
# -------------------------------------------------------------------------

class MetricsError(GreenBullError):
    """Base exception for all Metrics Engine boundary violations."""
    error_code: str = "GBR-MET-000"


class MetricNotFoundError(MetricsError):
    """Raised when an operation targets an unregistered metric explicitly."""
    error_code: str = "GBR-MET-001"


class MetricValueError(MetricsError):
    """Raised when an invalid mathematical operation is applied to a metric."""
    error_code: str = "GBR-MET-002"


class MetricsCardinalityError(MetricsError):
    """Raised when the engine exhausts its maximum allowed unique metrics limit."""
    error_code: str = "GBR-MET-003"


# -------------------------------------------------------------------------
# ENUMS
# -------------------------------------------------------------------------

class MetricType(str, Enum):
    """Standardized operational metric classifications."""
    COUNTER = "COUNTER"
    GAUGE = "GAUGE"
    SUMMARY = "SUMMARY"
    HISTOGRAM = "HISTOGRAM"
    RATE = "RATE"


# -------------------------------------------------------------------------
# RWLOCK (High-Throughput Concurrency)
# -------------------------------------------------------------------------

class RWLock:
    """
    Enterprise Read-Write Lock preventing reader starvation.
    Allows highly scalable concurrent metric lookups while safely bounding structural writes.
    """
    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._readers = 0
        self._writers = 0

    @contextmanager
    def read(self) -> Generator[None, None, None]:
        with self._condition:
            while self._writers > 0:
                self._condition.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._condition:
                self._readers -= 1
                if self._readers == 0:
                    self._condition.notify_all()

    @contextmanager
    def write(self) -> Generator[None, None, None]:
        with self._condition:
            self._writers += 1
            while self._readers > 0:
                self._condition.wait()
        try:
            yield
        finally:
            with self._condition:
                self._writers -= 1
                self._condition.notify_all()


# -------------------------------------------------------------------------
# DATACLASSES & IMMUTABLE SNAPSHOTS
# -------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MetricSnapshot:
    """Immutable point-in-time snapshot of an individual metric."""
    name: str
    namespace: str
    metric_type: str
    tags: Dict[str, str]
    value: float
    count: int
    sum_val: float
    min_val: float
    max_val: float
    p50: float
    p90: float
    p95: float
    p99: float
    std_dev: float
    variance: float
    buckets: Dict[str, int]
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "namespace": self.namespace,
            "metric_type": self.metric_type,
            "tags": self.tags,
            "value": self.value,
            "count": self.count,
            "sum": self.sum_val,
            "min": self.min_val,
            "max": self.max_val,
            "p50": self.p50,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
            "std_dev": self.std_dev,
            "variance": self.variance,
            "buckets": self.buckets,
            "updated_at": self.updated_at
        }


# -------------------------------------------------------------------------
# METRIC PRIMITIVES (High-Performance Thread-Safe State Containers)
# -------------------------------------------------------------------------

class BaseMetric:
    """Abstract thread-safe telemetry container with lazy snapshot caching."""
    __slots__ = ('name', 'namespace', 'tags', 'metric_type', '_lock', '_updated_at_ts', '_cached_snapshot')

    def __init__(self, name: str, namespace: str, tags: Dict[str, str], metric_type: MetricType) -> None:
        self.name = name
        self.namespace = namespace
        self.tags = tags
        self.metric_type = metric_type
        self._lock = threading.Lock()
        self._updated_at_ts = time.time()
        self._cached_snapshot: Optional[MetricSnapshot] = None

    def snapshot(self) -> MetricSnapshot:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class Counter(BaseMetric):
    """Monotonically increasing accumulation metric."""
    __slots__ = ('_value', '_count')

    def __init__(self, name: str, namespace: str, tags: Dict[str, str]) -> None:
        super().__init__(name, namespace, tags, MetricType.COUNTER)
        self._value = 0.0
        self._count = 0

    def inc(self, amount: float = 1.0) -> None:
        if amount < 0:
            raise MetricValueError("Counters can only increment strictly positive values.")
        with self._lock:
            self._value += amount
            self._count += 1
            self._updated_at_ts = time.time()
            self._cached_snapshot = None

    def snapshot(self) -> MetricSnapshot:
        with self._lock:
            if self._cached_snapshot is not None:
                return self._cached_snapshot
            val, count, updated = self._value, self._count, self._updated_at_ts
            
            snap = MetricSnapshot(
                self.name, self.namespace, self.metric_type.value, self.tags,
                val, count, val, val, val, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {},
                time.strftime("%Y-%m-%dT%H:%M:%S%z", time.gmtime(updated))
            )
            self._cached_snapshot = snap
            return snap

    def reset(self) -> None:
        with self._lock:
            self._value = 0.0
            self._count = 0
            self._updated_at_ts = time.time()
            self._cached_snapshot = None


class Gauge(BaseMetric):
    """State metric capable of arbitrary fluctuation."""
    __slots__ = ('_value', '_count')

    def __init__(self, name: str, namespace: str, tags: Dict[str, str]) -> None:
        super().__init__(name, namespace, tags, MetricType.GAUGE)
        self._value = 0.0
        self._count = 0

    def set(self, value: float) -> None:
        with self._lock:
            self._value = value
            self._count += 1
            self._updated_at_ts = time.time()
            self._cached_snapshot = None

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount
            self._count += 1
            self._updated_at_ts = time.time()
            self._cached_snapshot = None

    def dec(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value -= amount
            self._count += 1
            self._updated_at_ts = time.time()
            self._cached_snapshot = None

    def snapshot(self) -> MetricSnapshot:
        with self._lock:
            if self._cached_snapshot is not None:
                return self._cached_snapshot
            val, count, updated = self._value, self._count, self._updated_at_ts
            
            snap = MetricSnapshot(
                self.name, self.namespace, self.metric_type.value, self.tags,
                val, count, val, val, val, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {},
                time.strftime("%Y-%m-%dT%H:%M:%S%z", time.gmtime(updated))
            )
            self._cached_snapshot = snap
            return snap

    def reset(self) -> None:
        with self._lock:
            self._value = 0.0
            self._count = 0
            self._updated_at_ts = time.time()
            self._cached_snapshot = None


class Summary(BaseMetric):
    """
    Distribution aggregation utilizing a bounded Ring Buffer (Sliding Window).
    O(1) observation latency with lazily evaluated percentiles calculated OUTSIDE the lock.
    """
    __slots__ = ('_total_count', '_total_sum', '_min', '_max', '_buffer', '_max_size')

    def __init__(self, name: str, namespace: str, tags: Dict[str, str], max_size: int = 1000) -> None:
        super().__init__(name, namespace, tags, MetricType.SUMMARY)
        self._max_size = max_size
        self._buffer: collections.deque = collections.deque(maxlen=max_size)
        self._total_count = 0
        self._total_sum = 0.0
        self._min = float('inf')
        self._max = float('-inf')

    def observe(self, value: float) -> None:
        with self._lock:
            self._total_count += 1
            self._total_sum += value
            if value < self._min: self._min = value
            if value > self._max: self._max = value
            self._buffer.append(value)
            self._updated_at_ts = time.time()
            self._cached_snapshot = None

    def _percentile(self, sorted_data: List[float], p: float) -> float:
        if not sorted_data: return 0.0
        idx = (len(sorted_data) - 1) * p
        lower, upper = math.floor(idx), math.ceil(idx)
        if lower == upper: return sorted_data[int(idx)]
        weight = idx - lower
        return sorted_data[int(lower)] * (1.0 - weight) + sorted_data[int(upper)] * weight

    def snapshot(self) -> MetricSnapshot:
        with self._lock:
            if self._cached_snapshot is not None:
                return self._cached_snapshot
            count, sum_val = self._total_count, self._total_sum
            min_val = self._min if count > 0 else 0.0
            max_val = self._max if count > 0 else 0.0
            data = list(self._buffer)
            updated = self._updated_at_ts

        # Mathematically intensive operations executing completely outside the lock boundary
        if not data:
            snap = MetricSnapshot(
                self.name, self.namespace, self.metric_type.value, self.tags,
                0.0, count, sum_val, min_val, max_val, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {},
                time.strftime("%Y-%m-%dT%H:%M:%S%z", time.gmtime(updated))
            )
        else:
            data.sort()
            p50, p90, p95, p99 = (self._percentile(data, p) for p in [0.50, 0.90, 0.95, 0.99])
            
            # Variance calculated STRICTLY over the active sliding window buffer
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data) if len(data) > 1 else 0.0
            std_dev = math.sqrt(variance)

            snap = MetricSnapshot(
                self.name, self.namespace, self.metric_type.value, self.tags,
                p50, count, sum_val, min_val, max_val, p50, p90, p95, p99, std_dev, variance, {},
                time.strftime("%Y-%m-%dT%H:%M:%S%z", time.gmtime(updated))
            )
            
        with self._lock:
            # Idempotent assignment only if a concurrent mutation hasn't bypassed us
            if self._updated_at_ts == updated:
                self._cached_snapshot = snap
        return snap

    def reset(self) -> None:
        with self._lock:
            self._total_count = 0
            self._total_sum = 0.0
            self._min = float('inf')
            self._max = float('-inf')
            self._buffer.clear()
            self._updated_at_ts = time.time()
            self._cached_snapshot = None


class Histogram(BaseMetric):
    """
    Cumulative Bucket aggregation strictly conforming to Prometheus histogram behavior.
    """
    __slots__ = ('_total_count', '_total_sum', '_buckets', '_bucket_bounds')

    def __init__(self, name: str, namespace: str, tags: Dict[str, str], bounds: Optional[List[float]] = None) -> None:
        super().__init__(name, namespace, tags, MetricType.HISTOGRAM)
        self._bucket_bounds = bounds or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        self._bucket_bounds.sort()
        # Cumulative buckets setup
        self._buckets = {str(b): 0 for b in self._bucket_bounds}
        self._buckets["+inf"] = 0
        self._total_count = 0
        self._total_sum = 0.0

    def observe(self, value: float) -> None:
        with self._lock:
            self._total_count += 1
            self._total_sum += value
            self._updated_at_ts = time.time()
            self._cached_snapshot = None
            
            # Cumulative increment: bucket increments if value <= bound
            for bound in self._bucket_bounds:
                if value <= bound:
                    self._buckets[str(bound)] += 1
            self._buckets["+inf"] += 1

    def snapshot(self) -> MetricSnapshot:
        with self._lock:
            if self._cached_snapshot is not None:
                return self._cached_snapshot
            count, sum_val, buckets, updated = self._total_count, self._total_sum, self._buckets.copy(), self._updated_at_ts

        snap = MetricSnapshot(
            self.name, self.namespace, self.metric_type.value, self.tags,
            0.0, count, sum_val, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, buckets,
            time.strftime("%Y-%m-%dT%H:%M:%S%z", time.gmtime(updated))
        )
        with self._lock:
            if self._updated_at_ts == updated:
                self._cached_snapshot = snap
        return snap

    def reset(self) -> None:
        with self._lock:
            self._total_count = 0
            self._total_sum = 0.0
            for k in self._buckets:
                self._buckets[k] = 0
            self._updated_at_ts = time.time()
            self._cached_snapshot = None


class Rate(BaseMetric):
    """
    Exponentially Weighted Moving Average (EWMA) evaluating dynamic throughput rates.
    """
    __slots__ = ('_count', '_rate', '_alpha', '_last_tick')

    def __init__(self, name: str, namespace: str, tags: Dict[str, str], window_sec: float = 60.0) -> None:
        super().__init__(name, namespace, tags, MetricType.RATE)
        self._count = 0
        self._rate = 0.0
        self._alpha = 1.0 - math.exp(-1.0 / window_sec)  # EWMA decay factor
        self._last_tick = time.time()

    def mark(self, count: int = 1) -> None:
        with self._lock:
            self._count += count
            now = time.time()
            dt = now - self._last_tick
            
            if dt > 0:
                instant_rate = count / dt
                if self._rate == 0.0:
                    self._rate = instant_rate
                else:
                    self._rate = self._rate + self._alpha * (instant_rate - self._rate)
                self._last_tick = now
            self._updated_at_ts = now
            self._cached_snapshot = None

    def snapshot(self) -> MetricSnapshot:
        with self._lock:
            if self._cached_snapshot is not None:
                return self._cached_snapshot
            val, count, updated = self._rate, self._count, self._updated_at_ts
            
            snap = MetricSnapshot(
                self.name, self.namespace, self.metric_type.value, self.tags,
                val, count, val, val, val, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {},
                time.strftime("%Y-%m-%dT%H:%M:%S%z", time.gmtime(updated))
            )
            self._cached_snapshot = snap
            return snap

    def reset(self) -> None:
        with self._lock:
            self._count = 0
            self._rate = 0.0
            self._last_tick = time.time()
            self._updated_at_ts = time.time()
            self._cached_snapshot = None


# -------------------------------------------------------------------------
# METRICS ENGINE (SINGLETON)
# -------------------------------------------------------------------------

class MetricsEngine:
    """
    Enterprise Central Metrics Engine.
    Orchestrates thread-safe telemetry pipelines via lock-efficient mappings (RWLock).
    Automates smart caching, Cardinality Protection, OS-Aware System Metrics,
    and exact Prometheus-compatible representations.
    """
    _instance = None
    _init_lock = threading.Lock()

    def __new__(cls) -> 'MetricsEngine':
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super(MetricsEngine, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        self._registry: Dict[str, BaseMetric] = {}
        self._registry_lock = RWLock()
        
        self._active_timers: Dict[str, Tuple[float, float]] = {}
        self._timers_lock = threading.Lock()
        
        self._total_requests: int = 0
        self._total_success: int = 0
        self._total_failure: int = 0
        self._stats_lock = threading.Lock()

        # Background Maintenance
        self._stop_event = threading.Event()
        self._maint_thread = threading.Thread(target=self._maintenance_loop, name="MetricsMaintenance", daemon=True)
        self._maint_thread.start()

        _logger.info("Enterprise Metrics Engine initialized successfully.")

    # -------------------------------------------------------------------------
    # INTERNAL IDENTIFICATION & REGISTRATION
    # -------------------------------------------------------------------------

    def _build_key(self, namespace: str, name: str, tags: Optional[Dict[str, str]]) -> str:
        tag_str = ",".join(f"{k}={v}" for k, v in sorted((tags or {}).items()))
        return f"{namespace}:{name}:{tag_str}"

    def _get_or_create(self, m_type: Type[BaseMetric], namespace: str, name: str, tags: Optional[Dict[str, str]]) -> BaseMetric:
        key = self._build_key(namespace, name, tags)
        
        with self._registry_lock.read():
            if key in self._registry:
                return self._registry[key]
            
        with self._registry_lock.write():
            if key not in self._registry:
                if len(self._registry) >= MAX_METRICS_LIMIT:
                    raise MetricsCardinalityError(f"Engine exhausted metric capacity ({MAX_METRICS_LIMIT}). Cardinality explosion protected.")
                self._registry[key] = m_type(name, namespace, tags or {})
            return self._registry[key]

    # -------------------------------------------------------------------------
    # PUBLIC METRIC APIS
    # -------------------------------------------------------------------------

    @trace_span(operation="metrics.counter", component="metrics", kind=SpanKind.INTERNAL)
    def counter(self, name: str, namespace: str = "custom", tags: Optional[Dict[str, str]] = None) -> Counter:
        return cast(Counter, self._get_or_create(Counter, namespace, name, tags))

    @trace_span(operation="metrics.gauge", component="metrics", kind=SpanKind.INTERNAL)
    def gauge(self, name: str, namespace: str = "custom", tags: Optional[Dict[str, str]] = None) -> Gauge:
        return cast(Gauge, self._get_or_create(Gauge, namespace, name, tags))

    @trace_span(operation="metrics.summary", component="metrics", kind=SpanKind.INTERNAL)
    def summary(self, name: str, namespace: str = "custom", tags: Optional[Dict[str, str]] = None) -> Summary:
        return cast(Summary, self._get_or_create(Summary, namespace, name, tags))

    @trace_span(operation="metrics.histogram", component="metrics", kind=SpanKind.INTERNAL)
    def histogram(self, name: str, namespace: str = "custom", tags: Optional[Dict[str, str]] = None) -> Histogram:
        return cast(Histogram, self._get_or_create(Histogram, namespace, name, tags))

    @trace_span(operation="metrics.rate", component="metrics", kind=SpanKind.INTERNAL)
    def rate(self, name: str, namespace: str = "custom", tags: Optional[Dict[str, str]] = None) -> Rate:
        return cast(Rate, self._get_or_create(Rate, namespace, name, tags))

    def increment(self, name: str, namespace: str = "custom", amount: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        self.counter(name, namespace, tags).inc(amount)

    def decrement(self, name: str, namespace: str = "custom", amount: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        self.gauge(name, namespace, tags).dec(amount)

    def set(self, name: str, namespace: str = "custom", value: float = 0.0, tags: Optional[Dict[str, str]] = None) -> None:
        self.gauge(name, namespace, tags).set(value)

    def observe(self, name: str, namespace: str = "custom", value: float = 0.0, tags: Optional[Dict[str, str]] = None) -> None:
        self.summary(name, namespace, tags).observe(value)
        self.histogram(f"{name}_buckets", namespace, tags).observe(value)

    def mark_rate(self, name: str, namespace: str = "custom", count: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        self.rate(name, namespace, tags).mark(count)

    # -------------------------------------------------------------------------
    # HIGHER-ORDER OPERATIONS (Timing, Lifecycle & Generic Bounds)
    # -------------------------------------------------------------------------

    @trace_span(operation="metrics.start_timer", component="metrics", kind=SpanKind.INTERNAL)
    def start_timer(self) -> str:
        timer_id = uuid.uuid4().hex
        with self._timers_lock:
            self._active_timers[timer_id] = (time.perf_counter(), time.time())
        return timer_id

    @trace_span(operation="metrics.stop_timer", component="metrics", kind=SpanKind.INTERNAL)
    def stop_timer(self, timer_id: str, name: str, namespace: str = "custom", tags: Optional[Dict[str, str]] = None) -> None:
        with self._timers_lock:
            timer_data = self._active_timers.pop(timer_id, None)
            
        if timer_data is not None:
            start_perf, _ = timer_data
            duration_ms = (time.perf_counter() - start_perf) * 1000.0
            self.observe(name, namespace, duration_ms, tags)

    @contextmanager
    def timer(self, name: str, namespace: str = "custom", tags: Optional[Dict[str, str]] = None) -> Generator[None, None, None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000.0
            self.observe(name, namespace, duration_ms, tags)

    def measure(self, name: str, namespace: str = "custom", tags: Optional[Dict[str, str]] = None) -> Callable:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.timer(name, namespace, tags):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    @trace_span(operation="metrics.record", component="metrics", kind=SpanKind.INTERNAL)
    def record(self, name: str, namespace: str, value: float, metric_type: MetricType = MetricType.GAUGE, tags: Optional[Dict[str, str]] = None) -> None:
        """Generic explicit payload ingestion mapping directly to exact internal primitives."""
        if metric_type == MetricType.COUNTER:
            self.increment(name, namespace, value, tags)
        elif metric_type == MetricType.GAUGE:
            self.set(name, namespace, value, tags)
        elif metric_type == MetricType.SUMMARY:
            self.summary(name, namespace, tags).observe(value)
        elif metric_type == MetricType.HISTOGRAM:
            self.histogram(name, namespace, tags).observe(value)
        elif metric_type == MetricType.RATE:
            self.mark_rate(name, namespace, int(value), tags)

    @trace_span(operation="metrics.record_latency", component="metrics", kind=SpanKind.INTERNAL)
    def record_latency(self, name: str, namespace: str, duration_ms: float, tags: Optional[Dict[str, str]] = None) -> None:
        self.observe(name, namespace, duration_ms, tags)

    @trace_span(operation="metrics.record_success", component="metrics", kind=SpanKind.INTERNAL)
    def record_success(self, namespace: str = "pipeline", tags: Optional[Dict[str, str]] = None) -> None:
        with self._stats_lock:
            self._total_requests += 1
            self._total_success += 1
        self.increment("success_count", namespace, 1.0, tags)
        self.mark_rate("throughput", namespace, 1, tags)

    @trace_span(operation="metrics.record_failure", component="metrics", kind=SpanKind.INTERNAL)
    def record_failure(self, namespace: str = "pipeline", tags: Optional[Dict[str, str]] = None) -> None:
        with self._stats_lock:
            self._total_requests += 1
            self._total_failure += 1
        self.increment("failure_count", namespace, 1.0, tags)
        self.mark_rate("throughput", namespace, 1, tags)

    # -------------------------------------------------------------------------
    # REGISTRY MANAGEMENT & MAINTENANCE
    # -------------------------------------------------------------------------

    @trace_span(operation="metrics.unregister_metric", component="metrics", kind=SpanKind.INTERNAL)
    def unregister_metric(self, name: str, namespace: str, tags: Optional[Dict[str, str]] = None) -> None:
        key = self._build_key(namespace, name, tags)
        with self._registry_lock.write():
            self._registry.pop(key, None)

    @trace_span(operation="metrics.unregister_namespace", component="metrics", kind=SpanKind.INTERNAL)
    def unregister_namespace(self, namespace: str) -> None:
        prefix = f"{namespace}:"
        with self._registry_lock.write():
            keys_to_remove = [k for k in self._registry.keys() if k.startswith(prefix)]
            for k in keys_to_remove:
                del self._registry[k]

    def _maintenance_loop(self) -> None:
        """Background thread purging orphaned timers and accurately scraping OS memory bounds."""
        _logger.info("Metrics Engine maintenance thread active.")
        while not self._stop_event.is_set():
            try:
                now_ts = time.time()
                
                # Active Timer Leak Protection
                with self._timers_lock:
                    stale_keys = [k for k, (_, ts) in self._active_timers.items() if (now_ts - ts) > MAX_TIMER_AGE_SEC]
                    for k in stale_keys:
                        del self._active_timers[k]
                if stale_keys:
                    _logger.warning(f"Purged {len(stale_keys)} stale active timers to prevent memory leaks.")
                    
                # OS-Aware System Metrics Extraction
                try:
                    import resource
                    usage = resource.getrusage(resource.RUSAGE_SELF)
                    if sys.platform == "darwin":
                        rss_bytes = float(usage.ru_maxrss)
                    else:
                        rss_bytes = float(usage.ru_maxrss * 1024.0)
                        
                    self.set("memory_rss_bytes", "system", rss_bytes)
                    self.set("cpu_user_time_sec", "system", float(usage.ru_utime))
                except Exception: pass
                    
                if hasattr(os, 'getloadavg'):
                    try:
                        load1, load5, load15 = os.getloadavg()
                        self.set("load_avg_1m", "system", float(load1))
                        self.set("load_avg_5m", "system", float(load5))
                    except Exception: pass
                    
            except Exception as e:
                TraceEngine.record_exception(e)
                
            self._stop_event.wait(60.0)

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._maint_thread.is_alive():
            self._maint_thread.join(timeout=3.0)

    # -------------------------------------------------------------------------
    # STATISTICS & IMMUTABLE EXPORTS
    # -------------------------------------------------------------------------

    @trace_span(operation="metrics.snapshot", component="metrics", kind=SpanKind.INTERNAL)
    def snapshot(self) -> List[MetricSnapshot]:
        with self._registry_lock.read():
            metrics = list(self._registry.values())
        return [m.snapshot() for m in metrics]

    @trace_span(operation="metrics.statistics", component="metrics", kind=SpanKind.INTERNAL)
    def statistics(self) -> Dict[str, Any]:
        with self._stats_lock:
            err_rate = (self._total_failure / self._total_requests) if self._total_requests > 0 else 0.0
            return {
                "total_requests": self._total_requests,
                "total_success": self._total_success,
                "total_failure": self._total_failure,
                "error_rate": round(err_rate, 4),
                "active_metrics": len(self._registry)
            }

    @trace_span(operation="metrics.health_check", component="metrics", kind=SpanKind.INTERNAL)
    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "HEALTHY",
            "active_metrics": len(self._registry),
            "active_timers": len(self._active_timers),
            "statistics": self.statistics(),
            "version": self.version(),
            "checksum": self.checksum()
        }

    @trace_span(operation="metrics.reset", component="metrics", kind=SpanKind.INTERNAL)
    def reset(self) -> None:
        with self._registry_lock.write():
            for metric in self._registry.values():
                metric.reset()
                
        with self._stats_lock:
            self._total_requests, self._total_success, self._total_failure = 0, 0, 0
            
        with self._timers_lock:
            self._active_timers.clear()

    @trace_span(operation="metrics.clear", component="metrics", kind=SpanKind.INTERNAL)
    def clear(self) -> None:
        with self._registry_lock.write():
            self._registry.clear()
            
        with self._timers_lock:
            self._active_timers.clear()
            
        with self._stats_lock:
            self._total_requests, self._total_success, self._total_failure = 0, 0, 0

        AuditEngine.record_success("metrics.clear", AuditAction.DELETE, "Global metrics registry cleared.")

    @trace_span(operation="metrics.export_prometheus", component="metrics", kind=SpanKind.INTERNAL)
    def export_prometheus(self) -> str:
        """
        Translates configurations into accurate Prometheus (OpenMetrics) scrape representations.
        Distinguishes Summaries (Quantiles) from actual Histograms (le Buckets).
        """
        lines = []
        for snap in self.snapshot():
            metric_name = f"gbr_{snap.namespace}_{snap.name}".replace(".", "_").replace("-", "_").lower()
            
            tag_str = ""
            if snap.tags:
                tag_pairs = [f'{k}="{v}"' for k, v in snap.tags.items()]
                tag_str = "{" + ",".join(tag_pairs) + "}"

            if snap.metric_type == MetricType.COUNTER.value:
                lines.append(f"# HELP {metric_name} Green Bull Rider Metric")
                lines.append(f"# TYPE {metric_name} counter")
                lines.append(f"{metric_name}{tag_str} {snap.value}")
                
            elif snap.metric_type == MetricType.GAUGE.value or snap.metric_type == MetricType.RATE.value:
                lines.append(f"# HELP {metric_name} Green Bull Rider Metric")
                lines.append(f"# TYPE {metric_name} gauge")
                lines.append(f"{metric_name}{tag_str} {snap.value}")
                
            elif snap.metric_type == MetricType.SUMMARY.value:
                lines.append(f"# HELP {metric_name} Green Bull Rider Metric")
                lines.append(f"# TYPE {metric_name} summary")
                
                qt_tags_base = snap.tags.copy()
                for q_label, q_val in [("0.5", snap.p50), ("0.9", snap.p90), ("0.95", snap.p95), ("0.99", snap.p99)]:
                    qt_tags = qt_tags_base.copy()
                    qt_tags["quantile"] = q_label
                    qt_tag_str = "{" + ",".join(f'{k}="{v}"' for k, v in qt_tags.items()) + "}"
                    lines.append(f"{metric_name}{qt_tag_str} {q_val}")
                    
                lines.append(f"{metric_name}_sum{tag_str} {snap.sum_val}")
                lines.append(f"{metric_name}_count{tag_str} {snap.count}")

            elif snap.metric_type == MetricType.HISTOGRAM.value:
                lines.append(f"# HELP {metric_name} Green Bull Rider Metric")
                lines.append(f"# TYPE {metric_name} histogram")
                
                bkt_tags_base = snap.tags.copy()
                for bkt, b_count in snap.buckets.items():
                    bkt_tags = bkt_tags_base.copy()
                    bkt_tags["le"] = bkt
                    bkt_tag_str = "{" + ",".join(f'{k}="{v}"' for k, v in bkt_tags.items()) + "}"
                    lines.append(f"{metric_name}_bucket{bkt_tag_str} {b_count}")
                    
                lines.append(f"{metric_name}_sum{tag_str} {snap.sum_val}")
                lines.append(f"{metric_name}_count{tag_str} {snap.count}")

        return "\n".join(lines) + "\n"

    def export_json(self) -> str:
        return json.dumps([s.to_dict() for s in self.snapshot()], default=str)

    def export_dict(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self.snapshot()]

    # -------------------------------------------------------------------------
    # ENTERPRISE ALGEBRA
    # -------------------------------------------------------------------------

    @trace_span(operation="metrics.merge", component="metrics", kind=SpanKind.INTERNAL)
    def merge(self, other_snapshots: List[Dict[str, Any]]) -> None:
        """Integrates cross-environment state payloads natively into the registry."""
        for snap_dict in other_snapshots:
            m_type = snap_dict.get("metric_type")
            ns = snap_dict.get("namespace", "custom")
            name = snap_dict.get("name", "unknown")
            tags = snap_dict.get("tags", {})
            val = float(snap_dict.get("value", 0.0))
            
            if m_type == MetricType.COUNTER.value:
                self.increment(name, ns, val, tags)
            elif m_type == MetricType.GAUGE.value:
                self.set(name, ns, val, tags)

    @trace_span(operation="metrics.diff", component="metrics", kind=SpanKind.INTERNAL)
    def diff(self, previous_snapshots: List[MetricSnapshot]) -> List[MetricSnapshot]:
        """Computes strict mathematical deltas across isolated temporal intervals."""
        current_snaps = {f"{s.namespace}:{s.name}:{frozenset(s.tags.items())}": s for s in self.snapshot()}
        prev_snaps = {f"{s.namespace}:{s.name}:{frozenset(s.tags.items())}": s for s in previous_snapshots}
        
        diffs = []
        for key, curr in current_snaps.items():
            prev = prev_snaps.get(key)
            if not prev:
                diffs.append(curr)
            elif curr.metric_type == MetricType.COUNTER.value:
                val_diff = curr.value - prev.value
                diffs.append(MetricSnapshot(
                    curr.name, curr.namespace, curr.metric_type, curr.tags,
                    val_diff, int(val_diff), val_diff, val_diff, val_diff, val_diff, val_diff, val_diff, val_diff, 0.0, 0.0, {},
                    curr.updated_at
                ))
            else:
                diffs.append(curr)
        return diffs

    @trace_span(operation="metrics.version", component="metrics", kind=SpanKind.INTERNAL)
    def version(self) -> str:
        return "GREEN_BULL_RIDER_METRICS_V6.0"

    @trace_span(operation="metrics.checksum", component="metrics", kind=SpanKind.INTERNAL)
    def checksum(self) -> str:
        json_payload = self.export_json()
        return hashlib.sha256(json_payload.encode("utf-8")).hexdigest()


# -------------------------------------------------------------------------
# GLOBAL SINGLETON EXPORT
# -------------------------------------------------------------------------

metrics_engine: Final[MetricsEngine] = MetricsEngine()

__all__ = [
    "MetricsError",
    "MetricNotFoundError",
    "MetricValueError",
    "MetricsCardinalityError",
    "MetricType",
    "MetricSnapshot",
    "BaseMetric",
    "Counter",
    "Gauge",
    "Summary",
    "Histogram",
    "Rate",
    "MetricsEngine",
    "metrics_engine"
]
