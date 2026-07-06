"""
GREEN BULL RIDER V6 - Institutional-grade AI Stock Analysis Platform
Module: backend/core/cache.py
Description: Enterprise-grade in-memory cache engine.
             Provides high-performance, thread-safe, and deterministic caching 
             with support for namespaces, TTL expiration via Min-Heap, strict LRU/FIFO 
             eviction policies, deep memory sizing, and bulk operational efficiency.
             Fully decoupled from business logic. Production Locked.
"""

import sys
import time
import threading
import collections
import re
import heapq
from enum import Enum
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, 
    Final, Union, Set
)
from functools import wraps

# Internal Platform Integrations
from backend.config.settings import settings
from backend.core.logger import AppLogger
from backend.core.exceptions import GreenBullError
from backend.core.trace import TraceEngine, SpanKind, trace_span
from backend.core.audit import AuditEngine, AuditAction, AuditResult, AuditSeverity

# -------------------------------------------------------------------------
# LOGGER INITIALIZATION
# -------------------------------------------------------------------------
_logger = AppLogger("CacheEngine")


# -------------------------------------------------------------------------
# EXCEPTIONS
# -------------------------------------------------------------------------

class CacheError(GreenBullError):
    """Base exception for all Cache Engine boundaries."""
    error_code: str = "GBR-CAC-000"


class CacheMissError(CacheError):
    """Raised when an explicit cache retrieval fails to locate a key."""
    error_code: str = "GBR-CAC-001"


class CacheExpiredError(CacheError):
    """Raised when an accessed cache entry exists but has exceeded its TTL."""
    error_code: str = "GBR-CAC-002"


class CacheMemoryError(CacheError):
    """Raised when the cache engine exhausts designated memory or item limits."""
    error_code: str = "GBR-CAC-003"


# -------------------------------------------------------------------------
# ENUMS
# -------------------------------------------------------------------------

class CachePolicy(str, Enum):
    """Eviction strategies governing cache lifecycle under memory pressure."""
    LRU = "LRU"            # Least Recently Used (moves to end on access)
    FIFO = "FIFO"          # First In First Out (strict insertion order)
    TTL_ONLY = "TTL_ONLY"  # No active eviction based on size, strictly time-based


# -------------------------------------------------------------------------
# DATACLASSES
# -------------------------------------------------------------------------

@dataclass(slots=True)
class CacheEntry:
    """Mutable state tracker for a single cached value."""
    key: str
    value: Any
    namespace: str
    size_bytes: int
    ttl_seconds: Optional[float]
    created_at_perf: float = field(default_factory=time.perf_counter)
    created_at_timestamp: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    @property
    def expires_at(self) -> Optional[float]:
        if self.ttl_seconds is None:
            return None
        return self.created_at_timestamp + self.ttl_seconds

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() >= self.expires_at


@dataclass(slots=True)
class NamespaceStatistics:
    """Isolated telemetry tracking per domain namespace."""
    current_entries: int = 0
    memory_usage_bytes: int = 0
    total_hits: int = 0
    total_misses: int = 0
    evictions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_entries": self.current_entries,
            "memory_usage_bytes": self.memory_usage_bytes,
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "evictions": self.evictions
        }


@dataclass(slots=True)
class CacheStatistics:
    """Mutable telemetry tracking for the global cache matrix."""
    total_hits: int = 0
    total_misses: int = 0
    evictions: int = 0
    expired_entries: int = 0
    cleanup_runs: int = 0
    
    current_entries: int = 0
    memory_usage_bytes: int = 0

    @property
    def hit_ratio(self) -> float:
        total = self.total_hits + self.total_misses
        return self.total_hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "hit_ratio": round(self.hit_ratio, 4),
            "evictions": self.evictions,
            "expired_entries": self.expired_entries,
            "cleanup_runs": self.cleanup_runs,
            "current_entries": self.current_entries,
            "memory_usage_bytes": self.memory_usage_bytes
        }


# -------------------------------------------------------------------------
# UTILITIES
# -------------------------------------------------------------------------

def _deterministic_repr(obj: Any) -> str:
    """Recursively generates a stable, deterministic representation of an object for caching keys."""
    if isinstance(obj, (int, float, str, bool, type(None))):
        return repr(obj)
    elif isinstance(obj, (list, tuple)):
        return "[" + ",".join(_deterministic_repr(i) for i in obj) + "]"
    elif isinstance(obj, dict):
        return "{" + ",".join(f"{_deterministic_repr(k)}:{_deterministic_repr(v)}" 
                             for k, v in sorted(obj.items(), key=lambda x: str(x[0]))) + "}"
    elif isinstance(obj, (set, frozenset)):
        return "{" + ",".join(sorted(_deterministic_repr(i) for i in obj)) + "}"
    else:
        return f"<{type(obj).__name__}_{id(obj)}>"


# -------------------------------------------------------------------------
# CACHE ENGINE (SINGLETON)
# -------------------------------------------------------------------------

class CacheEngine:
    """
    Enterprise Central In-Memory Cache Engine.
    Delivers O(1) thread-safe data persistence with integrated telemetry,
    configurable lifecycle policies, deep memory inspection, 
    and Min-Heap based smart background maintenance.
    """
    _instance = None
    _init_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super(CacheEngine, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Constructs safe internal registries, metrics, and background structures."""
        self._data: collections.OrderedDict[str, CacheEntry] = collections.OrderedDict()
        self._expiry_heap: List[Tuple[float, str]] = []
        
        self._lock = threading.RLock()
        self._stats = CacheStatistics()
        self._namespace_stats: Dict[str, NamespaceStatistics] = collections.defaultdict(NamespaceStatistics)

        self.policy: CachePolicy = CachePolicy(getattr(settings, "cache_policy", "LRU"))
        self.max_items: int = getattr(settings, "cache_max_items", 100000)
        self.max_memory_bytes: int = getattr(settings, "cache_max_memory_bytes", 512 * 1024 * 1024)
        self.cleanup_interval_sec: float = getattr(settings, "cache_cleanup_interval", 60.0)
        
        self._stop_event = threading.Event()
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, name="CacheCleanup", daemon=True)
        self._cleanup_thread.start()

        _logger.info("Enterprise Cache Engine initialized.", metadata={"policy": self.policy.value, "max_memory_mb": self.max_memory_bytes // 1048576})

    # -------------------------------------------------------------------------
    # INTERNAL MECHANICS
    # -------------------------------------------------------------------------

    def _build_key(self, namespace: str, key: str) -> str:
        return f"{namespace}:{key}"

    def _approximate_size(self, obj: Any, seen: Optional[Set[int]] = None) -> int:
        """Recursively calculates deep memory footprint avoiding cyclical references."""
        if seen is None:
            seen = set()
            
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)
        
        try:
            size = sys.getsizeof(obj)
        except TypeError:
            size = 64
            
        if isinstance(obj, dict):
            size += sum(self._approximate_size(k, seen) + self._approximate_size(v, seen) for k, v in obj.items())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            size += sum(self._approximate_size(i, seen) for i in obj)
            
        return size

    def _compact_heap(self) -> None:
        """Purges orphaned elements from the min-heap to prevent bloat during high touch/set rates."""
        with self._lock:
            new_heap = []
            for full_key, entry in self._data.items():
                if entry.expires_at is not None:
                    new_heap.append((entry.expires_at, full_key))
            heapq.heapify(new_heap)
            self._expiry_heap = new_heap

    def _internal_delete_accounting(self, entry: CacheEntry) -> None:
        self._stats.current_entries -= 1
        self._stats.memory_usage_bytes -= entry.size_bytes
        
        ns_stats = self._namespace_stats[entry.namespace]
        ns_stats.current_entries -= 1
        ns_stats.memory_usage_bytes -= entry.size_bytes
        if ns_stats.current_entries <= 0:
            self._namespace_stats.pop(entry.namespace, None)

    def _enforce_eviction(self, required_bytes: int) -> None:
        if self.policy == CachePolicy.TTL_ONLY:
            if (self._stats.current_entries >= self.max_items) or \
               (self._stats.memory_usage_bytes + required_bytes > self.max_memory_bytes):
                raise CacheMemoryError("Cache memory exhausted under TTL_ONLY policy.")
            return

        while (self._stats.current_entries >= self.max_items) or \
              (self._stats.memory_usage_bytes + required_bytes > self.max_memory_bytes):
            if not self._data:
                break
            
            key, entry = self._data.popitem(last=False)
            self._internal_delete_accounting(entry)
            self._stats.evictions += 1
            self._namespace_stats[entry.namespace].evictions += 1
            
            _logger.debug(f"Evicted key {key} to maintain memory bounds.")

        if self._stats.memory_usage_bytes + required_bytes > self.max_memory_bytes:
            raise CacheMemoryError(f"Single payload ({required_bytes}B) exceeds total cache bounds.")

    # -------------------------------------------------------------------------
    # INTERNAL PRIMITIVES (Lock assumed acquired by caller)
    # -------------------------------------------------------------------------

    def _set_internal(self, full_key: str, value: Any, namespace: str, ttl: Optional[float]) -> int:
        """Internal execution core for injections. Returns consumed byte size."""
        size = self._approximate_size(value)
        if full_key in self._data:
            old_entry = self._data.pop(full_key)
            self._internal_delete_accounting(old_entry)

        self._enforce_eviction(required_bytes=size)

        entry = CacheEntry(
            key=full_key,
            value=value,
            namespace=namespace,
            size_bytes=size,
            ttl_seconds=ttl
        )
        
        self._data[full_key] = entry
        self._stats.current_entries += 1
        self._stats.memory_usage_bytes += size
        self._namespace_stats[namespace].current_entries += 1
        self._namespace_stats[namespace].memory_usage_bytes += size
        
        if entry.expires_at is not None:
            heapq.heappush(self._expiry_heap, (entry.expires_at, full_key))
            # Trigger compaction if heap bloats exponentially
            if len(self._expiry_heap) > max(1000, self._stats.current_entries * 2):
                self._compact_heap()
                
        return size

    def _get_internal(self, full_key: str, namespace: str) -> Tuple[bool, Any, bool]:
        """Internal core for retrievals. Returns (is_hit, value, is_expired)."""
        entry = self._data.get(full_key)
        
        if entry is None:
            self._stats.total_misses += 1
            self._namespace_stats[namespace].total_misses += 1
            return False, None, False

        if entry.is_expired:
            self._stats.expired_entries += 1
            self._stats.total_misses += 1
            self._namespace_stats[namespace].total_misses += 1
            del self._data[full_key]
            self._internal_delete_accounting(entry)
            return False, None, True

        self._stats.total_hits += 1
        self._namespace_stats[namespace].total_hits += 1
        entry.last_accessed = time.time()
        
        if self.policy == CachePolicy.LRU:
            self._data.move_to_end(full_key)
            
        return True, entry.value, False

    def _delete_internal(self, full_key: str) -> bool:
        """Internal core for removals."""
        entry = self._data.pop(full_key, None)
        if entry:
            self._internal_delete_accounting(entry)
            return True
        return False

    # -------------------------------------------------------------------------
    # PUBLIC APIS
    # -------------------------------------------------------------------------

    @trace_span(operation="cache.set", component="cache", kind=SpanKind.INTERNAL)
    def set(self, key: str, value: Any, namespace: str = "default", ttl: Optional[float] = None) -> None:
        full_key = self._build_key(namespace, key)
        start_time = time.perf_counter()

        try:
            with self._lock:
                size = self._set_internal(full_key, value, namespace, ttl)

            duration = (time.perf_counter() - start_time) * 1000.0
            TraceEngine.attach_metadata("cache_key", full_key)
            TraceEngine.attach_metadata("namespace", namespace)
            TraceEngine.attach_metadata("cache_size", size)
            TraceEngine.attach_metadata("execution_time_ms", round(duration, 3))
            
            _logger.debug("Cache entry established.", metadata={"key": full_key, "size_bytes": size})
        except Exception as e:
            TraceEngine.record_exception(e)
            raise CacheError(f"Failed to set cache entry: {e}") from e

    @trace_span(operation="cache.get", component="cache", kind=SpanKind.INTERNAL)
    def get(self, key: str, namespace: str = "default", raise_on_miss: bool = False, default: Any = None) -> Any:
        full_key = self._build_key(namespace, key)
        start_time = time.perf_counter()
        
        with self._lock:
            is_hit, val, is_expired = self._get_internal(full_key, namespace)

        duration = (time.perf_counter() - start_time) * 1000.0
        TraceEngine.attach_metadata("cache_key", full_key)
        TraceEngine.attach_metadata("namespace", namespace)
        TraceEngine.attach_metadata("cache_hit", is_hit)
        TraceEngine.attach_metadata("execution_time_ms", round(duration, 3))

        if is_hit:
            AuditEngine.record_success("cache.get", AuditAction.READ, "Cache hit.", metadata={"key": full_key})
            return val
        else:
            reason = "expired" if is_expired else "missing"
            AuditEngine.record_success("cache.get", AuditAction.READ, f"Cache miss ({reason}).", metadata={"key": full_key})
            if raise_on_miss:
                if is_expired:
                    raise CacheExpiredError(f"Key {full_key} expired.")
                raise CacheMissError(f"Key {full_key} not found.")
            return default

    @trace_span(operation="cache.delete", component="cache", kind=SpanKind.INTERNAL)
    def delete(self, key: str, namespace: str = "default") -> bool:
        full_key = self._build_key(namespace, key)
        start_time = time.perf_counter()
        
        with self._lock:
            deleted = self._delete_internal(full_key)
                
        duration = (time.perf_counter() - start_time) * 1000.0
        TraceEngine.attach_metadata("cache_key", full_key)
        TraceEngine.attach_metadata("deleted", deleted)
        TraceEngine.attach_metadata("execution_time_ms", round(duration, 3))
        
        if deleted:
            AuditEngine.record_success("cache.delete", AuditAction.DELETE, "Entry deleted.", metadata={"key": full_key})
        return deleted

    @trace_span(operation="cache.clear", component="cache", kind=SpanKind.INTERNAL)
    def clear(self) -> None:
        try:
            with self._lock:
                self._data.clear()
                self._expiry_heap.clear()
                self._stats.current_entries = 0
                self._stats.memory_usage_bytes = 0
                self._namespace_stats.clear()

            _logger.warning("Cache Engine memory completely purged.")
            AuditEngine.record_success("cache.clear", AuditAction.DELETE, "Global cache cleared.")
        except Exception as e:
            TraceEngine.record_exception(e)
            raise CacheError(f"Global cache clear failed: {e}")

    @trace_span(operation="cache.exists", component="cache", kind=SpanKind.INTERNAL)
    def exists(self, key: str, namespace: str = "default") -> bool:
        full_key = self._build_key(namespace, key)
        with self._lock:
            entry = self._data.get(full_key)
            return entry is not None and not entry.is_expired

    @trace_span(operation="cache.touch", component="cache", kind=SpanKind.INTERNAL)
    def touch(self, key: str, namespace: str = "default", ttl: float = 3600.0) -> bool:
        full_key = self._build_key(namespace, key)
        try:
            with self._lock:
                entry = self._data.get(full_key)
                if entry is None or entry.is_expired:
                    return False
                
                entry.ttl_seconds = ttl
                entry.created_at_timestamp = time.time()
                heapq.heappush(self._expiry_heap, (entry.expires_at, full_key))
                
            _logger.debug("Cache entry TTL extended.", metadata={"key": full_key, "ttl": ttl})
            return True
        except Exception as e:
            TraceEngine.record_exception(e)
            raise CacheError(f"Touch failed for {full_key}: {e}")

    @trace_span(operation="cache.ttl", component="cache", kind=SpanKind.INTERNAL)
    def ttl(self, key: str, namespace: str = "default") -> Optional[float]:
        full_key = self._build_key(namespace, key)
        with self._lock:
            entry = self._data.get(full_key)
            if entry is None or entry.is_expired:
                return None
            if entry.expires_at is None:
                return -1.0  
            return max(0.0, entry.expires_at - time.time())

    @trace_span(operation="cache.invalidate", component="cache", kind=SpanKind.INTERNAL)
    def invalidate(self, key: str, namespace: str = "default") -> None:
        self.delete(key, namespace)

    # -------------------------------------------------------------------------
    # ITERATION APIS
    # -------------------------------------------------------------------------

    @trace_span(operation="cache.size", component="cache", kind=SpanKind.INTERNAL)
    def size(self) -> int:
        with self._lock:
            return self._stats.current_entries

    def keys(self, namespace: Optional[str] = None) -> List[str]:
        with self._lock:
            prefix = f"{namespace}:" if namespace else ""
            return [k for k, e in self._data.items() if k.startswith(prefix) and not e.is_expired]

    def values(self, namespace: Optional[str] = None) -> List[Any]:
        with self._lock:
            prefix = f"{namespace}:" if namespace else ""
            return [e.value for k, e in self._data.items() if k.startswith(prefix) and not e.is_expired]

    def items(self, namespace: Optional[str] = None) -> List[Tuple[str, Any]]:
        with self._lock:
            prefix = f"{namespace}:" if namespace else ""
            return [(k, e.value) for k, e in self._data.items() if k.startswith(prefix) and not e.is_expired]

    # -------------------------------------------------------------------------
    # BULK OPERATIONS (Optimized Internals)
    # -------------------------------------------------------------------------

    @trace_span(operation="cache.get_many", component="cache", kind=SpanKind.INTERNAL)
    def get_many(self, keys: List[str], namespace: str = "default") -> Dict[str, Any]:
        results = {}
        start_time = time.perf_counter()
        
        with self._lock:
            for key in keys:
                full_key = self._build_key(namespace, key)
                is_hit, val, _ = self._get_internal(full_key, namespace)
                if is_hit:
                    results[key] = val
                    
        duration = (time.perf_counter() - start_time) * 1000.0
        TraceEngine.attach_metadata("batch_size", len(keys))
        TraceEngine.attach_metadata("hits", len(results))
        TraceEngine.attach_metadata("execution_time_ms", round(duration, 3))
        
        AuditEngine.record_success("cache.get_many", AuditAction.READ, "Bulk retrieval.", metadata={"hits": len(results)})
        return results

    @trace_span(operation="cache.set_many", component="cache", kind=SpanKind.INTERNAL)
    def set_many(self, mapping: Dict[str, Any], namespace: str = "default", ttl: Optional[float] = None) -> None:
        start_time = time.perf_counter()
        try:
            with self._lock:
                for key, value in mapping.items():
                    full_key = self._build_key(namespace, key)
                    self._set_internal(full_key, value, namespace, ttl)
                    
            duration = (time.perf_counter() - start_time) * 1000.0
            TraceEngine.attach_metadata("batch_size", len(mapping))
            TraceEngine.attach_metadata("execution_time_ms", round(duration, 3))
            
            _logger.info("Bulk cache injection successful.", metadata={"count": len(mapping)})
            AuditEngine.record_success("cache.set_many", AuditAction.CREATE, "Bulk injection.", metadata={"count": len(mapping)})
            
        except Exception as e:
            TraceEngine.record_exception(e)
            raise CacheError(f"set_many failed: {e}")

    @trace_span(operation="cache.delete_many", component="cache", kind=SpanKind.INTERNAL)
    def delete_many(self, keys: List[str], namespace: str = "default") -> None:
        start_time = time.perf_counter()
        deleted = 0
        with self._lock:
            for key in keys:
                full_key = self._build_key(namespace, key)
                if self._delete_internal(full_key):
                    deleted += 1
                    
        duration = (time.perf_counter() - start_time) * 1000.0
        TraceEngine.attach_metadata("deleted_count", deleted)
        TraceEngine.attach_metadata("execution_time_ms", round(duration, 3))
        AuditEngine.record_success("cache.delete_many", AuditAction.DELETE, "Bulk deletion.", metadata={"count": deleted})

    # -------------------------------------------------------------------------
    # PATTERN MATCHING
    # -------------------------------------------------------------------------

    @trace_span(operation="cache.delete_namespace", component="cache", kind=SpanKind.INTERNAL)
    def delete_namespace(self, namespace: str) -> int:
        prefix = f"{namespace}:"
        deleted_count = 0
        try:
            with self._lock:
                keys_to_delete = [k for k in self._data.keys() if k.startswith(prefix)]
                for k in keys_to_delete:
                    self._delete_internal(k)
                    deleted_count += 1
                    
            AuditEngine.record_success("cache.delete_namespace", AuditAction.DELETE, f"Namespace purged.", metadata={"count": deleted_count})
            _logger.info("Namespace purged.", metadata={"namespace": namespace, "count": deleted_count})
            return deleted_count
        except Exception as e:
            TraceEngine.record_exception(e)
            raise CacheError(f"Namespace deletion failed: {e}")

    @trace_span(operation="cache.delete_pattern", component="cache", kind=SpanKind.INTERNAL)
    def delete_pattern(self, pattern: str) -> int:
        try:
            regex = re.compile(pattern)
        except re.error as e:
            TraceEngine.record_exception(e)
            AuditEngine.record_failure("cache.delete_pattern", AuditAction.DELETE, f"Invalid Regex: {e}")
            raise CacheError(f"Invalid regex compilation pattern '{pattern}': {e}")

        deleted_count = 0
        with self._lock:
            keys_to_delete = [k for k in self._data.keys() if regex.search(k)]
            for k in keys_to_delete:
                self._delete_internal(k)
                deleted_count += 1
                
        AuditEngine.record_success("cache.delete_pattern", AuditAction.DELETE, f"Pattern purged.", metadata={"count": deleted_count})
        return deleted_count

    # -------------------------------------------------------------------------
    # BACKGROUND MAINTENANCE & HEALTH
    # -------------------------------------------------------------------------

    @trace_span(operation="cache.cleanup", component="cache", kind=SpanKind.INTERNAL)
    def cleanup(self) -> None:
        """Manually invokes Min-Heap driven fast maintenance to eradicate expired limits."""
        now = time.time()
        expired_count = 0
        
        with self._lock:
            # O(log N) expiration resolution without full linear scans
            while self._expiry_heap and self._expiry_heap[0][0] <= now:
                exp_time, fkey = heapq.heappop(self._expiry_heap)
                entry = self._data.get(fkey)
                
                if entry and entry.expires_at == exp_time:
                    del self._data[fkey]
                    self._internal_delete_accounting(entry)
                    self._stats.expired_entries += 1
                    expired_count += 1
                    
            self._stats.cleanup_runs += 1

        if expired_count > 0:
            TraceEngine.attach_metadata("keys_removed", expired_count)
            AuditEngine.record_success("cache.cleanup", AuditAction.SYSTEM, "Maintenance executed.", metadata={"removed": expired_count})
            _logger.debug(f"Cache cleanup eradicated {expired_count} expired entries.")

    def _cleanup_loop(self) -> None:
        """Smart Maintenance Loop mapping natively against heap expiration bounds."""
        _logger.info("Cache Maintenance Engine active.")
        while not self._stop_event.is_set():
            try:
                self.cleanup()
                
                # Smart Sleep calculating bounds mapped against next imminent expiration
                with self._lock:
                    if self._expiry_heap:
                        next_expiry = self._expiry_heap[0][0]
                        sleep_time = max(0.1, min(next_expiry - time.time(), self.cleanup_interval_sec))
                    else:
                        sleep_time = self.cleanup_interval_sec
                        
                self._stop_event.wait(sleep_time)
                
            except Exception as e:
                _logger.error(f"Critical failure inside background cache maintenance: {e}", exc_info=e)
                TraceEngine.record_exception(e)
                AuditEngine.record_failure("cache.cleanup_loop", AuditAction.SYSTEM, "Maintenance crash.", AuditSeverity.CRITICAL)
                self._stop_event.wait(5.0)

        _logger.info("Cache Maintenance Engine safely terminated.")

    def shutdown(self) -> None:
        """Signals graceful teardown and systematically purges all residual memory structures."""
        _logger.info("Initiating cache shutdown sequences.")
        self._stop_event.set()
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=3.0)
        self.clear()

    @trace_span(operation="cache.health_check", component="cache", kind=SpanKind.INTERNAL)
    def health_check(self) -> Dict[str, Any]:
        """Provides Read-Only status validation determining Lock Contention and Memory constraints."""
        try:
            locked = self._lock.acquire(timeout=1.0)
            if locked:
                self._lock.release()
                
            return {
                "status": "HEALTHY" if locked else "DEGRADED",
                "lock_contention": not locked,
                "cleanup_thread_active": self._cleanup_thread.is_alive(),
                "policy": self.policy.value,
                "memory_saturation": f"{(self._stats.memory_usage_bytes / self.max_memory_bytes) * 100:.1f}%",
                "statistics": self.get_statistics(),
                "namespace_statistics": {k: v.to_dict() for k, v in self._namespace_stats.items()}
            }
        except Exception as e:
            TraceEngine.record_exception(e)
            return {"status": "UNHEALTHY", "error": str(e)}

    # -------------------------------------------------------------------------
    # STATISTICS API
    # -------------------------------------------------------------------------

    @trace_span(operation="cache.get_statistics", component="cache", kind=SpanKind.INTERNAL)
    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            return self._stats.to_dict()

    @trace_span(operation="cache.reset_statistics", component="cache", kind=SpanKind.INTERNAL)
    def reset_statistics(self) -> None:
        with self._lock:
            current_entries = self._stats.current_entries
            mem_usage = self._stats.memory_usage_bytes
            self._stats = CacheStatistics(
                current_entries=current_entries,
                memory_usage_bytes=mem_usage
            )


# -------------------------------------------------------------------------
# GLOBAL SINGLETON EXPORT
# -------------------------------------------------------------------------

cache_engine: Final[CacheEngine] = CacheEngine()


# -------------------------------------------------------------------------
# DECORATORS
# -------------------------------------------------------------------------

def cached(namespace: str = "default", ttl: Optional[float] = None) -> Callable[..., Any]:
    """
    Enterprise-grade decorator orchestrating synchronous caching boundaries natively.
    Wraps expensive evaluations inside deterministic temporal lookups mapping robust keys.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                args_repr = ",".join(_deterministic_repr(a) for a in args)
                kwargs_repr = ",".join(f"{k}={_deterministic_repr(v)}" for k, v in sorted(kwargs.items()))
                key = f"{func.__name__}({args_repr}|{kwargs_repr})"
            except Exception:
                key = f"{func.__name__}_unhashable_{time.time()}"
            
            try:
                return cache_engine.get(key, namespace=namespace, raise_on_miss=True)
            except (CacheMissError, CacheExpiredError):
                result = func(*args, **kwargs)
                cache_engine.set(key, result, namespace=namespace, ttl=ttl)
                return result

        return wrapper
    return decorator


# -------------------------------------------------------------------------
# MODULE EXPORTS
# -------------------------------------------------------------------------

__all__ = [
    "CacheError",
    "CacheMissError",
    "CacheExpiredError",
    "CacheMemoryError",
    "CachePolicy",
    "CacheEntry",
    "NamespaceStatistics",
    "CacheStatistics",
    "CacheEngine",
    "cache_engine",
    "cached"
]
