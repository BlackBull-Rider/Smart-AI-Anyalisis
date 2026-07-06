"""
GREEN BULL RIDER V6 - Institutional-grade AI Stock Analysis Platform
Module: backend/core/event_bus.py
Description: Centralized Enterprise Event Bus.
             Orchestrates asynchronous, fully decoupled communication across all
             platform subsystems using high-performance O(1) priority queues.
             Provides deep isolation, dead-letter processing, and native 
             integration with Trace, Audit, and Logging telemetry pipelines.
"""

import time
import uuid
import threading
import collections
import queue
from enum import IntEnum, Enum
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Final

# Internal Platform Integrations
from backend.core.logger import AppLogger
from backend.core.trace import TraceEngine, SpanKind, TraceLevel
from backend.core.audit import AuditEngine, AuditAction, AuditResult, AuditSeverity
from backend.core.exceptions import EventBusError


# -------------------------------------------------------------------------
# LOGGER INITIALIZATION
# -------------------------------------------------------------------------
_logger = AppLogger("EventBus")


# -------------------------------------------------------------------------
# EXCEPTIONS
# -------------------------------------------------------------------------

class EventPublishError(EventBusError):
    """Raised when an event fails to enter the routing queue safely."""
    error_code: str = "GBR-EVT-001"


class EventHandlerError(EventBusError):
    """Raised when an isolated subscriber handler fails during execution."""
    error_code: str = "GBR-EVT-002"


# -------------------------------------------------------------------------
# ENUMS
# -------------------------------------------------------------------------

class EventPriority(IntEnum):
    """
    Mathematical priority definition for deterministic queue sorting.
    Lower numerical values indicate higher execution priority.
    """
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


class EventStatus(str, Enum):
    """Lifecycle tracking states for a distinct event payload."""
    CREATED = "CREATED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


# -------------------------------------------------------------------------
# DATACLASSES
# -------------------------------------------------------------------------

@dataclass(slots=True)
class Event:
    """
    Mutable state container for an actionable event passing through the bus.
    Memory-optimized using slots for high-throughput routing.
    """
    name: str
    payload: Dict[str, Any]
    source: str
    priority: EventPriority = EventPriority.NORMAL
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: EventStatus = EventStatus.CREATED
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: 'Event') -> bool:
        """Determines routing precedence based on priority bounds and temporal creation."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp


@dataclass(frozen=True, slots=True)
class EventStatistics:
    """Immutable operational telemetry for the Event Bus infrastructure."""
    published_events: int
    processed_events: int
    failed_events: int
    registered_handlers: int
    queued_events: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "published_events": self.published_events,
            "processed_events": self.processed_events,
            "failed_events": self.failed_events,
            "registered_handlers": self.registered_handlers,
            "queued_events": self.queued_events
        }


# -------------------------------------------------------------------------
# HIGH-PERFORMANCE QUEUE ABSTRACTION
# -------------------------------------------------------------------------

class FastPriorityQueue:
    """
    O(1) strict time complexity Priority Queue mapped explicitly to the EventPriority bounds.
    Utilizes localized deque endpoints inside a Condition lock to outperform stdlib PriorityQueue.
    """
    def __init__(self) -> None:
        self._queues: Dict[EventPriority, collections.deque] = {
            EventPriority.CRITICAL: collections.deque(),
            EventPriority.HIGH: collections.deque(),
            EventPriority.NORMAL: collections.deque(),
            EventPriority.LOW: collections.deque(),
        }
        self._size: int = 0
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    def put(self, event: Event) -> None:
        with self._not_empty:
            self._queues[event.priority].append(event)
            self._size += 1
            self._not_empty.notify()

    def get(self, timeout: Optional[float] = None) -> Event:
        with self._not_empty:
            if self._size == 0:
                self._not_empty.wait(timeout)
            if self._size == 0:
                raise queue.Empty

            for priority in EventPriority:
                if self._queues[priority]:
                    self._size -= 1
                    return self._queues[priority].popleft()
            
            raise queue.Empty

    def qsize(self) -> int:
        with self._lock:
            return self._size


# -------------------------------------------------------------------------
# EVENT BUS ENGINE (SINGLETON)
# -------------------------------------------------------------------------

class EventBus:
    """
    Enterprise Central Event Bus.
    Provides strict decoupled integration layers across database, sync engines, 
    AI matrices, and APIs via asynchronous, thread-safe message routing.
    """
    _instance = None
    _init_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super(EventBus, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        # Registry mapping Event Names to Thread-Safe Sets of Callables
        self._subscribers: Dict[str, Set[Callable[[Event], None]]] = {}
        self._subscribers_lock = threading.Lock()
        
        self._queue = FastPriorityQueue()
        
        # Dead-letter sink for unroutable or permanently failed messages
        self._dead_letters: List[Dict[str, Any]] = []
        self._dead_letters_lock = threading.Lock()

        # Telemetry Metrics
        self._stats_lock = threading.Lock()
        self._published_events: int = 0
        self._processed_events: int = 0
        self._failed_events: int = 0

        # Dedicated Dispatcher Thread Engine
        self._stop_event = threading.Event()
        self._dispatcher_thread = threading.Thread(
            target=self._dispatch_loop,
            name="EventBusDispatcher",
            daemon=True
        )
        self._dispatcher_thread.start()

    # -------------------------------------------------------------------------
    # SUBSCRIBER MANAGEMENT
    # -------------------------------------------------------------------------

    def subscribe(self, event_name: str, handler: Callable[[Event], None]) -> None:
        """Safely attaches an execution handler to an explicit event endpoint."""
        with self._subscribers_lock:
            if event_name not in self._subscribers:
                self._subscribers[event_name] = set()
            self._subscribers[event_name].add(handler)

    def unsubscribe(self, event_name: str, handler: Callable[[Event], None]) -> None:
        """Safely detaches an active execution handler from an event endpoint."""
        with self._subscribers_lock:
            if event_name in self._subscribers:
                self._subscribers[event_name].discard(handler)
                if not self._subscribers[event_name]:
                    del self._subscribers[event_name]

    def clear_subscribers(self) -> None:
        """Purges the entire routing matrix cleanly."""
        with self._subscribers_lock:
            self._subscribers.clear()

    def has_subscribers(self, event_name: str) -> bool:
        """Evaluates whether an endpoint possesses actionable listener targets."""
        with self._subscribers_lock:
            return event_name in self._subscribers and len(self._subscribers[event_name]) > 0

    def handler_count(self, event_name: str) -> int:
        """Quantifies the exact listener weight bounded to an event endpoint."""
        with self._subscribers_lock:
            return len(self._subscribers.get(event_name, set()))

    def event_queue_size(self) -> int:
        """Evaluates the current operational latency load."""
        return self._queue.qsize()

    # -------------------------------------------------------------------------
    # PUBLICATION ENDPOINTS
    # -------------------------------------------------------------------------

    def publish(self, event: Event) -> None:
        """
        Ingests an actionable payload into the core operational pipeline. O(1) timing.
        Automatically instruments Trace Engine bounds and structural Auditing.
        """
        try:
            event.status = EventStatus.QUEUED
            
            with TraceEngine.nested_span(operation="event_bus.publish", component="event_bus", kind=SpanKind.PRODUCER):
                TraceEngine.attach_metadata("event_id", event.event_id)
                TraceEngine.attach_metadata("event_name", event.name)
                TraceEngine.attach_metadata("priority", event.priority.name)
                TraceEngine.attach_metadata("source", event.source)
                
                self._queue.put(event)
                
                with self._stats_lock:
                    self._published_events += 1

                _logger.debug(f"Event published correctly into pipeline: {event.name}", metadata={"event_id": event.event_id})
                
                AuditEngine.record_success(
                    operation="event_bus.publish",
                    action=AuditAction.SYSTEM,
                    message=f"Event {event.name} queued successfully.",
                    metadata={"event_id": event.event_id, "priority": event.priority.name}
                )

        except Exception as e:
            TraceEngine.record_exception(e)
            _logger.error(f"Event publication failure for {event.name}: {e}", exc_info=e)
            raise EventPublishError(f"Failed to publish event {event.name}: {e}", cause=e) from e

    def publish_many(self, events: List[Event]) -> None:
        """Iterative wrapper optimizing consecutive bulk publication requests."""
        for event in events:
            self.publish(event)

    # -------------------------------------------------------------------------
    # DISPATCHER ENGINE
    # -------------------------------------------------------------------------

    def _dispatch_loop(self) -> None:
        """
        Continuous background processor stripping priority queues dynamically.
        Isolates execution failures explicitly from disrupting structural timelines.
        """
        while not self._stop_event.is_set():
            try:
                event = self._queue.get(timeout=1.0)
                self._handle_event(event)
            except queue.Empty:
                continue
            except Exception as e:
                _logger.critical(f"Critical dispatcher isolation breach: {e}", exc_info=e)

    def _handle_event(self, event: Event) -> None:
        """
        Executes bounded listeners dynamically. Emits strict trace spans and evaluates
        execution resilience. Flags dead-letters for unroutable requests.
        """
        event.status = EventStatus.RUNNING
        start_time = time.perf_counter()

        with self._subscribers_lock:
            # Shallow copy the handlers to avoid deadlocks during execution
            handlers = list(self._subscribers.get(event.name, []))

        if not handlers:
            event.status = EventStatus.CANCELLED
            self._send_to_dead_letter(event, reason="No registered handlers mapped to endpoint.")
            return

        with TraceEngine.nested_span(operation="event_bus.process", component="event_bus", kind=SpanKind.CONSUMER):
            TraceEngine.attach_metadata("event_id", event.event_id)
            TraceEngine.attach_metadata("event_name", event.name)
            TraceEngine.attach_metadata("priority", event.priority.name)
            TraceEngine.attach_metadata("handler_count", len(handlers))

            failed_handlers = 0

            # Absolute Isolation execution barrier
            for handler in handlers:
                try:
                    handler(event)
                except Exception as e:
                    failed_handlers += 1
                    handler_name = getattr(handler, "__name__", str(handler))
                    
                    err = EventHandlerError(
                        message=f"Handler {handler_name} failed unexpectedly for event {event.name}",
                        cause=e,
                        operation="event_bus.process"
                    )
                    TraceEngine.record_exception(err)
                    _logger.error(f"Event handler runtime exception: {err.message}", exc_info=e)

            duration_ms = (time.perf_counter() - start_time) * 1000.0
            TraceEngine.attach_metadata("processing_time_ms", round(duration_ms, 3))
            TraceEngine.attach_metadata("failed_handlers", failed_handlers)

            # Resolve terminal state matrices safely
            with self._stats_lock:
                if failed_handlers == 0:
                    self._processed_events += 1
                    event.status = EventStatus.COMPLETED
                    AuditEngine.record_success(
                        operation="event_bus.process",
                        action=AuditAction.EXECUTE,
                        message=f"Event {event.name} processed flawlessly by {len(handlers)} handlers."
                    )
                else:
                    self._failed_events += 1
                    event.status = EventStatus.FAILED
                    self._send_to_dead_letter(event, reason=f"{failed_handlers} internal handlers collapsed.")
                    AuditEngine.record_failure(
                        operation="event_bus.process",
                        action=AuditAction.EXECUTE,
                        message=f"Event {event.name} suffered {failed_handlers} endpoint collapses.",
                        severity=AuditSeverity.WARNING
                    )

    def _send_to_dead_letter(self, event: Event, reason: str) -> None:
        """Routes failed or unbound execution payloads into a restricted audit trace memory block."""
        with self._dead_letters_lock:
            if len(self._dead_letters) >= 1000:
                self._dead_letters.pop(0)  # Capped FIFO ring buffer logic
            self._dead_letters.append({"event": event, "reason": reason, "timestamp": datetime.now(timezone.utc).isoformat()})
            
        _logger.warning(f"Event sent securely to Dead-Letter Sink: {event.name} | Reason: {reason}")
        TraceEngine.record_event("DEAD_LETTER_ROUTED", TraceLevel.WARNING, metadata={"event_id": event.event_id, "reason": reason})

    # -------------------------------------------------------------------------
    # STATISTICS & TEARDOWN
    # -------------------------------------------------------------------------

    def get_statistics(self) -> EventStatistics:
        """Retrieves a thread-safe snapshot of global routing telemetry."""
        with self._stats_lock:
            with self._subscribers_lock:
                total_handlers = sum(len(h) for h in self._subscribers.values())
            
            return EventStatistics(
                published_events=self._published_events,
                processed_events=self._processed_events,
                failed_events=self._failed_events,
                registered_handlers=total_handlers,
                queued_events=self._queue.qsize()
            )

    def reset_statistics(self) -> None:
        """Purges global telemetry routing metrics cleanly."""
        with self._stats_lock:
            self._published_events = 0
            self._processed_events = 0
            self._failed_events = 0

    def shutdown(self) -> None:
        """Safely terminates the active internal processor mapping dispatchers."""
        self._stop_event.set()
        if self._dispatcher_thread.is_alive():
            # Release any blocking lock inside the queue cleanly via a mock entry
            self._queue.put(Event(name="SYSTEM_SHUTDOWN", payload={}, source="system", priority=EventPriority.CRITICAL))
            self._dispatcher_thread.join(timeout=3.0)


# -------------------------------------------------------------------------
# GLOBAL SINGLETON EXPORT
# -------------------------------------------------------------------------

event_bus: Final[EventBus] = EventBus()

__all__ = [
    "EventPriority",
    "EventStatus",
    "EventPublishError",
    "EventHandlerError",
    "Event",
    "EventStatistics",
    "FastPriorityQueue",
    "EventBus",
    "event_bus"
]
