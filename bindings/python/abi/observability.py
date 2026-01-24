"""Observability module for metrics, tracing, and profiling.

This module provides Python bindings for the ABI observability system,
including Prometheus-compatible metrics, distributed tracing, and
performance profiling.

Example:
    >>> from abi.observability import MetricsRegistry, Counter, Gauge, Histogram
    >>> registry = MetricsRegistry()
    >>> requests = registry.counter("http_requests_total", "Total HTTP requests")
    >>> requests.inc()
    >>> latency = registry.histogram("request_latency_seconds", "Request latency")
    >>> latency.observe(0.123)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Iterator
from contextlib import contextmanager
from enum import Enum
import threading
import time
import json


class MetricType(Enum):
    """Type of metric."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricLabels:
    """Labels for a metric."""

    labels: Dict[str, str] = field(default_factory=dict)

    def with_label(self, key: str, value: str) -> "MetricLabels":
        """Add a label and return new MetricLabels."""
        new_labels = dict(self.labels)
        new_labels[key] = value
        return MetricLabels(labels=new_labels)

    def to_prometheus(self) -> str:
        """Format labels for Prometheus output."""
        if not self.labels:
            return ""
        pairs = [f'{k}="{v}"' for k, v in sorted(self.labels.items())]
        return "{" + ",".join(pairs) + "}"


class Counter:
    """A counter metric that can only increase.

    Thread-safe counter implementation compatible with Prometheus.

    Example:
        >>> counter = Counter("requests_total", "Total requests")
        >>> counter.inc()
        >>> counter.inc(5)
        >>> print(counter.get())
        6
    """

    def __init__(self, name: str, help_text: str = "", labels: Optional[MetricLabels] = None):
        self.name = name
        self.help_text = help_text
        self.labels = labels or MetricLabels()
        self._value: float = 0.0
        self._lock = threading.Lock()

    def inc(self, delta: float = 1.0) -> None:
        """Increment the counter by delta (default 1)."""
        if delta < 0:
            raise ValueError("Counter can only increase")
        with self._lock:
            self._value += delta

    def get(self) -> float:
        """Get the current counter value."""
        with self._lock:
            return self._value

    def reset(self) -> None:
        """Reset the counter to zero (use sparingly)."""
        with self._lock:
            self._value = 0.0

    def to_prometheus(self) -> str:
        """Format as Prometheus text format."""
        label_str = self.labels.to_prometheus()
        return f"{self.name}{label_str} {self.get()}"


class Gauge:
    """A gauge metric that can increase or decrease.

    Thread-safe gauge implementation compatible with Prometheus.

    Example:
        >>> gauge = Gauge("temperature", "Current temperature")
        >>> gauge.set(23.5)
        >>> gauge.inc(1.5)
        >>> print(gauge.get())
        25.0
    """

    def __init__(self, name: str, help_text: str = "", labels: Optional[MetricLabels] = None):
        self.name = name
        self.help_text = help_text
        self.labels = labels or MetricLabels()
        self._value: float = 0.0
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        """Set the gauge to a specific value."""
        with self._lock:
            self._value = value

    def inc(self, delta: float = 1.0) -> None:
        """Increment the gauge by delta."""
        with self._lock:
            self._value += delta

    def dec(self, delta: float = 1.0) -> None:
        """Decrement the gauge by delta."""
        with self._lock:
            self._value -= delta

    def get(self) -> float:
        """Get the current gauge value."""
        with self._lock:
            return self._value

    def set_to_current_time(self) -> None:
        """Set the gauge to the current Unix timestamp."""
        self.set(time.time())

    @contextmanager
    def track_inprogress(self):
        """Context manager to track in-progress operations."""
        self.inc()
        try:
            yield
        finally:
            self.dec()

    def to_prometheus(self) -> str:
        """Format as Prometheus text format."""
        label_str = self.labels.to_prometheus()
        return f"{self.name}{label_str} {self.get()}"


@dataclass
class HistogramBucket:
    """A histogram bucket with upper bound and count."""

    le: float  # less than or equal
    count: int = 0


class Histogram:
    """A histogram metric for measuring distributions.

    Thread-safe histogram implementation with configurable buckets.

    Example:
        >>> histogram = Histogram("request_duration", "Request duration",
        ...                       buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0])
        >>> histogram.observe(0.123)
        >>> print(histogram.get_count())
        1
    """

    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)

    def __init__(
        self,
        name: str,
        help_text: str = "",
        buckets: Optional[tuple] = None,
        labels: Optional[MetricLabels] = None,
    ):
        self.name = name
        self.help_text = help_text
        self.labels = labels or MetricLabels()
        self._buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self._bucket_counts: List[int] = [0] * len(self._buckets)
        self._sum: float = 0.0
        self._count: int = 0
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        """Record an observation."""
        with self._lock:
            self._sum += value
            self._count += 1
            # Find the first bucket where value <= bound
            for i, bound in enumerate(self._buckets):
                if value <= bound:
                    self._bucket_counts[i] += 1
                    break  # Only count in one bucket, cumulative is computed on read

    def get_count(self) -> int:
        """Get the total number of observations."""
        with self._lock:
            return self._count

    def get_sum(self) -> float:
        """Get the sum of all observations."""
        with self._lock:
            return self._sum

    def get_buckets(self) -> List[tuple]:
        """Get bucket bounds and counts as (le, count) pairs."""
        with self._lock:
            cumulative = 0
            result = []
            for i, bound in enumerate(self._buckets):
                cumulative += self._bucket_counts[i]
                result.append((bound, cumulative))
            result.append((float("inf"), self._count))
            return result

    @contextmanager
    def time(self):
        """Context manager to time an operation."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe(time.perf_counter() - start)

    def to_prometheus(self) -> str:
        """Format as Prometheus text format."""
        lines = []
        label_str = self.labels.to_prometheus()
        base_labels = self.labels.labels

        for le, count in self.get_buckets():
            le_str = "+Inf" if le == float("inf") else str(le)
            bucket_labels = dict(base_labels)
            bucket_labels["le"] = le_str
            label_part = "{" + ",".join(f'{k}="{v}"' for k, v in sorted(bucket_labels.items())) + "}"
            lines.append(f"{self.name}_bucket{label_part} {count}")

        lines.append(f"{self.name}_sum{label_str} {self.get_sum()}")
        lines.append(f"{self.name}_count{label_str} {self.get_count()}")
        return "\n".join(lines)


class MetricsRegistry:
    """Registry for managing metrics.

    Example:
        >>> registry = MetricsRegistry()
        >>> counter = registry.counter("requests_total", "Total requests")
        >>> gauge = registry.gauge("active_connections", "Active connections")
        >>> histogram = registry.histogram("request_duration", "Request duration")
    """

    def __init__(self, prefix: str = ""):
        self._prefix = prefix
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._lock = threading.Lock()

    def _full_name(self, name: str) -> str:
        """Get full metric name with prefix."""
        return f"{self._prefix}{name}" if self._prefix else name

    def counter(
        self, name: str, help_text: str = "", labels: Optional[MetricLabels] = None
    ) -> Counter:
        """Create or get a counter metric."""
        full_name = self._full_name(name)
        with self._lock:
            if full_name not in self._counters:
                self._counters[full_name] = Counter(full_name, help_text, labels)
            return self._counters[full_name]

    def gauge(
        self, name: str, help_text: str = "", labels: Optional[MetricLabels] = None
    ) -> Gauge:
        """Create or get a gauge metric."""
        full_name = self._full_name(name)
        with self._lock:
            if full_name not in self._gauges:
                self._gauges[full_name] = Gauge(full_name, help_text, labels)
            return self._gauges[full_name]

    def histogram(
        self,
        name: str,
        help_text: str = "",
        buckets: Optional[tuple] = None,
        labels: Optional[MetricLabels] = None,
    ) -> Histogram:
        """Create or get a histogram metric."""
        full_name = self._full_name(name)
        with self._lock:
            if full_name not in self._histograms:
                self._histograms[full_name] = Histogram(full_name, help_text, buckets, labels)
            return self._histograms[full_name]

    def collect(self) -> Dict[str, Any]:
        """Collect all metrics as a dictionary."""
        with self._lock:
            return {
                "counters": {name: c.get() for name, c in self._counters.items()},
                "gauges": {name: g.get() for name, g in self._gauges.items()},
                "histograms": {
                    name: {
                        "count": h.get_count(),
                        "sum": h.get_sum(),
                        "buckets": h.get_buckets(),
                    }
                    for name, h in self._histograms.items()
                },
            }

    def to_prometheus(self) -> str:
        """Export all metrics in Prometheus text format."""
        lines = []
        with self._lock:
            for name, counter in self._counters.items():
                if counter.help_text:
                    lines.append(f"# HELP {name} {counter.help_text}")
                lines.append(f"# TYPE {name} counter")
                lines.append(counter.to_prometheus())

            for name, gauge in self._gauges.items():
                if gauge.help_text:
                    lines.append(f"# HELP {name} {gauge.help_text}")
                lines.append(f"# TYPE {name} gauge")
                lines.append(gauge.to_prometheus())

            for name, histogram in self._histograms.items():
                if histogram.help_text:
                    lines.append(f"# HELP {name} {histogram.help_text}")
                lines.append(f"# TYPE {name} histogram")
                lines.append(histogram.to_prometheus())

        return "\n".join(lines)

    def to_json(self) -> str:
        """Export all metrics as JSON."""
        return json.dumps(self.collect(), indent=2)


# ============================================================================
# Tracing
# ============================================================================


@dataclass
class SpanContext:
    """Context for distributed tracing."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)


class Span:
    """A span representing a unit of work in a trace.

    Example:
        >>> with tracer.start_span("http_request") as span:
        ...     span.set_attribute("http.method", "GET")
        ...     span.set_attribute("http.url", "/api/users")
        ...     # do work
        ...     span.set_status("ok")
    """

    def __init__(
        self,
        name: str,
        context: SpanContext,
        parent: Optional["Span"] = None,
    ):
        self.name = name
        self.context = context
        self.parent = parent
        self.start_time: float = time.time()
        self.end_time: Optional[float] = None
        self.attributes: Dict[str, Any] = {}
        self.events: List[Dict[str, Any]] = []
        self.status: str = "unset"
        self._children: List["Span"] = []

    def set_attribute(self, key: str, value: Any) -> "Span":
        """Set an attribute on this span."""
        self.attributes[key] = value
        return self

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> "Span":
        """Add an event to this span."""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })
        return self

    def set_status(self, status: str, description: str = "") -> "Span":
        """Set the status of this span (ok, error, unset)."""
        self.status = status
        if description:
            self.attributes["status.description"] = description
        return self

    def end(self) -> None:
        """End this span."""
        self.end_time = time.time()

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary."""
        return {
            "name": self.name,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status,
        }

    def __enter__(self) -> "Span":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self.set_status("error", str(exc_val))
        elif self.status == "unset":
            self.set_status("ok")
        self.end()


class Tracer:
    """Distributed tracing implementation.

    Example:
        >>> tracer = Tracer("my-service")
        >>> with tracer.start_span("process_request") as span:
        ...     span.set_attribute("user.id", "123")
        ...     # do work
    """

    def __init__(self, service_name: str):
        self.service_name = service_name
        self._spans: List[Span] = []
        self._current_span: Optional[Span] = None
        self._lock = threading.Lock()
        self._span_counter = 0

    def _generate_id(self) -> str:
        """Generate a unique ID for traces/spans."""
        import hashlib
        self._span_counter += 1
        data = f"{self.service_name}-{time.time()}-{self._span_counter}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def start_span(
        self,
        name: str,
        parent: Optional[Span] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """Start a new span."""
        parent = parent or self._current_span

        if parent:
            context = SpanContext(
                trace_id=parent.context.trace_id,
                span_id=self._generate_id(),
                parent_span_id=parent.context.span_id,
            )
        else:
            context = SpanContext(
                trace_id=self._generate_id(),
                span_id=self._generate_id(),
            )

        span = Span(name, context, parent)
        span.set_attribute("service.name", self.service_name)

        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        with self._lock:
            self._spans.append(span)
            self._current_span = span

        return span

    def get_current_span(self) -> Optional[Span]:
        """Get the current active span."""
        return self._current_span

    def get_spans(self) -> List[Span]:
        """Get all recorded spans."""
        with self._lock:
            return list(self._spans)

    def clear(self) -> None:
        """Clear all recorded spans."""
        with self._lock:
            self._spans.clear()
            self._current_span = None

    def export_json(self) -> str:
        """Export all spans as JSON."""
        with self._lock:
            return json.dumps([s.to_dict() for s in self._spans], indent=2)


# ============================================================================
# Profiling
# ============================================================================


@dataclass
class ProfileSample:
    """A single profiling sample."""

    name: str
    start_time: float
    end_time: float
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class Profiler:
    """Simple profiler for measuring execution time.

    Example:
        >>> profiler = Profiler()
        >>> with profiler.measure("database_query"):
        ...     # do work
        >>> print(profiler.get_stats("database_query"))
    """

    def __init__(self):
        self._samples: Dict[str, List[ProfileSample]] = {}
        self._lock = threading.Lock()

    @contextmanager
    def measure(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager to measure execution time."""
        start = time.perf_counter()
        start_time = time.time()
        try:
            yield
        finally:
            end = time.perf_counter()
            duration_ms = (end - start) * 1000

            sample = ProfileSample(
                name=name,
                start_time=start_time,
                end_time=time.time(),
                duration_ms=duration_ms,
                metadata=metadata or {},
            )

            with self._lock:
                if name not in self._samples:
                    self._samples[name] = []
                self._samples[name].append(sample)

    def get_samples(self, name: str) -> List[ProfileSample]:
        """Get all samples for a given name."""
        with self._lock:
            return list(self._samples.get(name, []))

    def _compute_stats(self, samples: List[ProfileSample]) -> Dict[str, float]:
        """Compute stats from samples (internal, no lock)."""
        if not samples:
            return {"count": 0, "min_ms": 0, "max_ms": 0, "avg_ms": 0, "total_ms": 0}

        durations = [s.duration_ms for s in samples]
        return {
            "count": len(durations),
            "min_ms": min(durations),
            "max_ms": max(durations),
            "avg_ms": sum(durations) / len(durations),
            "total_ms": sum(durations),
        }

    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a given name."""
        samples = self.get_samples(name)
        return self._compute_stats(samples)

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all measured operations."""
        with self._lock:
            return {
                name: self._compute_stats(list(samples))
                for name, samples in self._samples.items()
            }

    def clear(self) -> None:
        """Clear all samples."""
        with self._lock:
            self._samples.clear()

    def to_json(self) -> str:
        """Export all stats as JSON."""
        return json.dumps(self.get_all_stats(), indent=2)


# ============================================================================
# Health Checks
# ============================================================================


class HealthStatus(Enum):
    """Health check status."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    name: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0


class HealthChecker:
    """Health check manager.

    Example:
        >>> checker = HealthChecker()
        >>> checker.register("database", lambda: HealthCheckResult(
        ...     name="database", status=HealthStatus.HEALTHY))
        >>> results = checker.check_all()
    """

    def __init__(self):
        self._checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self._lock = threading.Lock()

    def register(self, name: str, check: Callable[[], HealthCheckResult]) -> None:
        """Register a health check."""
        with self._lock:
            self._checks[name] = check

    def unregister(self, name: str) -> None:
        """Unregister a health check."""
        with self._lock:
            self._checks.pop(name, None)

    def check(self, name: str) -> HealthCheckResult:
        """Run a single health check."""
        with self._lock:
            check = self._checks.get(name)
            if not check:
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check '{name}' not found",
                )

        start = time.perf_counter()
        try:
            result = check()
            result.duration_ms = (time.perf_counter() - start) * 1000
            return result
        except Exception as e:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    def check_all(self) -> List[HealthCheckResult]:
        """Run all health checks."""
        with self._lock:
            names = list(self._checks.keys())
        return [self.check(name) for name in names]

    def is_healthy(self) -> bool:
        """Check if all health checks pass."""
        results = self.check_all()
        return all(r.status == HealthStatus.HEALTHY for r in results)

    def to_dict(self) -> Dict[str, Any]:
        """Export health status as dictionary."""
        results = self.check_all()
        overall = HealthStatus.HEALTHY
        for r in results:
            if r.status == HealthStatus.UNHEALTHY:
                overall = HealthStatus.UNHEALTHY
                break
            elif r.status == HealthStatus.DEGRADED:
                overall = HealthStatus.DEGRADED

        return {
            "status": overall.value,
            "checks": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "message": r.message,
                    "details": r.details,
                    "duration_ms": r.duration_ms,
                }
                for r in results
            ],
        }

    def to_json(self) -> str:
        """Export health status as JSON."""
        return json.dumps(self.to_dict(), indent=2)


# ============================================================================
# Global instances
# ============================================================================

# Default global instances
_default_registry: Optional[MetricsRegistry] = None
_default_tracer: Optional[Tracer] = None
_default_profiler: Optional[Profiler] = None
_default_health_checker: Optional[HealthChecker] = None


def get_registry() -> MetricsRegistry:
    """Get the default metrics registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = MetricsRegistry(prefix="abi_")
    return _default_registry


def get_tracer(service_name: str = "abi") -> Tracer:
    """Get the default tracer."""
    global _default_tracer
    if _default_tracer is None:
        _default_tracer = Tracer(service_name)
    return _default_tracer


def get_profiler() -> Profiler:
    """Get the default profiler."""
    global _default_profiler
    if _default_profiler is None:
        _default_profiler = Profiler()
    return _default_profiler


def get_health_checker() -> HealthChecker:
    """Get the default health checker."""
    global _default_health_checker
    if _default_health_checker is None:
        _default_health_checker = HealthChecker()
    return _default_health_checker


__all__ = [
    # Types
    "MetricType",
    "MetricLabels",
    "HealthStatus",
    "HealthCheckResult",
    "SpanContext",
    "ProfileSample",
    # Metrics
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsRegistry",
    # Tracing
    "Span",
    "Tracer",
    # Profiling
    "Profiler",
    # Health
    "HealthChecker",
    # Global accessors
    "get_registry",
    "get_tracer",
    "get_profiler",
    "get_health_checker",
]
