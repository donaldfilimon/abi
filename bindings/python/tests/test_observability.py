"""Tests for the observability module."""

import pytest
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from abi.observability import (
    MetricType,
    MetricLabels,
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    SpanContext,
    Span,
    Tracer,
    Profiler,
    ProfileSample,
    HealthStatus,
    HealthCheckResult,
    HealthChecker,
    get_registry,
    get_tracer,
    get_profiler,
    get_health_checker,
)


class TestMetricLabels:
    """Tests for MetricLabels."""

    def test_empty_labels(self):
        labels = MetricLabels()
        assert labels.labels == {}
        assert labels.to_prometheus() == ""

    def test_with_label(self):
        labels = MetricLabels()
        labels2 = labels.with_label("method", "GET")
        assert labels2.labels == {"method": "GET"}
        assert labels.labels == {}  # Original unchanged

    def test_multiple_labels(self):
        labels = MetricLabels(labels={"method": "GET", "path": "/api"})
        prometheus = labels.to_prometheus()
        assert 'method="GET"' in prometheus
        assert 'path="/api"' in prometheus


class TestCounter:
    """Tests for Counter metric."""

    def test_increment(self):
        counter = Counter("test_counter", "Test counter")
        assert counter.get() == 0
        counter.inc()
        assert counter.get() == 1
        counter.inc(5)
        assert counter.get() == 6

    def test_increment_negative_raises(self):
        counter = Counter("test_counter")
        with pytest.raises(ValueError):
            counter.inc(-1)

    def test_reset(self):
        counter = Counter("test_counter")
        counter.inc(10)
        counter.reset()
        assert counter.get() == 0

    def test_thread_safety(self):
        counter = Counter("test_counter")
        num_threads = 10
        increments_per_thread = 1000

        def increment():
            for _ in range(increments_per_thread):
                counter.inc()

        threads = [threading.Thread(target=increment) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert counter.get() == num_threads * increments_per_thread

    def test_prometheus_format(self):
        counter = Counter("http_requests_total", "Total HTTP requests")
        counter.inc(42)
        prometheus = counter.to_prometheus()
        assert "http_requests_total 42" in prometheus

    def test_prometheus_format_with_labels(self):
        labels = MetricLabels(labels={"method": "GET"})
        counter = Counter("http_requests_total", "Total HTTP requests", labels)
        counter.inc(10)
        prometheus = counter.to_prometheus()
        assert 'http_requests_total{method="GET"} 10' in prometheus


class TestGauge:
    """Tests for Gauge metric."""

    def test_set_and_get(self):
        gauge = Gauge("temperature", "Current temperature")
        gauge.set(23.5)
        assert gauge.get() == 23.5

    def test_inc_dec(self):
        gauge = Gauge("active_connections")
        gauge.set(10)
        gauge.inc()
        assert gauge.get() == 11
        gauge.dec()
        assert gauge.get() == 10
        gauge.inc(5)
        assert gauge.get() == 15
        gauge.dec(3)
        assert gauge.get() == 12

    def test_set_to_current_time(self):
        gauge = Gauge("last_update_time")
        before = time.time()
        gauge.set_to_current_time()
        after = time.time()
        assert before <= gauge.get() <= after

    def test_track_inprogress(self):
        gauge = Gauge("in_progress")
        assert gauge.get() == 0
        with gauge.track_inprogress():
            assert gauge.get() == 1
        assert gauge.get() == 0

    def test_track_inprogress_with_exception(self):
        gauge = Gauge("in_progress")
        try:
            with gauge.track_inprogress():
                assert gauge.get() == 1
                raise ValueError("test")
        except ValueError:
            pass
        assert gauge.get() == 0

    def test_prometheus_format(self):
        gauge = Gauge("temperature")
        gauge.set(23.5)
        assert "temperature 23.5" in gauge.to_prometheus()


class TestHistogram:
    """Tests for Histogram metric."""

    def test_observe(self):
        histogram = Histogram("request_duration", "Request duration")
        histogram.observe(0.123)
        assert histogram.get_count() == 1
        assert histogram.get_sum() == pytest.approx(0.123)

    def test_buckets(self):
        histogram = Histogram("duration", buckets=(0.1, 0.5, 1.0))
        histogram.observe(0.05)  # <= 0.1 (falls in bucket 0)
        histogram.observe(0.3)  # <= 0.5 but not <= 0.1 (falls in bucket 0 and 1)
        histogram.observe(0.8)  # <= 1.0 but not <= 0.5 (falls in buckets 0, 1, 2)
        histogram.observe(2.0)  # > 1.0 (only +inf)

        buckets = histogram.get_buckets()
        # Buckets are cumulative: count of observations <= bound
        assert buckets[0] == (0.1, 1)  # 1 observation <= 0.1
        assert buckets[1] == (0.5, 2)  # 2 observations <= 0.5 (0.05 + 0.3)
        assert buckets[2] == (1.0, 3)  # 3 observations <= 1.0 (0.05 + 0.3 + 0.8)
        assert buckets[3] == (float("inf"), 4)  # 4 total

    def test_time_context_manager(self):
        histogram = Histogram("operation_time")
        with histogram.time():
            time.sleep(0.01)
        assert histogram.get_count() == 1
        assert histogram.get_sum() >= 0.01

    def test_prometheus_format(self):
        histogram = Histogram("duration", buckets=(0.1, 0.5))
        histogram.observe(0.3)
        prometheus = histogram.to_prometheus()
        assert "duration_bucket" in prometheus
        assert "duration_sum" in prometheus
        assert "duration_count" in prometheus


class TestMetricsRegistry:
    """Tests for MetricsRegistry."""

    def test_counter_creation(self):
        registry = MetricsRegistry()
        counter1 = registry.counter("requests", "Total requests")
        counter2 = registry.counter("requests")
        assert counter1 is counter2

    def test_gauge_creation(self):
        registry = MetricsRegistry()
        gauge1 = registry.gauge("temperature", "Current temp")
        gauge2 = registry.gauge("temperature")
        assert gauge1 is gauge2

    def test_histogram_creation(self):
        registry = MetricsRegistry()
        h1 = registry.histogram("duration", "Request duration")
        h2 = registry.histogram("duration")
        assert h1 is h2

    def test_prefix(self):
        registry = MetricsRegistry(prefix="app_")
        counter = registry.counter("requests")
        assert counter.name == "app_requests"

    def test_collect(self):
        registry = MetricsRegistry()
        counter = registry.counter("requests")
        gauge = registry.gauge("active")
        histogram = registry.histogram("duration")

        counter.inc(10)
        gauge.set(5)
        histogram.observe(0.1)

        collected = registry.collect()
        assert collected["counters"]["requests"] == 10
        assert collected["gauges"]["active"] == 5
        assert collected["histograms"]["duration"]["count"] == 1

    def test_to_prometheus(self):
        registry = MetricsRegistry()
        registry.counter("requests", "Total requests").inc(5)
        prometheus = registry.to_prometheus()
        assert "# TYPE requests counter" in prometheus
        assert "requests 5" in prometheus

    def test_to_json(self):
        registry = MetricsRegistry()
        registry.counter("requests").inc(10)
        json_str = registry.to_json()
        data = json.loads(json_str)
        assert data["counters"]["requests"] == 10


class TestSpanAndTracer:
    """Tests for Span and Tracer."""

    def test_span_creation(self):
        context = SpanContext(trace_id="abc123", span_id="def456")
        span = Span("test_operation", context)
        assert span.name == "test_operation"
        assert span.context.trace_id == "abc123"

    def test_span_attributes(self):
        context = SpanContext(trace_id="abc", span_id="def")
        span = Span("test", context)
        span.set_attribute("http.method", "GET")
        span.set_attribute("http.url", "/api")
        assert span.attributes["http.method"] == "GET"
        assert span.attributes["http.url"] == "/api"

    def test_span_events(self):
        context = SpanContext(trace_id="abc", span_id="def")
        span = Span("test", context)
        span.add_event("cache_hit", {"key": "user:123"})
        assert len(span.events) == 1
        assert span.events[0]["name"] == "cache_hit"

    def test_span_status(self):
        context = SpanContext(trace_id="abc", span_id="def")
        span = Span("test", context)
        span.set_status("ok")
        assert span.status == "ok"

    def test_span_context_manager(self):
        context = SpanContext(trace_id="abc", span_id="def")
        with Span("test", context) as span:
            span.set_attribute("key", "value")
        assert span.end_time is not None
        assert span.status == "ok"

    def test_span_context_manager_with_error(self):
        context = SpanContext(trace_id="abc", span_id="def")
        try:
            with Span("test", context) as span:
                raise ValueError("test error")
        except ValueError:
            pass
        assert span.status == "error"

    def test_span_duration(self):
        context = SpanContext(trace_id="abc", span_id="def")
        span = Span("test", context)
        time.sleep(0.01)
        span.end()
        assert span.duration_ms >= 10

    def test_tracer_start_span(self):
        tracer = Tracer("test-service")
        span = tracer.start_span("operation")
        assert span.name == "operation"
        assert span.attributes["service.name"] == "test-service"

    def test_tracer_nested_spans(self):
        tracer = Tracer("test-service")
        parent = tracer.start_span("parent")
        child = tracer.start_span("child")
        assert child.context.parent_span_id == parent.context.span_id
        assert child.context.trace_id == parent.context.trace_id

    def test_tracer_get_spans(self):
        tracer = Tracer("test-service")
        tracer.start_span("op1")
        tracer.start_span("op2")
        spans = tracer.get_spans()
        assert len(spans) == 2

    def test_tracer_clear(self):
        tracer = Tracer("test-service")
        tracer.start_span("operation")
        tracer.clear()
        assert len(tracer.get_spans()) == 0

    def test_tracer_export_json(self):
        tracer = Tracer("test-service")
        with tracer.start_span("operation") as span:
            span.set_attribute("key", "value")
        json_str = tracer.export_json()
        data = json.loads(json_str)
        assert len(data) == 1
        assert data[0]["name"] == "operation"


class TestProfiler:
    """Tests for Profiler."""

    def test_measure(self):
        profiler = Profiler()
        with profiler.measure("operation"):
            time.sleep(0.01)
        samples = profiler.get_samples("operation")
        assert len(samples) == 1
        assert samples[0].duration_ms >= 10

    def test_measure_with_metadata(self):
        profiler = Profiler()
        with profiler.measure("operation", {"key": "value"}):
            pass
        samples = profiler.get_samples("operation")
        assert samples[0].metadata == {"key": "value"}

    def test_get_stats(self):
        profiler = Profiler()
        for _ in range(5):
            with profiler.measure("operation"):
                time.sleep(0.001)

        stats = profiler.get_stats("operation")
        assert stats["count"] == 5
        assert stats["min_ms"] > 0
        assert stats["max_ms"] >= stats["min_ms"]
        assert stats["avg_ms"] > 0

    def test_get_stats_empty(self):
        profiler = Profiler()
        stats = profiler.get_stats("nonexistent")
        assert stats["count"] == 0

    def test_get_all_stats(self):
        profiler = Profiler()
        with profiler.measure("op1"):
            pass
        with profiler.measure("op2"):
            pass
        all_stats = profiler.get_all_stats()
        assert "op1" in all_stats
        assert "op2" in all_stats

    def test_clear(self):
        profiler = Profiler()
        with profiler.measure("operation"):
            pass
        profiler.clear()
        assert len(profiler.get_samples("operation")) == 0

    def test_to_json(self):
        profiler = Profiler()
        with profiler.measure("operation"):
            pass
        json_str = profiler.to_json()
        data = json.loads(json_str)
        assert "operation" in data


class TestHealthChecker:
    """Tests for HealthChecker."""

    def test_register_and_check(self):
        checker = HealthChecker()
        checker.register(
            "test",
            lambda: HealthCheckResult(name="test", status=HealthStatus.HEALTHY),
        )
        result = checker.check("test")
        assert result.status == HealthStatus.HEALTHY

    def test_check_not_found(self):
        checker = HealthChecker()
        result = checker.check("nonexistent")
        assert result.status == HealthStatus.UNHEALTHY
        assert "not found" in result.message

    def test_check_exception(self):
        checker = HealthChecker()

        def failing_check():
            raise RuntimeError("Connection failed")

        checker.register("failing", failing_check)
        result = checker.check("failing")
        assert result.status == HealthStatus.UNHEALTHY
        assert "Connection failed" in result.message

    def test_check_all(self):
        checker = HealthChecker()
        checker.register(
            "db",
            lambda: HealthCheckResult(name="db", status=HealthStatus.HEALTHY),
        )
        checker.register(
            "cache",
            lambda: HealthCheckResult(name="cache", status=HealthStatus.HEALTHY),
        )
        results = checker.check_all()
        assert len(results) == 2

    def test_is_healthy(self):
        checker = HealthChecker()
        checker.register(
            "healthy",
            lambda: HealthCheckResult(name="healthy", status=HealthStatus.HEALTHY),
        )
        assert checker.is_healthy()

        checker.register(
            "unhealthy",
            lambda: HealthCheckResult(name="unhealthy", status=HealthStatus.UNHEALTHY),
        )
        assert not checker.is_healthy()

    def test_unregister(self):
        checker = HealthChecker()
        checker.register(
            "test",
            lambda: HealthCheckResult(name="test", status=HealthStatus.HEALTHY),
        )
        checker.unregister("test")
        result = checker.check("test")
        assert result.status == HealthStatus.UNHEALTHY

    def test_to_dict(self):
        checker = HealthChecker()
        checker.register(
            "db",
            lambda: HealthCheckResult(name="db", status=HealthStatus.HEALTHY),
        )
        data = checker.to_dict()
        assert data["status"] == "healthy"
        assert len(data["checks"]) == 1

    def test_to_json(self):
        checker = HealthChecker()
        checker.register(
            "db",
            lambda: HealthCheckResult(name="db", status=HealthStatus.HEALTHY),
        )
        json_str = checker.to_json()
        data = json.loads(json_str)
        assert data["status"] == "healthy"


class TestGlobalAccessors:
    """Tests for global accessor functions."""

    def test_get_registry(self):
        registry = get_registry()
        assert isinstance(registry, MetricsRegistry)
        # Should return same instance
        assert get_registry() is registry

    def test_get_tracer(self):
        tracer = get_tracer()
        assert isinstance(tracer, Tracer)

    def test_get_profiler(self):
        profiler = get_profiler()
        assert isinstance(profiler, Profiler)

    def test_get_health_checker(self):
        checker = get_health_checker()
        assert isinstance(checker, HealthChecker)
