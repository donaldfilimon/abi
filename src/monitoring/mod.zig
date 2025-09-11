//! WDBX Monitoring Module
//!
//! Provides comprehensive monitoring capabilities including:
//! - Prometheus metrics export
//! - CPU and memory sampling
//! - Performance regression detection
//! - Health checks and alerting
//! - Real-time performance tracking

const std = @import("std");

pub const prometheus = @import("prometheus.zig");
pub const sampling = @import("sampling.zig");
pub const regression = @import("regression.zig");
pub const health = @import("health.zig");
pub const tracing = @import("tracing.zig");
pub const memory_tracker = @import("memory_tracker.zig");
pub const performance_profiler = @import("performance_profiler.zig");
pub const performance = @import("performance.zig");

// Re-export main types
pub const PrometheusServer = prometheus.PrometheusServer;
pub const MetricsCollector = prometheus.MetricsCollector;
pub const PerformanceSampler = sampling.PerformanceSampler;
pub const RegressionDetector = regression.RegressionDetector;
pub const HealthChecker = health.HealthChecker;

// Re-export tracing utilities
pub const Tracer = tracing.Tracer;
pub const TraceId = tracing.TraceId;
pub const Span = tracing.Span;
pub const SpanId = tracing.SpanId;
pub const TraceContext = tracing.TraceContext;
pub const TracingError = tracing.TracingError;

// Re-export performance utilities
pub const MemoryTracker = memory_tracker.MemoryTracker;
pub const PerformanceProfiler = performance_profiler.PerformanceProfiler;
pub const PerformanceMetrics = performance.PerformanceMetrics;
