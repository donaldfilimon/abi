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

// Re-export main types
pub const PrometheusServer = prometheus.PrometheusServer;
pub const MetricsCollector = prometheus.MetricsCollector;
pub const PerformanceSampler = sampling.PerformanceSampler;
pub const RegressionDetector = regression.RegressionDetector;
pub const HealthChecker = health.HealthChecker;
