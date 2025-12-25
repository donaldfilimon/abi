//! Disabled profiling stubs
//!
//! Provides no-op metrics collectors when profiling is disabled so that the
//! compute surface area remains usable without the feature flag.

const std = @import("std");

pub const MetricsConfig = struct {
    sample_rate_ns: u64 = 1_000_000,
    histogram_buckets: []const u64 = &.{},
    enable_worker_stats: bool = false,
    enable_memory_stats: bool = false,
};

pub const WorkerMetrics = struct {
    tasks_executed: u64 = 0,
    total_execution_ns: u64 = 0,
    min_execution_ns: u64 = 0,
    max_execution_ns: u64 = 0,
};

pub const MetricsSummary = struct {
    total_tasks: u64,
    total_execution_ns: u64,
    avg_execution_ns: u64,
    min_execution_ns: u64,
    max_execution_ns: u64,
    task_histogram: []const u64,
};

pub const Histogram = struct {
    pub fn init(_: std.mem.Allocator, _: []const u64) !Histogram {
        return Histogram{};
    }

    pub fn deinit(self: *Histogram, _: std.mem.Allocator) void {
        _ = self;
    }

    pub fn record(self: *Histogram, _: u64) void {
        _ = self;
    }

    pub fn cloneSnapshot(self: *Histogram, _: std.mem.Allocator) ![]u64 {
        _ = self;
        return &.{};
    }
};

pub const MetricsCollector = struct {
    config: MetricsConfig,

    pub fn init(_: std.mem.Allocator, cfg: MetricsConfig, _: usize) !MetricsCollector {
        return MetricsCollector{ .config = cfg };
    }

    pub fn deinit(self: *MetricsCollector) void {
        _ = self;
    }

    pub fn recordTaskExecution(self: *MetricsCollector, _: u32, _: u64) void {
        _ = self;
    }

    pub fn recordTaskComplete(self: *MetricsCollector, _: u32, _: u64) void {
        _ = self;
    }

    pub fn getWorkerStats(self: *MetricsCollector, _: u32) ?WorkerMetrics {
        _ = self;
        return null;
    }

    pub fn getSummary(self: *MetricsCollector) MetricsSummary {
        _ = self;
        return MetricsSummary{
            .total_tasks = 0,
            .total_execution_ns = 0,
            .avg_execution_ns = 0,
            .min_execution_ns = 0,
            .max_execution_ns = 0,
            .task_histogram = &.{},
        };
    }
};

pub const DEFAULT_METRICS_CONFIG = MetricsConfig{};
