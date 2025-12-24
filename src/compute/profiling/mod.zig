//! Profiling and metrics collection module
//!
//! Thread-safe metrics collection for compute engine.
//! Feature-gated: only compiled when enable_profiling is true.

const std = @import("std");

const build_options = @import("build_options");

pub const MetricsConfig = struct {
    sample_rate_ns: u64 = 1_000_000,
    histogram_buckets: []const u64 = &.{ 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000 },
    enable_worker_stats: bool = true,
    enable_memory_stats: bool = true,
};

pub const MetricsCollector = struct {
    config: MetricsConfig,
    allocator: std.mem.Allocator,
    worker_stats: []WorkerMetrics,
    task_histogram: Histogram,
    mutex: std.Thread.Mutex,

    pub fn init(allocator: std.mem.Allocator, cfg: MetricsConfig, worker_count: usize) !MetricsCollector {
        const worker_stats = try allocator.alloc(WorkerMetrics, worker_count);
        @memset(worker_stats, WorkerMetrics{});

        const histogram = try Histogram.init(allocator, cfg.histogram_buckets);

        return MetricsCollector{
            .config = cfg,
            .allocator = allocator,
            .worker_stats = worker_stats,
            .task_histogram = histogram,
            .mutex = std.Thread.Mutex{},
        };
    }

    pub fn deinit(self: *MetricsCollector) void {
        self.allocator.free(self.worker_stats);
        self.task_histogram.deinit(self.allocator);
    }

    pub fn recordTaskExecution(self: *MetricsCollector, worker_id: u32, duration_ns: u64) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (worker_id < self.worker_stats.len) {
            const stats = &self.worker_stats[worker_id];
            stats.tasks_executed += 1;
            stats.total_execution_ns += duration_ns;
            stats.min_execution_ns = @min(stats.min_execution_ns, duration_ns);
            stats.max_execution_ns = @max(stats.max_execution_ns, duration_ns);
        }

        self.task_histogram.record(duration_ns);
    }

    pub fn getWorkerStats(self: *MetricsCollector, worker_id: u32) ?WorkerMetrics {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (worker_id < self.worker_stats.len) {
            return self.worker_stats[worker_id];
        }

        return null;
    }

    pub fn getSummary(self: *MetricsCollector) MetricsSummary {
        self.mutex.lock();
        defer self.mutex.unlock();

        var total_tasks: u64 = 0;
        var total_execution_ns: u64 = 0;
        var min_duration_ns: u64 = std.math.maxInt(u64);
        var max_duration_ns: u64 = 0;

        for (self.worker_stats) |stats| {
            total_tasks += stats.tasks_executed;
            total_execution_ns += stats.total_execution_ns;
            if (stats.tasks_executed > 0) {
                min_duration_ns = @min(min_duration_ns, stats.min_execution_ns);
                max_duration_ns = @max(max_duration_ns, stats.max_execution_ns);
            }
        }

        const avg_duration_ns = if (total_tasks > 0) total_execution_ns / total_tasks else 0;

        return MetricsSummary{
            .total_tasks = total_tasks,
            .total_execution_ns = total_execution_ns,
            .avg_execution_ns = avg_duration_ns,
            .min_execution_ns = if (total_tasks > 0) min_duration_ns else 0,
            .max_execution_ns = max_duration_ns,
            .task_histogram = self.task_histogram.cloneSnapshot(self.allocator) catch &.{},
        };
    }
};

pub const WorkerMetrics = struct {
    tasks_executed: u64 = 0,
    total_execution_ns: u64 = 0,
    min_execution_ns: u64 = std.math.maxInt(u64),
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
    buckets: []const u64,
    counts: []u64,
    total: u64,

    pub fn init(allocator: std.mem.Allocator, bucket_boundaries: []const u64) !Histogram {
        const counts = try allocator.alloc(u64, bucket_boundaries.len + 1);
        @memset(counts, 0);

        return Histogram{
            .buckets = bucket_boundaries,
            .counts = counts,
            .total = 0,
        };
    }

    pub fn deinit(self: *Histogram, allocator: std.mem.Allocator) void {
        allocator.free(self.counts);
    }

    pub fn record(self: *Histogram, value: u64) void {
        self.total += 1;

        var bucket_index: usize = 0;
        while (bucket_index < self.buckets.len) : (bucket_index += 1) {
            if (value <= self.buckets[bucket_index]) {
                self.counts[bucket_index] += 1;
                return;
            }
        }

        self.counts[self.buckets.len] += 1;
    }

    pub fn cloneSnapshot(self: *Histogram, allocator: std.mem.Allocator) ![]u64 {
        const snapshot = try allocator.alloc(u64, self.counts.len);
        @memcpy(snapshot, self.counts);
        return snapshot;
    }
};

pub const DEFAULT_METRICS_CONFIG = MetricsConfig{};
