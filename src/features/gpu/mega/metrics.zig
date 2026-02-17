//! GPU Metrics and Prometheus Export
//!
//! Provides metrics collection for GPU workloads with Prometheus-compatible
//! export. Uses shared observability primitives from the centralized metrics module.

const std = @import("std");
const time = @import("../../../services/shared/time.zig");
const sync = @import("../../../services/shared/sync.zig");
const backend_mod = @import("../backend.zig");
const core_metrics = @import("../../../services/shared/utils/metric_types.zig");

// Re-export shared types for API compatibility
pub const Counter = core_metrics.Counter;
pub const Gauge = core_metrics.Gauge;
pub const FloatGauge = core_metrics.FloatGauge;
pub const HistogramValue = core_metrics.LatencyHistogram;
pub const default_latency_buckets = core_metrics.default_latency_buckets;

/// Per-backend metrics.
pub const BackendMetrics = struct {
    workload_count: Counter = .{},
    workload_success: Counter = .{},
    workload_failure: Counter = .{},
    failover_count: Counter = .{},
    latency_histogram: HistogramValue = HistogramValue.initDefault(),
    active_workloads: FloatGauge = .{},
    energy_wh: FloatGauge = .{},

    /// Record a completed workload.
    pub fn recordWorkload(self: *BackendMetrics, latency_ms: f64, success: bool) void {
        self.workload_count.inc();
        if (success) {
            self.workload_success.inc();
        } else {
            self.workload_failure.inc();
        }
        self.latency_histogram.observe(latency_ms);
    }

    /// Record a failover event.
    pub fn recordFailover(self: *BackendMetrics) void {
        self.failover_count.inc();
    }
};

/// Metrics exporter with Prometheus format support.
pub const MetricsExporter = struct {
    allocator: std.mem.Allocator,
    metrics: std.AutoHashMap(backend_mod.Backend, BackendMetrics),
    global_workload_count: Counter = .{},
    global_failover_count: Counter = .{},
    mutex: sync.Mutex,

    pub fn init(allocator: std.mem.Allocator) !*MetricsExporter {
        const self = try allocator.create(MetricsExporter);
        self.* = .{
            .allocator = allocator,
            .metrics = std.AutoHashMap(backend_mod.Backend, BackendMetrics).init(allocator),
            .global_workload_count = .{},
            .global_failover_count = .{},
            .mutex = .{},
        };
        return self;
    }

    pub fn deinit(self: *MetricsExporter) void {
        self.metrics.deinit();
        self.allocator.destroy(self);
    }

    /// Record a workload for a backend.
    /// Uses getOrPut to avoid panics and ensure best-effort recording.
    pub fn recordWorkload(self: *MetricsExporter, backend: backend_mod.Backend, latency_ms: f64, success: bool) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Use getOrPut (not getOrPutAssumeCapacity) to prevent panic on growth
        const result = self.metrics.getOrPut(backend) catch {
            // Best-effort: if allocation fails, just skip recording
            return;
        };
        if (!result.found_existing) {
            result.value_ptr.* = .{};
        }
        result.value_ptr.recordWorkload(latency_ms, success);
        self.global_workload_count.inc();
    }

    /// Record a failover event.
    pub fn recordFailover(self: *MetricsExporter, from_backend: backend_mod.Backend, to_backend: backend_mod.Backend) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Record failover on source backend
        const from_result = self.metrics.getOrPut(from_backend) catch return;
        if (!from_result.found_existing) {
            from_result.value_ptr.* = .{};
        }
        from_result.value_ptr.recordFailover();

        // Ensure destination backend is tracked
        const to_result = self.metrics.getOrPut(to_backend) catch return;
        if (!to_result.found_existing) {
            to_result.value_ptr.* = .{};
        }

        self.global_failover_count.inc();
    }

    /// Get metrics for a specific backend.
    pub fn getBackendMetrics(self: *MetricsExporter, backend: backend_mod.Backend) ?BackendMetrics {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.metrics.get(backend);
    }

    /// Export metrics in Prometheus text format.
    /// Uses snapshot pattern: clones data under lock, writes without lock.
    pub fn exportPrometheus(self: *MetricsExporter, allocator: std.mem.Allocator) ![]u8 {
        // Snapshot under lock
        const snapshot = blk: {
            self.mutex.lock();
            defer self.mutex.unlock();

            var cloned = std.AutoHashMap(backend_mod.Backend, BackendMetrics).init(allocator);
            errdefer cloned.deinit();

            var iter = self.metrics.iterator();
            while (iter.next()) |entry| {
                try cloned.put(entry.key_ptr.*, entry.value_ptr.*);
            }

            break :blk .{
                .metrics = cloned,
                .global_workload_count = self.global_workload_count.get(),
                .global_failover_count = self.global_failover_count.get(),
            };
        };
        defer {
            var metrics = snapshot.metrics;
            metrics.deinit();
        }

        // Build output without holding lock
        var output = std.ArrayListUnmanaged(u8).empty;
        errdefer output.deinit(allocator);

        var aw: std.Io.Writer.Allocating = .fromArrayList(allocator, &output);
        const writer = &aw.writer;

        // Global metrics
        try writer.print("# HELP gpu_mega_workload_total Total workloads processed\n", .{});
        try writer.print("# TYPE gpu_mega_workload_total counter\n", .{});
        try writer.print("gpu_mega_workload_total {d}\n\n", .{snapshot.global_workload_count});

        try writer.print("# HELP gpu_mega_failover_total Total failover events\n", .{});
        try writer.print("# TYPE gpu_mega_failover_total counter\n", .{});
        try writer.print("gpu_mega_failover_total {d}\n\n", .{snapshot.global_failover_count});

        // Per-backend metrics
        try writer.print("# HELP gpu_mega_backend_workload_total Workloads per backend\n", .{});
        try writer.print("# TYPE gpu_mega_backend_workload_total counter\n", .{});

        var iter = snapshot.metrics.iterator();
        while (iter.next()) |entry| {
            const m = entry.value_ptr;

            try writer.print("gpu_mega_backend_workload_total{{backend=\"{t}\"}} {d}\n", .{ entry.key_ptr.*, m.workload_count.get() });
        }
        try writer.print("\n", .{});

        try writer.print("# HELP gpu_mega_backend_success_total Successful workloads per backend\n", .{});
        try writer.print("# TYPE gpu_mega_backend_success_total counter\n", .{});

        iter = snapshot.metrics.iterator();
        while (iter.next()) |entry| {
            const m = entry.value_ptr;

            try writer.print("gpu_mega_backend_success_total{{backend=\"{t}\"}} {d}\n", .{ entry.key_ptr.*, m.workload_success.get() });
        }
        try writer.print("\n", .{});

        try writer.print("# HELP gpu_mega_backend_failure_total Failed workloads per backend\n", .{});
        try writer.print("# TYPE gpu_mega_backend_failure_total counter\n", .{});

        iter = snapshot.metrics.iterator();
        while (iter.next()) |entry| {
            const m = entry.value_ptr;

            try writer.print("gpu_mega_backend_failure_total{{backend=\"{t}\"}} {d}\n", .{ entry.key_ptr.*, m.workload_failure.get() });
        }
        try writer.print("\n", .{});

        try writer.print("# HELP gpu_mega_backend_failover_total Failover events per backend\n", .{});
        try writer.print("# TYPE gpu_mega_backend_failover_total counter\n", .{});

        iter = snapshot.metrics.iterator();
        while (iter.next()) |entry| {
            const m = entry.value_ptr;

            try writer.print("gpu_mega_backend_failover_total{{backend=\"{t}\"}} {d}\n", .{ entry.key_ptr.*, m.failover_count.get() });
        }
        try writer.print("\n", .{});

        // Histogram metrics
        try writer.print("# HELP gpu_mega_backend_latency_ms Workload latency histogram\n", .{});
        try writer.print("# TYPE gpu_mega_backend_latency_ms histogram\n", .{});

        iter = snapshot.metrics.iterator();
        while (iter.next()) |entry| {
            const backend = entry.key_ptr.*;
            const m = entry.value_ptr;

            var cumulative: u64 = 0;
            for (m.latency_histogram.buckets, 0..) |bucket, i| {
                cumulative += bucket;
                try writer.print("gpu_mega_backend_latency_ms_bucket{{backend=\"{t}\",le=\"{d}\"}} {d}\n", .{ backend, default_latency_buckets[i], cumulative });
            }
            try writer.print("gpu_mega_backend_latency_ms_bucket{{backend=\"{t}\",le=\"+Inf\"}} {d}\n", .{ backend, m.latency_histogram.count });
            try writer.print("gpu_mega_backend_latency_ms_sum{{backend=\"{t}\"}} {d}\n", .{ backend, m.latency_histogram.sum });
            try writer.print("gpu_mega_backend_latency_ms_count{{backend=\"{t}\"}} {d}\n", .{ backend, m.latency_histogram.count });
        }

        var al = aw.toArrayList();
        return al.toOwnedSlice(allocator);
    }

    /// Get global workload count.
    pub fn getGlobalWorkloadCount(self: *MetricsExporter) u64 {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.global_workload_count.get();
    }

    /// Get global failover count.
    pub fn getGlobalFailoverCount(self: *MetricsExporter) u64 {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.global_failover_count.get();
    }

    /// Reset all metrics.
    pub fn reset(self: *MetricsExporter) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.metrics.clearRetainingCapacity();
        self.global_workload_count = .{};
        self.global_failover_count = .{};
    }
};

test "histogram value" {
    var h = HistogramValue.initDefault();

    h.observe(5.0);
    h.observe(15.0);
    h.observe(50.0);

    try std.testing.expectEqual(@as(u64, 3), h.getCount());
    try std.testing.expect(h.mean() > 23.0 and h.mean() < 24.0);
}

test "backend metrics" {
    var m = BackendMetrics{};

    m.recordWorkload(10.0, true);
    m.recordWorkload(20.0, false);
    m.recordFailover();

    try std.testing.expectEqual(@as(u64, 2), m.workload_count.get());
    try std.testing.expectEqual(@as(u64, 1), m.workload_success.get());
    try std.testing.expectEqual(@as(u64, 1), m.workload_failure.get());
    try std.testing.expectEqual(@as(u64, 1), m.failover_count.get());
}

test "metrics exporter" {
    const allocator = std.testing.allocator;
    const exporter = try MetricsExporter.init(allocator);
    defer exporter.deinit();

    // Record some workloads
    exporter.recordWorkload(.cuda, 10.0, true);
    exporter.recordWorkload(.cuda, 20.0, true);
    exporter.recordWorkload(.vulkan, 15.0, false);

    // Record a failover
    exporter.recordFailover(.cuda, .vulkan);

    // Check metrics
    try std.testing.expectEqual(@as(u64, 3), exporter.getGlobalWorkloadCount());
    try std.testing.expectEqual(@as(u64, 1), exporter.getGlobalFailoverCount());

    const cuda_metrics = exporter.getBackendMetrics(.cuda);
    try std.testing.expect(cuda_metrics != null);
    try std.testing.expectEqual(@as(u64, 2), cuda_metrics.?.workload_count.get());
    try std.testing.expectEqual(@as(u64, 1), cuda_metrics.?.failover_count.get());

    const vulkan_metrics = exporter.getBackendMetrics(.vulkan);
    try std.testing.expect(vulkan_metrics != null);
    try std.testing.expectEqual(@as(u64, 1), vulkan_metrics.?.workload_count.get());
}

test "prometheus export" {
    const allocator = std.testing.allocator;
    const exporter = try MetricsExporter.init(allocator);
    defer exporter.deinit();

    exporter.recordWorkload(.cuda, 10.0, true);
    exporter.recordWorkload(.vulkan, 5.0, true);

    const output = try exporter.exportPrometheus(allocator);
    defer allocator.free(output);

    // Verify output contains expected metrics
    try std.testing.expect(std.mem.indexOf(u8, output, "gpu_mega_workload_total") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "backend=\"cuda\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "backend=\"vulkan\"") != null);
}
