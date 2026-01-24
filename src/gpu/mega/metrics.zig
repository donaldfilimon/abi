//! GPU Metrics and Prometheus Export
//!
//! Provides metrics collection for GPU workloads with Prometheus-compatible
//! export. Uses a snapshot pattern to avoid blocking during I/O operations.
//!
//! Note: This module uses local Counter/Gauge/Histogram types that are protected
//! by MetricsExporter.mutex. For standalone thread-safe primitives, use the
//! types from `observability.Counter`, `observability.Gauge`, etc.

const std = @import("std");
const backend_mod = @import("../backend.zig");
const obs = @import("../../observability/mod.zig");

/// Standard latency buckets in milliseconds for histogram tracking.
pub const default_latency_buckets = [_]f64{ 0.5, 1, 2.5, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000 };

/// A histogram value with bucket counts.
pub const HistogramValue = struct {
    buckets: [@sizeOf(@TypeOf(default_latency_buckets)) / @sizeOf(f64)]u64 = [_]u64{0} ** (default_latency_buckets.len),
    sum: f64 = 0,
    count: u64 = 0,

    /// Record a value into the histogram.
    pub fn observe(self: *HistogramValue, value: f64) void {
        self.sum += value;
        self.count += 1;
        for (&self.buckets, 0..) |*bucket, i| {
            if (value <= default_latency_buckets[i]) {
                bucket.* += 1;
            }
        }
    }

    /// Get the mean value.
    pub fn mean(self: HistogramValue) f64 {
        if (self.count == 0) return 0;
        return self.sum / @as(f64, @floatFromInt(self.count));
    }

    /// Estimate a percentile (approximate based on buckets).
    pub fn percentile(self: HistogramValue, p: f64) f64 {
        if (self.count == 0) return 0;
        const target = @as(f64, @floatFromInt(self.count)) * p;
        var cumulative: u64 = 0;
        for (self.buckets, 0..) |bucket, i| {
            cumulative += bucket;
            if (@as(f64, @floatFromInt(cumulative)) >= target) {
                return default_latency_buckets[i];
            }
        }
        return default_latency_buckets[default_latency_buckets.len - 1];
    }
};

/// Counter metric type.
pub const Counter = struct {
    value: u64 = 0,

    pub fn inc(self: *Counter) void {
        self.value += 1;
    }

    pub fn add(self: *Counter, n: u64) void {
        self.value += n;
    }
};

/// Gauge metric type.
pub const Gauge = struct {
    value: f64 = 0,

    pub fn set(self: *Gauge, v: f64) void {
        self.value = v;
    }

    pub fn inc(self: *Gauge) void {
        self.value += 1;
    }

    pub fn dec(self: *Gauge) void {
        self.value -= 1;
    }

    pub fn add(self: *Gauge, v: f64) void {
        self.value += v;
    }
};

/// Per-backend metrics.
pub const BackendMetrics = struct {
    workload_count: Counter = .{},
    workload_success: Counter = .{},
    workload_failure: Counter = .{},
    failover_count: Counter = .{},
    latency_histogram: HistogramValue = .{},
    active_workloads: Gauge = .{},
    energy_wh: Gauge = .{},

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
    mutex: std.Thread.Mutex,

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
                .global_workload_count = self.global_workload_count.value,
                .global_failover_count = self.global_failover_count.value,
            };
        };
        defer snapshot.metrics.deinit();

        // Build output without holding lock
        var output = std.ArrayList(u8).init(allocator);
        errdefer output.deinit();

        const writer = output.writer();

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
            const backend_name = @tagName(entry.key_ptr.*);
            const m = entry.value_ptr;

            try writer.print("gpu_mega_backend_workload_total{{backend=\"{s}\"}} {d}\n", .{ backend_name, m.workload_count.value });
        }
        try writer.print("\n", .{});

        try writer.print("# HELP gpu_mega_backend_success_total Successful workloads per backend\n", .{});
        try writer.print("# TYPE gpu_mega_backend_success_total counter\n", .{});

        iter = snapshot.metrics.iterator();
        while (iter.next()) |entry| {
            const backend_name = @tagName(entry.key_ptr.*);
            const m = entry.value_ptr;

            try writer.print("gpu_mega_backend_success_total{{backend=\"{s}\"}} {d}\n", .{ backend_name, m.workload_success.value });
        }
        try writer.print("\n", .{});

        try writer.print("# HELP gpu_mega_backend_failure_total Failed workloads per backend\n", .{});
        try writer.print("# TYPE gpu_mega_backend_failure_total counter\n", .{});

        iter = snapshot.metrics.iterator();
        while (iter.next()) |entry| {
            const backend_name = @tagName(entry.key_ptr.*);
            const m = entry.value_ptr;

            try writer.print("gpu_mega_backend_failure_total{{backend=\"{s}\"}} {d}\n", .{ backend_name, m.workload_failure.value });
        }
        try writer.print("\n", .{});

        try writer.print("# HELP gpu_mega_backend_failover_total Failover events per backend\n", .{});
        try writer.print("# TYPE gpu_mega_backend_failover_total counter\n", .{});

        iter = snapshot.metrics.iterator();
        while (iter.next()) |entry| {
            const backend_name = @tagName(entry.key_ptr.*);
            const m = entry.value_ptr;

            try writer.print("gpu_mega_backend_failover_total{{backend=\"{s}\"}} {d}\n", .{ backend_name, m.failover_count.value });
        }
        try writer.print("\n", .{});

        // Histogram metrics
        try writer.print("# HELP gpu_mega_backend_latency_ms Workload latency histogram\n", .{});
        try writer.print("# TYPE gpu_mega_backend_latency_ms histogram\n", .{});

        iter = snapshot.metrics.iterator();
        while (iter.next()) |entry| {
            const backend_name = @tagName(entry.key_ptr.*);
            const m = entry.value_ptr;

            var cumulative: u64 = 0;
            for (m.latency_histogram.buckets, 0..) |bucket, i| {
                cumulative += bucket;
                try writer.print("gpu_mega_backend_latency_ms_bucket{{backend=\"{s}\",le=\"{d}\"}} {d}\n", .{ backend_name, default_latency_buckets[i], cumulative });
            }
            try writer.print("gpu_mega_backend_latency_ms_bucket{{backend=\"{s}\",le=\"+Inf\"}} {d}\n", .{ backend_name, m.latency_histogram.count });
            try writer.print("gpu_mega_backend_latency_ms_sum{{backend=\"{s}\"}} {d}\n", .{ backend_name, m.latency_histogram.sum });
            try writer.print("gpu_mega_backend_latency_ms_count{{backend=\"{s}\"}} {d}\n", .{ backend_name, m.latency_histogram.count });
        }

        return try output.toOwnedSlice();
    }

    /// Get global workload count.
    pub fn getGlobalWorkloadCount(self: *MetricsExporter) u64 {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.global_workload_count.value;
    }

    /// Get global failover count.
    pub fn getGlobalFailoverCount(self: *MetricsExporter) u64 {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.global_failover_count.value;
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
    var h = HistogramValue{};

    h.observe(5.0);
    h.observe(15.0);
    h.observe(50.0);

    try std.testing.expectEqual(@as(u64, 3), h.count);
    try std.testing.expect(h.sum == 70.0);
    try std.testing.expect(h.mean() > 23.0 and h.mean() < 24.0);
}

test "backend metrics" {
    var m = BackendMetrics{};

    m.recordWorkload(10.0, true);
    m.recordWorkload(20.0, false);
    m.recordFailover();

    try std.testing.expectEqual(@as(u64, 2), m.workload_count.value);
    try std.testing.expectEqual(@as(u64, 1), m.workload_success.value);
    try std.testing.expectEqual(@as(u64, 1), m.workload_failure.value);
    try std.testing.expectEqual(@as(u64, 1), m.failover_count.value);
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
    try std.testing.expectEqual(@as(u64, 2), cuda_metrics.?.workload_count.value);
    try std.testing.expectEqual(@as(u64, 1), cuda_metrics.?.failover_count.value);

    const vulkan_metrics = exporter.getBackendMetrics(.vulkan);
    try std.testing.expect(vulkan_metrics != null);
    try std.testing.expectEqual(@as(u64, 1), vulkan_metrics.?.workload_count.value);
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
