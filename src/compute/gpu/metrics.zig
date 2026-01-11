//! Comprehensive GPU metrics collection and monitoring.
//!
//! Provides detailed performance metrics, resource tracking, and
//! real-time monitoring for GPU operations across all backends.

const std = @import("std");

/// Metric type classification.
pub const MetricType = enum {
    counter,
    gauge,
    histogram,
    summary,
};

/// Time series data point.
pub const DataPoint = struct {
    timestamp: i64,
    value: f64,
};

/// Histogram bucket for latency tracking.
pub const HistogramBucket = struct {
    upper_bound: f64,
    count: u64,
};

/// Percentile calculation for latency metrics.
pub const Percentiles = struct {
    p50: f64 = 0,
    p90: f64 = 0,
    p95: f64 = 0,
    p99: f64 = 0,
    p999: f64 = 0,
};

/// Kernel execution metrics.
pub const KernelMetrics = struct {
    name: []const u8,
    invocation_count: u64 = 0,
    total_time_ns: u64 = 0,
    min_time_ns: u64 = std.math.maxInt(u64),
    max_time_ns: u64 = 0,
    avg_time_ns: f64 = 0,
    percentiles: Percentiles = .{},
    last_invocation: i64 = 0,

    pub fn record(self: *KernelMetrics, duration_ns: u64) void {
        self.invocation_count += 1;
        self.total_time_ns += duration_ns;
        self.last_invocation = std.time.timestamp();

        if (duration_ns < self.min_time_ns) {
            self.min_time_ns = duration_ns;
        }
        if (duration_ns > self.max_time_ns) {
            self.max_time_ns = duration_ns;
        }

        self.avg_time_ns = @as(f64, @floatFromInt(self.total_time_ns)) /
            @as(f64, @floatFromInt(self.invocation_count));
    }

    pub fn throughput(self: *const KernelMetrics) f64 {
        if (self.avg_time_ns == 0) return 0;
        return 1.0e9 / self.avg_time_ns; // Operations per second
    }
};

/// Memory transfer metrics.
pub const TransferMetrics = struct {
    direction: enum { host_to_device, device_to_host, device_to_device },
    transfer_count: u64 = 0,
    total_bytes: u64 = 0,
    total_time_ns: u64 = 0,
    bandwidth_gbps: f64 = 0,
    avg_latency_ns: f64 = 0,

    pub fn record(self: *TransferMetrics, bytes: u64, duration_ns: u64) void {
        self.transfer_count += 1;
        self.total_bytes += bytes;
        self.total_time_ns += duration_ns;

        // Calculate bandwidth in GB/s
        if (duration_ns > 0) {
            const seconds = @as(f64, @floatFromInt(duration_ns)) / 1.0e9;
            const gigabytes = @as(f64, @floatFromInt(bytes)) / 1.0e9;
            self.bandwidth_gbps = gigabytes / seconds;
        }

        self.avg_latency_ns = @as(f64, @floatFromInt(self.total_time_ns)) /
            @as(f64, @floatFromInt(self.transfer_count));
    }
};

/// Device utilization metrics.
pub const DeviceMetrics = struct {
    device_id: i32,
    compute_utilization_percent: f64 = 0,
    memory_utilization_percent: f64 = 0,
    temperature_celsius: f64 = 0,
    power_usage_watts: f64 = 0,
    memory_allocated_bytes: u64 = 0,
    memory_free_bytes: u64 = 0,
    clock_speed_mhz: u32 = 0,
    last_updated: i64 = 0,

    pub fn update(self: *DeviceMetrics) void {
        self.last_updated = std.time.timestamp();
    }
};

/// Error rate tracking.
pub const ErrorMetrics = struct {
    total_errors: u64 = 0,
    errors_by_type: std.EnumArray(ErrorType, u64),
    last_error_time: i64 = 0,
    error_rate_per_second: f64 = 0,

    pub const ErrorType = enum {
        initialization,
        memory_allocation,
        kernel_launch,
        synchronization,
        device_lost,
        timeout,
        other,
    };

    pub fn init() ErrorMetrics {
        return .{
            .errors_by_type = std.EnumArray(ErrorType, u64).initFill(0),
        };
    }

    pub fn recordError(self: *ErrorMetrics, error_type: ErrorType) void {
        self.total_errors += 1;
        const current = self.errors_by_type.get(error_type);
        self.errors_by_type.set(error_type, current + 1);
        self.last_error_time = std.time.timestamp();
    }
};

/// Comprehensive GPU metrics collector.
pub const MetricsCollector = struct {
    allocator: std.mem.Allocator,
    kernel_metrics: std.StringHashMapUnmanaged(KernelMetrics),
    transfer_metrics: [3]TransferMetrics,
    device_metrics: std.AutoHashMapUnmanaged(i32, DeviceMetrics),
    error_metrics: ErrorMetrics,
    collection_start: i64,
    total_kernel_invocations: u64,
    total_memory_allocated: u64,
    total_memory_freed: u64,
    peak_memory_usage: u64,
    mutex: std.Thread.Mutex,

    /// Initialize the metrics collector.
    pub fn init(allocator: std.mem.Allocator) MetricsCollector {
        return .{
            .allocator = allocator,
            .kernel_metrics = .{},
            .transfer_metrics = [_]TransferMetrics{
                .{ .direction = .host_to_device },
                .{ .direction = .device_to_host },
                .{ .direction = .device_to_device },
            },
            .device_metrics = .{},
            .error_metrics = ErrorMetrics.init(),
            .collection_start = std.time.timestamp(),
            .total_kernel_invocations = 0,
            .total_memory_allocated = 0,
            .total_memory_freed = 0,
            .peak_memory_usage = 0,
            .mutex = .{},
        };
    }

    /// Deinitialize the collector.
    pub fn deinit(self: *MetricsCollector) void {
        var iter = self.kernel_metrics.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.kernel_metrics.deinit(self.allocator);
        self.device_metrics.deinit(self.allocator);
        self.* = undefined;
    }

    /// Record a kernel execution.
    pub fn recordKernel(self: *MetricsCollector, name: []const u8, duration_ns: u64) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const result = try self.kernel_metrics.getOrPut(self.allocator, name);
        if (!result.found_existing) {
            result.key_ptr.* = try self.allocator.dupe(u8, name);
            result.value_ptr.* = .{ .name = result.key_ptr.* };
        }

        result.value_ptr.record(duration_ns);
        self.total_kernel_invocations += 1;
    }

    /// Record a memory transfer.
    pub fn recordTransfer(
        self: *MetricsCollector,
        direction: enum { host_to_device, device_to_host, device_to_device },
        bytes: u64,
        duration_ns: u64,
    ) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const idx: usize = switch (direction) {
            .host_to_device => 0,
            .device_to_host => 1,
            .device_to_device => 2,
        };

        self.transfer_metrics[idx].record(bytes, duration_ns);
    }

    /// Record memory allocation.
    pub fn recordAllocation(self: *MetricsCollector, bytes: u64) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.total_memory_allocated += bytes;
        const current_usage = self.total_memory_allocated - self.total_memory_freed;
        if (current_usage > self.peak_memory_usage) {
            self.peak_memory_usage = current_usage;
        }
    }

    /// Record memory deallocation.
    pub fn recordDeallocation(self: *MetricsCollector, bytes: u64) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.total_memory_freed += bytes;
    }

    /// Update device metrics.
    pub fn updateDevice(self: *MetricsCollector, device_id: i32, metrics: DeviceMetrics) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        try self.device_metrics.put(self.allocator, device_id, metrics);
    }

    /// Record an error.
    pub fn recordError(self: *MetricsCollector, error_type: ErrorMetrics.ErrorType) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.error_metrics.recordError(error_type);
    }

    /// Get summary of all metrics.
    pub fn getSummary(self: *MetricsCollector) Summary {
        self.mutex.lock();
        defer self.mutex.unlock();

        const uptime_seconds = @as(f64, @floatFromInt(std.time.timestamp() - self.collection_start));
        const current_memory = self.total_memory_allocated - self.total_memory_freed;

        var total_kernel_time_ns: u64 = 0;
        var kernel_count: usize = 0;
        var iter = self.kernel_metrics.valueIterator();
        while (iter.next()) |metrics| {
            total_kernel_time_ns += metrics.total_time_ns;
            kernel_count += 1;
        }

        const avg_kernel_time_ns = if (self.total_kernel_invocations > 0)
            @as(f64, @floatFromInt(total_kernel_time_ns)) / @as(f64, @floatFromInt(self.total_kernel_invocations))
        else
            0.0;

        return .{
            .uptime_seconds = uptime_seconds,
            .total_kernels = kernel_count,
            .total_kernel_invocations = self.total_kernel_invocations,
            .avg_kernel_time_ns = avg_kernel_time_ns,
            .kernels_per_second = if (uptime_seconds > 0)
                @as(f64, @floatFromInt(self.total_kernel_invocations)) / uptime_seconds
            else
                0.0,
            .total_memory_allocated = self.total_memory_allocated,
            .total_memory_freed = self.total_memory_freed,
            .current_memory_usage = current_memory,
            .peak_memory_usage = self.peak_memory_usage,
            .total_h2d_transfers = self.transfer_metrics[0].transfer_count,
            .total_d2h_transfers = self.transfer_metrics[1].transfer_count,
            .total_d2d_transfers = self.transfer_metrics[2].transfer_count,
            .avg_h2d_bandwidth_gbps = self.transfer_metrics[0].bandwidth_gbps,
            .avg_d2h_bandwidth_gbps = self.transfer_metrics[1].bandwidth_gbps,
            .avg_d2d_bandwidth_gbps = self.transfer_metrics[2].bandwidth_gbps,
            .total_errors = self.error_metrics.total_errors,
            .error_rate_per_second = if (uptime_seconds > 0)
                @as(f64, @floatFromInt(self.error_metrics.total_errors)) / uptime_seconds
            else
                0.0,
        };
    }

    /// Get metrics for a specific kernel.
    pub fn getKernelMetrics(self: *MetricsCollector, name: []const u8) ?KernelMetrics {
        self.mutex.lock();
        defer self.mutex.unlock();

        return self.kernel_metrics.get(name);
    }

    /// Get all kernel names.
    pub fn getKernelNames(self: *MetricsCollector, allocator: std.mem.Allocator) ![][]const u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        var names = std.ArrayListUnmanaged([]const u8).empty;
        errdefer names.deinit(allocator);

        var iter = self.kernel_metrics.keyIterator();
        while (iter.next()) |name| {
            try names.append(allocator, name.*);
        }

        return names.toOwnedSlice(allocator);
    }

    /// Reset all metrics.
    pub fn reset(self: *MetricsCollector) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        var iter = self.kernel_metrics.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.kernel_metrics.clearRetainingCapacity();

        for (&self.transfer_metrics) |*metrics| {
            metrics.* = .{ .direction = metrics.direction };
        }

        self.error_metrics = ErrorMetrics.init();
        self.collection_start = std.time.timestamp();
        self.total_kernel_invocations = 0;
        self.total_memory_allocated = 0;
        self.total_memory_freed = 0;
        self.peak_memory_usage = 0;
    }

    /// Export metrics in JSON format.
    pub fn exportJson(self: *MetricsCollector, writer: anytype) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const summary = self.getSummary();

        try writer.writeAll("{\n");
        try writer.print("  \"uptime_seconds\": {d:.2},\n", .{summary.uptime_seconds});
        try writer.print("  \"total_kernel_invocations\": {d},\n", .{summary.total_kernel_invocations});
        try writer.print("  \"kernels_per_second\": {d:.2},\n", .{summary.kernels_per_second});
        try writer.print("  \"avg_kernel_time_ns\": {d:.2},\n", .{summary.avg_kernel_time_ns});
        try writer.print("  \"current_memory_usage\": {d},\n", .{summary.current_memory_usage});
        try writer.print("  \"peak_memory_usage\": {d},\n", .{summary.peak_memory_usage});
        try writer.print("  \"total_errors\": {d},\n", .{summary.total_errors});
        try writer.print("  \"error_rate_per_second\": {d:.4}\n", .{summary.error_rate_per_second});
        try writer.writeAll("}\n");
    }
};

/// Metrics summary.
pub const Summary = struct {
    uptime_seconds: f64,
    total_kernels: usize,
    total_kernel_invocations: u64,
    avg_kernel_time_ns: f64,
    kernels_per_second: f64,
    total_memory_allocated: u64,
    total_memory_freed: u64,
    current_memory_usage: u64,
    peak_memory_usage: u64,
    total_h2d_transfers: u64,
    total_d2h_transfers: u64,
    total_d2d_transfers: u64,
    avg_h2d_bandwidth_gbps: f64,
    avg_d2h_bandwidth_gbps: f64,
    avg_d2d_bandwidth_gbps: f64,
    total_errors: u64,
    error_rate_per_second: f64,
};

test "metrics collection" {
    const allocator = std.testing.allocator;
    var collector = MetricsCollector.init(allocator);
    defer collector.deinit();

    // Record some kernel executions
    try collector.recordKernel("test_kernel", 1000);
    try collector.recordKernel("test_kernel", 1500);
    try collector.recordKernel("other_kernel", 2000);

    const summary = collector.getSummary();
    try std.testing.expectEqual(@as(u64, 3), summary.total_kernel_invocations);

    const kernel_metrics = collector.getKernelMetrics("test_kernel").?;
    try std.testing.expectEqual(@as(u64, 2), kernel_metrics.invocation_count);
    try std.testing.expectApproxEqAbs(@as(f64, 1250.0), kernel_metrics.avg_time_ns, 1.0);
}

test "memory tracking" {
    const allocator = std.testing.allocator;
    var collector = MetricsCollector.init(allocator);
    defer collector.deinit();

    collector.recordAllocation(1024);
    collector.recordAllocation(2048);
    collector.recordDeallocation(1024);

    const summary = collector.getSummary();
    try std.testing.expectEqual(@as(u64, 2048), summary.current_memory_usage);
    try std.testing.expectEqual(@as(u64, 3072), summary.peak_memory_usage);
}

test "transfer metrics" {
    const allocator = std.testing.allocator;
    var collector = MetricsCollector.init(allocator);
    defer collector.deinit();

    collector.recordTransfer(.host_to_device, 1024 * 1024, 1_000_000); // 1 MB in 1 ms
    collector.recordTransfer(.device_to_host, 2048 * 1024, 2_000_000); // 2 MB in 2 ms

    const summary = collector.getSummary();
    try std.testing.expectEqual(@as(u64, 1), summary.total_h2d_transfers);
    try std.testing.expectEqual(@as(u64, 1), summary.total_d2h_transfers);
}
