//! High-performance monitoring and profiling system
//!
//! This module provides comprehensive performance monitoring capabilities including:
//! - Real-time metrics collection
//! - CPU profiling with sampling
//! - Memory allocation tracking
//! - Lock-free metric aggregation
//! - Tracy profiler integration
//! - Platform-specific optimizations

const std = @import("std");
// Note: core functionality is now imported through module dependencies
// TODO: Fix module import for Zig 0.16
// const platform = @import("../../shared/platform/mod.zig").platform;
const builtin = @import("builtin");
const lockfree = @import("../ai/data_structures/lockfree.zig");

/// Re-export commonly used types
pub const Allocator = std.mem.Allocator;

/// Performance monitoring specific error types
pub const PerformanceError = error{
    ProfilerNotAvailable,
    InvalidMetricName,
    BufferOverflow,
    SamplingFailed,
    TimerNotStarted,
    InvalidConfiguration,
    InsufficientPermissions,
    PlatformNotSupported,
    ResourceLimitExceeded,
};

/// Performance metric types
pub const MetricType = enum {
    counter,
    gauge,
    histogram,
    timer,
};

/// Performance metric value
pub const MetricValue = union(MetricType) {
    counter: u64,
    gauge: f64,
    histogram: HistogramData,
    timer: TimerData,
};

/// Histogram data for latency measurements
pub const HistogramData = struct {
    buckets: [16]u64 = [_]u64{0} ** 16,
    total_count: u64 = 0,
    total_sum: f64 = 0.0,

    const bucket_bounds = [_]f64{
        0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,  0.2,
        0.5,   1.0,   2.0,   5.0,  10.0, 20.0, 50.0, 100.0,
    };

    pub fn record(self: *HistogramData, value: f64) void {
        self.total_count += 1;
        self.total_sum += value;

        for (bucket_bounds, 0..) |bound, i| {
            if (value <= bound) {
                self.buckets[i] += 1;
                return;
            }
        }
        // Value exceeds all bounds, put in last bucket
        self.buckets[bucket_bounds.len - 1] += 1;
    }

    pub fn percentile(self: *const HistogramData, p: f64) f64 {
        const target = @as(u64, @intFromFloat(@as(f64, @floatFromInt(self.total_count)) * p));
        var cumulative: u64 = 0;

        for (self.buckets, 0..) |count, i| {
            cumulative += count;
            if (cumulative >= target) {
                return bucket_bounds[i];
            }
        }

        return bucket_bounds[bucket_bounds.len - 1];
    }
};

/// Timer data for duration measurements
pub const TimerData = struct {
    start_time: i128,
    total_duration: i128 = 0,
    count: u64 = 0,

    pub fn start(self: *TimerData) void {
        self.start_time = std.time.nanoTimestamp;
    }

    pub fn stop(self: *TimerData) void {
        const duration = std.time.nanoTimestamp - self.start_time;
        self.total_duration += duration;
        self.count += 1;
    }

    pub fn averageDuration(self: *const TimerData) f64 {
        if (self.count == 0) return 0.0;
        return @as(f64, @floatFromInt(self.total_duration)) / @as(f64, @floatFromInt(self.count));
    }
};

/// Performance metric entry
pub const Metric = struct {
    name: []const u8,
    value: MetricValue,
    timestamp: i128,
    labels: std.StringHashMap([]const u8),

    pub fn init(allocator: std.mem.Allocator, name: []const u8, value: MetricValue) !Metric {
        return Metric{
            .name = try allocator.dupe(u8, name),
            .value = value,
            .timestamp = std.time.nanoTimestamp,
            .labels = std.StringHashMap([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *Metric, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        var iterator = self.labels.iterator();
        while (iterator.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        self.labels.deinit();
    }

    pub fn addLabel(self: *Metric, allocator: std.mem.Allocator, key: []const u8, value: []const u8) !void {
        const key_copy = try allocator.dupe(u8, key);
        const value_copy = try allocator.dupe(u8, value);
        try self.labels.put(key_copy, value_copy);
    }
};

/// CPU profiler with sampling
pub const CPUProfiler = struct {
    samples: std.ArrayList(Sample),
    sampling_rate: u32,
    running: std.atomic.Value(bool),
    thread: ?std.Thread = null,

    const Sample = struct {
        timestamp: i128,
        instruction_pointer: usize,
        thread_id: u32,
        cpu_id: u32,
    };

    pub fn init(allocator: std.mem.Allocator, sampling_rate: u32) CPUProfiler {
        return CPUProfiler{
            .samples = std.ArrayList(Sample).initCapacity(allocator, 0) catch return error.OutOfMemory,
            .sampling_rate = sampling_rate,
            .running = std.atomic.Value(bool).init(false),
        };
    }

    pub fn deinit(self: *CPUProfiler) void {
        self.stop();
        self.samples.deinit();
    }

    pub fn start(self: *CPUProfiler) !void {
        if (self.running.load(.acquire)) return;

        self.running.store(true, .release);
        self.thread = try std.Thread.spawn(.{}, samplingLoop, .{self});
    }

    pub fn stop(self: *CPUProfiler) void {
        if (!self.running.load(.acquire)) return;

        self.running.store(false, .release);
        if (self.thread) |thread| {
            thread.join();
            self.thread = null;
        }
    }

    fn samplingLoop(self: *CPUProfiler) void {
        const interval_ns = 1_000_000_000 / self.sampling_rate;

        while (self.running.load(.acquire)) {
            // Collect sample
            const sample = Sample{
                .timestamp = std.time.nanoTimestamp,
                .instruction_pointer = @returnAddress(),
                .thread_id = @intCast(std.Thread.getCurrentId()),
                .cpu_id = getCurrentCPU(),
            };

            self.samples.append(sample) catch continue;

            std.Thread.sleep(interval_ns);
        }
    }

    fn getCurrentCPU() u32 {
        return switch (builtin.os.tag) {
            .linux => 0,
            else => 0, // Fallback for other platforms
        };
    }
};

/// Memory allocation tracker
pub const MemoryTracker = struct {
    allocations: lockfree.lockFreeHashMap(usize, AllocationInfo),
    total_allocated: std.atomic.Value(u64),
    total_freed: std.atomic.Value(u64),
    peak_usage: std.atomic.Value(u64),

    const AllocationInfo = struct {
        size: usize,
        timestamp: i128,
        stack_trace: [8]usize,
    };

    pub fn init(allocator: std.mem.Allocator) !MemoryTracker {
        return MemoryTracker{
            .allocations = try lockfree.lockFreeHashMap(usize, AllocationInfo).init(allocator, 4096),
            .total_allocated = std.atomic.Value(u64).init(0),
            .total_freed = std.atomic.Value(u64).init(0),
            .peak_usage = std.atomic.Value(u64).init(0),
        };
    }

    pub fn deinit(self: *MemoryTracker) void {
        self.allocations.deinit();
    }

    pub fn recordAllocation(self: *MemoryTracker, ptr: usize, size: usize) void {
        const info = AllocationInfo{
            .size = size,
            .timestamp = std.time.nanoTimestamp,
            .stack_trace = captureStackTrace(),
        };

        _ = self.allocations.put(ptr, info) catch return;

        const new_total = self.total_allocated.fetchAdd(@intCast(size), .release) + @as(u64, @intCast(size));
        const current_peak = self.peak_usage.load(.acquire);
        if (new_total > current_peak) {
            _ = self.peak_usage.compareAndSwap(current_peak, new_total, .release, .acquire);
        }
    }

    pub fn recordDeallocation(self: *MemoryTracker, ptr: usize) void {
        if (self.allocations.get(ptr)) |info| {
            _ = self.total_freed.fetchAdd(@intCast(info.size), .release);
        }
    }

    fn captureStackTrace() [8]usize {
        var stack_trace = [_]usize{0} ** 8;
        var stack_iterator = std.debug.StackIterator.init(@returnAddress(), @frameAddress());

        var i: usize = 0;
        while (stack_iterator.next()) |address| {
            if (i >= stack_trace.len) break;
            stack_trace[i] = address;
            i += 1;
        }

        return stack_trace;
    }

    pub fn getCurrentUsage(self: *const MemoryTracker) u64 {
        return self.total_allocated.load(.acquire) - self.total_freed.load(.acquire);
    }

    pub fn getPeakUsage(self: *const MemoryTracker) u64 {
        return self.peak_usage.load(.acquire);
    }
};

/// Global performance monitoring system
pub const PerformanceMonitor = struct {
    metrics: lockfree.lockFreeHashMap([]const u8, Metric),
    cpu_profiler: CPUProfiler,
    memory_tracker: MemoryTracker,
    allocator: std.mem.Allocator,
    enabled: bool = false,

    var instance: ?*PerformanceMonitor = null;

    pub fn init(allocator: std.mem.Allocator) !*PerformanceMonitor {
        if (instance) |existing| return existing;

        const self = try allocator.create(PerformanceMonitor);
        self.* = PerformanceMonitor{
            .metrics = try lockfree.lockFreeHashMap([]const u8, Metric).init(allocator, 1024),
            .cpu_profiler = CPUProfiler.init(allocator, 1000), // 1kHz sampling
            .memory_tracker = try MemoryTracker.init(allocator),
            .allocator = allocator,
            .enabled = true,
        };

        instance = self;
        return self;
    }

    pub fn deinit(self: *PerformanceMonitor) void {
        self.cpu_profiler.deinit();
        self.memory_tracker.deinit();
        self.metrics.deinit();
        self.allocator.destroy(self);
        instance = null;
    }

    pub fn recordMetric(self: *PerformanceMonitor, name: []const u8, value: MetricValue) !void {
        if (!self.enabled) return;

        const metric = try Metric.init(self.allocator, name, value);
        _ = try self.metrics.put(name, metric);
    }

    pub fn startProfiling(self: *PerformanceMonitor) !void {
        try self.cpu_profiler.start();
    }

    pub fn stopProfiling(self: *PerformanceMonitor) void {
        self.cpu_profiler.stop();
    }

    pub fn getMetric(self: *PerformanceMonitor, name: []const u8) ?Metric {
        return self.metrics.get(name);
    }
};

/// Tracy profiler integration (when enabled)
pub const TracyProfiler = struct {
    pub fn zoneName(comptime name: []const u8) void {
        if (@hasDecl(@import("build_options"), "enable_tracy") and @import("build_options").enable_tracy) {
            // Tracy zone implementation would go here
            _ = name;
        }
    }

    pub fn zoneStart() void {
        if (@hasDecl(@import("build_options"), "enable_tracy") and @import("build_options").enable_tracy) {
            // Tracy zone start implementation
        }
    }

    pub fn zoneEnd() void {
        if (@hasDecl(@import("build_options"), "enable_tracy") and @import("build_options").enable_tracy) {
            // Tracy zone end implementation
        }
    }

    pub fn plot(name: []const u8, value: f64) void {
        if (@hasDecl(@import("build_options"), "enable_tracy") and @import("build_options").enable_tracy) {
            _ = name;
            _ = value;
            // Tracy plot implementation
        }
    }
};

/// Global performance monitoring functions
var global_monitor: ?*PerformanceMonitor = null;

pub fn init() !void {
    if (global_monitor != null) return;
    global_monitor = try PerformanceMonitor.init(std.heap.page_allocator);
}

pub fn deinit() void {
    if (global_monitor) |monitor| {
        monitor.deinit();
        global_monitor = null;
    }
}

pub fn recordMetric(name: []const u8, value: f64) void {
    if (global_monitor) |monitor| {
        monitor.recordMetric(name, MetricValue{ .gauge = value }) catch return;
    }
}

pub fn recordCounter(name: []const u8, value: u64) void {
    if (global_monitor) |monitor| {
        monitor.recordMetric(name, MetricValue{ .counter = value }) catch return;
    }
}

pub fn recordLatency(name: []const u8, duration_ns: u64) void {
    if (global_monitor) |monitor| {
        var histogram = HistogramData{};
        histogram.record(@as(f64, @floatFromInt(duration_ns)) / 1_000_000.0); // Convert to ms
        monitor.recordMetric(name, MetricValue{ .histogram = histogram }) catch return;
    }
}

/// Timer utility for measuring execution time
pub const Timer = struct {
    start_time: i128,
    name: []const u8,

    pub fn start(comptime name: []const u8) Timer {
        TracyProfiler.zoneName(name);
        TracyProfiler.zoneStart();

        return Timer{
            .start_time = std.time.nanoTimestamp,
            .name = name,
        };
    }

    pub fn stop(self: Timer) void {
        const duration = std.time.nanoTimestamp - self.start_time;
        recordLatency(self.name, @intCast(duration));

        TracyProfiler.zoneEnd();
    }
};

/// Convenient macro for timing function execution
pub fn timed(comptime name: []const u8, func: anytype) @TypeOf(func()) {
    const timer = Timer.start(name);
    defer timer.stop();
    return func();
}

test "performance monitoring" {
    try init();
    defer deinit();

    // Test metric recording
    recordMetric("test_gauge", 42.0);
    recordCounter("test_counter", 100);
    recordLatency("test_latency", 1_000_000); // 1ms

    // Test timer
    const timer = Timer.start("test_timer");
    std.time.sleep(1_000_000); // 1ms
    timer.stop();
}

test "histogram percentiles" {
    const testing = std.testing;

    var histogram = HistogramData{};

    // Record some values
    histogram.record(0.001);
    histogram.record(0.005);
    histogram.record(0.01);
    histogram.record(0.1);
    histogram.record(1.0);

    try testing.expect(histogram.total_count == 5);
    try testing.expect(histogram.percentile(0.5) >= 0.01);
}
