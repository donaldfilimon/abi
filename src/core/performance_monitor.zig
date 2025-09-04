//! Advanced Performance Monitoring Module
//! Modern performance tracking with inline optimizations and minimal overhead

const std = @import("std");
const core = @import("mod.zig");
const Allocator = core.Allocator;
const Logger = core.Logger;

/// Performance measurement scope with automatic timing
pub const PerformanceScope = struct {
    name: []const u8,
    start_time: i64,
    logger: ?*Logger,

    pub fn init(name: []const u8, logger: ?*Logger) PerformanceScope {
        return .{
            .name = name,
            .start_time = std.time.microTimestamp(),
            .logger = logger,
        };
    }

    pub fn deinit(self: *PerformanceScope) void {
        const elapsed = std.time.microTimestamp() - self.start_time;
        if (self.logger) |logger| {
            logger.info("Performance: {s} took {d:.3}ms", .{ self.name, @as(f64, @floatFromInt(elapsed)) / 1000.0 });
        }
    }
};

/// High-resolution timer with minimal overhead
pub const HighResTimer = struct {
    start_time: i64,

    pub fn start() HighResTimer {
        return .{ .start_time = std.time.microTimestamp() };
    }

    pub fn elapsed(self: HighResTimer) i64 {
        return std.time.microTimestamp() - self.start_time;
    }

    pub fn elapsedNs(self: HighResTimer) u64 {
        return @intCast(self.elapsed() * 1000);
    }

    pub fn elapsedMs(self: HighResTimer) f64 {
        return @as(f64, @floatFromInt(self.elapsed())) / 1000.0;
    }

    pub fn elapsedUs(self: HighResTimer) f64 {
        return @as(f64, @floatFromInt(self.elapsed()));
    }
};

/// Performance counter with automatic statistics
pub const PerformanceCounter = struct {
    count: u64 = 0,
    total_time: i64 = 0,
    min_time: i64 = std.math.maxInt(i64),
    max_time: i64 = 0,
    name: []const u8,

    pub fn init(name: []const u8) PerformanceCounter {
        return .{ .name = name };
    }

    pub fn record(self: *PerformanceCounter, elapsed_time: i64) void {
        self.count += 1;
        self.total_time += elapsed_time;
        self.min_time = @min(self.min_time, elapsed_time);
        self.max_time = @max(self.max_time, elapsed_time);
    }

    pub fn getStats(self: *const PerformanceCounter) PerformanceStats {
        return .{
            .name = self.name,
            .count = self.count,
            .total_time = self.total_time,
            .min_time = self.min_time,
            .max_time = self.max_time,
            .average_time = if (self.count > 0) @as(f64, @floatFromInt(self.total_time)) / @as(f64, @floatFromInt(self.count)) else 0.0,
        };
    }

    pub fn reset(self: *PerformanceCounter) void {
        self.count = 0;
        self.total_time = 0;
        self.min_time = std.math.maxInt(i64);
        self.max_time = 0;
    }
};

/// Performance statistics
pub const PerformanceStats = struct {
    name: []const u8,
    count: u64,
    total_time: i64,
    min_time: i64,
    max_time: i64,
    average_time: f64,

    pub fn print(self: *const PerformanceStats) void {
        std.debug.print("Performance Stats - {s}:\n", .{self.name});
        std.debug.print("  Count: {d}\n", .{self.count});
        std.debug.print("  Total Time: {d:.3}ms\n", .{@as(f64, @floatFromInt(self.total_time)) / 1000.0});
        std.debug.print("  Min Time: {d:.3}ms\n", .{@as(f64, @floatFromInt(self.min_time)) / 1000.0});
        std.debug.print("  Max Time: {d:.3}ms\n", .{@as(f64, @floatFromInt(self.max_time)) / 1000.0});
        std.debug.print("  Average Time: {d:.3}ms\n", .{self.average_time / 1000.0});
    }
};

/// Memory usage tracker
pub const MemoryTracker = struct {
    allocator: Allocator,
    allocations: std.AutoHashMap(usize, AllocationInfo),
    total_allocated: usize = 0,
    peak_allocated: usize = 0,
    allocation_count: u64 = 0,
    deallocation_count: u64 = 0,

    const AllocationInfo = struct {
        size: usize,
        timestamp: i64,
        stack_trace: ?[]const u8,
    };

    pub fn init(allocator: Allocator) MemoryTracker {
        return .{
            .allocator = allocator,
            .allocations = std.AutoHashMap(usize, AllocationInfo).init(allocator),
        };
    }

    pub fn deinit(self: *MemoryTracker) void {
        self.allocations.deinit();
    }

    pub fn trackAllocation(self: *MemoryTracker, ptr: [*]u8, size: usize) void {
        const addr = @intFromPtr(ptr);
        self.total_allocated += size;
        self.peak_allocated = @max(self.peak_allocated, self.total_allocated);
        self.allocation_count += 1;

        self.allocations.put(addr, .{
            .size = size,
            .timestamp = std.time.microTimestamp(),
            .stack_trace = null, // Could capture stack trace here
        }) catch {};
    }

    pub fn trackDeallocation(self: *MemoryTracker, ptr: [*]u8) void {
        const addr = @intFromPtr(ptr);
        if (self.allocations.get(addr)) |info| {
            self.total_allocated -= info.size;
            self.deallocation_count += 1;
            _ = self.allocations.remove(addr);
        }
    }

    pub fn getStats(self: *const MemoryTracker) MemoryStats {
        return .{
            .total_allocated = self.total_allocated,
            .peak_allocated = self.peak_allocated,
            .allocation_count = self.allocation_count,
            .deallocation_count = self.deallocation_count,
            .active_allocations = self.allocations.count(),
        };
    }

    pub fn printStats(self: *const MemoryTracker) void {
        const stats = self.getStats();
        std.debug.print("Memory Stats:\n", .{});
        std.debug.print("  Total Allocated: {d} bytes\n", .{stats.total_allocated});
        std.debug.print("  Peak Allocated: {d} bytes\n", .{stats.peak_allocated});
        std.debug.print("  Allocation Count: {d}\n", .{stats.allocation_count});
        std.debug.print("  Deallocation Count: {d}\n", .{stats.deallocation_count});
        std.debug.print("  Active Allocations: {d}\n", .{stats.active_allocations});
    }
};

/// Memory statistics
pub const MemoryStats = struct {
    total_allocated: usize,
    peak_allocated: usize,
    allocation_count: u64,
    deallocation_count: u64,
    active_allocations: usize,
};

/// Performance profiler with automatic scoping
pub const PerformanceProfiler = struct {
    allocator: Allocator,
    counters: std.AutoHashMap([]const u8, PerformanceCounter),
    memory_tracker: ?*MemoryTracker,
    logger: ?*Logger,

    pub fn init(allocator: Allocator, enable_memory_tracking: bool, logger: ?*Logger) !*PerformanceProfiler {
        const self = try allocator.create(PerformanceProfiler);
        errdefer allocator.destroy(self);

        self.* = .{
            .allocator = allocator,
            .counters = std.AutoHashMap([]const u8, PerformanceCounter).init(allocator),
            .memory_tracker = if (enable_memory_tracking) try allocator.create(MemoryTracker) else null,
            .logger = logger,
        };

        if (self.memory_tracker) |tracker| {
            tracker.* = MemoryTracker.init(allocator);
        }

        return self;
    }

    pub fn deinit(self: *PerformanceProfiler) void {
        if (self.memory_tracker) |tracker| {
            tracker.deinit();
            self.allocator.destroy(tracker);
        }

        var it = self.counters.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.counters.deinit();

        self.allocator.destroy(self);
    }

    /// Start timing a named operation
    pub fn startTimer(self: *PerformanceProfiler, name: []const u8) PerformanceScope {
        return PerformanceScope.init(name, self.logger);
    }

    /// Record timing for a named operation
    pub fn recordTiming(self: *PerformanceProfiler, name: []const u8, elapsed_time: i64) !void {
        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);

        const result = try self.counters.getOrPut(name_copy);
        if (!result.found_existing) {
            result.value_ptr.* = PerformanceCounter.init(name_copy);
        }

        result.value_ptr.record(elapsed_time);
    }

    /// Get performance counter for a named operation
    pub fn getCounter(self: *const PerformanceProfiler, name: []const u8) ?*const PerformanceCounter {
        var it = self.counters.iterator();
        while (it.next()) |entry| {
            if (std.mem.eql(u8, entry.key_ptr.*, name)) {
                return entry.value_ptr;
            }
        }
        return null;
    }

    /// Get all performance counters
    pub fn getAllCounters(self: *const PerformanceProfiler) []const PerformanceCounter {
        var result = std.ArrayList(PerformanceCounter).init(self.allocator);
        defer result.deinit();

        var it = self.counters.iterator();
        while (it.next()) |entry| {
            result.append(entry.value_ptr.*) catch continue;
        }

        return result.toOwnedSlice();
    }

    /// Print all performance statistics
    pub fn printAllStats(self: *const PerformanceProfiler) void {
        std.debug.print("\n=== Performance Profile ===\n", .{});

        var it = self.counters.iterator();
        while (it.next()) |entry| {
            const stats = entry.value_ptr.getStats();
            stats.print();
            std.debug.print("\n", .{});
        }

        if (self.memory_tracker) |tracker| {
            tracker.printStats();
        }

        std.debug.print("========================\n", .{});
    }

    /// Reset all performance counters
    pub fn resetAll(self: *PerformanceProfiler) void {
        var it = self.counters.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.reset();
        }

        if (self.memory_tracker) |tracker| {
            tracker.* = MemoryTracker.init(self.allocator);
        }
    }

    /// Get memory tracker
    pub fn getMemoryTracker(self: *PerformanceProfiler) ?*MemoryTracker {
        return self.memory_tracker;
    }
};

/// Inline performance measurement macro
pub const perf = struct {
    /// Measure performance of a block
    pub fn measure(name: []const u8, logger: ?*Logger, block: anytype) @TypeOf(block) {
        const timer = HighResTimer.start();
        defer {
            const elapsed = timer.elapsed();
            if (logger) |l| {
                l.info("Performance: {s} took {d:.3}ms", .{ name, @as(f64, @floatFromInt(elapsed)) / 1000.0 });
            }
        }
        return block;
    }

    /// Measure performance and return result
    pub fn measureResult(name: []const u8, logger: ?*Logger, block: anytype) @TypeOf(block) {
        const timer = HighResTimer.start();
        defer {
            const elapsed = timer.elapsed();
            if (logger) |l| {
                l.info("Performance: {s} took {d:.3}ms", .{ name, @as(f64, @floatFromInt(elapsed)) / 1000.0 });
            }
        }
        return block;
    }
};

test "performance monitoring basic functionality" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var profiler = try PerformanceProfiler.init(allocator, true, null);
    defer profiler.deinit();

    // Test timer
    const timer = HighResTimer.start();
    std.time.sleep(1000); // 1ms
    const elapsed = timer.elapsed();
    try testing.expect(elapsed > 0);

    // Test performance counter
    try profiler.recordTiming("test_operation", elapsed);
    const counter = profiler.getCounter("test_operation");
    try testing.expect(counter != null);
    try testing.expectEqual(@as(u64, 1), counter.?.count);

    // Test memory tracking
    const memory_tracker = profiler.getMemoryTracker();
    try testing.expect(memory_tracker != null);

    // Test performance scope
    {
        const scope = profiler.startTimer("test_scope");
        defer scope.deinit();
        std.time.sleep(1000); // 1ms
    }
}

test "performance monitoring inline macros" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var logger = Logger.init(allocator, .debug);
    defer logger.deinit();

    // Test measure macro
    const result = perf.measure("test_measure", &logger, struct {
        fn operation() u32 {
            std.time.sleep(1000); // 1ms
            return 42;
        }
    }.operation());

    try testing.expectEqual(@as(u32, 42), result);

    // Test measureResult macro
    const result2 = perf.measureResult("test_measure_result", &logger, struct {
        fn operation() u32 {
            std.time.sleep(1000); // 1ms
            return 84;
        }
    }.operation());

    try testing.expectEqual(@as(u32, 84), result2);
}
