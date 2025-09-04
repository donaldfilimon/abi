//! Profiling utilities for WDBX-AI utils
//!
//! Provides profiling and performance measurement utilities.

const std = @import("std");
const core = @import("../core/mod.zig");

// Re-export core performance functionality
pub const Timer = core.time.Timer;
pub const PerformanceMetrics = core.performance.PerformanceMetrics;

/// Simple profiler for tracking function performance
pub const Profiler = struct {
    name: []const u8,
    timer: Timer,
    
    const Self = @This();
    
    pub fn start(name: []const u8) Self {
        return Self{
            .name = name,
            .timer = Timer.start(),
        };
    }
    
    pub fn end(self: Self) u64 {
        const elapsed = self.timer.elapsed();
        core.log.debug("Profile [{s}]: {d}ns ({d:.2}ms)", .{
            self.name, elapsed, @as(f64, @floatFromInt(elapsed)) / 1_000_000.0
        });
        return elapsed;
    }
    
    pub fn endAndRecord(self: Self) !void {
        const elapsed = self.timer.elapsed();
        try core.performance.endMeasurement(self.name, self.timer);
    }
};

/// Profile a function call
pub fn profile(comptime name: []const u8, comptime func: anytype, args: anytype) @TypeOf(@call(.auto, func, args)) {
    const profiler = Profiler.start(name);
    defer _ = profiler.end();
    return @call(.auto, func, args);
}

/// Benchmark a function multiple times
pub fn benchmark(comptime name: []const u8, comptime func: anytype, args: anytype, iterations: u32) struct {
    avg_ns: f64,
    min_ns: u64,
    max_ns: u64,
    total_ns: u64,
} {
    var min_time: u64 = std.math.maxInt(u64);
    var max_time: u64 = 0;
    var total_time: u64 = 0;
    
    for (0..iterations) |_| {
        const timer = Timer.start();
        _ = @call(.auto, func, args);
        const elapsed = timer.elapsed();
        
        min_time = @min(min_time, elapsed);
        max_time = @max(max_time, elapsed);
        total_time += elapsed;
    }
    
    const avg_time = @as(f64, @floatFromInt(total_time)) / @as(f64, @floatFromInt(iterations));
    
    core.log.info("Benchmark [{s}] ({d} iterations):", .{ name, iterations });
    core.log.info("  Average: {d:.2}ns ({d:.2}ms)", .{ avg_time, avg_time / 1_000_000.0 });
    core.log.info("  Min: {d}ns", .{min_time});
    core.log.info("  Max: {d}ns", .{max_time});
    core.log.info("  Total: {d}ns", .{total_time});
    
    return .{
        .avg_ns = avg_time,
        .min_ns = min_time,
        .max_ns = max_time,
        .total_ns = total_time,
    };
}

test "profiling utilities" {
    const testing = std.testing;
    
    const TestFn = struct {
        fn testFunc(x: u32) u32 {
            return x * 2;
        }
    };
    
    const result = profile("test_func", TestFn.testFunc, .{5});
    try testing.expectEqual(@as(u32, 10), result);
    
    const profiler = Profiler.start("manual_test");
    std.time.sleep(100_000); // 0.1ms
    const elapsed = profiler.end();
    try testing.expect(elapsed >= 100_000);
}