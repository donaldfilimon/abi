const std = @import("std");

pub const BenchmarkResult = struct {
    name: []const u8,
    iterations: u64,
    duration_ns: u64,
    ops_per_sec: f64,
};

pub fn executeBenchmark(allocator: std.mem.Allocator, comptime name: []const u8, benchmark_fn: fn (*anyopaque) void) BenchmarkResult!void {
    std.debug.print("Running: {s}\n", .{name});

    var timer = try std.time.Timer.start();

    var i: u64 = 0;
    const iterations: u64 = 100_000;
    const start = timer.read();

    while (i < iterations) : (i += 1) {
        benchmark_fn(&{});
    }

    const duration_ns = timer.read() - start;
    const seconds = @as(f64, @floatFromInt(duration_ns)) / 1_000_000_000.0;
    const ops_per_sec = @as(f64, @floatFromInt(iterations)) / seconds;

    std.debug.print("  iterations: {d}\n", .{iterations});
    std.debug.print("  duration: {d} ns\n", .{duration_ns});
    std.debug.print("  avg: {d} ns/op\n", .{duration_ns / iterations});
    std.debug.print("  ops/sec: {d:.0}\n", .{ops_per_sec});

    return BenchmarkResult{
        .name = try allocator.dupe(u8, name),
        .iterations = iterations,
        .duration_ns = duration_ns,
        .ops_per_sec = ops_per_sec,
    };
}
