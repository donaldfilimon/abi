const std = @import("std");
const crypto = @import("../../shared/utils/crypto/mod.zig");

pub const BenchmarkResult = struct {
    name: []const u8,
    iterations: u64,
    duration_ns: u64,
    ops_per_sec: f64,
};

pub fn runBenchmarks(allocator: std.mem.Allocator) ![]BenchmarkResult {
    var list = std.ArrayList(BenchmarkResult).empty;
    errdefer list.deinit(allocator);

    try list.append(allocator, runHashBenchmark());
    try list.append(allocator, runVectorBenchmark());

    return list.toOwnedSlice(allocator);
}

fn runHashBenchmark() BenchmarkResult {
    const iterations: u64 = 20000;
    const data = "abi-benchmark";
    var hash: u64 = 0;
    const start = std.time.nanoTimestamp();
    var i: u64 = 0;
    while (i < iterations) : (i += 1) {
        hash ^= crypto.fnv1a64(data);
    }
    const elapsed = std.time.nanoTimestamp() - start;
    std.mem.doNotOptimizeAway(hash);
    return buildResult("fnv1a64", iterations, elapsed);
}

fn runVectorBenchmark() BenchmarkResult {
    const iterations: u64 = 50000;
    const vec_a = [_]f32{ 0.1, 0.2, 0.3, 0.4 };
    const vec_b = [_]f32{ 0.4, 0.3, 0.2, 0.1 };
    var sum: f32 = 0;
    const start = std.time.nanoTimestamp();
    var i: u64 = 0;
    while (i < iterations) : (i += 1) {
        sum += dot4(&vec_a, &vec_b);
    }
    const elapsed = std.time.nanoTimestamp() - start;
    std.mem.doNotOptimizeAway(sum);
    return buildResult("dot4", iterations, elapsed);
}

fn dot4(a: *const [4]f32, b: *const [4]f32) f32 {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

fn buildResult(name: []const u8, iterations: u64, elapsed_ns: i128) BenchmarkResult {
    const elapsed: u64 = @intCast(elapsed_ns);
    const seconds = @as(f64, @floatFromInt(elapsed)) / 1_000_000_000.0;
    const ops = if (seconds == 0) 0 else @as(f64, @floatFromInt(iterations)) / seconds;
    return .{
        .name = name,
        .iterations = iterations,
        .duration_ns = elapsed,
        .ops_per_sec = ops,
    };
}
