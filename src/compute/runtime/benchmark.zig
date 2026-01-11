const std = @import("std");
const crypto = @import("../../shared/utils/crypto/mod.zig");
const simd = @import("../../shared/simd.zig");

pub const BenchmarkResult = struct {
    name: []const u8,
    iterations: u64,
    duration_ns: u64,
    ops_per_sec: f64,
};

pub fn runBenchmarks(allocator: std.mem.Allocator) ![]BenchmarkResult {
    var list = std.ArrayListUnmanaged(BenchmarkResult).empty;
    errdefer list.deinit(allocator);

    try list.append(allocator, try runHashBenchmark());
    try list.append(allocator, try runVectorBenchmark());
    try list.append(allocator, try runMatrixMultiplyBenchmark());
    try list.append(allocator, try runCosineSimilarityBenchmark());
    try list.append(allocator, try runSIMDVectorDotBenchmark());

    return list.toOwnedSlice(allocator);
}

fn runHashBenchmark() !BenchmarkResult {
    const iterations: u64 = 500_000;
    const warmup_iterations: u64 = 50_000;
    const data = "abi-benchmark";
    warmUpHash(data, warmup_iterations);

    var hash: u64 = 0;
    var timer = try std.time.Timer.start();
    var i: u64 = 0;
    while (i < iterations) : (i += 1) {
        hash ^= crypto.fnv1a64(data);
    }
    const elapsed = timer.read();
    std.mem.doNotOptimizeAway(hash);
    return buildResult("fnv1a64", iterations, elapsed);
}

fn runVectorBenchmark() !BenchmarkResult {
    const iterations: u64 = 50000;
    const warmup_iterations: u64 = 10_000;
    const vec_a = [_]f32{ 0.1, 0.2, 0.3, 0.4 };
    const vec_b = [_]f32{ 0.4, 0.3, 0.2, 0.1 };
    warmUpDot4(&vec_a, &vec_b, warmup_iterations);

    var sum: f32 = 0;
    var timer = try std.time.Timer.start();
    var i: u64 = 0;
    while (i < iterations) : (i += 1) {
        sum += dot4(&vec_a, &vec_b);
    }
    const elapsed = timer.read();
    std.mem.doNotOptimizeAway(sum);
    return buildResult("dot4", iterations, elapsed);
}

fn dot4(a: *const [4]f32, b: *const [4]f32) f32 {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

fn warmUpHash(data: []const u8, iterations: u64) void {
    var hash: u64 = 0;
    var i: u64 = 0;
    while (i < iterations) : (i += 1) {
        hash ^= crypto.fnv1a64(data);
    }
    std.mem.doNotOptimizeAway(hash);
}

fn warmUpDot4(a: *const [4]f32, b: *const [4]f32, iterations: u64) void {
    var sum: f32 = 0;
    var i: u64 = 0;
    while (i < iterations) : (i += 1) {
        sum += dot4(a, b);
    }
    std.mem.doNotOptimizeAway(sum);
}

fn buildResult(name: []const u8, iterations: u64, elapsed_ns: u64) BenchmarkResult {
    const seconds = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;
    const ops = if (seconds == 0) 0 else @as(f64, @floatFromInt(iterations)) / seconds;
    return .{
        .name = name,
        .iterations = iterations,
        .duration_ns = elapsed_ns,
        .ops_per_sec = ops,
    };
}

/// Benchmark SIMD-vectorized matrix multiplication
fn runMatrixMultiplyBenchmark() !BenchmarkResult {
    const iterations: u64 = 1000;
    const size = 64;

    // Initialize matrices
    var a: [size * size]f32 = undefined;
    var b: [size * size]f32 = undefined;
    var result: [size * size]f32 = undefined;

    for (0..size * size) |i| {
        a[i] = @floatFromInt(i % 17);
        b[i] = @floatFromInt(i % 13);
    }

    // Warmup
    for (0..100) |_| {
        simd.matrixMultiply(&a, &b, &result, size, size, size);
    }
    std.mem.doNotOptimizeAway(&result);

    // Benchmark
    var timer = try std.time.Timer.start();
    var i: u64 = 0;
    while (i < iterations) : (i += 1) {
        simd.matrixMultiply(&a, &b, &result, size, size, size);
    }
    const elapsed = timer.read();
    std.mem.doNotOptimizeAway(&result);

    return buildResult("matmul_64x64_simd", iterations, elapsed);
}

/// Benchmark SIMD cosine similarity
fn runCosineSimilarityBenchmark() !BenchmarkResult {
    const iterations: u64 = 100_000;
    const vec_len = 128;

    var a: [vec_len]f32 = undefined;
    var b: [vec_len]f32 = undefined;

    for (0..vec_len) |i| {
        a[i] = @as(f32, @floatFromInt(i)) / @as(f32, vec_len);
        b[i] = @as(f32, @floatFromInt(vec_len - i)) / @as(f32, vec_len);
    }

    // Warmup
    var warmup_sum: f32 = 0;
    for (0..10_000) |_| {
        warmup_sum += simd.cosineSimilarity(&a, &b);
    }
    std.mem.doNotOptimizeAway(warmup_sum);

    // Benchmark
    var sum: f32 = 0;
    var timer = try std.time.Timer.start();
    var i: u64 = 0;
    while (i < iterations) : (i += 1) {
        sum += simd.cosineSimilarity(&a, &b);
    }
    const elapsed = timer.read();
    std.mem.doNotOptimizeAway(sum);

    return buildResult("cosine_sim_128d", iterations, elapsed);
}

/// Benchmark SIMD vector dot product with larger vectors
fn runSIMDVectorDotBenchmark() !BenchmarkResult {
    const iterations: u64 = 100_000;
    const vec_len = 256;

    var a: [vec_len]f32 = undefined;
    var b: [vec_len]f32 = undefined;

    for (0..vec_len) |i| {
        a[i] = @as(f32, @floatFromInt(i)) * 0.01;
        b[i] = @as(f32, @floatFromInt(vec_len - i)) * 0.01;
    }

    // Warmup
    var warmup_sum: f32 = 0;
    for (0..10_000) |_| {
        warmup_sum += simd.vectorDot(&a, &b);
    }
    std.mem.doNotOptimizeAway(warmup_sum);

    // Benchmark
    var sum: f32 = 0;
    var timer = try std.time.Timer.start();
    var i: u64 = 0;
    while (i < iterations) : (i += 1) {
        sum += simd.vectorDot(&a, &b);
    }
    const elapsed = timer.read();
    std.mem.doNotOptimizeAway(sum);

    return buildResult("dot_product_256d", iterations, elapsed);
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    const results = try runBenchmarks(allocator);
    defer allocator.free(results);

    std.debug.print("ABI Benchmarks\n", .{});
    for (results) |result| {
        std.debug.print(
            "  {s}: {d} iters, {d} ns, {d:.2} ops/sec\n",
            .{ result.name, result.iterations, result.duration_ns, result.ops_per_sec },
        );
    }
}
