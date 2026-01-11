//! SIMD/Vector Operation Benchmarks
//!
//! Industry-standard benchmarks for SIMD operations:
//! - Dot product (various sizes)
//! - Vector addition/subtraction
//! - Matrix multiplication
//! - Normalization
//! - Distance calculations (Euclidean, Cosine, Manhattan)
//! - Memory bandwidth tests
//! - Cache hierarchy analysis

const std = @import("std");
const framework = @import("framework.zig");

/// SIMD benchmark configuration
pub const SIMDBenchConfig = struct {
    /// Vector dimensions to benchmark
    dimensions: []const usize = &.{ 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096 },
    /// Number of vectors for batch operations
    batch_sizes: []const usize = &.{ 1, 10, 100, 1000, 10000 },
    /// Whether to include unaligned memory tests
    include_unaligned: bool = true,
    /// Whether to benchmark scalar fallbacks
    include_scalar_comparison: bool = true,
};

/// Benchmark results for SIMD operations
pub const SIMDResults = struct {
    dot_product: []framework.BenchResult,
    vector_add: []framework.BenchResult,
    vector_mul: []framework.BenchResult,
    normalize: []framework.BenchResult,
    euclidean_distance: []framework.BenchResult,
    cosine_similarity: []framework.BenchResult,
    matrix_mul: []framework.BenchResult,
    memory_bandwidth: []framework.BenchResult,
};

// ============================================================================
// Core SIMD Operations (for benchmarking)
// ============================================================================

/// Scalar dot product (baseline)
fn scalarDotProduct(a: []const f32, b: []const f32) f32 {
    var sum: f32 = 0.0;
    for (a, b) |x, y| {
        sum += x * y;
    }
    return sum;
}

/// SIMD dot product using Zig's vector types
fn simdDotProduct(comptime N: usize, a: []const f32, b: []const f32) f32 {
    const Vec = @Vector(N, f32);
    var sum: f32 = 0.0;
    var i: usize = 0;

    // Process N elements at a time
    while (i + N <= a.len) : (i += N) {
        const va: Vec = a[i..][0..N].*;
        const vb: Vec = b[i..][0..N].*;
        sum += @reduce(.Add, va * vb);
    }

    // Handle remainder
    while (i < a.len) : (i += 1) {
        sum += a[i] * b[i];
    }

    return sum;
}

/// SIMD vector addition
fn simdVectorAdd(comptime N: usize, a: []const f32, b: []const f32, result: []f32) void {
    const Vec = @Vector(N, f32);
    var i: usize = 0;

    while (i + N <= a.len) : (i += N) {
        const va: Vec = a[i..][0..N].*;
        const vb: Vec = b[i..][0..N].*;
        result[i..][0..N].* = va + vb;
    }

    while (i < a.len) : (i += 1) {
        result[i] = a[i] + b[i];
    }
}

/// SIMD vector multiplication (element-wise)
fn simdVectorMul(comptime N: usize, a: []const f32, b: []const f32, result: []f32) void {
    const Vec = @Vector(N, f32);
    var i: usize = 0;

    while (i + N <= a.len) : (i += N) {
        const va: Vec = a[i..][0..N].*;
        const vb: Vec = b[i..][0..N].*;
        result[i..][0..N].* = va * vb;
    }

    while (i < a.len) : (i += 1) {
        result[i] = a[i] * b[i];
    }
}

/// SIMD vector normalization (L2)
fn simdNormalize(comptime N: usize, v: []const f32, result: []f32) void {
    const norm_sq = simdDotProduct(N, v, v);
    const norm = @sqrt(norm_sq);

    if (norm == 0) {
        for (result) |*r| {
            r.* = 0;
        }
        return;
    }

    const inv_norm = 1.0 / norm;
    const Vec = @Vector(N, f32);
    const inv_vec: Vec = @splat(inv_norm);
    var i: usize = 0;

    while (i + N <= v.len) : (i += N) {
        const va: Vec = v[i..][0..N].*;
        result[i..][0..N].* = va * inv_vec;
    }

    while (i < v.len) : (i += 1) {
        result[i] = v[i] * inv_norm;
    }
}

/// SIMD Euclidean distance
fn simdEuclideanDistance(comptime N: usize, a: []const f32, b: []const f32) f32 {
    const Vec = @Vector(N, f32);
    var sum_sq: f32 = 0.0;
    var i: usize = 0;

    while (i + N <= a.len) : (i += N) {
        const va: Vec = a[i..][0..N].*;
        const vb: Vec = b[i..][0..N].*;
        const diff = va - vb;
        sum_sq += @reduce(.Add, diff * diff);
    }

    while (i < a.len) : (i += 1) {
        const diff = a[i] - b[i];
        sum_sq += diff * diff;
    }

    return @sqrt(sum_sq);
}

/// SIMD Cosine similarity
fn simdCosineSimilarity(comptime N: usize, a: []const f32, b: []const f32) f32 {
    const dot = simdDotProduct(N, a, b);
    const norm_a = @sqrt(simdDotProduct(N, a, a));
    const norm_b = @sqrt(simdDotProduct(N, b, b));

    if (norm_a == 0 or norm_b == 0) return 0;
    return dot / (norm_a * norm_b);
}

/// Naive matrix multiplication (for comparison)
fn scalarMatMul(a: []const f32, b: []const f32, c: []f32, n: usize) void {
    for (0..n) |i| {
        for (0..n) |j| {
            var sum: f32 = 0.0;
            for (0..n) |k| {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Cache-optimized blocked matrix multiplication
fn blockedMatMul(a: []const f32, b: []const f32, c: []f32, n: usize, block_size: usize) void {
    // Zero output
    @memset(c, 0);

    // Blocked multiplication for better cache utilization
    var i: usize = 0;
    while (i < n) : (i += block_size) {
        var j: usize = 0;
        while (j < n) : (j += block_size) {
            var k: usize = 0;
            while (k < n) : (k += block_size) {
                // Multiply blocks
                const i_end = @min(i + block_size, n);
                const j_end = @min(j + block_size, n);
                const k_end = @min(k + block_size, n);

                var ii = i;
                while (ii < i_end) : (ii += 1) {
                    var kk = k;
                    while (kk < k_end) : (kk += 1) {
                        const a_val = a[ii * n + kk];
                        var jj = j;
                        while (jj < j_end) : (jj += 1) {
                            c[ii * n + jj] += a_val * b[kk * n + jj];
                        }
                    }
                }
            }
        }
    }
}

/// SIMD-accelerated blocked matrix multiplication
fn simdMatMul(comptime N: usize, a: []const f32, b: []const f32, c: []f32, n: usize) void {
    const Vec = @Vector(N, f32);
    @memset(c, 0);

    for (0..n) |i| {
        for (0..n) |k| {
            const a_val: Vec = @splat(a[i * n + k]);
            var j: usize = 0;

            while (j + N <= n) : (j += N) {
                const b_vec: Vec = b[k * n + j ..][0..N].*;
                const c_vec: Vec = c[i * n + j ..][0..N].*;
                c[i * n + j ..][0..N].* = c_vec + a_val * b_vec;
            }

            // Handle remainder
            while (j < n) : (j += 1) {
                c[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }
}

// ============================================================================
// Benchmark Functions
// ============================================================================

pub fn runSIMDBenchmarks(allocator: std.mem.Allocator, config: SIMDBenchConfig) !void {
    var runner = framework.BenchmarkRunner.init(allocator);
    defer runner.deinit();

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("                    SIMD/VECTOR OPERATION BENCHMARKS\n", .{});
    std.debug.print("================================================================================\n\n", .{});

    // Dot product benchmarks
    try benchmarkDotProduct(allocator, &runner, config);

    // Vector operations
    try benchmarkVectorOps(allocator, &runner, config);

    // Distance calculations
    try benchmarkDistanceCalcs(allocator, &runner, config);

    // Matrix multiplication
    try benchmarkMatrixMul(allocator, &runner, config);

    // Memory bandwidth
    try benchmarkMemoryBandwidth(allocator, &runner);

    // Print summary
    runner.printSummaryDebug();
}

fn benchmarkDotProduct(
    allocator: std.mem.Allocator,
    runner: *framework.BenchmarkRunner,
    config: SIMDBenchConfig,
) !void {
    std.debug.print("[Dot Product Benchmarks]\n", .{});

    for (config.dimensions) |dim| {
        // Allocate aligned vectors
        const a = try allocator.alignedAlloc(f32, .@"64", dim);
        defer allocator.free(a);
        const b = try allocator.alignedAlloc(f32, .@"64", dim);
        defer allocator.free(b);

        // Initialize with random-ish values
        for (a, 0..) |*val, i| {
            val.* = @sin(@as(f32, @floatFromInt(i)) * 0.1);
        }
        for (b, 0..) |*val, i| {
            val.* = @cos(@as(f32, @floatFromInt(i)) * 0.1);
        }

        // Scalar benchmark (baseline)
        if (config.include_scalar_comparison) {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "dot_scalar_{d}", .{dim}) catch "dot_scalar";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "simd/dot_product",
                    .bytes_per_op = dim * @sizeOf(f32) * 2,
                    .warmup_iterations = 1000,
                    .min_time_ns = 500_000_000,
                },
                struct {
                    fn bench(va: []const f32, vb: []const f32) f32 {
                        return scalarDotProduct(va, vb);
                    }
                }.bench,
                .{ a, b },
            );

            std.debug.print("  {s}: {d:.0} ops/sec, {d:.2} GB/s\n", .{
                name,
                result.stats.opsPerSecond(),
                result.stats.throughputMBps(dim * @sizeOf(f32) * 2) / 1024.0,
            });
        }

        // SIMD benchmarks with different vector widths
        inline for ([_]usize{ 4, 8, 16 }) |width| {
            if (dim >= width) {
                var name_buf: [64]u8 = undefined;
                const name = std.fmt.bufPrint(&name_buf, "dot_simd{d}_{d}", .{ width, dim }) catch "dot_simd";

                const result = try runner.run(
                    .{
                        .name = name,
                        .category = "simd/dot_product",
                        .bytes_per_op = dim * @sizeOf(f32) * 2,
                        .warmup_iterations = 1000,
                        .min_time_ns = 500_000_000,
                    },
                    struct {
                        fn bench(va: []const f32, vb: []const f32) f32 {
                            return simdDotProduct(width, va, vb);
                        }
                    }.bench,
                    .{ a, b },
                );

                std.debug.print("  {s}: {d:.0} ops/sec, {d:.2} GB/s\n", .{
                    name,
                    result.stats.opsPerSecond(),
                    result.stats.throughputMBps(dim * @sizeOf(f32) * 2) / 1024.0,
                });
            }
        }
    }

    std.debug.print("\n", .{});
}

fn benchmarkVectorOps(
    allocator: std.mem.Allocator,
    runner: *framework.BenchmarkRunner,
    config: SIMDBenchConfig,
) !void {
    std.debug.print("[Vector Addition/Multiplication Benchmarks]\n", .{});

    for (config.dimensions) |dim| {
        const a = try allocator.alignedAlloc(f32, .@"64", dim);
        defer allocator.free(a);
        const b = try allocator.alignedAlloc(f32, .@"64", dim);
        defer allocator.free(b);
        const result_buf = try allocator.alignedAlloc(f32, .@"64", dim);
        defer allocator.free(result_buf);

        for (a, 0..) |*val, i| {
            val.* = @floatFromInt(i);
        }
        for (b, 0..) |*val, i| {
            val.* = @floatFromInt(i * 2);
        }

        // Vector add
        inline for ([_]usize{ 8, 16 }) |width| {
            if (dim >= width) {
                var name_buf: [64]u8 = undefined;
                const name = std.fmt.bufPrint(&name_buf, "vecadd_simd{d}_{d}", .{ width, dim }) catch "vecadd";

                const bench_result = try runner.run(
                    .{
                        .name = name,
                        .category = "simd/vector_ops",
                        .bytes_per_op = dim * @sizeOf(f32) * 3, // 2 read + 1 write
                        .warmup_iterations = 1000,
                        .min_time_ns = 500_000_000,
                    },
                    struct {
                        fn bench(va: []const f32, vb: []const f32, out: []f32) void {
                            simdVectorAdd(width, va, vb, out);
                        }
                    }.bench,
                    .{ a, b, result_buf },
                );

                std.debug.print("  {s}: {d:.0} ops/sec, {d:.2} GB/s\n", .{
                    name,
                    bench_result.stats.opsPerSecond(),
                    bench_result.stats.throughputMBps(dim * @sizeOf(f32) * 3) / 1024.0,
                });
            }
        }
    }

    std.debug.print("\n", .{});
}

fn benchmarkDistanceCalcs(
    allocator: std.mem.Allocator,
    runner: *framework.BenchmarkRunner,
    config: SIMDBenchConfig,
) !void {
    std.debug.print("[Distance Calculation Benchmarks]\n", .{});

    // Typical embedding dimensions used in ML
    const ml_dims = [_]usize{ 64, 128, 256, 384, 512, 768, 1024, 1536, 4096 };

    for (ml_dims) |dim| {
        var found = false;
        for (config.dimensions) |d| {
            if (d == dim) {
                found = true;
                break;
            }
        }
        if (!found and dim > config.dimensions[config.dimensions.len - 1]) continue;

        const a = try allocator.alignedAlloc(f32, .@"64", dim);
        defer allocator.free(a);
        const b = try allocator.alignedAlloc(f32, .@"64", dim);
        defer allocator.free(b);

        // Initialize normalized vectors (common for embeddings)
        var norm_a: f32 = 0;
        var norm_b: f32 = 0;
        for (a, 0..) |*val, i| {
            val.* = @sin(@as(f32, @floatFromInt(i)) * 0.01);
            norm_a += val.* * val.*;
        }
        for (b, 0..) |*val, i| {
            val.* = @cos(@as(f32, @floatFromInt(i)) * 0.01);
            norm_b += val.* * val.*;
        }
        norm_a = @sqrt(norm_a);
        norm_b = @sqrt(norm_b);
        for (a) |*val| val.* /= norm_a;
        for (b) |*val| val.* /= norm_b;

        // Euclidean distance
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "euclidean_{d}", .{dim}) catch "euclidean";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "simd/distance",
                    .bytes_per_op = dim * @sizeOf(f32) * 2,
                    .warmup_iterations = 1000,
                    .min_time_ns = 500_000_000,
                },
                struct {
                    fn bench(va: []const f32, vb: []const f32) f32 {
                        return simdEuclideanDistance(8, va, vb);
                    }
                }.bench,
                .{ a, b },
            );

            std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
        }

        // Cosine similarity
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "cosine_{d}", .{dim}) catch "cosine";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "simd/distance",
                    .bytes_per_op = dim * @sizeOf(f32) * 2,
                    .warmup_iterations = 1000,
                    .min_time_ns = 500_000_000,
                },
                struct {
                    fn bench(va: []const f32, vb: []const f32) f32 {
                        return simdCosineSimilarity(8, va, vb);
                    }
                }.bench,
                .{ a, b },
            );

            std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
        }
    }

    std.debug.print("\n", .{});
}

fn benchmarkMatrixMul(
    allocator: std.mem.Allocator,
    runner: *framework.BenchmarkRunner,
    _: SIMDBenchConfig,
) !void {
    std.debug.print("[Matrix Multiplication Benchmarks]\n", .{});

    const matrix_sizes = [_]usize{ 64, 128, 256, 512 };

    for (matrix_sizes) |n| {
        const size = n * n;
        const a = try allocator.alignedAlloc(f32, .@"64", size);
        defer allocator.free(a);
        const b = try allocator.alignedAlloc(f32, .@"64", size);
        defer allocator.free(b);
        const c = try allocator.alignedAlloc(f32, .@"64", size);
        defer allocator.free(c);

        // Initialize matrices
        for (a, 0..) |*val, i| {
            val.* = @floatFromInt(@mod(i, 10));
        }
        for (b, 0..) |*val, i| {
            val.* = @floatFromInt(@mod(i + 1, 10));
        }

        // Naive (scalar) matrix multiplication
        if (n <= 256) {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "matmul_naive_{d}x{d}", .{ n, n }) catch "matmul_naive";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "simd/matrix",
                    .bytes_per_op = size * @sizeOf(f32) * 3,
                    .warmup_iterations = 10,
                    .min_time_ns = 500_000_000,
                    .max_iterations = 1000,
                },
                struct {
                    fn bench(ma: []const f32, mb: []const f32, mc: []f32, dim: usize) void {
                        scalarMatMul(ma, mb, mc, dim);
                    }
                }.bench,
                .{ a, b, c, n },
            );

            const gflops = @as(f64, @floatFromInt(2 * n * n * n)) * result.stats.opsPerSecond() / 1e9;
            std.debug.print("  {s}: {d:.0} ops/sec ({d:.2} GFLOPS)\n", .{
                name,
                result.stats.opsPerSecond(),
                gflops,
            });
        }

        // Blocked matrix multiplication
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "matmul_blocked_{d}x{d}", .{ n, n }) catch "matmul_blocked";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "simd/matrix",
                    .bytes_per_op = size * @sizeOf(f32) * 3,
                    .warmup_iterations = 10,
                    .min_time_ns = 500_000_000,
                    .max_iterations = 1000,
                },
                struct {
                    fn bench(ma: []const f32, mb: []const f32, mc: []f32, dim: usize) void {
                        blockedMatMul(ma, mb, mc, dim, 32);
                    }
                }.bench,
                .{ a, b, c, n },
            );

            const gflops = @as(f64, @floatFromInt(2 * n * n * n)) * result.stats.opsPerSecond() / 1e9;
            std.debug.print("  {s}: {d:.0} ops/sec ({d:.2} GFLOPS)\n", .{
                name,
                result.stats.opsPerSecond(),
                gflops,
            });
        }

        // SIMD matrix multiplication
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "matmul_simd_{d}x{d}", .{ n, n }) catch "matmul_simd";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "simd/matrix",
                    .bytes_per_op = size * @sizeOf(f32) * 3,
                    .warmup_iterations = 10,
                    .min_time_ns = 500_000_000,
                    .max_iterations = 1000,
                },
                struct {
                    fn bench(ma: []const f32, mb: []const f32, mc: []f32, dim: usize) void {
                        simdMatMul(8, ma, mb, mc, dim);
                    }
                }.bench,
                .{ a, b, c, n },
            );

            const gflops = @as(f64, @floatFromInt(2 * n * n * n)) * result.stats.opsPerSecond() / 1e9;
            std.debug.print("  {s}: {d:.0} ops/sec ({d:.2} GFLOPS)\n", .{
                name,
                result.stats.opsPerSecond(),
                gflops,
            });
        }
    }

    std.debug.print("\n", .{});
}

fn benchmarkMemoryBandwidth(allocator: std.mem.Allocator, runner: *framework.BenchmarkRunner) !void {
    std.debug.print("[Memory Bandwidth Benchmarks]\n", .{});

    // Test different buffer sizes to analyze cache behavior
    const sizes = [_]usize{
        32 * 1024, // 32KB - L1 cache
        256 * 1024, // 256KB - L2 cache
        4 * 1024 * 1024, // 4MB - L3 cache
        32 * 1024 * 1024, // 32MB - Main memory
    };

    const size_names = [_][]const u8{ "L1_32KB", "L2_256KB", "L3_4MB", "RAM_32MB" };

    for (sizes, size_names) |size, size_name| {
        const buf = try allocator.alignedAlloc(u8, .@"64", size);
        defer allocator.free(buf);
        const dest = try allocator.alignedAlloc(u8, .@"64", size);
        defer allocator.free(dest);

        // Initialize
        @memset(buf, 0xAA);

        // Sequential read
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "memread_{s}", .{size_name}) catch "memread";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "memory/bandwidth",
                    .bytes_per_op = size,
                    .warmup_iterations = 10,
                    .min_time_ns = 500_000_000,
                },
                struct {
                    fn bench(b: []const u8) u64 {
                        var sum: u64 = 0;
                        for (b) |byte| {
                            sum +%= byte;
                        }
                        return sum;
                    }
                }.bench,
                .{buf},
            );

            std.debug.print("  {s}: {d:.2} GB/s\n", .{
                name,
                result.stats.throughputMBps(size) / 1024.0,
            });
        }

        // Sequential write
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "memwrite_{s}", .{size_name}) catch "memwrite";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "memory/bandwidth",
                    .bytes_per_op = size,
                    .warmup_iterations = 10,
                    .min_time_ns = 500_000_000,
                },
                struct {
                    fn bench(b: []u8) void {
                        @memset(b, 0x55);
                    }
                }.bench,
                .{dest},
            );

            std.debug.print("  {s}: {d:.2} GB/s\n", .{
                name,
                result.stats.throughputMBps(size) / 1024.0,
            });
        }

        // Copy (memcpy)
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "memcopy_{s}", .{size_name}) catch "memcopy";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "memory/bandwidth",
                    .bytes_per_op = size * 2, // Read + Write
                    .warmup_iterations = 10,
                    .min_time_ns = 500_000_000,
                },
                struct {
                    fn bench(d: []u8, s: []const u8) void {
                        @memcpy(d, s);
                    }
                }.bench,
                .{ dest, buf },
            );

            std.debug.print("  {s}: {d:.2} GB/s\n", .{
                name,
                result.stats.throughputMBps(size * 2) / 1024.0,
            });
        }
    }

    std.debug.print("\n", .{});
}

// ============================================================================
// Entry Point
// ============================================================================

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    try runSIMDBenchmarks(allocator, .{});
}

test "simd operations correctness" {
    const allocator = std.testing.allocator;

    // Test vectors
    const a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const b = [_]f32{ 8, 7, 6, 5, 4, 3, 2, 1 };

    // Dot product should match scalar
    const scalar_dot = scalarDotProduct(&a, &b);
    const simd_dot = simdDotProduct(4, &a, &b);
    try std.testing.expectApproxEqAbs(scalar_dot, simd_dot, 0.001);

    // Vector add
    var result = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    simdVectorAdd(4, &a, &b, &result);
    for (a, b, result) |va, vb, vr| {
        try std.testing.expectApproxEqAbs(va + vb, vr, 0.001);
    }

    // Euclidean distance
    const dist = simdEuclideanDistance(4, &a, &b);
    try std.testing.expect(dist > 0);

    // Cosine similarity (should be between -1 and 1)
    const cos_sim = simdCosineSimilarity(4, &a, &b);
    try std.testing.expect(cos_sim >= -1 and cos_sim <= 1);

    // Normalization
    var norm_result: [8]f32 = undefined;
    simdNormalize(4, &a, &norm_result);
    const norm_check = simdDotProduct(4, &norm_result, &norm_result);
    try std.testing.expectApproxEqAbs(norm_check, 1.0, 0.01);

    _ = allocator;
}
