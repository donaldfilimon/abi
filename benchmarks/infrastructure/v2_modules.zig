//! v2 Module Benchmarks
//!
//! Performance benchmarks for modules integrated from abi-system-v2.0:
//! - SIMD activation functions (softmax, GELU, SiLU, ReLU)
//! - SIMD BLAS operations (SAXPY, euclidean distance, reductions)
//! - Matrix operations (multiply, transpose, matvec)
//! - SwissMap hash table (put, get, iterate)
//! - Allocator combinators (tracking, limiting, fallback)

const std = @import("std");
const abi = @import("abi");
const framework = @import("../system/framework.zig");

// v2 modules accessed via abi re-exports
const simd = abi.services.shared.simd;
const matrix_mod = abi.services.shared.matrix;
const tensor_mod = abi.services.shared.tensor;
const utils = abi.services.shared.utils;

pub const V2BenchConfig = struct {
    /// Vector sizes for SIMD benchmarks
    vector_sizes: []const usize = &.{ 64, 256, 1024, 4096 },
    /// Matrix dimensions (square)
    matrix_sizes: []const usize = &.{ 32, 64, 128, 256 },
    /// Number of entries for SwissMap
    map_sizes: []const usize = &.{ 100, 1000, 10000 },
    /// Number of messages for channel throughput
    channel_messages: usize = 10_000,
};

// ============================================================================
// SIMD Activation Function Benchmarks
// ============================================================================

fn benchSoftmax(allocator: std.mem.Allocator, runner: *framework.BenchmarkRunner, config: V2BenchConfig) !void {
    std.debug.print("[v2 SIMD Activation Functions]\n", .{});

    for (config.vector_sizes) |dim| {
        const data = try allocator.alloc(f32, dim);
        defer allocator.free(data);
        const out = try allocator.alloc(f32, dim);
        defer allocator.free(out);

        // Init with values in a reasonable range for softmax
        for (data, 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt(i % 20)) * 0.1 - 1.0;
        }

        // softmax (out-of-place, v2 kernel)
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "softmax_{d}", .{dim}) catch "softmax";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "v2/activation",
                    .bytes_per_op = dim * @sizeOf(f32) * 2,
                    .warmup_iterations = 500,
                    .min_time_ns = 100_000_000,
                },
                struct {
                    fn bench(src: []const f32, dst: []f32) void {
                        simd.softmax(src, dst);
                    }
                }.bench,
                .{ data, out },
            );
            std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
        }

        // softmaxInPlace
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "softmax_inplace_{d}", .{dim}) catch "softmax_ip";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "v2/activation",
                    .bytes_per_op = dim * @sizeOf(f32),
                    .warmup_iterations = 500,
                    .min_time_ns = 100_000_000,
                },
                struct {
                    fn bench(src: []const f32, dst: []f32) void {
                        @memcpy(dst, src);
                        simd.softmaxInPlace(dst);
                    }
                }.bench,
                .{ data, out },
            );
            std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
        }

        // GELU
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "gelu_{d}", .{dim}) catch "gelu";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "v2/activation",
                    .bytes_per_op = dim * @sizeOf(f32),
                    .warmup_iterations = 500,
                    .min_time_ns = 100_000_000,
                },
                struct {
                    fn bench(src: []const f32, dst: []f32) void {
                        @memcpy(dst, src);
                        simd.geluInPlace(dst);
                    }
                }.bench,
                .{ data, out },
            );
            std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
        }

        // SiLU
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "silu_{d}", .{dim}) catch "silu";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "v2/activation",
                    .bytes_per_op = dim * @sizeOf(f32),
                    .warmup_iterations = 500,
                    .min_time_ns = 100_000_000,
                },
                struct {
                    fn bench(src: []const f32, dst: []f32) void {
                        @memcpy(dst, src);
                        simd.siluInPlace(dst);
                    }
                }.bench,
                .{ data, out },
            );
            std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
        }

        // ReLU
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "relu_{d}", .{dim}) catch "relu";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "v2/activation",
                    .bytes_per_op = dim * @sizeOf(f32),
                    .warmup_iterations = 500,
                    .min_time_ns = 100_000_000,
                },
                struct {
                    fn bench(src: []const f32, dst: []f32) void {
                        @memcpy(dst, src);
                        simd.reluInPlace(dst);
                    }
                }.bench,
                .{ data, out },
            );
            std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
        }

        // RMSNorm
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "rmsnorm_{d}", .{dim}) catch "rmsnorm";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "v2/activation",
                    .bytes_per_op = dim * @sizeOf(f32),
                    .warmup_iterations = 500,
                    .min_time_ns = 100_000_000,
                },
                struct {
                    fn bench(src: []const f32, dst: []f32) void {
                        @memcpy(dst, src);
                        simd.rmsNormInPlace(dst, null, 1e-6);
                    }
                }.bench,
                .{ data, out },
            );
            std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
        }
    }
    std.debug.print("\n", .{});
}

// ============================================================================
// SIMD BLAS-Level Operations
// ============================================================================

fn benchBlasOps(allocator: std.mem.Allocator, runner: *framework.BenchmarkRunner, config: V2BenchConfig) !void {
    std.debug.print("[v2 SIMD BLAS Operations]\n", .{});

    for (config.vector_sizes) |dim| {
        const x = try allocator.alloc(f32, dim);
        defer allocator.free(x);
        const y = try allocator.alloc(f32, dim);
        defer allocator.free(y);
        const out = try allocator.alloc(f32, dim);
        defer allocator.free(out);

        for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 100)) * 0.01;
        for (y, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i + 37) % 100)) * 0.01;

        // SAXPY
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "saxpy_{d}", .{dim}) catch "saxpy";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "v2/blas",
                    .bytes_per_op = dim * @sizeOf(f32) * 2,
                    .warmup_iterations = 500,
                    .min_time_ns = 100_000_000,
                },
                struct {
                    fn bench(vx: []const f32, vy: []f32) void {
                        simd.saxpy(2.5, vx, vy);
                    }
                }.bench,
                .{ x, y },
            );
            std.debug.print("  {s}: {d:.0} ops/sec, {d:.2} GB/s\n", .{
                name,
                result.stats.opsPerSecond(),
                result.stats.throughputMBps(dim * @sizeOf(f32) * 2) / 1024.0,
            });
        }

        // Euclidean distance
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "euclidean_{d}", .{dim}) catch "euclid";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "v2/blas",
                    .bytes_per_op = dim * @sizeOf(f32) * 2,
                    .warmup_iterations = 500,
                    .min_time_ns = 100_000_000,
                },
                struct {
                    fn bench(va: []const f32, vb: []const f32) f32 {
                        return simd.euclideanDistance(va, vb);
                    }
                }.bench,
                .{ x, y },
            );
            std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
        }

        // FMA
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "fma_{d}", .{dim}) catch "fma";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "v2/blas",
                    .bytes_per_op = dim * @sizeOf(f32) * 3,
                    .warmup_iterations = 500,
                    .min_time_ns = 100_000_000,
                },
                struct {
                    fn bench(va: []const f32, vb: []const f32, vc: []f32) void {
                        simd.fma(va, vb, va, vc);
                    }
                }.bench,
                .{ x, y, out },
            );
            std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
        }

        // Reduce sum
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "reduce_sum_{d}", .{dim}) catch "rsum";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "v2/blas",
                    .bytes_per_op = dim * @sizeOf(f32),
                    .warmup_iterations = 500,
                    .min_time_ns = 100_000_000,
                },
                struct {
                    fn bench(data: []const f32) f32 {
                        return simd.reduceSum(data);
                    }
                }.bench,
                .{x},
            );
            std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
        }

        // Scale (out-of-place)
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "scale_{d}", .{dim}) catch "scale";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "v2/blas",
                    .bytes_per_op = dim * @sizeOf(f32) * 2,
                    .warmup_iterations = 500,
                    .min_time_ns = 100_000_000,
                },
                struct {
                    fn bench(data: []const f32, dst: []f32) void {
                        simd.scale(data, 3.14, dst);
                    }
                }.bench,
                .{ x, out },
            );
            std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
        }
    }
    std.debug.print("\n", .{});
}

// ============================================================================
// Matrix Operation Benchmarks
// ============================================================================

fn benchMatrixOps(allocator: std.mem.Allocator, runner: *framework.BenchmarkRunner, config: V2BenchConfig) !void {
    std.debug.print("[v2 Matrix Operations]\n", .{});

    const Mat = matrix_mod.Matrix(f32);

    for (config.matrix_sizes) |n| {
        // Matrix multiply
        {
            var a = try Mat.alloc(allocator, n, n);
            defer a.free(allocator);
            var b = try Mat.alloc(allocator, n, n);
            defer b.free(allocator);
            var c = try Mat.alloc(allocator, n, n);
            defer c.free(allocator);

            // Initialize with non-trivial values
            for (0..n) |i| {
                for (0..n) |j| {
                    a.set(i, j, @as(f32, @floatFromInt((i * n + j) % 17)) * 0.1);
                    b.set(i, j, @as(f32, @floatFromInt((i + j * 3) % 13)) * 0.1);
                }
            }

            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "matmul_{d}x{d}", .{ n, n }) catch "matmul";

            const flops = 2 * n * n * n;
            const result = try runner.run(
                .{
                    .name = name,
                    .category = "v2/matrix",
                    .bytes_per_op = n * n * @sizeOf(f32) * 3,
                    .warmup_iterations = 10,
                    .min_time_ns = 100_000_000,
                    .max_iterations = 1000,
                },
                struct {
                    fn bench(ma: *const Mat, mb: *const Mat, mc: *Mat) void {
                        Mat.multiply(ma, mb, mc);
                    }
                }.bench,
                .{ &a, &b, &c },
            );
            const gflops = @as(f64, @floatFromInt(flops)) * result.stats.opsPerSecond() / 1e9;
            std.debug.print("  {s}: {d:.0} ops/sec ({d:.2} GFLOPS)\n", .{
                name,
                result.stats.opsPerSecond(),
                gflops,
            });
        }

        // Matrix-vector multiply
        {
            var a = try Mat.alloc(allocator, n, n);
            defer a.free(allocator);
            const vec = try allocator.alloc(f32, n);
            defer allocator.free(vec);
            const out = try allocator.alloc(f32, n);
            defer allocator.free(out);

            for (0..n) |i| {
                for (0..n) |j| {
                    a.set(i, j, @as(f32, @floatFromInt((i * n + j) % 17)) * 0.1);
                }
                vec[i] = @as(f32, @floatFromInt(i % 11)) * 0.1;
            }

            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "matvec_{d}", .{n}) catch "matvec";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "v2/matrix",
                    .bytes_per_op = (n * n + n) * @sizeOf(f32),
                    .warmup_iterations = 100,
                    .min_time_ns = 100_000_000,
                },
                struct {
                    fn bench(ma: *const Mat, v: []const f32, o: []f32) void {
                        ma.matvec(v, o);
                    }
                }.bench,
                .{ &a, vec, out },
            );
            std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
        }

        // Transpose
        {
            var a = try Mat.alloc(allocator, n, n);
            defer a.free(allocator);
            var t = try Mat.alloc(allocator, n, n);
            defer t.free(allocator);

            for (0..n) |i| {
                for (0..n) |j| {
                    a.set(i, j, @as(f32, @floatFromInt(i * n + j)));
                }
            }

            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "transpose_{d}x{d}", .{ n, n }) catch "transp";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "v2/matrix",
                    .bytes_per_op = n * n * @sizeOf(f32) * 2,
                    .warmup_iterations = 100,
                    .min_time_ns = 100_000_000,
                },
                struct {
                    fn bench(ma: *const Mat, mt: *Mat) void {
                        ma.transpose(mt);
                    }
                }.bench,
                .{ &a, &t },
            );
            std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
        }
    }
    std.debug.print("\n", .{});
}

// ============================================================================
// SwissMap Hash Table Benchmarks
// ============================================================================

fn benchSwissMap(allocator: std.mem.Allocator, runner: *framework.BenchmarkRunner, config: V2BenchConfig) !void {
    std.debug.print("[v2 SwissMap Hash Table]\n", .{});

    const IntMap = utils.swiss_map.SwissMap(u64, u64);

    for (config.map_sizes) |n| {
        // Insert benchmark
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "swissmap_put_{d}", .{n}) catch "put";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "v2/swissmap",
                    .bytes_per_op = n * (@sizeOf(u64) * 2),
                    .warmup_iterations = 10,
                    .min_time_ns = 100_000_000,
                    .max_iterations = 10000,
                },
                struct {
                    fn bench(count: usize, alloc: std.mem.Allocator) void {
                        var map = IntMap.init(alloc);
                        defer map.deinit();
                        for (0..count) |i| {
                            map.put(i, i *% 0x9e3779b97f4a7c15) catch return;
                        }
                    }
                }.bench,
                .{ n, allocator },
            );
            std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
        }

        // Lookup benchmark (pre-populated)
        {
            var map = IntMap.init(allocator);
            defer map.deinit();
            for (0..n) |i| {
                try map.put(i, i *% 0x9e3779b97f4a7c15);
            }

            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "swissmap_get_{d}", .{n}) catch "get";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "v2/swissmap",
                    .warmup_iterations = 100,
                    .min_time_ns = 100_000_000,
                },
                struct {
                    fn bench(m: *const IntMap, count: usize) u64 {
                        var sum: u64 = 0;
                        for (0..count) |i| {
                            if (m.get(i)) |v| sum +%= v;
                        }
                        return sum;
                    }
                }.bench,
                .{ &map, n },
            );
            std.debug.print("  {s}: {d:.0} ops/sec ({d:.0} lookups/sec)\n", .{
                name,
                result.stats.opsPerSecond(),
                result.stats.opsPerSecond() * @as(f64, @floatFromInt(n)),
            });
        }

        // Iteration benchmark
        {
            var map = IntMap.init(allocator);
            defer map.deinit();
            for (0..n) |i| {
                try map.put(i, i);
            }

            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "swissmap_iter_{d}", .{n}) catch "iter";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "v2/swissmap",
                    .warmup_iterations = 100,
                    .min_time_ns = 100_000_000,
                },
                struct {
                    fn bench(m: *const IntMap) u64 {
                        var sum: u64 = 0;
                        var it = m.iterator();
                        while (it.next()) |entry| {
                            sum +%= entry.value;
                        }
                        return sum;
                    }
                }.bench,
                .{&map},
            );
            std.debug.print("  {s}: {d:.0} ops/sec\n", .{ name, result.stats.opsPerSecond() });
        }
    }
    std.debug.print("\n", .{});
}

// ============================================================================
// SIMD matrixMultiply (from simd.zig, the row-major version)
// ============================================================================

fn benchSimdMatMul(allocator: std.mem.Allocator, runner: *framework.BenchmarkRunner, config: V2BenchConfig) !void {
    std.debug.print("[v2 SIMD Matrix Multiply (row-major)]\n", .{});

    for (config.matrix_sizes) |n| {
        const size = n * n;
        const a = try allocator.alloc(f32, size);
        defer allocator.free(a);
        const b = try allocator.alloc(f32, size);
        defer allocator.free(b);
        const c = try allocator.alloc(f32, size);
        defer allocator.free(c);

        for (a, 0..) |*v, i| v.* = @floatFromInt(@mod(i, 10));
        for (b, 0..) |*v, i| v.* = @floatFromInt(@mod(i + 1, 10));

        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "simd_matmul_{d}x{d}", .{ n, n }) catch "simd_mm";

        const flops = 2 * n * n * n;
        const result = try runner.run(
            .{
                .name = name,
                .category = "v2/simd_matrix",
                .bytes_per_op = size * @sizeOf(f32) * 3,
                .warmup_iterations = 10,
                .min_time_ns = 100_000_000,
                .max_iterations = 1000,
            },
            struct {
                fn bench(ma: []const f32, mb: []const f32, mc: []f32, dim: usize) void {
                    simd.matrixMultiply(ma, mb, mc, dim, dim, dim);
                }
            }.bench,
            .{ a, b, c, n },
        );
        const gflops = @as(f64, @floatFromInt(flops)) * result.stats.opsPerSecond() / 1e9;
        std.debug.print("  {s}: {d:.0} ops/sec ({d:.2} GFLOPS)\n", .{
            name,
            result.stats.opsPerSecond(),
            gflops,
        });
    }
    std.debug.print("\n", .{});
}

// ============================================================================
// v2 Primitives (RingBuffer, String hash)
// ============================================================================

fn benchPrimitives(runner: *framework.BenchmarkRunner) !void {
    std.debug.print("[v2 Utility Primitives]\n", .{});

    // FNV-1a string hash
    {
        const test_strings = [_][]const u8{
            "hello world",
            "the quick brown fox jumps over the lazy dog",
            "ABI Framework v2.0 benchmark key",
            "a]b/c:d-e_f.g+h@i#j$k%l^m&n*o(p)q",
        };

        const result = try runner.run(
            .{
                .name = "fnv1a_hash",
                .category = "v2/primitives",
                .warmup_iterations = 1000,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench(strings: []const []const u8) u64 {
                    var sum: u64 = 0;
                    for (strings) |s| {
                        sum +%= utils.crypto.fnv1a64(s);
                    }
                    return sum;
                }
            }.bench,
            .{&test_strings},
        );
        std.debug.print("  fnv1a_hash: {d:.0} ops/sec\n", .{result.stats.opsPerSecond()});
    }

    // RingBuffer throughput
    {
        const result = try runner.run(
            .{
                .name = "ringbuffer_1024",
                .category = "v2/primitives",
                .warmup_iterations = 500,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench() u64 {
                    var ring: utils.primitives.RingBuffer(u64, 1024) = .{};
                    var sum: u64 = 0;
                    // Fill and drain 10 times
                    for (0..10) |round| {
                        for (0..1024) |i| {
                            _ = ring.push(i +% round);
                        }
                        while (ring.pop()) |v| {
                            sum +%= v;
                        }
                    }
                    return sum;
                }
            }.bench,
            .{},
        );
        std.debug.print("  ringbuffer_1024: {d:.0} ops/sec\n", .{result.stats.opsPerSecond()});
    }

    // Math.alignUp (power-of-two alignment)
    {
        const result = try runner.run(
            .{
                .name = "align_up",
                .category = "v2/primitives",
                .warmup_iterations = 1000,
                .min_time_ns = 100_000_000,
            },
            struct {
                fn bench() u64 {
                    var sum: u64 = 0;
                    for (1..10001) |i| {
                        sum +%= utils.primitives.Math.alignUp(usize, i, 64);
                    }
                    return sum;
                }
            }.bench,
            .{},
        );
        std.debug.print("  align_up: {d:.0} ops/sec\n", .{result.stats.opsPerSecond()});
    }

    std.debug.print("\n", .{});
}

// ============================================================================
// Main Benchmark Runner
// ============================================================================

pub fn runV2Benchmarks(allocator: std.mem.Allocator, config: V2BenchConfig) !void {
    var runner = framework.BenchmarkRunner.init(allocator);
    defer runner.deinit();

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("                    v2 MODULE BENCHMARKS\n", .{});
    std.debug.print("================================================================================\n\n", .{});

    try benchSoftmax(allocator, &runner, config);
    try benchBlasOps(allocator, &runner, config);
    try benchMatrixOps(allocator, &runner, config);
    try benchSimdMatMul(allocator, &runner, config);
    try benchSwissMap(allocator, &runner, config);
    try benchPrimitives(&runner);

    runner.printSummaryDebug();
}

/// Convenience alias for infrastructure mod.zig
pub fn run(allocator: std.mem.Allocator) !void {
    try runV2Benchmarks(allocator, .{});
}

test "v2 benchmark imports" {
    _ = simd;
    _ = matrix_mod;
    _ = tensor_mod;
    _ = utils;
    _ = framework;
}
