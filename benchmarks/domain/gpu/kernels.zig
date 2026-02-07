//! GPU Kernel Benchmarks
//!
//! Comprehensive benchmarks for GPU kernel operations:
//! - Matrix operations (matmul, transpose)
//! - Vector operations (add, mul, dot, normalize)
//! - Reduction operations (sum, max, min, argmax)
//! - Memory operations (copy, fill, bandwidth)
//!
//! Benchmarks run on GPU when available, falling back to CPU SIMD/scalar
//! implementations for comparison and baseline measurements.

const std = @import("std");
const build_options = @import("build_options");
const framework = @import("../../system/framework.zig");
const mod = @import("mod.zig");

const GpuBenchConfig = mod.GpuBenchConfig;

// Conditional GPU imports based on build options
const gpu = if (build_options.enable_gpu) @import("abi").gpu else struct {
    pub const Gpu = void;
    pub const GpuConfig = struct {};
    pub const moduleEnabled = struct {
        pub fn call() bool {
            return false;
        }
    }.call;
};

// ============================================================================
// CPU Baseline Implementations (for comparison and fallback)
// ============================================================================

/// Scalar matrix multiplication: C = A * B
fn scalarMatmul(A: []const f32, B: []const f32, C: []f32, M: usize, N: usize, K: usize) void {
    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f32 = 0;
            for (0..K) |k| {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/// Blocked matrix multiplication for better cache utilization
fn blockedMatmul(A: []const f32, B: []const f32, C: []f32, M: usize, N: usize, K: usize, block_size: usize) void {
    @memset(C, 0);

    var i: usize = 0;
    while (i < M) : (i += block_size) {
        var j: usize = 0;
        while (j < N) : (j += block_size) {
            var k: usize = 0;
            while (k < K) : (k += block_size) {
                const i_end = @min(i + block_size, M);
                const j_end = @min(j + block_size, N);
                const k_end = @min(k + block_size, K);

                var ii = i;
                while (ii < i_end) : (ii += 1) {
                    var kk = k;
                    while (kk < k_end) : (kk += 1) {
                        const a_val = A[ii * K + kk];
                        var jj = j;
                        while (jj < j_end) : (jj += 1) {
                            C[ii * N + jj] += a_val * B[kk * N + jj];
                        }
                    }
                }
            }
        }
    }
}

/// SIMD matrix multiplication
fn simdMatmul(comptime VEC_SIZE: usize, A: []const f32, B: []const f32, C: []f32, M: usize, N: usize, K: usize) void {
    const Vec = @Vector(VEC_SIZE, f32);
    @memset(C, 0);

    for (0..M) |i| {
        for (0..K) |k| {
            const a_val: Vec = @splat(A[i * K + k]);
            var j: usize = 0;

            while (j + VEC_SIZE <= N) : (j += VEC_SIZE) {
                const b_vec: Vec = B[k * N + j ..][0..VEC_SIZE].*;
                const c_vec: Vec = C[i * N + j ..][0..VEC_SIZE].*;
                C[i * N + j ..][0..VEC_SIZE].* = c_vec + a_val * b_vec;
            }

            while (j < N) : (j += 1) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

/// Scalar vector addition
fn scalarVectorAdd(a: []const f32, b: []const f32, result: []f32) void {
    for (a, b, result) |va, vb, *r| {
        r.* = va + vb;
    }
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

/// Scalar vector multiplication (element-wise)
fn scalarVectorMul(a: []const f32, b: []const f32, result: []f32) void {
    for (a, b, result) |va, vb, *r| {
        r.* = va * vb;
    }
}

/// SIMD vector multiplication
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

/// Scalar dot product
fn scalarDotProduct(a: []const f32, b: []const f32) f32 {
    var sum: f32 = 0;
    for (a, b) |va, vb| {
        sum += va * vb;
    }
    return sum;
}

/// SIMD dot product
fn simdDotProduct(comptime N: usize, a: []const f32, b: []const f32) f32 {
    const Vec = @Vector(N, f32);
    var sum: f32 = 0;
    var i: usize = 0;

    while (i + N <= a.len) : (i += N) {
        const va: Vec = a[i..][0..N].*;
        const vb: Vec = b[i..][0..N].*;
        sum += @reduce(.Add, va * vb);
    }

    while (i < a.len) : (i += 1) {
        sum += a[i] * b[i];
    }

    return sum;
}

/// Scalar vector normalization (L2)
fn scalarNormalize(v: []const f32, result: []f32) void {
    var sum_sq: f32 = 0;
    for (v) |x| {
        sum_sq += x * x;
    }
    const norm = @sqrt(sum_sq);
    if (norm == 0) {
        @memset(result, 0);
        return;
    }
    const inv_norm = 1.0 / norm;
    for (v, result) |x, *r| {
        r.* = x * inv_norm;
    }
}

/// SIMD vector normalization
fn simdNormalize(comptime N: usize, v: []const f32, result: []f32) void {
    const sum_sq = simdDotProduct(N, v, v);
    const norm = @sqrt(sum_sq);
    if (norm == 0) {
        @memset(result, 0);
        return;
    }

    const Vec = @Vector(N, f32);
    const inv_norm: Vec = @splat(1.0 / norm);
    var i: usize = 0;

    while (i + N <= v.len) : (i += N) {
        const vec: Vec = v[i..][0..N].*;
        result[i..][0..N].* = vec * inv_norm;
    }

    const scalar_inv = 1.0 / norm;
    while (i < v.len) : (i += 1) {
        result[i] = v[i] * scalar_inv;
    }
}

/// Scalar reduction sum
fn scalarReduceSum(v: []const f32) f32 {
    var sum: f32 = 0;
    for (v) |x| {
        sum += x;
    }
    return sum;
}

/// SIMD reduction sum
fn simdReduceSum(comptime N: usize, v: []const f32) f32 {
    const Vec = @Vector(N, f32);
    var sum: f32 = 0;
    var i: usize = 0;

    while (i + N <= v.len) : (i += N) {
        const vec: Vec = v[i..][0..N].*;
        sum += @reduce(.Add, vec);
    }

    while (i < v.len) : (i += 1) {
        sum += v[i];
    }

    return sum;
}

/// Scalar reduction max
fn scalarReduceMax(v: []const f32) f32 {
    var max_val: f32 = v[0];
    for (v[1..]) |x| {
        if (x > max_val) max_val = x;
    }
    return max_val;
}

/// SIMD reduction max
fn simdReduceMax(comptime N: usize, v: []const f32) f32 {
    const Vec = @Vector(N, f32);
    var max_val: f32 = v[0];
    var i: usize = 0;

    while (i + N <= v.len) : (i += N) {
        const vec: Vec = v[i..][0..N].*;
        const local_max = @reduce(.Max, vec);
        if (local_max > max_val) max_val = local_max;
    }

    while (i < v.len) : (i += 1) {
        if (v[i] > max_val) max_val = v[i];
    }

    return max_val;
}

/// Scalar reduction min
fn scalarReduceMin(v: []const f32) f32 {
    var min_val: f32 = v[0];
    for (v[1..]) |x| {
        if (x < min_val) min_val = x;
    }
    return min_val;
}

/// SIMD reduction min
fn simdReduceMin(comptime N: usize, v: []const f32) f32 {
    const Vec = @Vector(N, f32);
    var min_val: f32 = v[0];
    var i: usize = 0;

    while (i + N <= v.len) : (i += N) {
        const vec: Vec = v[i..][0..N].*;
        const local_min = @reduce(.Min, vec);
        if (local_min < min_val) min_val = local_min;
    }

    while (i < v.len) : (i += 1) {
        if (v[i] < min_val) min_val = v[i];
    }

    return min_val;
}

/// Scalar argmax
fn scalarArgmax(v: []const f32) usize {
    var max_idx: usize = 0;
    var max_val: f32 = v[0];
    for (v[1..], 1..) |x, i| {
        if (x > max_val) {
            max_val = x;
            max_idx = i;
        }
    }
    return max_idx;
}

/// Matrix transpose
fn matrixTranspose(src: []const f32, dst: []f32, rows: usize, cols: usize) void {
    for (0..rows) |i| {
        for (0..cols) |j| {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

/// Blocked matrix transpose for better cache utilization
fn blockedTranspose(src: []const f32, dst: []f32, rows: usize, cols: usize, block_size: usize) void {
    var i: usize = 0;
    while (i < rows) : (i += block_size) {
        var j: usize = 0;
        while (j < cols) : (j += block_size) {
            const i_end = @min(i + block_size, rows);
            const j_end = @min(j + block_size, cols);

            var ii = i;
            while (ii < i_end) : (ii += 1) {
                var jj = j;
                while (jj < j_end) : (jj += 1) {
                    dst[jj * rows + ii] = src[ii * cols + jj];
                }
            }
        }
    }
}

// ============================================================================
// Benchmark Utilities
// ============================================================================

/// Initialize data with deterministic pseudo-random values
fn initRandomData(data: []f32, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    for (data) |*x| {
        x.* = prng.random().float(f32) * 2.0 - 1.0;
    }
}

/// Format size as human-readable string
fn formatSize(size: usize) struct { value: f64, unit: []const u8 } {
    if (size >= 1024 * 1024 * 1024) {
        return .{ .value = @as(f64, @floatFromInt(size)) / (1024.0 * 1024.0 * 1024.0), .unit = "GB" };
    } else if (size >= 1024 * 1024) {
        return .{ .value = @as(f64, @floatFromInt(size)) / (1024.0 * 1024.0), .unit = "MB" };
    } else if (size >= 1024) {
        return .{ .value = @as(f64, @floatFromInt(size)) / 1024.0, .unit = "KB" };
    } else {
        return .{ .value = @as(f64, @floatFromInt(size)), .unit = "B" };
    }
}

// ============================================================================
// Benchmark Functions
// ============================================================================

/// Run matrix multiplication benchmarks
fn benchmarkMatmul(
    allocator: std.mem.Allocator,
    runner: *framework.BenchmarkRunner,
    config: GpuBenchConfig,
) !void {
    std.debug.print("\n[Matrix Multiplication Benchmarks]\n", .{});

    const gpu_available = mod.hasHardwareGpu(allocator);
    if (build_options.enable_gpu and !gpu_available) {
        std.debug.print("  [GPU SKIP] No hardware GPU detected; skipping GPU matmul benchmarks.\n", .{});
    }

    for (config.matrix_sizes) |size| {
        const matrix_size = size * size;

        // Allocate matrices with 64-byte alignment for SIMD
        const A = try allocator.alignedAlloc(f32, .@"64", matrix_size);
        defer allocator.free(A);
        const B = try allocator.alignedAlloc(f32, .@"64", matrix_size);
        defer allocator.free(B);
        const C = try allocator.alignedAlloc(f32, .@"64", matrix_size);
        defer allocator.free(C);

        // Initialize with deterministic data
        initRandomData(A, 42);
        initRandomData(B, 43);

        const flops_per_op = 2 * size * size * size; // multiply-add for each element
        const bytes_per_op = matrix_size * @sizeOf(f32) * 3; // A, B read + C write

        // Scalar matmul (baseline)
        if (size <= 512) { // Skip large sizes for naive version
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "matmul_scalar_{d}x{d}", .{ size, size }) catch "matmul_scalar";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "gpu/matmul",
                    .bytes_per_op = bytes_per_op,
                    .warmup_iterations = @min(config.warmup_iterations, 5),
                    .min_time_ns = config.min_time_ns,
                    .max_iterations = @min(config.max_iterations, 100),
                },
                struct {
                    fn bench(a: []const f32, b: []const f32, c: []f32, n: usize) void {
                        scalarMatmul(a, b, c, n, n, n);
                    }
                }.bench,
                .{ A, B, C, size },
            );

            const gflops = @as(f64, @floatFromInt(flops_per_op)) * result.stats.opsPerSecond() / 1e9;
            std.debug.print("  {s}: {d:.2} GFLOPS ({d:.2} ms/op)\n", .{
                name,
                gflops,
                result.stats.mean_ns / 1e6,
            });
        }

        // Blocked matmul
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "matmul_blocked_{d}x{d}", .{ size, size }) catch "matmul_blocked";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "gpu/matmul",
                    .bytes_per_op = bytes_per_op,
                    .warmup_iterations = config.warmup_iterations,
                    .min_time_ns = config.min_time_ns,
                    .max_iterations = config.max_iterations,
                },
                struct {
                    fn bench(a: []const f32, b: []const f32, c: []f32, n: usize) void {
                        blockedMatmul(a, b, c, n, n, n, 32);
                    }
                }.bench,
                .{ A, B, C, size },
            );

            const gflops = @as(f64, @floatFromInt(flops_per_op)) * result.stats.opsPerSecond() / 1e9;
            std.debug.print("  {s}: {d:.2} GFLOPS ({d:.2} ms/op)\n", .{
                name,
                gflops,
                result.stats.mean_ns / 1e6,
            });
        }

        // SIMD matmul
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "matmul_simd8_{d}x{d}", .{ size, size }) catch "matmul_simd";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "gpu/matmul",
                    .bytes_per_op = bytes_per_op,
                    .warmup_iterations = config.warmup_iterations,
                    .min_time_ns = config.min_time_ns,
                    .max_iterations = config.max_iterations,
                },
                struct {
                    fn bench(a: []const f32, b: []const f32, c: []f32, n: usize) void {
                        simdMatmul(8, a, b, c, n, n, n);
                    }
                }.bench,
                .{ A, B, C, size },
            );

            const gflops = @as(f64, @floatFromInt(flops_per_op)) * result.stats.opsPerSecond() / 1e9;
            std.debug.print("  {s}: {d:.2} GFLOPS ({d:.2} ms/op)\n", .{
                name,
                gflops,
                result.stats.mean_ns / 1e6,
            });
        }

        // GPU matmul (if available)
        if (build_options.enable_gpu and gpu_available) {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "matmul_gpu_{d}x{d}", .{ size, size }) catch "matmul_gpu";

            // Try to initialize GPU and run benchmark
            const gpu_result = runGpuMatmulBenchmark(allocator, A, B, C, size, config) catch |err| {
                std.debug.print("  {s}: GPU unavailable ({t})\n", .{ name, err });
                continue;
            };

            const gflops = @as(f64, @floatFromInt(flops_per_op)) / (gpu_result.mean_ns / 1e9);
            std.debug.print("  {s}: {d:.2} GFLOPS ({d:.2} ms/op)\n", .{
                name,
                gflops,
                gpu_result.mean_ns / 1e6,
            });
        }
    }
}

/// GPU matmul benchmark result
const GpuBenchmarkResult = struct {
    mean_ns: f64,
    min_ns: u64,
    max_ns: u64,
    iterations: u32,
};

/// Run GPU matmul benchmark (only called when GPU is enabled)
fn runGpuMatmulBenchmark(
    allocator: std.mem.Allocator,
    A: []const f32,
    B: []const f32,
    C: []f32,
    size: usize,
    config: GpuBenchConfig,
) !GpuBenchmarkResult {
    if (!build_options.enable_gpu) return error.GpuNotEnabled;

    // This is where we would use the actual GPU API
    // For now, use the unified GPU API if available
    const abi = @import("abi");

    var gpu_ctx = abi.gpu.Gpu.init(allocator, .{}) catch return error.GpuInitFailed;
    defer gpu_ctx.deinit();

    if (!gpu_ctx.isAvailable()) return error.NoGpuDevice;

    // Create GPU buffers
    const buf_a = try gpu_ctx.createBufferFromSlice(f32, A, .{ .mode = .explicit });
    defer gpu_ctx.destroyBuffer(buf_a);

    const buf_b = try gpu_ctx.createBufferFromSlice(f32, B, .{ .mode = .explicit });
    defer gpu_ctx.destroyBuffer(buf_b);

    const buf_c = try gpu_ctx.createBuffer(C.len * @sizeOf(f32), .{ .mode = .explicit });
    defer gpu_ctx.destroyBuffer(buf_c);

    // Warmup
    for (0..config.warmup_iterations) |_| {
        _ = try gpu_ctx.matrixMultiply(buf_a, buf_b, buf_c, .{
            .m = size,
            .n = size,
            .k = size,
        });
    }

    // Benchmark
    var total_ns: u64 = 0;
    var min_ns: u64 = std.math.maxInt(u64);
    var max_ns: u64 = 0;
    var iterations: u32 = 0;

    while (total_ns < config.min_time_ns and iterations < config.benchmark_iterations) : (iterations += 1) {
        var timer = abi.shared.time.Timer.start() catch return error.TimerFailed;
        _ = try gpu_ctx.matrixMultiply(buf_a, buf_b, buf_c, .{
            .m = size,
            .n = size,
            .k = size,
        });
        try gpu_ctx.synchronize();
        const elapsed = timer.read();

        total_ns += elapsed;
        if (elapsed < min_ns) min_ns = elapsed;
        if (elapsed > max_ns) max_ns = elapsed;
    }

    return GpuBenchmarkResult{
        .mean_ns = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(iterations)),
        .min_ns = min_ns,
        .max_ns = max_ns,
        .iterations = iterations,
    };
}

/// Run vector operation benchmarks
fn benchmarkVectorOps(
    allocator: std.mem.Allocator,
    runner: *framework.BenchmarkRunner,
    config: GpuBenchConfig,
) !void {
    std.debug.print("\n[Vector Operation Benchmarks]\n", .{});

    for (config.vector_sizes) |size| {
        const a = try allocator.alignedAlloc(f32, .@"64", size);
        defer allocator.free(a);
        const b = try allocator.alignedAlloc(f32, .@"64", size);
        defer allocator.free(b);
        const result = try allocator.alignedAlloc(f32, .@"64", size);
        defer allocator.free(result);

        initRandomData(a, 42);
        initRandomData(b, 43);

        const size_info = formatSize(size * @sizeOf(f32));
        const bytes_per_op = size * @sizeOf(f32) * 3; // 2 read + 1 write

        // Vector Add - Scalar
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "vecadd_scalar_{d:.1}{s}", .{ size_info.value, size_info.unit }) catch "vecadd_scalar";

            const bench_result = try runner.run(
                .{
                    .name = name,
                    .category = "gpu/vector",
                    .bytes_per_op = bytes_per_op,
                    .warmup_iterations = config.warmup_iterations,
                    .min_time_ns = config.min_time_ns,
                },
                struct {
                    fn bench(va: []const f32, vb: []const f32, vr: []f32) void {
                        scalarVectorAdd(va, vb, vr);
                    }
                }.bench,
                .{ a, b, result },
            );

            std.debug.print("  {s}: {d:.2} GB/s ({d:.0} ops/sec)\n", .{
                name,
                bench_result.stats.throughputMBps(bytes_per_op) / 1024.0,
                bench_result.stats.opsPerSecond(),
            });
        }

        // Vector Add - SIMD
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "vecadd_simd8_{d:.1}{s}", .{ size_info.value, size_info.unit }) catch "vecadd_simd";

            const bench_result = try runner.run(
                .{
                    .name = name,
                    .category = "gpu/vector",
                    .bytes_per_op = bytes_per_op,
                    .warmup_iterations = config.warmup_iterations,
                    .min_time_ns = config.min_time_ns,
                },
                struct {
                    fn bench(va: []const f32, vb: []const f32, vr: []f32) void {
                        simdVectorAdd(8, va, vb, vr);
                    }
                }.bench,
                .{ a, b, result },
            );

            std.debug.print("  {s}: {d:.2} GB/s ({d:.0} ops/sec)\n", .{
                name,
                bench_result.stats.throughputMBps(bytes_per_op) / 1024.0,
                bench_result.stats.opsPerSecond(),
            });
        }

        // Vector Mul - SIMD
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "vecmul_simd8_{d:.1}{s}", .{ size_info.value, size_info.unit }) catch "vecmul_simd";

            const bench_result = try runner.run(
                .{
                    .name = name,
                    .category = "gpu/vector",
                    .bytes_per_op = bytes_per_op,
                    .warmup_iterations = config.warmup_iterations,
                    .min_time_ns = config.min_time_ns,
                },
                struct {
                    fn bench(va: []const f32, vb: []const f32, vr: []f32) void {
                        simdVectorMul(8, va, vb, vr);
                    }
                }.bench,
                .{ a, b, result },
            );

            std.debug.print("  {s}: {d:.2} GB/s ({d:.0} ops/sec)\n", .{
                name,
                bench_result.stats.throughputMBps(bytes_per_op) / 1024.0,
                bench_result.stats.opsPerSecond(),
            });
        }

        // Dot Product - Scalar vs SIMD
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "dot_simd8_{d:.1}{s}", .{ size_info.value, size_info.unit }) catch "dot_simd";

            const bench_result = try runner.run(
                .{
                    .name = name,
                    .category = "gpu/vector",
                    .bytes_per_op = size * @sizeOf(f32) * 2,
                    .warmup_iterations = config.warmup_iterations,
                    .min_time_ns = config.min_time_ns,
                },
                struct {
                    fn bench(va: []const f32, vb: []const f32) f32 {
                        return simdDotProduct(8, va, vb);
                    }
                }.bench,
                .{ a, b },
            );

            std.debug.print("  {s}: {d:.2} GB/s ({d:.0} ops/sec)\n", .{
                name,
                bench_result.stats.throughputMBps(size * @sizeOf(f32) * 2) / 1024.0,
                bench_result.stats.opsPerSecond(),
            });
        }

        // Normalize
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "normalize_simd8_{d:.1}{s}", .{ size_info.value, size_info.unit }) catch "normalize_simd";

            const bench_result = try runner.run(
                .{
                    .name = name,
                    .category = "gpu/vector",
                    .bytes_per_op = size * @sizeOf(f32) * 2,
                    .warmup_iterations = config.warmup_iterations,
                    .min_time_ns = config.min_time_ns,
                },
                struct {
                    fn bench(va: []const f32, vr: []f32) void {
                        simdNormalize(8, va, vr);
                    }
                }.bench,
                .{ a, result },
            );

            std.debug.print("  {s}: {d:.2} GB/s ({d:.0} ops/sec)\n", .{
                name,
                bench_result.stats.throughputMBps(size * @sizeOf(f32) * 2) / 1024.0,
                bench_result.stats.opsPerSecond(),
            });
        }
    }
}

/// Run reduction operation benchmarks
fn benchmarkReductions(
    allocator: std.mem.Allocator,
    runner: *framework.BenchmarkRunner,
    config: GpuBenchConfig,
) !void {
    std.debug.print("\n[Reduction Operation Benchmarks]\n", .{});

    for (config.vector_sizes) |size| {
        const v = try allocator.alignedAlloc(f32, .@"64", size);
        defer allocator.free(v);

        initRandomData(v, 42);

        const size_info = formatSize(size * @sizeOf(f32));
        const bytes_per_op = size * @sizeOf(f32);

        // Reduce Sum
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "reduce_sum_simd8_{d:.1}{s}", .{ size_info.value, size_info.unit }) catch "reduce_sum";

            const bench_result = try runner.run(
                .{
                    .name = name,
                    .category = "gpu/reduction",
                    .bytes_per_op = bytes_per_op,
                    .warmup_iterations = config.warmup_iterations,
                    .min_time_ns = config.min_time_ns,
                },
                struct {
                    fn bench(data: []const f32) f32 {
                        return simdReduceSum(8, data);
                    }
                }.bench,
                .{v},
            );

            std.debug.print("  {s}: {d:.2} GB/s ({d:.0} ops/sec)\n", .{
                name,
                bench_result.stats.throughputMBps(bytes_per_op) / 1024.0,
                bench_result.stats.opsPerSecond(),
            });
        }

        // Reduce Max
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "reduce_max_simd8_{d:.1}{s}", .{ size_info.value, size_info.unit }) catch "reduce_max";

            const bench_result = try runner.run(
                .{
                    .name = name,
                    .category = "gpu/reduction",
                    .bytes_per_op = bytes_per_op,
                    .warmup_iterations = config.warmup_iterations,
                    .min_time_ns = config.min_time_ns,
                },
                struct {
                    fn bench(data: []const f32) f32 {
                        return simdReduceMax(8, data);
                    }
                }.bench,
                .{v},
            );

            std.debug.print("  {s}: {d:.2} GB/s ({d:.0} ops/sec)\n", .{
                name,
                bench_result.stats.throughputMBps(bytes_per_op) / 1024.0,
                bench_result.stats.opsPerSecond(),
            });
        }

        // Reduce Min
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "reduce_min_simd8_{d:.1}{s}", .{ size_info.value, size_info.unit }) catch "reduce_min";

            const bench_result = try runner.run(
                .{
                    .name = name,
                    .category = "gpu/reduction",
                    .bytes_per_op = bytes_per_op,
                    .warmup_iterations = config.warmup_iterations,
                    .min_time_ns = config.min_time_ns,
                },
                struct {
                    fn bench(data: []const f32) f32 {
                        return simdReduceMin(8, data);
                    }
                }.bench,
                .{v},
            );

            std.debug.print("  {s}: {d:.2} GB/s ({d:.0} ops/sec)\n", .{
                name,
                bench_result.stats.throughputMBps(bytes_per_op) / 1024.0,
                bench_result.stats.opsPerSecond(),
            });
        }

        // Argmax
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "argmax_scalar_{d:.1}{s}", .{ size_info.value, size_info.unit }) catch "argmax";

            const bench_result = try runner.run(
                .{
                    .name = name,
                    .category = "gpu/reduction",
                    .bytes_per_op = bytes_per_op,
                    .warmup_iterations = config.warmup_iterations,
                    .min_time_ns = config.min_time_ns,
                },
                struct {
                    fn bench(data: []const f32) usize {
                        return scalarArgmax(data);
                    }
                }.bench,
                .{v},
            );

            std.debug.print("  {s}: {d:.2} GB/s ({d:.0} ops/sec)\n", .{
                name,
                bench_result.stats.throughputMBps(bytes_per_op) / 1024.0,
                bench_result.stats.opsPerSecond(),
            });
        }
    }
}

/// Run memory operation benchmarks
fn benchmarkMemoryOps(
    allocator: std.mem.Allocator,
    runner: *framework.BenchmarkRunner,
    config: GpuBenchConfig,
) !void {
    std.debug.print("\n[Memory Operation Benchmarks]\n", .{});

    for (config.memory_sizes) |size| {
        const src = try allocator.alignedAlloc(u8, .@"64", size);
        defer allocator.free(src);
        const dst = try allocator.alignedAlloc(u8, .@"64", size);
        defer allocator.free(dst);

        // Initialize source
        @memset(src, 0xAA);

        const size_info = formatSize(size);

        // Memory Copy
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "memcpy_{d:.1}{s}", .{ size_info.value, size_info.unit }) catch "memcpy";

            const bench_result = try runner.run(
                .{
                    .name = name,
                    .category = "gpu/memory",
                    .bytes_per_op = size * 2, // read + write
                    .warmup_iterations = config.warmup_iterations,
                    .min_time_ns = config.min_time_ns,
                },
                struct {
                    fn bench(d: []u8, s: []const u8) void {
                        @memcpy(d, s);
                    }
                }.bench,
                .{ dst, src },
            );

            std.debug.print("  {s}: {d:.2} GB/s\n", .{
                name,
                bench_result.stats.throughputMBps(size * 2) / 1024.0,
            });
        }

        // Memory Set
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "memset_{d:.1}{s}", .{ size_info.value, size_info.unit }) catch "memset";

            const bench_result = try runner.run(
                .{
                    .name = name,
                    .category = "gpu/memory",
                    .bytes_per_op = size,
                    .warmup_iterations = config.warmup_iterations,
                    .min_time_ns = config.min_time_ns,
                },
                struct {
                    fn bench(d: []u8) void {
                        @memset(d, 0x55);
                    }
                }.bench,
                .{dst},
            );

            std.debug.print("  {s}: {d:.2} GB/s\n", .{
                name,
                bench_result.stats.throughputMBps(size) / 1024.0,
            });
        }

        // Memory Read (sequential)
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "memread_{d:.1}{s}", .{ size_info.value, size_info.unit }) catch "memread";

            const bench_result = try runner.run(
                .{
                    .name = name,
                    .category = "gpu/memory",
                    .bytes_per_op = size,
                    .warmup_iterations = config.warmup_iterations,
                    .min_time_ns = config.min_time_ns,
                },
                struct {
                    fn bench(s: []const u8) u64 {
                        var sum: u64 = 0;
                        for (s) |byte| {
                            sum +%= byte;
                        }
                        return sum;
                    }
                }.bench,
                .{src},
            );

            std.debug.print("  {s}: {d:.2} GB/s\n", .{
                name,
                bench_result.stats.throughputMBps(size) / 1024.0,
            });
        }
    }

    // Matrix transpose benchmarks
    std.debug.print("\n[Matrix Transpose Benchmarks]\n", .{});
    for (config.matrix_sizes) |size| {
        if (size > 2048) continue; // Skip very large sizes for transpose

        const matrix_size = size * size;
        const src = try allocator.alignedAlloc(f32, .@"64", matrix_size);
        defer allocator.free(src);
        const dst = try allocator.alignedAlloc(f32, .@"64", matrix_size);
        defer allocator.free(dst);

        initRandomData(src, 42);

        const bytes_per_op = matrix_size * @sizeOf(f32) * 2;

        // Naive transpose
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "transpose_naive_{d}x{d}", .{ size, size }) catch "transpose_naive";

            const bench_result = try runner.run(
                .{
                    .name = name,
                    .category = "gpu/memory",
                    .bytes_per_op = bytes_per_op,
                    .warmup_iterations = config.warmup_iterations,
                    .min_time_ns = config.min_time_ns,
                },
                struct {
                    fn bench(s: []const f32, d: []f32, n: usize) void {
                        matrixTranspose(s, d, n, n);
                    }
                }.bench,
                .{ src, dst, size },
            );

            std.debug.print("  {s}: {d:.2} GB/s ({d:.2} ms/op)\n", .{
                name,
                bench_result.stats.throughputMBps(bytes_per_op) / 1024.0,
                bench_result.stats.mean_ns / 1e6,
            });
        }

        // Blocked transpose
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "transpose_blocked_{d}x{d}", .{ size, size }) catch "transpose_blocked";

            const bench_result = try runner.run(
                .{
                    .name = name,
                    .category = "gpu/memory",
                    .bytes_per_op = bytes_per_op,
                    .warmup_iterations = config.warmup_iterations,
                    .min_time_ns = config.min_time_ns,
                },
                struct {
                    fn bench(s: []const f32, d: []f32, n: usize) void {
                        blockedTranspose(s, d, n, n, 32);
                    }
                }.bench,
                .{ src, dst, size },
            );

            std.debug.print("  {s}: {d:.2} GB/s ({d:.2} ms/op)\n", .{
                name,
                bench_result.stats.throughputMBps(bytes_per_op) / 1024.0,
                bench_result.stats.mean_ns / 1e6,
            });
        }
    }
}

/// Run all kernel benchmarks
pub fn runKernelBenchmarks(allocator: std.mem.Allocator, config: GpuBenchConfig) !void {
    var runner = framework.BenchmarkRunner.init(allocator);
    defer runner.deinit();

    // Matrix multiplication benchmarks
    try benchmarkMatmul(allocator, &runner, config);

    // Vector operation benchmarks
    try benchmarkVectorOps(allocator, &runner, config);

    // Reduction operation benchmarks
    try benchmarkReductions(allocator, &runner, config);

    // Memory operation benchmarks
    try benchmarkMemoryOps(allocator, &runner, config);

    // Print summary
    runner.printSummaryDebug();
}

// ============================================================================
// Tests
// ============================================================================

test "scalar operations correctness" {
    // Test matmul
    const A = [_]f32{ 1, 2, 3, 4 };
    const B = [_]f32{ 1, 0, 0, 1 };
    var C: [4]f32 = undefined;

    scalarMatmul(&A, &B, &C, 2, 2, 2);

    try std.testing.expectApproxEqAbs(@as(f32, 1), C[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2), C[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3), C[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4), C[3], 0.001);
}

test "simd operations correctness" {
    // Test dot product
    const a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const b = [_]f32{ 8, 7, 6, 5, 4, 3, 2, 1 };

    const scalar_dot = scalarDotProduct(&a, &b);
    const simd_dot = simdDotProduct(4, &a, &b);

    try std.testing.expectApproxEqAbs(scalar_dot, simd_dot, 0.001);
}

test "reduction operations correctness" {
    const v = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };

    const sum = simdReduceSum(4, &v);
    try std.testing.expectApproxEqAbs(@as(f32, 36), sum, 0.001);

    const max = simdReduceMax(4, &v);
    try std.testing.expectApproxEqAbs(@as(f32, 8), max, 0.001);

    const min = simdReduceMin(4, &v);
    try std.testing.expectApproxEqAbs(@as(f32, 1), min, 0.001);
}

test "normalize correctness" {
    const v = [_]f32{ 3, 4 };
    var result: [2]f32 = undefined;

    simdNormalize(2, &v, &result);

    // 3/5 = 0.6, 4/5 = 0.8
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), result[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), result[1], 0.001);
}

test "transpose correctness" {
    const src = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var dst: [6]f32 = undefined;

    matrixTranspose(&src, &dst, 2, 3);

    // [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
    try std.testing.expectApproxEqAbs(@as(f32, 1), dst[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4), dst[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2), dst[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 5), dst[3], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3), dst[4], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 6), dst[5], 0.001);
}
