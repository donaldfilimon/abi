//! GPU vs CPU Performance Comparison Benchmarks
//!
//! Provides detailed comparison between GPU and CPU implementations
//! of common operations, including:
//! - Speedup ratios
//! - Break-even points (data sizes where GPU becomes faster)
//! - Efficiency analysis
//!
//! This module helps determine when to offload computation to GPU
//! and provides insights into GPU utilization efficiency.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");
const framework = @import("../../system/framework.zig");
const mod = @import("mod.zig");

const GpuBenchConfig = mod.GpuBenchConfig;

/// Comparison result for a single operation
pub const ComparisonResult = struct {
    operation: []const u8,
    data_size: usize,
    cpu_time_ns: f64,
    gpu_time_ns: f64,
    speedup: f64,
    gpu_efficiency: f64, // Percentage of theoretical peak
    cpu_throughput_gbps: f64,
    gpu_throughput_gbps: f64,
    gpu_available: bool,
    breakeven_estimated: bool,

    pub fn format(self: ComparisonResult) [256]u8 {
        var buf: [256]u8 = undefined;
        if (self.gpu_available) {
            _ = std.fmt.bufPrint(&buf, "{s}: CPU={d:.2}ms GPU={d:.2}ms Speedup={d:.1}x", .{
                self.operation,
                self.cpu_time_ns / 1e6,
                self.gpu_time_ns / 1e6,
                self.speedup,
            }) catch {};
        } else {
            _ = std.fmt.bufPrint(&buf, "{s}: CPU={d:.2}ms GPU=N/A", .{
                self.operation,
                self.cpu_time_ns / 1e6,
            }) catch {};
        }
        return buf;
    }
};

/// Break-even analysis result
pub const BreakevenAnalysis = struct {
    operation: []const u8,
    breakeven_size: ?usize, // Size where GPU becomes faster
    small_data_winner: []const u8, // "cpu" or "gpu"
    large_data_winner: []const u8, // "cpu" or "gpu"
    recommendation: []const u8,
};

// ============================================================================
// CPU Reference Implementations
// ============================================================================

/// CPU SIMD matrix multiplication
fn cpuSimdMatmul(A: []const f32, B: []const f32, C: []f32, M: usize, N: usize, K: usize) void {
    const Vec = @Vector(8, f32);
    @memset(C, 0);

    for (0..M) |i| {
        for (0..K) |k| {
            const a_val: Vec = @splat(A[i * K + k]);
            var j: usize = 0;

            while (j + 8 <= N) : (j += 8) {
                const b_vec: Vec = B[k * N + j ..][0..8].*;
                const c_vec: Vec = C[i * N + j ..][0..8].*;
                C[i * N + j ..][0..8].* = c_vec + a_val * b_vec;
            }

            while (j < N) : (j += 1) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

/// CPU SIMD vector addition
fn cpuSimdVectorAdd(a: []const f32, b: []const f32, result: []f32) void {
    const Vec = @Vector(8, f32);
    var i: usize = 0;

    while (i + 8 <= a.len) : (i += 8) {
        const va: Vec = a[i..][0..8].*;
        const vb: Vec = b[i..][0..8].*;
        result[i..][0..8].* = va + vb;
    }

    while (i < a.len) : (i += 1) {
        result[i] = a[i] + b[i];
    }
}

/// CPU SIMD reduction sum
fn cpuSimdReduceSum(v: []const f32) f32 {
    const Vec = @Vector(8, f32);
    var sum: f32 = 0;
    var i: usize = 0;

    while (i + 8 <= v.len) : (i += 8) {
        const vec: Vec = v[i..][0..8].*;
        sum += @reduce(.Add, vec);
    }

    while (i < v.len) : (i += 1) {
        sum += v[i];
    }

    return sum;
}

// ============================================================================
// Comparison Benchmarks
// ============================================================================

/// Compare matrix multiplication performance
pub fn compareMatmul(
    allocator: std.mem.Allocator,
    size: usize,
    config: GpuBenchConfig,
) !ComparisonResult {
    const matrix_size = size * size;
    const flops = 2 * size * size * size;
    const bytes = matrix_size * @sizeOf(f32) * 3;

    // Allocate matrices
    const A = try allocator.alignedAlloc(f32, .@"64", matrix_size);
    defer allocator.free(A);
    const B = try allocator.alignedAlloc(f32, .@"64", matrix_size);
    defer allocator.free(B);
    const C = try allocator.alignedAlloc(f32, .@"64", matrix_size);
    defer allocator.free(C);

    // Initialize
    var prng = std.Random.DefaultPrng.init(42);
    for (A) |*x| x.* = prng.random().float(f32);
    for (B) |*x| x.* = prng.random().float(f32);

    // CPU benchmark
    var cpu_total: u64 = 0;
    var cpu_iters: u32 = 0;

    // Warmup
    for (0..config.warmup_iterations) |_| {
        cpuSimdMatmul(A, B, C, size, size, size);
    }

    // Benchmark
    while (cpu_total < config.min_time_ns and cpu_iters < config.benchmark_iterations) : (cpu_iters += 1) {
        var timer = abi.services.shared.time.Timer.start() catch continue;
        cpuSimdMatmul(A, B, C, size, size, size);
        cpu_total += timer.read();
    }

    if (cpu_iters == 0) return error.BenchmarkFailed;

    const cpu_time_ns = @as(f64, @floatFromInt(cpu_total)) / @as(f64, @floatFromInt(cpu_iters));
    const cpu_throughput = @as(f64, @floatFromInt(bytes)) / (cpu_time_ns / 1e9) / 1e9;

    const hardware_gpu = mod.hasHardwareGpu(allocator);

    // GPU benchmark (if available)
    var gpu_time_ns: f64 = 0;
    var gpu_throughput: f64 = 0;
    var gpu_available = false;

    if (build_options.enable_gpu and hardware_gpu) {
        var gpu_instance_storage: abi.features.gpu.Gpu = abi.features.gpu.Gpu.init(allocator, .{}) catch {
            return ComparisonResult{
                .operation = "matmul",
                .data_size = size,
                .cpu_time_ns = cpu_time_ns,
                .gpu_time_ns = 0,
                .speedup = 0,
                .gpu_efficiency = 0,
                .cpu_throughput_gbps = cpu_throughput,
                .gpu_throughput_gbps = 0,
                .gpu_available = false,
                .breakeven_estimated = false,
            };
        };
        {
            var gpu_ctx = &gpu_instance_storage;
            defer gpu_ctx.deinit();

            if (gpu_ctx.isAvailable()) {
                gpu_available = true;

                const buf_a = gpu_ctx.createBufferFromSlice(f32, A, .{ .mode = .explicit }) catch null;
                const buf_b = gpu_ctx.createBufferFromSlice(f32, B, .{ .mode = .explicit }) catch null;
                const buf_c = gpu_ctx.createBuffer(C.len * @sizeOf(f32), .{ .mode = .explicit }) catch null;

                if (buf_a != null and buf_b != null and buf_c != null) {
                    defer {
                        if (buf_a) |b| gpu_ctx.destroyBuffer(b);
                        if (buf_b) |b| gpu_ctx.destroyBuffer(b);
                        if (buf_c) |b| gpu_ctx.destroyBuffer(b);
                    }

                    // Warmup
                    for (0..config.warmup_iterations) |_| {
                        _ = gpu_ctx.matrixMultiply(buf_a.?, buf_b.?, buf_c.?, .{
                            .m = size,
                            .n = size,
                            .k = size,
                        }) catch continue;
                    }

                    // Benchmark
                    var gpu_total: u64 = 0;
                    var gpu_iters: u32 = 0;

                    while (gpu_total < config.min_time_ns and gpu_iters < config.benchmark_iterations) : (gpu_iters += 1) {
                        var timer = abi.services.shared.time.Timer.start() catch continue;
                        _ = gpu_ctx.matrixMultiply(buf_a.?, buf_b.?, buf_c.?, .{
                            .m = size,
                            .n = size,
                            .k = size,
                        }) catch continue;
                        gpu_ctx.synchronize() catch {};
                        gpu_total += timer.read();
                    }

                    if (gpu_iters > 0) {
                        gpu_time_ns = @as(f64, @floatFromInt(gpu_total)) / @as(f64, @floatFromInt(gpu_iters));
                        gpu_throughput = @as(f64, @floatFromInt(bytes)) / (gpu_time_ns / 1e9) / 1e9;
                    }
                }
            }
        }
    }

    const speedup = if (gpu_available and gpu_time_ns > 0) cpu_time_ns / gpu_time_ns else 0;

    // Estimate GPU efficiency (assuming ~10 TFLOPS theoretical peak for modern GPU)
    const theoretical_peak_gflops: f64 = 10000;
    const actual_gflops = if (gpu_time_ns > 0)
        @as(f64, @floatFromInt(flops)) / (gpu_time_ns / 1e9) / 1e9
    else
        0;
    const gpu_efficiency = (actual_gflops / theoretical_peak_gflops) * 100;

    return ComparisonResult{
        .operation = "matmul",
        .data_size = size,
        .cpu_time_ns = cpu_time_ns,
        .gpu_time_ns = gpu_time_ns,
        .speedup = speedup,
        .gpu_efficiency = gpu_efficiency,
        .cpu_throughput_gbps = cpu_throughput,
        .gpu_throughput_gbps = gpu_throughput,
        .gpu_available = gpu_available,
        .breakeven_estimated = false,
    };
}

/// Compare vector addition performance
pub fn compareVectorAdd(
    allocator: std.mem.Allocator,
    size: usize,
    config: GpuBenchConfig,
) !ComparisonResult {
    const bytes = size * @sizeOf(f32) * 3;

    // Allocate vectors
    const a = try allocator.alignedAlloc(f32, .@"64", size);
    defer allocator.free(a);
    const b = try allocator.alignedAlloc(f32, .@"64", size);
    defer allocator.free(b);
    const result = try allocator.alignedAlloc(f32, .@"64", size);
    defer allocator.free(result);

    // Initialize
    var prng = std.Random.DefaultPrng.init(42);
    for (a) |*x| x.* = prng.random().float(f32);
    for (b) |*x| x.* = prng.random().float(f32);

    // CPU benchmark
    var cpu_total: u64 = 0;
    var cpu_iters: u32 = 0;

    for (0..config.warmup_iterations) |_| {
        cpuSimdVectorAdd(a, b, result);
    }

    while (cpu_total < config.min_time_ns and cpu_iters < config.benchmark_iterations) : (cpu_iters += 1) {
        var timer = abi.services.shared.time.Timer.start() catch continue;
        cpuSimdVectorAdd(a, b, result);
        cpu_total += timer.read();
    }

    if (cpu_iters == 0) return error.BenchmarkFailed;

    const cpu_time_ns = @as(f64, @floatFromInt(cpu_total)) / @as(f64, @floatFromInt(cpu_iters));
    const cpu_throughput = @as(f64, @floatFromInt(bytes)) / (cpu_time_ns / 1e9) / 1e9;

    // GPU would be benchmarked here if available
    // For now, return CPU-only result
    return ComparisonResult{
        .operation = "vector_add",
        .data_size = size,
        .cpu_time_ns = cpu_time_ns,
        .gpu_time_ns = 0,
        .speedup = 0,
        .gpu_efficiency = 0,
        .cpu_throughput_gbps = cpu_throughput,
        .gpu_throughput_gbps = 0,
        .gpu_available = false,
        .breakeven_estimated = false,
    };
}

/// Compare reduction sum performance
pub fn compareReduceSum(
    allocator: std.mem.Allocator,
    size: usize,
    config: GpuBenchConfig,
) !ComparisonResult {
    const bytes = size * @sizeOf(f32);

    // Allocate vector
    const v = try allocator.alignedAlloc(f32, .@"64", size);
    defer allocator.free(v);

    // Initialize
    var prng = std.Random.DefaultPrng.init(42);
    for (v) |*x| x.* = prng.random().float(f32);

    // CPU benchmark
    var cpu_total: u64 = 0;
    var cpu_iters: u32 = 0;

    for (0..config.warmup_iterations) |_| {
        _ = cpuSimdReduceSum(v);
    }

    while (cpu_total < config.min_time_ns and cpu_iters < config.benchmark_iterations) : (cpu_iters += 1) {
        var timer = abi.services.shared.time.Timer.start() catch continue;
        _ = cpuSimdReduceSum(v);
        cpu_total += timer.read();
    }

    if (cpu_iters == 0) return error.BenchmarkFailed;

    const cpu_time_ns = @as(f64, @floatFromInt(cpu_total)) / @as(f64, @floatFromInt(cpu_iters));
    const cpu_throughput = @as(f64, @floatFromInt(bytes)) / (cpu_time_ns / 1e9) / 1e9;

    return ComparisonResult{
        .operation = "reduce_sum",
        .data_size = size,
        .cpu_time_ns = cpu_time_ns,
        .gpu_time_ns = 0,
        .speedup = 0,
        .gpu_efficiency = 0,
        .cpu_throughput_gbps = cpu_throughput,
        .gpu_throughput_gbps = 0,
        .gpu_available = false,
        .breakeven_estimated = false,
    };
}

/// Analyze break-even point for an operation
pub fn analyzeBreakeven(
    allocator: std.mem.Allocator,
    operation: []const u8,
    sizes: []const usize,
    config: GpuBenchConfig,
) !BreakevenAnalysis {
    var breakeven_size: ?usize = null;
    var small_winner: []const u8 = "cpu";
    var large_winner: []const u8 = "cpu";

    for (sizes) |size| {
        const result = if (std.mem.eql(u8, operation, "matmul"))
            try compareMatmul(allocator, size, config)
        else if (std.mem.eql(u8, operation, "vector_add"))
            try compareVectorAdd(allocator, size, config)
        else
            try compareReduceSum(allocator, size, config);

        if (result.gpu_available and result.speedup > 1.0) {
            if (breakeven_size == null) {
                breakeven_size = size;
            }
            large_winner = "gpu";
        }

        if (size == sizes[0] and result.gpu_available and result.speedup > 1.0) {
            small_winner = "gpu";
        }
    }

    const recommendation = if (breakeven_size) |be|
        if (be <= 1024)
            "GPU recommended for most workloads"
        else if (be <= 4096)
            "GPU recommended for medium to large workloads"
        else
            "GPU only beneficial for large workloads"
    else
        "CPU recommended (GPU not available or not faster)";

    return BreakevenAnalysis{
        .operation = operation,
        .breakeven_size = breakeven_size,
        .small_data_winner = small_winner,
        .large_data_winner = large_winner,
        .recommendation = recommendation,
    };
}

/// Run all GPU vs CPU comparison benchmarks
pub fn runComparisonBenchmarks(allocator: std.mem.Allocator, config: GpuBenchConfig) !void {
    std.debug.print("\n[GPU vs CPU Comparison Benchmarks]\n", .{});
    std.debug.print("=====================================\n", .{});

    // Matrix multiplication comparison
    std.debug.print("\n  Matrix Multiplication:\n", .{});
    for (config.matrix_sizes[0..@min(4, config.matrix_sizes.len)]) |size| {
        const result = try compareMatmul(allocator, size, config);
        if (result.gpu_available) {
            std.debug.print("    {d}x{d}: CPU={d:.2}ms GPU={d:.2}ms Speedup={d:.1}x Efficiency={d:.1}%\n", .{
                size,
                size,
                result.cpu_time_ns / 1e6,
                result.gpu_time_ns / 1e6,
                result.speedup,
                result.gpu_efficiency,
            });
        } else {
            std.debug.print("    {d}x{d}: CPU={d:.2}ms GPU=N/A\n", .{
                size,
                size,
                result.cpu_time_ns / 1e6,
            });
        }
    }

    // Vector addition comparison
    std.debug.print("\n  Vector Addition:\n", .{});
    for (config.vector_sizes[0..@min(4, config.vector_sizes.len)]) |size| {
        const result = try compareVectorAdd(allocator, size, config);
        std.debug.print("    {d} elements: CPU={d:.2}ms ({d:.2} GB/s)\n", .{
            size,
            result.cpu_time_ns / 1e6,
            result.cpu_throughput_gbps,
        });
    }

    // Reduction comparison
    std.debug.print("\n  Reduction Sum:\n", .{});
    for (config.vector_sizes[0..@min(4, config.vector_sizes.len)]) |size| {
        const result = try compareReduceSum(allocator, size, config);
        std.debug.print("    {d} elements: CPU={d:.2}ms ({d:.2} GB/s)\n", .{
            size,
            result.cpu_time_ns / 1e6,
            result.cpu_throughput_gbps,
        });
    }

    // Break-even analysis
    std.debug.print("\n  Break-even Analysis:\n", .{});
    const matmul_analysis = try analyzeBreakeven(allocator, "matmul", config.matrix_sizes, config);
    if (matmul_analysis.breakeven_size) |be| {
        std.debug.print("    matmul: GPU faster at {d}x{d} and above\n", .{ be, be });
    } else {
        std.debug.print("    matmul: {s}\n", .{matmul_analysis.recommendation});
    }
}

// ============================================================================
// Tests
// ============================================================================

test "comparison result format" {
    const result = ComparisonResult{
        .operation = "test_op",
        .data_size = 1024,
        .cpu_time_ns = 1_000_000,
        .gpu_time_ns = 100_000,
        .speedup = 10.0,
        .gpu_efficiency = 50.0,
        .cpu_throughput_gbps = 1.0,
        .gpu_throughput_gbps = 10.0,
        .gpu_available = true,
        .breakeven_estimated = false,
    };

    _ = result.format();
}

test "cpu simd operations" {
    // Test vector add
    const a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const b = [_]f32{ 8, 7, 6, 5, 4, 3, 2, 1 };
    var result: [8]f32 = undefined;

    cpuSimdVectorAdd(&a, &b, &result);

    for (0..8) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, 9), result[i], 0.001);
    }
}

test "cpu simd reduce" {
    const v = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const sum = cpuSimdReduceSum(&v);
    try std.testing.expectApproxEqAbs(@as(f32, 36), sum, 0.001);
}
