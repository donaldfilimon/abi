//! GPU Backend Comparison Benchmarks
//!
//! Compares performance across different GPU backends:
//! - CUDA vs Vulkan vs Metal vs OpenGL
//! - GPU vs CPU performance ratios
//! - Backend-specific optimizations
//!
//! When multiple backends are available, runs the same operations
//! on each backend and reports comparative performance metrics.

const std = @import("std");
const build_options = @import("build_options");
const framework = @import("../../system/framework.zig");
const mod = @import("mod.zig");

const GpuBenchConfig = mod.GpuBenchConfig;

// Conditional imports based on build options
const gpu_mod = if (build_options.enable_gpu) @import("abi").features.gpu else struct {
    pub const Backend = enum { none, auto, cuda, vulkan, metal, webgpu, opengl, stdgpu, fpga };
    pub const moduleEnabled = struct {
        pub fn call() bool {
            return false;
        }
    }.call;
    pub const availableBackends = struct {
        pub fn call(_: std.mem.Allocator) ![]Backend {
            return &.{};
        }
    }.call;
    pub const backendName = struct {
        pub fn call(_: Backend) []const u8 {
            return "none";
        }
    }.call;
};

/// Backend benchmark result
pub const BackendBenchResult = struct {
    backend: gpu_mod.Backend,
    operation: []const u8,
    size: usize,
    mean_ns: f64,
    min_ns: u64,
    max_ns: u64,
    std_dev_ns: f64,
    iterations: u32,
    gflops: f64,
    bandwidth_gbps: f64,
    success: bool,
    error_message: ?[]const u8,

    pub fn throughputMBps(self: BackendBenchResult) f64 {
        if (self.mean_ns == 0) return 0;
        return self.bandwidth_gbps * 1024.0;
    }
};

/// Backend comparison summary
pub const BackendComparisonSummary = struct {
    operation: []const u8,
    size: usize,
    results: []BackendBenchResult,
    fastest_backend: ?gpu_mod.Backend,
    speedup_vs_cpu: f64,
};

/// Run benchmarks on a specific backend
fn runBackendBenchmark(
    allocator: std.mem.Allocator,
    backend: gpu_mod.Backend,
    operation: []const u8,
    size: usize,
    config: GpuBenchConfig,
) BackendBenchResult {
    _ = allocator;
    _ = config;

    // Default result for when GPU is not available
    var result = BackendBenchResult{
        .backend = backend,
        .operation = operation,
        .size = size,
        .mean_ns = 0,
        .min_ns = 0,
        .max_ns = 0,
        .std_dev_ns = 0,
        .iterations = 0,
        .gflops = 0,
        .bandwidth_gbps = 0,
        .success = false,
        .error_message = "GPU not enabled",
    };

    if (!build_options.enable_gpu) {
        return result;
    }

    // Try to initialize the specific backend
    // Note: Full implementation would initialize GPU context with specific backend
    // and run the actual benchmark
    result.error_message = "Backend benchmarking not yet implemented";
    return result;
}

/// Detect available GPU backends
fn detectAvailableBackends(allocator: std.mem.Allocator) ![]gpu_mod.Backend {
    if (!build_options.enable_gpu) {
        return &.{};
    }

    var backends = std.ArrayListUnmanaged(gpu_mod.Backend).empty;
    errdefer backends.deinit(allocator);

    // Check each backend's availability
    // Note: This uses compile-time flags; runtime detection would need driver probing

    if (comptime build_options.gpu_cuda) {
        try backends.append(allocator, .cuda);
    }
    if (comptime build_options.gpu_vulkan) {
        try backends.append(allocator, .vulkan);
    }
    if (comptime build_options.gpu_metal) {
        try backends.append(allocator, .metal);
    }
    if (comptime build_options.gpu_webgpu) {
        try backends.append(allocator, .webgpu);
    }
    if (comptime build_options.gpu_opengl) {
        try backends.append(allocator, .opengl);
    }

    // Always include stdgpu as fallback
    try backends.append(allocator, .stdgpu);

    return backends.toOwnedSlice(allocator);
}

/// Print backend detection results
fn printBackendDetection(backends: []const gpu_mod.Backend) void {
    std.debug.print("\n[GPU Backend Detection]\n", .{});

    if (backends.len == 0) {
        std.debug.print("  No GPU backends available\n", .{});
        std.debug.print("  Rebuild with GPU backend flags:\n", .{});
        std.debug.print("    -Dgpu-backend=cuda,vulkan\n", .{});
        return;
    }

    std.debug.print("  Available backends: ", .{});
    for (backends, 0..) |backend, i| {
        if (i > 0) std.debug.print(", ", .{});
        std.debug.print("{t}", .{backend});
    }
    std.debug.print("\n", .{});
}

/// Compare GPU performance vs CPU baseline
fn gpuVsCpuComparison(
    allocator: std.mem.Allocator,
    runner: *framework.BenchmarkRunner,
    config: GpuBenchConfig,
) !void {
    std.debug.print("\n[GPU vs CPU Comparison]\n", .{});

    if (!build_options.enable_gpu) {
        std.debug.print("  GPU not enabled - skipping comparison\n", .{});
        return;
    }

    if (!mod.hasHardwareGpu(allocator)) {
        std.debug.print("  No hardware GPU detected - skipping comparison\n", .{});
        return;
    }

    // Matrix multiplication comparison
    for (config.matrix_sizes[0..@min(3, config.matrix_sizes.len)]) |size| {
        const matrix_size = size * size;
        const flops = 2 * size * size * size;

        // Allocate matrices
        const A = try allocator.alignedAlloc(f32, .@"64", matrix_size);
        defer allocator.free(A);
        const B = try allocator.alignedAlloc(f32, .@"64", matrix_size);
        defer allocator.free(B);
        const C = try allocator.alignedAlloc(f32, .@"64", matrix_size);
        defer allocator.free(C);

        // Initialize with random data
        var prng = std.Random.DefaultPrng.init(42);
        for (A) |*x| x.* = prng.random().float(f32);
        for (B) |*x| x.* = prng.random().float(f32);

        // CPU benchmark (SIMD)
        var cpu_time_ns: f64 = 0;
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "matmul_cpu_simd_{d}x{d}", .{ size, size }) catch "matmul_cpu";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "gpu/comparison",
                    .warmup_iterations = config.warmup_iterations,
                    .min_time_ns = config.min_time_ns,
                    .max_iterations = @min(config.max_iterations, 500),
                },
                struct {
                    fn bench(a: []const f32, b: []const f32, c: []f32, n: usize) void {
                        const Vec = @Vector(8, f32);
                        @memset(c, 0);

                        for (0..n) |i| {
                            for (0..n) |k| {
                                const a_val: Vec = @splat(a[i * n + k]);
                                var j: usize = 0;

                                while (j + 8 <= n) : (j += 8) {
                                    const b_vec: Vec = b[k * n + j ..][0..8].*;
                                    const c_vec: Vec = c[i * n + j ..][0..8].*;
                                    c[i * n + j ..][0..8].* = c_vec + a_val * b_vec;
                                }

                                while (j < n) : (j += 1) {
                                    c[i * n + j] += a[i * n + k] * b[k * n + j];
                                }
                            }
                        }
                    }
                }.bench,
                .{ A, B, C, size },
            );

            cpu_time_ns = result.stats.mean_ns;
            const cpu_gflops = @as(f64, @floatFromInt(flops)) / (cpu_time_ns / 1e9) / 1e9;
            std.debug.print("  {s}: {d:.2} GFLOPS ({d:.2} ms)\n", .{ name, cpu_gflops, cpu_time_ns / 1e6 });
        }

        // GPU benchmark (if available)
        if (build_options.enable_gpu) {
            const abi = @import("abi");

            var gpu_ctx = abi.features.gpu.Gpu.init(allocator, .{}) catch {
                std.debug.print("  matmul_gpu_{d}x{d}: GPU init failed\n", .{ size, size });
                continue;
            };
            defer gpu_ctx.deinit();

            if (!gpu_ctx.isAvailable()) {
                std.debug.print("  matmul_gpu_{d}x{d}: No GPU device\n", .{ size, size });
                continue;
            }

            // Create buffers
            const buf_a = gpu_ctx.createBufferFromSlice(f32, A, .{ .mode = .explicit }) catch {
                std.debug.print("  matmul_gpu_{d}x{d}: Buffer creation failed\n", .{ size, size });
                continue;
            };
            defer gpu_ctx.destroyBuffer(buf_a);

            const buf_b = gpu_ctx.createBufferFromSlice(f32, B, .{ .mode = .explicit }) catch {
                std.debug.print("  matmul_gpu_{d}x{d}: Buffer creation failed\n", .{ size, size });
                continue;
            };
            defer gpu_ctx.destroyBuffer(buf_b);

            const buf_c = gpu_ctx.createBuffer(C.len * @sizeOf(f32), .{ .mode = .explicit }) catch {
                std.debug.print("  matmul_gpu_{d}x{d}: Buffer creation failed\n", .{ size, size });
                continue;
            };
            defer gpu_ctx.destroyBuffer(buf_c);

            // Warmup
            for (0..config.warmup_iterations) |_| {
                _ = gpu_ctx.matrixMultiply(buf_a, buf_b, buf_c, .{
                    .m = size,
                    .n = size,
                    .k = size,
                }) catch continue;
            }

            // Benchmark
            var total_ns: u64 = 0;
            var iterations: u32 = 0;

            while (total_ns < config.min_time_ns and iterations < config.benchmark_iterations) : (iterations += 1) {
                var timer = abi.services.shared.time.Timer.start() catch continue;
                _ = gpu_ctx.matrixMultiply(buf_a, buf_b, buf_c, .{
                    .m = size,
                    .n = size,
                    .k = size,
                }) catch continue;
                gpu_ctx.synchronize() catch {};
                total_ns += timer.read();
            }

            if (iterations > 0) {
                const gpu_time_ns = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(iterations));
                const gpu_gflops = @as(f64, @floatFromInt(flops)) / (gpu_time_ns / 1e9) / 1e9;
                const speedup = cpu_time_ns / gpu_time_ns;

                if (gpu_ctx.getBackend()) |backend| {
                    std.debug.print("  matmul_gpu_{d}x{d} ({t}): {d:.2} GFLOPS ({d:.2} ms) - {d:.1}x speedup\n", .{
                        size,
                        size,
                        backend,
                        gpu_gflops,
                        gpu_time_ns / 1e6,
                        speedup,
                    });
                } else {
                    std.debug.print("  matmul_gpu_{d}x{d} (unknown): {d:.2} GFLOPS ({d:.2} ms) - {d:.1}x speedup\n", .{
                        size,
                        size,
                        gpu_gflops,
                        gpu_time_ns / 1e6,
                        speedup,
                    });
                }
            }
        }
    }
}

/// Compare performance across different backends
fn backendComparison(
    allocator: std.mem.Allocator,
    backends: []const gpu_mod.Backend,
    config: GpuBenchConfig,
) !void {
    std.debug.print("\n[Backend Performance Comparison]\n", .{});

    if (backends.len <= 1) {
        std.debug.print("  Only one backend available - skipping comparison\n", .{});
        return;
    }

    // For each test size
    for (config.matrix_sizes[0..@min(2, config.matrix_sizes.len)]) |size| {
        std.debug.print("\n  Matrix size: {d}x{d}\n", .{ size, size });

        var results = std.ArrayListUnmanaged(BackendBenchResult).empty;
        defer results.deinit(allocator);

        // Test each backend
        for (backends) |backend| {
            const result = runBackendBenchmark(allocator, backend, "matmul", size, config);
            try results.append(allocator, result);

            if (result.success) {
                std.debug.print("    {t}: {d:.2} GFLOPS\n", .{ backend, result.gflops });
            } else {
                std.debug.print("    {t}: {s}\n", .{ backend, result.error_message orelse "failed" });
            }
        }

        // Find fastest backend
        var fastest: ?BackendBenchResult = null;
        for (results.items) |result| {
            if (result.success) {
                if (fastest == null or result.mean_ns < fastest.?.mean_ns) {
                    fastest = result;
                }
            }
        }

        if (fastest) |f| {
            std.debug.print("    Fastest: {t}\n", .{f.backend});
        }
    }
}

/// Run all backend comparison benchmarks
pub fn runBackendBenchmarks(allocator: std.mem.Allocator, config: GpuBenchConfig) !void {
    var runner = framework.BenchmarkRunner.init(allocator);
    defer runner.deinit();

    // Detect available backends
    const backends = try detectAvailableBackends(allocator);
    defer if (backends.len > 0) allocator.free(backends);

    // Print detection results
    printBackendDetection(backends);

    // GPU vs CPU comparison
    if (config.include_cpu_comparison) {
        try gpuVsCpuComparison(allocator, &runner, config);
    }

    // Backend comparison (if multiple backends available)
    if (config.include_backend_comparison and backends.len > 1) {
        try backendComparison(allocator, backends, config);
    }
}

// ============================================================================
// Tests
// ============================================================================

test "backend detection" {
    const allocator = std.testing.allocator;
    const backends = try detectAvailableBackends(allocator);
    defer if (backends.len > 0) allocator.free(backends);

    // Should at least have stdgpu if GPU is enabled
    if (build_options.enable_gpu) {
        try std.testing.expect(backends.len >= 1);
    }
}

test "backend bench result" {
    const result = BackendBenchResult{
        .backend = .stdgpu,
        .operation = "matmul",
        .size = 1024,
        .mean_ns = 1_000_000, // 1ms
        .min_ns = 900_000,
        .max_ns = 1_100_000,
        .std_dev_ns = 50_000,
        .iterations = 100,
        .gflops = 10.0,
        .bandwidth_gbps = 50.0,
        .success = true,
        .error_message = null,
    };

    try std.testing.expectApproxEqAbs(@as(f64, 50.0 * 1024.0), result.throughputMBps(), 0.1);
}
