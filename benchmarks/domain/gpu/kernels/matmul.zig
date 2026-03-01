//! Matrix multiplication benchmarks — scalar, blocked, SIMD, and GPU.

const std = @import("std");
const build_options = @import("build_options");
const framework = @import("../../../system/framework.zig");
const parent_mod = @import("../mod.zig");
const mod = @import("mod.zig");
const cpu = mod.cpu_baselines;
const gpu = mod.gpu;

const GpuBenchConfig = parent_mod.GpuBenchConfig;

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

    const abi = @import("abi");

    var gpu_ctx = abi.features.gpu.Gpu.init(allocator, .{}) catch return error.GpuInitFailed;
    defer gpu_ctx.deinit();

    if (!gpu_ctx.isAvailable()) return error.NoGpuDevice;

    const buf_a = try gpu_ctx.createBufferFromSlice(f32, A, .{ .mode = .explicit });
    defer gpu_ctx.destroyBuffer(buf_a);
    const buf_b = try gpu_ctx.createBufferFromSlice(f32, B, .{ .mode = .explicit });
    defer gpu_ctx.destroyBuffer(buf_b);
    const buf_c = try gpu_ctx.createBuffer(C.len * @sizeOf(f32), .{ .mode = .explicit });
    defer gpu_ctx.destroyBuffer(buf_c);

    for (0..config.warmup_iterations) |_| {
        _ = try gpu_ctx.matrixMultiply(buf_a, buf_b, buf_c, .{ .m = size, .n = size, .k = size });
    }

    var total_ns: u64 = 0;
    var min_ns: u64 = std.math.maxInt(u64);
    var max_ns: u64 = 0;
    var iterations: u32 = 0;

    while (total_ns < config.min_time_ns and iterations < config.benchmark_iterations) : (iterations += 1) {
        var timer = abi.services.shared.time.Timer.start() catch return error.TimerFailed;
        _ = try gpu_ctx.matrixMultiply(buf_a, buf_b, buf_c, .{ .m = size, .n = size, .k = size });
        try gpu_ctx.synchronize();
        const elapsed = timer.read();
        total_ns += elapsed;
        if (elapsed < min_ns) min_ns = elapsed;
        if (elapsed > max_ns) max_ns = elapsed;
    }

    return .{
        .mean_ns = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(iterations)),
        .min_ns = min_ns,
        .max_ns = max_ns,
        .iterations = iterations,
    };
}

/// Run matrix multiplication benchmarks
pub fn benchmarkMatmul(
    allocator: std.mem.Allocator,
    runner: *framework.BenchmarkRunner,
    config: GpuBenchConfig,
) !void {
    std.debug.print("\n[Matrix Multiplication Benchmarks]\n", .{});

    const gpu_available = parent_mod.hasHardwareGpu(allocator);
    if (build_options.enable_gpu and !gpu_available) {
        std.debug.print("  [GPU SKIP] No hardware GPU detected; skipping GPU matmul benchmarks.\n", .{});
    }

    for (config.matrix_sizes) |size| {
        const matrix_size = size * size;
        const A = try allocator.alignedAlloc(f32, .@"64", matrix_size);
        defer allocator.free(A);
        const B = try allocator.alignedAlloc(f32, .@"64", matrix_size);
        defer allocator.free(B);
        const C = try allocator.alignedAlloc(f32, .@"64", matrix_size);
        defer allocator.free(C);

        mod.initRandomData(A, 42);
        mod.initRandomData(B, 43);

        const flops_per_op = 2 * size * size * size;
        const bytes_per_op = matrix_size * @sizeOf(f32) * 3;

        // Scalar matmul (baseline)
        if (size <= 512) {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "matmul_scalar_{d}x{d}", .{ size, size }) catch "matmul_scalar";
            const result = try runner.run(
                .{ .name = name, .category = "gpu/matmul", .bytes_per_op = bytes_per_op, .warmup_iterations = @min(config.warmup_iterations, 5), .min_time_ns = config.min_time_ns, .max_iterations = @min(config.max_iterations, 100) },
                struct {
                    fn bench(a: []const f32, b: []const f32, c: []f32, n: usize) void {
                        cpu.scalarMatmul(a, b, c, n, n, n);
                    }
                }.bench,
                .{ A, B, C, size },
            );
            const gflops = @as(f64, @floatFromInt(flops_per_op)) * result.stats.opsPerSecond() / 1e9;
            std.debug.print("  {s}: {d:.2} GFLOPS ({d:.2} ms/op)\n", .{ name, gflops, result.stats.mean_ns / 1e6 });
        }

        // Blocked matmul
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "matmul_blocked_{d}x{d}", .{ size, size }) catch "matmul_blocked";
            const result = try runner.run(
                .{ .name = name, .category = "gpu/matmul", .bytes_per_op = bytes_per_op, .warmup_iterations = config.warmup_iterations, .min_time_ns = config.min_time_ns, .max_iterations = config.max_iterations },
                struct {
                    fn bench(a: []const f32, b: []const f32, c: []f32, n: usize) void {
                        cpu.blockedMatmul(a, b, c, n, n, n, 32);
                    }
                }.bench,
                .{ A, B, C, size },
            );
            const gflops = @as(f64, @floatFromInt(flops_per_op)) * result.stats.opsPerSecond() / 1e9;
            std.debug.print("  {s}: {d:.2} GFLOPS ({d:.2} ms/op)\n", .{ name, gflops, result.stats.mean_ns / 1e6 });
        }

        // SIMD matmul
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "matmul_simd8_{d}x{d}", .{ size, size }) catch "matmul_simd";
            const result = try runner.run(
                .{ .name = name, .category = "gpu/matmul", .bytes_per_op = bytes_per_op, .warmup_iterations = config.warmup_iterations, .min_time_ns = config.min_time_ns, .max_iterations = config.max_iterations },
                struct {
                    fn bench(a: []const f32, b: []const f32, c: []f32, n: usize) void {
                        cpu.simdMatmul(8, a, b, c, n, n, n);
                    }
                }.bench,
                .{ A, B, C, size },
            );
            const gflops = @as(f64, @floatFromInt(flops_per_op)) * result.stats.opsPerSecond() / 1e9;
            std.debug.print("  {s}: {d:.2} GFLOPS ({d:.2} ms/op)\n", .{ name, gflops, result.stats.mean_ns / 1e6 });
        }

        // GPU matmul (if available)
        if (build_options.enable_gpu and gpu_available) {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "matmul_gpu_{d}x{d}", .{ size, size }) catch "matmul_gpu";
            const gpu_result = runGpuMatmulBenchmark(allocator, A, B, C, size, config) catch |err| {
                std.debug.print("  {s}: GPU unavailable ({t})\n", .{ name, err });
                continue;
            };
            const gflops = @as(f64, @floatFromInt(flops_per_op)) / (gpu_result.mean_ns / 1e9);
            std.debug.print("  {s}: {d:.2} GFLOPS ({d:.2} ms/op)\n", .{ name, gflops, gpu_result.mean_ns / 1e6 });
        }
    }
}
