//! Vector operation benchmarks — add, mul, dot product, normalize.

const std = @import("std");
const framework = @import("../../../system/framework.zig");
const parent_mod = @import("../mod.zig");
const mod = @import("mod.zig");
const cpu = mod.cpu_baselines;

const GpuBenchConfig = parent_mod.GpuBenchConfig;

/// Run vector operation benchmarks
pub fn benchmarkVectorOps(
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

        mod.initRandomData(a, 42);
        mod.initRandomData(b, 43);

        const size_info = mod.formatSize(size * @sizeOf(f32));
        const bytes_per_op = size * @sizeOf(f32) * 3; // 2 read + 1 write

        // Vector Add - Scalar
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "vecadd_scalar_{d:.1}{s}", .{ size_info.value, size_info.unit }) catch "vecadd_scalar";
            const bench_result = try runner.run(
                .{ .name = name, .category = "gpu/vector", .bytes_per_op = bytes_per_op, .warmup_iterations = config.warmup_iterations, .min_time_ns = config.min_time_ns },
                struct {
                    fn bench(va: []const f32, vb: []const f32, vr: []f32) void {
                        cpu.scalarVectorAdd(va, vb, vr);
                    }
                }.bench,
                .{ a, b, result },
            );
            std.debug.print("  {s}: {d:.2} GB/s ({d:.0} ops/sec)\n", .{ name, bench_result.stats.throughputMBps(bytes_per_op) / 1024.0, bench_result.stats.opsPerSecond() });
        }

        // Vector Add - SIMD
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "vecadd_simd8_{d:.1}{s}", .{ size_info.value, size_info.unit }) catch "vecadd_simd";
            const bench_result = try runner.run(
                .{ .name = name, .category = "gpu/vector", .bytes_per_op = bytes_per_op, .warmup_iterations = config.warmup_iterations, .min_time_ns = config.min_time_ns },
                struct {
                    fn bench(va: []const f32, vb: []const f32, vr: []f32) void {
                        cpu.simdVectorAdd(8, va, vb, vr);
                    }
                }.bench,
                .{ a, b, result },
            );
            std.debug.print("  {s}: {d:.2} GB/s ({d:.0} ops/sec)\n", .{ name, bench_result.stats.throughputMBps(bytes_per_op) / 1024.0, bench_result.stats.opsPerSecond() });
        }

        // Vector Mul - SIMD
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "vecmul_simd8_{d:.1}{s}", .{ size_info.value, size_info.unit }) catch "vecmul_simd";
            const bench_result = try runner.run(
                .{ .name = name, .category = "gpu/vector", .bytes_per_op = bytes_per_op, .warmup_iterations = config.warmup_iterations, .min_time_ns = config.min_time_ns },
                struct {
                    fn bench(va: []const f32, vb: []const f32, vr: []f32) void {
                        cpu.simdVectorMul(8, va, vb, vr);
                    }
                }.bench,
                .{ a, b, result },
            );
            std.debug.print("  {s}: {d:.2} GB/s ({d:.0} ops/sec)\n", .{ name, bench_result.stats.throughputMBps(bytes_per_op) / 1024.0, bench_result.stats.opsPerSecond() });
        }

        // Dot Product - SIMD
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "dot_simd8_{d:.1}{s}", .{ size_info.value, size_info.unit }) catch "dot_simd";
            const bench_result = try runner.run(
                .{ .name = name, .category = "gpu/vector", .bytes_per_op = size * @sizeOf(f32) * 2, .warmup_iterations = config.warmup_iterations, .min_time_ns = config.min_time_ns },
                struct {
                    fn bench(va: []const f32, vb: []const f32) f32 {
                        return cpu.simdDotProduct(8, va, vb);
                    }
                }.bench,
                .{ a, b },
            );
            std.debug.print("  {s}: {d:.2} GB/s ({d:.0} ops/sec)\n", .{ name, bench_result.stats.throughputMBps(size * @sizeOf(f32) * 2) / 1024.0, bench_result.stats.opsPerSecond() });
        }

        // Normalize
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "normalize_simd8_{d:.1}{s}", .{ size_info.value, size_info.unit }) catch "normalize_simd";
            const bench_result = try runner.run(
                .{ .name = name, .category = "gpu/vector", .bytes_per_op = size * @sizeOf(f32) * 2, .warmup_iterations = config.warmup_iterations, .min_time_ns = config.min_time_ns },
                struct {
                    fn bench(va: []const f32, vr: []f32) void {
                        cpu.simdNormalize(8, va, vr);
                    }
                }.bench,
                .{ a, result },
            );
            std.debug.print("  {s}: {d:.2} GB/s ({d:.0} ops/sec)\n", .{ name, bench_result.stats.throughputMBps(size * @sizeOf(f32) * 2) / 1024.0, bench_result.stats.opsPerSecond() });
        }
    }
}
