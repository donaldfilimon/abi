//! Reduction and memory operation benchmarks — sum, max, min, argmax, memcpy, transpose.

const std = @import("std");
const framework = @import("../../../system/framework.zig");
const parent_mod = @import("../mod.zig");
const mod = @import("mod.zig");
const cpu = mod.cpu_baselines;

const GpuBenchConfig = parent_mod.GpuBenchConfig;

/// Run reduction operation benchmarks
pub fn benchmarkReductions(
    allocator: std.mem.Allocator,
    runner: *framework.BenchmarkRunner,
    config: GpuBenchConfig,
) !void {
    std.debug.print("\n[Reduction Operation Benchmarks]\n", .{});

    for (config.vector_sizes) |size| {
        const v = try allocator.alignedAlloc(f32, .@"64", size);
        defer allocator.free(v);
        mod.initRandomData(v, 42);

        const size_info = mod.formatSize(size * @sizeOf(f32));
        const bytes_per_op = size * @sizeOf(f32);

        // Reduce Sum
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "reduce_sum_simd8_{d:.1}{s}", .{ size_info.value, size_info.unit }) catch "reduce_sum";
            const bench_result = try runner.run(
                .{ .name = name, .category = "gpu/reduction", .bytes_per_op = bytes_per_op, .warmup_iterations = config.warmup_iterations, .min_time_ns = config.min_time_ns },
                struct {
                    fn bench(data: []const f32) f32 {
                        return cpu.simdReduceSum(8, data);
                    }
                }.bench,
                .{v},
            );
            std.debug.print("  {s}: {d:.2} GB/s ({d:.0} ops/sec)\n", .{ name, bench_result.stats.throughputMBps(bytes_per_op) / 1024.0, bench_result.stats.opsPerSecond() });
        }

        // Reduce Max
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "reduce_max_simd8_{d:.1}{s}", .{ size_info.value, size_info.unit }) catch "reduce_max";
            const bench_result = try runner.run(
                .{ .name = name, .category = "gpu/reduction", .bytes_per_op = bytes_per_op, .warmup_iterations = config.warmup_iterations, .min_time_ns = config.min_time_ns },
                struct {
                    fn bench(data: []const f32) f32 {
                        return cpu.simdReduceMax(8, data);
                    }
                }.bench,
                .{v},
            );
            std.debug.print("  {s}: {d:.2} GB/s ({d:.0} ops/sec)\n", .{ name, bench_result.stats.throughputMBps(bytes_per_op) / 1024.0, bench_result.stats.opsPerSecond() });
        }

        // Reduce Min
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "reduce_min_simd8_{d:.1}{s}", .{ size_info.value, size_info.unit }) catch "reduce_min";
            const bench_result = try runner.run(
                .{ .name = name, .category = "gpu/reduction", .bytes_per_op = bytes_per_op, .warmup_iterations = config.warmup_iterations, .min_time_ns = config.min_time_ns },
                struct {
                    fn bench(data: []const f32) f32 {
                        return cpu.simdReduceMin(8, data);
                    }
                }.bench,
                .{v},
            );
            std.debug.print("  {s}: {d:.2} GB/s ({d:.0} ops/sec)\n", .{ name, bench_result.stats.throughputMBps(bytes_per_op) / 1024.0, bench_result.stats.opsPerSecond() });
        }

        // Argmax
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "argmax_scalar_{d:.1}{s}", .{ size_info.value, size_info.unit }) catch "argmax";
            const bench_result = try runner.run(
                .{ .name = name, .category = "gpu/reduction", .bytes_per_op = bytes_per_op, .warmup_iterations = config.warmup_iterations, .min_time_ns = config.min_time_ns },
                struct {
                    fn bench(data: []const f32) usize {
                        return cpu.scalarArgmax(data);
                    }
                }.bench,
                .{v},
            );
            std.debug.print("  {s}: {d:.2} GB/s ({d:.0} ops/sec)\n", .{ name, bench_result.stats.throughputMBps(bytes_per_op) / 1024.0, bench_result.stats.opsPerSecond() });
        }
    }
}

/// Run memory operation benchmarks
pub fn benchmarkMemoryOps(
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
        @memset(src, 0xAA);

        const size_info = mod.formatSize(size);

        // Memory Copy
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "memcpy_{d:.1}{s}", .{ size_info.value, size_info.unit }) catch "memcpy";
            const bench_result = try runner.run(
                .{ .name = name, .category = "gpu/memory", .bytes_per_op = size * 2, .warmup_iterations = config.warmup_iterations, .min_time_ns = config.min_time_ns },
                struct {
                    fn bench(d: []u8, s: []const u8) void {
                        @memcpy(d, s);
                    }
                }.bench,
                .{ dst, src },
            );
            std.debug.print("  {s}: {d:.2} GB/s\n", .{ name, bench_result.stats.throughputMBps(size * 2) / 1024.0 });
        }

        // Memory Set
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "memset_{d:.1}{s}", .{ size_info.value, size_info.unit }) catch "memset";
            const bench_result = try runner.run(
                .{ .name = name, .category = "gpu/memory", .bytes_per_op = size, .warmup_iterations = config.warmup_iterations, .min_time_ns = config.min_time_ns },
                struct {
                    fn bench(d: []u8) void {
                        @memset(d, 0x55);
                    }
                }.bench,
                .{dst},
            );
            std.debug.print("  {s}: {d:.2} GB/s\n", .{ name, bench_result.stats.throughputMBps(size) / 1024.0 });
        }

        // Memory Read (sequential)
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "memread_{d:.1}{s}", .{ size_info.value, size_info.unit }) catch "memread";
            const bench_result = try runner.run(
                .{ .name = name, .category = "gpu/memory", .bytes_per_op = size, .warmup_iterations = config.warmup_iterations, .min_time_ns = config.min_time_ns },
                struct {
                    fn bench(s: []const u8) u64 {
                        var sum: u64 = 0;
                        for (s) |byte| sum +%= byte;
                        return sum;
                    }
                }.bench,
                .{src},
            );
            std.debug.print("  {s}: {d:.2} GB/s\n", .{ name, bench_result.stats.throughputMBps(size) / 1024.0 });
        }
    }

    // Matrix transpose benchmarks
    std.debug.print("\n[Matrix Transpose Benchmarks]\n", .{});
    for (config.matrix_sizes) |size| {
        if (size > 2048) continue;

        const matrix_size = size * size;
        const src = try allocator.alignedAlloc(f32, .@"64", matrix_size);
        defer allocator.free(src);
        const dst = try allocator.alignedAlloc(f32, .@"64", matrix_size);
        defer allocator.free(dst);
        mod.initRandomData(src, 42);

        const bytes_per_op = matrix_size * @sizeOf(f32) * 2;

        // Naive transpose
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "transpose_naive_{d}x{d}", .{ size, size }) catch "transpose_naive";
            const bench_result = try runner.run(
                .{ .name = name, .category = "gpu/memory", .bytes_per_op = bytes_per_op, .warmup_iterations = config.warmup_iterations, .min_time_ns = config.min_time_ns },
                struct {
                    fn bench(s: []const f32, d: []f32, n: usize) void {
                        cpu.matrixTranspose(s, d, n, n);
                    }
                }.bench,
                .{ src, dst, size },
            );
            std.debug.print("  {s}: {d:.2} GB/s ({d:.2} ms/op)\n", .{ name, bench_result.stats.throughputMBps(bytes_per_op) / 1024.0, bench_result.stats.mean_ns / 1e6 });
        }

        // Blocked transpose
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "transpose_blocked_{d}x{d}", .{ size, size }) catch "transpose_blocked";
            const bench_result = try runner.run(
                .{ .name = name, .category = "gpu/memory", .bytes_per_op = bytes_per_op, .warmup_iterations = config.warmup_iterations, .min_time_ns = config.min_time_ns },
                struct {
                    fn bench(s: []const f32, d: []f32, n: usize) void {
                        cpu.blockedTranspose(s, d, n, n, 32);
                    }
                }.bench,
                .{ src, dst, size },
            );
            std.debug.print("  {s}: {d:.2} GB/s ({d:.2} ms/op)\n", .{ name, bench_result.stats.throughputMBps(bytes_per_op) / 1024.0, bench_result.stats.mean_ns / 1e6 });
        }
    }
}
