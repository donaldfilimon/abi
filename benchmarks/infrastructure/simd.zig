//! SIMD/Vector Operation Benchmarks
//!
//! Measures performance of the framework's canonical SIMD implementations
//! vs scalar fallbacks and naive loops.

const std = @import("std");
const abi = @import("abi");
const framework = @import("../system/framework.zig");
const simd_mod = abi.wdbx.neural; // Unified neural engine exposes SIMD

/// SIMD benchmark configuration
pub const SIMDBenchConfig = struct {
    /// Vector dimensions to benchmark
    dimensions: []const usize = &.{ 64, 128, 256, 384, 512, 768, 1024, 1536, 4096 },
    /// Whether to benchmark scalar fallbacks
    include_scalar_comparison: bool = true,
};

/// Scalar dot product (naive loop for baseline)
fn scalarDotProduct(a: []const f32, b: []const f32) f32 {
    var sum: f32 = 0.0;
    for (a, b) |x, y| {
        sum += x * y;
    }
    return sum;
}

pub fn runSIMDBenchmarks(allocator: std.mem.Allocator, config: SIMDBenchConfig) !void {
    var runner = framework.BenchmarkRunner.init(allocator);
    defer runner.deinit();

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("                    SIMD/VECTOR OPERATION BENCHMARKS\n", .{});
    std.debug.print("                    Detected Width: {d} floats\n", .{simd_mod.SIMDFeatures.vector_width});
    std.debug.print("================================================================================\n\n", .{});

    try benchmarkDotProduct(allocator, &runner, config);
    try benchmarkNormalization(allocator, &runner, config);
    try benchmarkQuantization(allocator, &runner, config);

    runner.printSummaryDebug();
}

fn benchmarkDotProduct(
    allocator: std.mem.Allocator,
    runner: *framework.BenchmarkRunner,
    config: SIMDBenchConfig,
) !void {
    std.debug.print("[Dot Product: Canonical SIMD vs Naive Scalar]\n", .{});

    for (config.dimensions) |dim| {
        const a = try allocator.alignedAlloc(f32, 64, dim);
        defer allocator.free(a);
        const b = try allocator.alignedAlloc(f32, 64, dim);
        defer allocator.free(b);

        // Fill with dummy data
        for (a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 10)) / 10.0;
        for (b, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i + 1) % 10)) / 10.0;

        // 1. Scalar Baseline
        if (config.include_scalar_comparison) {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "dot_scalar_{d}", .{dim}) catch "dot_scalar";

            _ = try runner.run(
                .{
                    .name = name,
                    .category = "simd/dot_product",
                    .bytes_per_op = dim * @sizeOf(f32) * 2,
                    .warmup_iterations = 100,
                },
                scalarDotProduct,
                .{ a, b },
            );
        }

        // 2. Canonical Framework SIMD
        {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "dot_framework_simd_{d}", .{dim}) catch "dot_simd";

            _ = try runner.run(
                .{
                    .name = name,
                    .category = "simd/dot_product",
                    .bytes_per_op = dim * @sizeOf(f32) * 2,
                    .warmup_iterations = 100,
                },
                simd_mod.dotProduct,
                .{ f32, a, b },
            );
        }
    }
}

fn benchmarkNormalization(
    allocator: std.mem.Allocator,
    runner: *framework.BenchmarkRunner,
    config: SIMDBenchConfig,
) !void {
    std.debug.print("\n[Vector Normalization]\n", .{});

    for (config.dimensions) |dim| {
        const a = try allocator.alignedAlloc(f32, 64, dim);
        defer allocator.free(a);
        for (a, 0..) |*v, i| v.* = @floatFromInt(i + 1);

        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "normalize_simd_{d}", .{dim}) catch "normalize";

        _ = try runner.run(
            .{
                .name = name,
                .category = "simd/normalize",
                .bytes_per_op = dim * @sizeOf(f32),
                .warmup_iterations = 100,
            },
            simd_mod.normalize,
            .{ f32, a },
        );
    }
}

fn benchmarkQuantization(
    allocator: std.mem.Allocator,
    runner: *framework.BenchmarkRunner,
    config: SIMDBenchConfig,
) !void {
    std.debug.print("\n[Vector Quantization]\n", .{});

    for (config.dimensions) |dim| {
        const src = try allocator.alignedAlloc(f32, 64, dim);
        defer allocator.free(src);
        const dst = try allocator.alloc(i8, dim);
        defer allocator.free(dst);
        
        for (src, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)));

        var scale: f32 = undefined;
        var zp: i8 = undefined;

        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "quantize_int8_{d}", .{dim}) catch "quantize";

        _ = try runner.run(
            .{
                .name = name,
                .category = "simd/quantize",
                .bytes_per_op = dim * @sizeOf(f32),
                .warmup_iterations = 50,
            },
            simd_mod.Quantize.toInt8,
            .{ src, dst, &scale, &zp },
        );
    }
}
