//! GPU Kernel Benchmarks
//!
//! Comprehensive benchmarks for GPU kernel operations:
//! - Matrix operations (matmul, transpose)
//! - Vector operations (add, mul, dot, normalize)
//! - Reduction operations (sum, max, min, argmax)
//! - Memory operations (copy, fill, bandwidth)
//!
//! Split into sub-modules for navigability:
//! - `cpu_baselines`: Scalar/SIMD reference implementations
//! - `matmul`: Matrix multiplication benchmarks
//! - `vector_ops`: Vector operation benchmarks
//! - `memory_ops`: Reduction and memory bandwidth benchmarks

const std = @import("std");
const build_options = @import("build_options");
const framework = @import("../../../system/framework.zig");
const parent_mod = @import("../mod.zig");

pub const cpu_baselines = @import("cpu_baselines.zig");
pub const matmul = @import("matmul.zig");
pub const vector_ops = @import("vector_ops.zig");
pub const memory_ops = @import("memory_ops.zig");

const GpuBenchConfig = parent_mod.GpuBenchConfig;

// Conditional GPU imports based on build options
pub const gpu = if (build_options.enable_gpu) @import("abi").features.gpu else struct {
    pub const Gpu = void;
    pub const GpuConfig = struct {};
    pub const moduleEnabled = struct {
        pub fn call() bool {
            return false;
        }
    }.call;
};

// ============================================================================
// Shared Utilities
// ============================================================================

/// Initialize data with deterministic pseudo-random values
pub fn initRandomData(data: []f32, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    for (data) |*x| {
        x.* = prng.random().float(f32) * 2.0 - 1.0;
    }
}

/// Format size as human-readable string
pub fn formatSize(size: usize) struct { value: f64, unit: []const u8 } {
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

/// Run all kernel benchmarks
pub fn runKernelBenchmarks(allocator: std.mem.Allocator, config: GpuBenchConfig) !void {
    var runner = framework.BenchmarkRunner.init(allocator);
    defer runner.deinit();

    try matmul.benchmarkMatmul(allocator, &runner, config);
    try vector_ops.benchmarkVectorOps(allocator, &runner, config);
    try memory_ops.benchmarkReductions(allocator, &runner, config);
    try memory_ops.benchmarkMemoryOps(allocator, &runner, config);

    runner.printSummaryDebug();
}

test {
    _ = cpu_baselines;
    _ = matmul;
    _ = vector_ops;
    _ = memory_ops;
}
