//! GPU Domain Benchmarks Module
//!
//! Comprehensive GPU performance testing including:
//! - Kernel operations (matmul, vector ops, reductions)
//! - Memory operations (copy, transpose, bandwidth)
//! - Backend comparisons (CUDA, Vulkan, Metal, etc.)
//! - GPU vs CPU performance comparisons

const std = @import("std");
const build_options = @import("build_options");
const framework = @import("../../system/framework.zig");

// GPU benchmark suites
pub const kernels = @import("kernels.zig");
pub const backends = @import("backends.zig");
pub const gpu_vs_cpu = @import("gpu_vs_cpu.zig");

/// GPU benchmark configuration presets
pub const GpuBenchmarkMode = enum {
    /// Quick benchmarks for CI (fewer iterations, smaller sizes)
    quick,
    /// Standard benchmarks (balanced)
    standard,
    /// Thorough benchmarks (comprehensive, more iterations)
    thorough,
};

/// Configuration for GPU benchmarks
pub const GpuBenchConfig = struct {
    /// Matrix sizes for matmul benchmarks [rows/cols]
    matrix_sizes: []const usize = &.{ 128, 256, 512, 1024, 2048, 4096 },
    /// Vector sizes for vector operation benchmarks
    vector_sizes: []const usize = &.{ 10_000, 100_000, 1_000_000, 10_000_000 },
    /// Memory sizes for bandwidth tests (in bytes)
    memory_sizes: []const usize = &.{
        32 * 1024, // 32KB - L1 cache
        256 * 1024, // 256KB - L2 cache
        4 * 1024 * 1024, // 4MB - L3 cache
        32 * 1024 * 1024, // 32MB - Main memory
        256 * 1024 * 1024, // 256MB - Large transfer
    },
    /// Number of warmup iterations
    warmup_iterations: u32 = 10,
    /// Number of benchmark iterations
    benchmark_iterations: u32 = 100,
    /// Minimum benchmark time in nanoseconds
    min_time_ns: u64 = 500_000_000,
    /// Maximum iterations regardless of time
    max_iterations: u64 = 10000,
    /// Whether to verify results for correctness
    verify_results: bool = true,
    /// Include GPU vs CPU comparison benchmarks
    include_cpu_comparison: bool = true,
    /// Include backend comparison benchmarks
    include_backend_comparison: bool = true,

    pub fn forMode(mode: GpuBenchmarkMode) GpuBenchConfig {
        return switch (mode) {
            .quick => .{
                .matrix_sizes = &.{ 128, 256, 512 },
                .vector_sizes = &.{ 10_000, 100_000 },
                .memory_sizes = &.{ 32 * 1024, 256 * 1024, 4 * 1024 * 1024 },
                .warmup_iterations = 5,
                .benchmark_iterations = 50,
                .min_time_ns = 100_000_000,
                .max_iterations = 1000,
                .verify_results = false,
                .include_cpu_comparison = false,
                .include_backend_comparison = false,
            },
            .standard => .{},
            .thorough => .{
                .matrix_sizes = &.{ 64, 128, 256, 512, 1024, 2048, 4096, 8192 },
                .vector_sizes = &.{ 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000 },
                .memory_sizes = &.{
                    32 * 1024,
                    64 * 1024,
                    128 * 1024,
                    256 * 1024,
                    512 * 1024,
                    1 * 1024 * 1024,
                    4 * 1024 * 1024,
                    16 * 1024 * 1024,
                    32 * 1024 * 1024,
                    64 * 1024 * 1024,
                    128 * 1024 * 1024,
                    256 * 1024 * 1024,
                },
                .warmup_iterations = 20,
                .benchmark_iterations = 200,
                .min_time_ns = 1_000_000_000,
                .max_iterations = 50000,
                .verify_results = true,
                .include_cpu_comparison = true,
                .include_backend_comparison = true,
            },
        };
    }
};

/// Detect if a real (non-emulated) GPU device is available.
pub fn hasHardwareGpu(allocator: std.mem.Allocator) bool {
    if (!build_options.enable_gpu) return false;

    const abi = @import("abi");
    var gpu_ctx = abi.gpu.Gpu.init(allocator, .{}) catch return false;
    defer gpu_ctx.deinit();

    if (!gpu_ctx.isAvailable()) return false;

    for (gpu_ctx.listDevices()) |device| {
        if (!device.is_emulated and
            device.device_type != .cpu and
            device.supportsFeature(.compute_shaders))
        {
            return true;
        }
    }

    return false;
}

/// Run all GPU benchmarks
pub fn runAllBenchmarks(allocator: std.mem.Allocator, mode: GpuBenchmarkMode) !void {
    const config = GpuBenchConfig.forMode(mode);

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("                         GPU KERNEL BENCHMARKS\n", .{});
    std.debug.print("================================================================================\n", .{});

    if (!build_options.enable_gpu) {
        std.debug.print("\n[GPU DISABLED] GPU feature is not enabled at compile time.\n", .{});
        std.debug.print("  Rebuild with: zig build -Denable-gpu=true\n", .{});
        std.debug.print("  Running CPU-only fallback benchmarks...\n\n", .{});
    }

    // Run kernel benchmarks (matmul, vector ops, reductions)
    try kernels.runKernelBenchmarks(allocator, config);

    // Run backend comparison if enabled
    if (config.include_backend_comparison) {
        try backends.runBackendBenchmarks(allocator, config);
    }

    // Run GPU vs CPU comparison if enabled
    if (config.include_cpu_comparison) {
        try gpu_vs_cpu.runComparisonBenchmarks(allocator, config);
    }

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("                         GPU BENCHMARKS COMPLETE\n", .{});
    std.debug.print("================================================================================\n", .{});
}

/// Quick benchmark run (for CI)
pub fn runQuick(allocator: std.mem.Allocator) !void {
    try runAllBenchmarks(allocator, .quick);
}

/// Standard benchmark run
pub fn run(allocator: std.mem.Allocator) !void {
    try runAllBenchmarks(allocator, .standard);
}

// ============================================================================
// Tests
// ============================================================================

test "gpu benchmark config modes" {
    const quick = GpuBenchConfig.forMode(.quick);
    const standard = GpuBenchConfig.forMode(.standard);
    const thorough = GpuBenchConfig.forMode(.thorough);

    try std.testing.expect(quick.matrix_sizes.len <= standard.matrix_sizes.len);
    try std.testing.expect(standard.matrix_sizes.len <= thorough.matrix_sizes.len);
    try std.testing.expect(quick.warmup_iterations <= standard.warmup_iterations);
    try std.testing.expect(standard.warmup_iterations <= thorough.warmup_iterations);
}

test "gpu benchmark module imports" {
    _ = kernels;
    _ = backends;
    _ = gpu_vs_cpu;
}
