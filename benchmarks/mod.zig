//! ABI Benchmark Suite Root Module
//!
//! Comprehensive benchmarking framework organized by category:
//! - domain/      : Feature-specific benchmarks (AI, database, GPU, network)
//! - infrastructure/: Infrastructure-level performance (memory, crypto, concurrency)
//! - competitive/ : Industry competition comparisons
//! - system/      : System/integration testing

const std = @import("std");

// Re-export all benchmark categories by domain
pub const domains = struct {
    pub const ai = @import("domain/ai/mod.zig");
    pub const database = @import("domain/database/mod.zig");
    pub const gpu = @import("domain/gpu/mod.zig");
    // Network domain not yet implemented
    pub const network = struct {};
};

// Infrastructure benchmarks (concurrency, crypto, memory, etc.)
pub const infrastructure = struct {
    pub const concurrency = @import("infrastructure/concurrency.zig");
    pub const crypto = @import("infrastructure/crypto.zig");
    pub const memory = @import("infrastructure/memory.zig");
    pub const simd = @import("infrastructure/simd.zig");
    pub const network = @import("infrastructure/network.zig");
    pub const v2_modules = @import("infrastructure/v2_modules.zig");

    // Utility functions
    pub const runner = struct {
        pub fn executeAll(allocator: std.mem.Allocator) !void {
            try concurrency.run(allocator);
            try crypto.run(allocator);
            try memory.run(allocator);
            try simd.run(allocator);
            try network.run(allocator);
        }
    };
};

// Competitive benchmarks (FAISS, LLM, vector DB comparisons)
pub const competitive = @import("competitive/mod.zig");

// System benchmarks (framework, CI, industry standards, baseline persistence)
pub const system = @import("system/mod.zig");

// Core utilities (config, vectors, distance, runner)
pub const core = @import("core/mod.zig");

// Benchmark utilities (reporter, runner, statistics)
pub const utilities = struct {
    pub const reporter = struct {};
    pub const runner = struct {};
    pub const statistics = struct {};
};

pub fn runAllBenchmarks(allocator: std.mem.Allocator) !void {
    std.debug.print("Running Complete ABI Benchmark Suite\n", .{});
    std.debug.print("===================================\n\n", .{});

    // Run domain benchmarks
    std.debug.print("Domain Benchmarks:\n", .{});
    try domains.ai.runAllBenchmarks(allocator, .standard);
    try domains.database.runAllBenchmarks(allocator, .standard);
    try domains.gpu.runAllBenchmarks(allocator, .standard);

    // Run infrastructure benchmarks
    std.debug.print("\nInfrastructure Benchmarks:\n", .{});
    try infrastructure.runner.executeAll(allocator);

    // Run system benchmarks
    std.debug.print("\nSystem Benchmarks:\n", .{});
    try system.framework.run(allocator);

    std.debug.print("\nBenchmark Suite Complete\n", .{});
    std.debug.print("===================================\n", .{});
}

// ============================================================================
// BenchmarkSuite — Convenience wrapper for quick benchmark scripts
// ============================================================================

/// Simple benchmark suite for running ad-hoc benchmark functions.
/// Each benchmark function takes `(std.mem.Allocator) !void`.
/// Wraps BenchmarkRunner with a simpler interface.
pub const BenchmarkSuite = struct {
    allocator: std.mem.Allocator,
    runner: system.BenchmarkRunner,

    pub fn init(allocator: std.mem.Allocator) BenchmarkSuite {
        return .{
            .allocator = allocator,
            .runner = system.BenchmarkRunner.init(allocator),
        };
    }

    pub fn deinit(self: *BenchmarkSuite) void {
        self.runner.deinit();
    }

    /// Run a benchmark function that takes (allocator) and returns !void.
    pub fn runBenchmark(
        self: *BenchmarkSuite,
        name: []const u8,
        comptime bench_fn: fn (std.mem.Allocator) anyerror!void,
        args: anytype,
    ) !void {
        _ = args; // Allocator passed via runWithAllocator
        _ = self.runner.runWithAllocator(.{
            .name = name,
            .category = "suite",
            .warmup_iterations = 3,
            .min_iterations = 5,
            .min_time_ns = 50_000_000, // 50ms
        }, bench_fn, .{}) catch |err| {
            std.debug.print("  {s}: FAILED ({t})\n", .{ name, err });
            return;
        };
        std.debug.print("  {s}: OK\n", .{name});
    }

    /// Print a formatted summary of results.
    pub fn printSummary(self: *BenchmarkSuite) void {
        self.runner.printSummaryDebug();
    }
};

// Legacy compatibility exports
pub const framework = system.framework;
pub const DatabaseBenchConfig = core.config.DatabaseBenchConfig;
pub const AIBenchConfig = core.config.AIBenchConfig;
pub const LLMBenchConfig = core.config.LLMBenchConfig;
pub const AnnDataset = core.config.AnnDataset;
pub const VectorDistribution = core.vectors.VectorDistribution;
pub const generateVector = core.vectors.generateVector;
pub const freeVector = core.vectors.freeVector;

// Baseline persistence exports
pub const BaselineStore = system.BaselineStore;
pub const BenchmarkResult = system.BenchmarkResult;
pub const ComparisonResult = system.ComparisonResult;
pub const RegressionReport = system.RegressionReport;
pub const ComparisonConfig = system.ComparisonConfig;
pub const compareAll = system.compareAll;
pub const compareAllWithConfig = system.compareAllWithConfig;

test "benchmark modules compile" {
    _ = domains;
    _ = infrastructure;
    _ = competitive;
    _ = system;
}
