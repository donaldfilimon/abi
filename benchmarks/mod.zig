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
    // GPU and network domains not yet implemented
    pub const gpu = struct {
        pub const backends = struct {
            pub const mod = struct {};
        };
        pub const kernels = struct {};
        pub const memory = struct {};
    };
    pub const network = struct {};
};

// Infrastructure benchmarks (concurrency, crypto, memory, etc.)
pub const infrastructure = struct {
    pub const concurrency = @import("infrastructure/concurrency.zig");
    pub const crypto = @import("infrastructure/crypto.zig");
    pub const memory = @import("infrastructure/memory.zig");
    pub const simd = @import("infrastructure/simd.zig");
    pub const network = @import("infrastructure/network.zig");

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

// System benchmarks (framework, CI, industry standards)
pub const system = struct {
    pub const framework = @import("system/framework.zig");
    pub const ci_integration = @import("system/ci_integration.zig");
    pub const industry_standard = @import("system/industry_standard.zig");
};

// Core utilities (config, vectors, distance, runner)
pub const core = @import("core/mod.zig");

// Benchmark utilities (reporter, runner, statistics)
pub const utilities = struct {
    pub const reporter = struct {};
    pub const runner = struct {};
    pub const statistics = struct {};
};

// Re-export commonly used types
pub const BenchmarkConfig = struct {
    iterations: u64 = 10000,
    warmup: u64 = 1000,
    timeout_seconds: u64 = 30,

    pub const quick = BenchmarkConfig{ .iterations = 1000, .warmup = 100 };
    pub const thorough = BenchmarkConfig{ .iterations = 100000, .warmup = 5000 };
};

pub fn runAllBenchmarks(allocator: std.mem.Allocator) !void {
    std.debug.print("Running Complete ABI Benchmark Suite\n", .{});
    std.debug.print("===================================\n\n", .{});

    // Run domain benchmarks
    std.debug.print("Domain Benchmarks:\n", .{});
    try domains.ai.runAllBenchmarks(allocator, .standard);
    try domains.database.runAllBenchmarks(allocator, .standard);

    // Run infrastructure benchmarks
    std.debug.print("\nInfrastructure Benchmarks:\n", .{});
    try infrastructure.runner.executeAll(allocator);

    // Run system benchmarks
    std.debug.print("\nSystem Benchmarks:\n", .{});
    try system.framework.run(allocator);

    std.debug.print("\nBenchmark Suite Complete\n", .{});
    std.debug.print("===================================\n", .{});
}

// Legacy compatibility exports
pub const framework = system.framework;
pub const DatabaseBenchConfig = core.config.DatabaseBenchConfig;
pub const AIBenchConfig = core.config.AIBenchConfig;
pub const LLMBenchConfig = core.config.LLMBenchConfig;
pub const AnnDataset = core.config.AnnDataset;
pub const VectorDistribution = core.vectors.VectorDistribution;
pub const generateVector = core.vectors.generateVector;
pub const freeVector = core.vectors.freeVector;

test "benchmark modules compile" {
    _ = domains;
    _ = infrastructure;
    _ = competitive;
    _ = system;
}
