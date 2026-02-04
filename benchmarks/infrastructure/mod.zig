//! Infrastructure Benchmarks Module
//!
//! Infrastructure-level performance testing for:
//! - Concurrency and parallelism
//! - Cryptographic operations
//! - Memory management
//! - SIMD/vectorization
//! - System operations

const std = @import("std");

// Infrastructure benchmark suites
pub const suites = struct {
    pub const concurrency = @import("concurrency.zig");
    pub const crypto = @import("crypto.zig");
    pub const memory = @import("memory.zig");
    pub const simd = @import("simd.zig");
    pub const network = @import("network.zig");
    pub const result_cache = @import("result_cache.zig");
    pub const registry = @import("registry.zig");
};

pub fn runAll(allocator: std.mem.Allocator) !void {
    std.debug.print("Running Infrastructure Benchmarks\n", .{});
    std.debug.print("==============================\n\n", .{});

    try suites.concurrency.run(allocator);
    try suites.crypto.run(allocator);
    try suites.memory.run(allocator);
    try suites.simd.run(allocator);
    try suites.network.run(allocator);
    try suites.result_cache.runAllBenchmarks(allocator, suites.result_cache.ResultCacheBenchConfig.quick);
    try suites.registry.runAllBenchmarks(allocator, suites.registry.RegistryBenchConfig.quick);

    std.debug.print("\nInfrastructure Benchmarks Complete\n", .{});
    std.debug.print("==============================\n", .{});
}

pub const InfrastructureBenchmarkConfig = struct {
    iterations: u64 = 10000,
    warmup_iterations: u64 = 1000,
    timeout_seconds: u64 = 30,

    pub const default = InfrastructureBenchmarkConfig{};
    pub const quick = InfrastructureBenchmarkConfig{ .iterations = 1000, .warmup_iterations = 100 };
    pub const thorough = InfrastructureBenchmarkConfig{ .iterations = 100000, .warmup_iterations = 5000 };
};

// Legacy compatibility exports
pub const BenchmarkSuite = struct {
    allocator: std.mem.Allocator,
    results: std.ArrayListUnmanaged(struct {
        name: []const u8,
        iterations: u64,
        duration_ns: u64,
        ops_per_sec: f64,
        error_count: u64,
    }),

    pub fn init(allocator: std.mem.Allocator) BenchmarkSuite {
        return .{
            .allocator = allocator,
            .results = .{},
        };
    }

    pub fn deinit(self: *BenchmarkSuite) void {
        for (self.results.items) |result| {
            self.allocator.free(result.name);
        }
        self.results.deinit(self.allocator);
    }
};

test {
    // Basic compilation test
    _ = suites.concurrency;
    _ = suites.crypto;
    _ = suites.memory;
    _ = suites.simd;
    _ = suites.network;
    _ = suites.result_cache;
    _ = suites.registry;
}
