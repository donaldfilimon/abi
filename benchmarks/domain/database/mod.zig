//! Database Benchmark Module
//!
//! Consolidated benchmarks for database and vector search operations:
//!
//! - **hnsw**: HNSW index construction and search
//! - **operations**: Insert, query, update, delete operations
//! - **ann_benchmarks**: ANN-Benchmarks compatible suite
//!
//! ## Usage
//!
//! ```zig
//! const db_bench = @import("database/mod.zig");
//!
//! // Run all database benchmarks
//! try db_bench.runAllBenchmarks(allocator, .standard);
//!
//! // Run specific benchmark suite
//! try db_bench.operations.runOperationsBenchmarks(allocator, config);
//! ```

const std = @import("std");
const core = @import("../../core/mod.zig");
const framework = @import("../../system/framework.zig");

pub const hnsw = @import("hnsw.zig");
pub const operations = @import("operations.zig");
pub const ann_benchmarks = @import("ann_benchmarks.zig");

// Re-export common types
pub const SearchResult = hnsw.SearchResult;
pub const EuclideanHNSW = hnsw.EuclideanHNSW;
pub const CosineHNSW = hnsw.CosineHNSW;
pub const AnnBenchmarkResult = ann_benchmarks.AnnBenchmarkResult;
pub const AnnBenchConfig = ann_benchmarks.AnnBenchConfig;

/// Configuration preset
pub const ConfigPreset = enum {
    quick,
    standard,
    comprehensive,
    ann,
};

/// Run all database benchmarks with the given preset
pub fn runAllBenchmarks(allocator: std.mem.Allocator, preset: ConfigPreset) !void {
    const config = switch (preset) {
        .quick => core.config.DatabaseBenchConfig.quick,
        .standard => core.config.DatabaseBenchConfig.standard,
        .comprehensive => core.config.DatabaseBenchConfig.comprehensive,
        .ann => core.config.DatabaseBenchConfig.ann_benchmarks,
    };

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("                    DATABASE/HNSW VECTOR SEARCH BENCHMARKS\n", .{});
    std.debug.print("================================================================================\n", .{});

    // Run operation benchmarks
    try operations.runOperationsBenchmarks(allocator, config);

    // Run HNSW-specific benchmarks
    try hnsw.runHnswBenchmarks(allocator, config);

    // Run ANN-Benchmarks if comprehensive or ann preset
    if (preset == .comprehensive or preset == .ann) {
        const ann_results = try ann_benchmarks.runAnnBenchmarks(allocator, .{
            .dataset = .custom,
            .custom_size = 5000,
            .custom_dimension = 128,
            .num_queries = 100,
            .k_values = &.{ 1, 10 },
            .hnsw_m_values = &.{16},
            .ef_construction_values = &.{200},
            .ef_search_values = &.{ 50, 100 },
        });
        defer allocator.free(ann_results);

        ann_benchmarks.printResults(ann_results);
    }

    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("                    DATABASE BENCHMARKS COMPLETE\n", .{});
    std.debug.print("================================================================================\n", .{});
}

/// Run database benchmarks with custom configuration
pub fn runDatabaseBenchmarks(allocator: std.mem.Allocator, config: core.config.DatabaseBenchConfig) !void {
    try operations.runOperationsBenchmarks(allocator, config);
    try hnsw.runHnswBenchmarks(allocator, config);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    try runAllBenchmarks(allocator, .standard);
}

test {
    _ = hnsw;
    _ = operations;
    _ = ann_benchmarks;
}
