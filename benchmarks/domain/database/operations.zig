//! Database Operations Benchmarks
//!
//! Measures performance of the production WDBX Engine.
//! Benchmarks: Insertion throughput, Query latency, and Batch operations.

const std = @import("std");
const abi = @import("abi");
const core = @import("../../core/mod.zig");
const framework = @import("../../system/framework.zig");

pub fn runOperationsBenchmarks(allocator: std.mem.Allocator, config: core.config.DatabaseBenchConfig) !void {
    var runner = framework.BenchmarkRunner.init(allocator);
    defer runner.deinit();

    std.debug.print("\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("                    DATABASE ENGINE BENCHMARKS\n", .{});
    std.debug.print("================================================================================\n\n", .{});

    const size = 1000;
    const dim = 384;
    const vectors = try core.vectors.generateNormalized(allocator, size, dim, config.seed);
    defer core.vectors.free(allocator, vectors);

    // 1. Insertion
    {
        _ = try runner.run(
            .{ .name = "engine_insert_throughput", .category = "database/ops" },
            struct {
                fn bench(a: std.mem.Allocator, vecs: [][]const f32) !void {
                    var engine = try abi.wdbx.neural.Engine.init(a, .{});
                    defer engine.deinit();
                    for (vecs, 0..) |v, i| {
                        var id_buf: [32]u8 = undefined;
                        const id = try std.fmt.bufPrint(&id_buf, "{d}", .{i});
                        try engine.indexByVector(id, v, .{ .text = "bench" });
                    }
                }
            }.bench,
            .{ allocator, vectors },
        );
    }

    // 2. Search Latency
    {
        var engine = try abi.wdbx.neural.Engine.init(allocator, .{});
        defer engine.deinit();
        for (vectors, 0..) |v, i| {
            var id_buf: [32]u8 = undefined;
            const id = try std.fmt.bufPrint(&id_buf, "{d}", .{i});
            try engine.indexByVector(id, v, .{ .text = "bench" });
        }

        const queries = try core.vectors.generateNormalized(allocator, 10, dim, config.seed + 1);
        defer core.vectors.free(allocator, queries);

        _ = try runner.run(
            .{ .name = "engine_search_latency", .category = "database/ops" },
            struct {
                fn bench(e: *abi.wdbx.neural.Engine, qs: [][]const f32, a: std.mem.Allocator) !void {
                    for (qs) |q| {
                        const results = try e.searchByVector(q, .{ .k = 10 });
                        a.free(results);
                    }
                }
            }.bench,
            .{ &engine, queries, allocator },
        );
    }

    runner.printSummaryDebug();
}
