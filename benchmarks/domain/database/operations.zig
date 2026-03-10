//! Database Operations Benchmarks
//!
//! Measures performance of the production WDBX Engine.
//! Benchmarks: Insertion throughput, Query latency, and Batch operations.

const std = @import("std");
const abi = @import("abi");
const core = @import("../../core/mod.zig");
const framework = @import("../../system/framework.zig");

const DistanceResult = struct {
    id: u64,
    distance: f32,
};

pub fn bruteForceSearch(
    allocator: std.mem.Allocator,
    vectors: [][]f32,
    query: []const f32,
    k: usize,
    metric: core.Metric,
) ![]u64 {
    var distances = try allocator.alloc(DistanceResult, vectors.len);
    defer allocator.free(distances);

    for (vectors, 0..) |vector, idx| {
        distances[idx] = .{
            .id = @intCast(idx),
            .distance = core.distance.compute(metric, vector, query),
        };
    }

    std.mem.sort(DistanceResult, distances, {}, struct {
        fn lessThan(_: void, lhs: DistanceResult, rhs: DistanceResult) bool {
            return lhs.distance < rhs.distance;
        }
    }.lessThan);

    const limit = @min(k, distances.len);
    const result = try allocator.alloc(u64, limit);
    for (distances[0..limit], 0..) |entry, idx| {
        result[idx] = entry.id;
    }
    return result;
}

pub fn calculateRecall(results: anytype, ground_truth: []const u64, k: usize) f64 {
    const limit = @min(k, @min(results.len, ground_truth.len));
    if (limit == 0) return 0.0;

    var matches: usize = 0;
    for (results[0..limit]) |result| {
        const result_id: u64 = @intCast(result.id);
        for (ground_truth[0..limit]) |gt_id| {
            if (result_id == gt_id) {
                matches += 1;
                break;
            }
        }
    }

    return @as(f64, @floatFromInt(matches)) / @as(f64, @floatFromInt(limit));
}

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
                fn bench(a: std.mem.Allocator, vecs: [][]f32) !void {
                    var engine = try abi.wdbx.Engine.init(a, .{});
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
        var engine = try abi.wdbx.Engine.init(allocator, .{});
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
                fn bench(e: *abi.wdbx.Engine, qs: [][]f32, a: std.mem.Allocator) !void {
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
