//! Database Operations Benchmarks
//!
//! Benchmarks for core database operations:
//! - Single and batch insertion
//! - Query latency and throughput
//! - Update and delete operations
//! - Brute force search (baseline)

const std = @import("std");
const core = @import("../core/mod.zig");
const framework = @import("../framework.zig");
const hnsw = @import("hnsw.zig");

// ============================================================================
// Insertion Benchmarks
// ============================================================================

/// Benchmark single vector insertion throughput
pub fn benchInsertionThroughput(
    allocator: std.mem.Allocator,
    vectors: [][]f32,
) !u64 {
    var index = hnsw.EuclideanHNSW.init(allocator, 16, 200);
    defer index.deinit();

    var prng = std.Random.DefaultPrng.init(12345);

    for (vectors, 0..) |v, i| {
        try index.insert(v, @intCast(i), prng.random());
    }

    return @intCast(vectors.len);
}

/// Benchmark batch insertion
pub fn benchBatchInsertion(
    allocator: std.mem.Allocator,
    vectors: [][]f32,
    batch_size: usize,
) !u64 {
    var index = hnsw.EuclideanHNSW.init(allocator, 16, 200);
    defer index.deinit();

    var prng = std.Random.DefaultPrng.init(12345);
    var inserted: u64 = 0;

    var i: usize = 0;
    while (i < vectors.len) {
        const batch_end = @min(i + batch_size, vectors.len);
        for (vectors[i..batch_end], i..) |v, idx| {
            try index.insert(v, @intCast(idx), prng.random());
            inserted += 1;
        }
        i = batch_end;
    }

    return inserted;
}

// ============================================================================
// Query Benchmarks
// ============================================================================

/// Benchmark query latency
pub fn benchQueryLatency(
    allocator: std.mem.Allocator,
    vectors: [][]f32,
    queries: [][]f32,
    k: usize,
    ef: usize,
) !u64 {
    var index = hnsw.EuclideanHNSW.init(allocator, 16, 200);
    defer index.deinit();

    var prng = std.Random.DefaultPrng.init(12345);

    // Build index
    for (vectors, 0..) |v, i| {
        try index.insert(v, @intCast(i), prng.random());
    }

    // Run queries
    var total_results: u64 = 0;
    for (queries) |q| {
        const results = try index.search(q, k, ef);
        defer allocator.free(results);
        total_results += results.len;
    }

    return total_results;
}

// ============================================================================
// Brute Force Search (Baseline)
// ============================================================================

/// Brute force k-NN search for recall calculation baseline
pub fn bruteForceSearch(
    allocator: std.mem.Allocator,
    vectors: [][]f32,
    query: []const f32,
    k: usize,
    comptime metric: core.Metric,
) ![]u64 {
    var distances = try allocator.alloc(hnsw.SearchResult, vectors.len);
    defer allocator.free(distances);

    for (vectors, 0..) |v, i| {
        distances[i] = .{ .id = @intCast(i), .dist = core.distance.compute(metric, query, v) };
    }

    std.mem.sort(hnsw.SearchResult, distances, {}, struct {
        fn cmp(_: void, a: hnsw.SearchResult, b: hnsw.SearchResult) bool {
            return a.dist < b.dist;
        }
    }.cmp);

    const result = try allocator.alloc(u64, k);
    for (result, 0..) |*r, i| {
        r.* = distances[i].id;
    }

    return result;
}

/// Calculate recall between approximate and exact results
pub fn calculateRecall(approximate: []const hnsw.SearchResult, exact: []const u64, k: usize) f64 {
    const limit = @min(k, @min(approximate.len, exact.len));
    var matches: usize = 0;

    for (approximate[0..limit]) |approx| {
        for (exact[0..limit]) |ex| {
            if (approx.id == ex) {
                matches += 1;
                break;
            }
        }
    }

    return @as(f64, @floatFromInt(matches)) / @as(f64, @floatFromInt(limit));
}

// ============================================================================
// Distance Metric Benchmarks
// ============================================================================

/// Benchmark different distance metrics
pub fn benchDistanceMetrics(
    _: std.mem.Allocator,
    vectors: [][]f32,
    runner: *framework.BenchmarkRunner,
) !void {
    std.debug.print("\n[Distance Metric Comparison]\n", .{});

    const dim = if (vectors.len > 0) vectors[0].len else 0;

    // Euclidean squared
    {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "euclidean_sq_d{d}", .{dim}) catch "euclidean_sq";

        const result = try runner.run(
            .{
                .name = name,
                .category = "database/distance",
                .warmup_iterations = 1000,
                .min_time_ns = 500_000_000,
            },
            struct {
                fn bench(vecs: [][]f32) f32 {
                    var sum: f32 = 0;
                    for (vecs[0 .. vecs.len - 1], vecs[1..]) |a, b| {
                        sum += core.distance.euclideanSq(a, b);
                    }
                    return sum;
                }
            }.bench,
            .{vectors},
        );

        std.debug.print("  {s}: {d:.0} pairs/sec\n", .{
            name,
            result.stats.opsPerSecond() * @as(f64, @floatFromInt(vectors.len - 1)),
        });
    }

    // Cosine
    {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "cosine_d{d}", .{dim}) catch "cosine";

        const result = try runner.run(
            .{
                .name = name,
                .category = "database/distance",
                .warmup_iterations = 1000,
                .min_time_ns = 500_000_000,
            },
            struct {
                fn bench(vecs: [][]f32) f32 {
                    var sum: f32 = 0;
                    for (vecs[0 .. vecs.len - 1], vecs[1..]) |a, b| {
                        sum += core.distance.cosine(a, b);
                    }
                    return sum;
                }
            }.bench,
            .{vectors},
        );

        std.debug.print("  {s}: {d:.0} pairs/sec\n", .{
            name,
            result.stats.opsPerSecond() * @as(f64, @floatFromInt(vectors.len - 1)),
        });
    }

    // Dot product
    {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "dot_product_d{d}", .{dim}) catch "dot_product";

        const result = try runner.run(
            .{
                .name = name,
                .category = "database/distance",
                .warmup_iterations = 1000,
                .min_time_ns = 500_000_000,
            },
            struct {
                fn bench(vecs: [][]f32) f32 {
                    var sum: f32 = 0;
                    for (vecs[0 .. vecs.len - 1], vecs[1..]) |a, b| {
                        sum += core.distance.dot(a, b);
                    }
                    return sum;
                }
            }.bench,
            .{vectors},
        );

        std.debug.print("  {s}: {d:.0} pairs/sec\n", .{
            name,
            result.stats.opsPerSecond() * @as(f64, @floatFromInt(vectors.len - 1)),
        });
    }

    // Manhattan
    {
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "manhattan_d{d}", .{dim}) catch "manhattan";

        const result = try runner.run(
            .{
                .name = name,
                .category = "database/distance",
                .warmup_iterations = 1000,
                .min_time_ns = 500_000_000,
            },
            struct {
                fn bench(vecs: [][]f32) f32 {
                    var sum: f32 = 0;
                    for (vecs[0 .. vecs.len - 1], vecs[1..]) |a, b| {
                        sum += core.distance.manhattan(a, b);
                    }
                    return sum;
                }
            }.bench,
            .{vectors},
        );

        std.debug.print("  {s}: {d:.0} pairs/sec\n", .{
            name,
            result.stats.opsPerSecond() * @as(f64, @floatFromInt(vectors.len - 1)),
        });
    }
}

// ============================================================================
// Main Benchmark Runner
// ============================================================================

/// Run all database operation benchmarks
pub fn runOperationsBenchmarks(allocator: std.mem.Allocator, config: core.config.DatabaseBenchConfig) !void {
    var runner = framework.BenchmarkRunner.init(allocator);
    defer runner.deinit();

    // Distance metric benchmarks
    for ([_]usize{ 128, 512, 1024 }) |dim| {
        const vectors = try core.vectors.generateNormalized(allocator, 1000, dim, config.seed);
        defer core.vectors.free(allocator, vectors);
        try benchDistanceMetrics(allocator, vectors, &runner);
    }

    // Insertion benchmarks
    std.debug.print("\n[Insertion Throughput]\n", .{});
    for (config.dataset_sizes[0..@min(3, config.dataset_sizes.len)]) |size| {
        for (config.dimensions[0..@min(2, config.dimensions.len)]) |dim| {
            const vectors = try core.vectors.generateNormalized(allocator, size, dim, config.seed);
            defer core.vectors.free(allocator, vectors);

            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "insert_{d}x{d}", .{ size, dim }) catch "insert";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "database/insert",
                    .warmup_iterations = 5,
                    .min_time_ns = config.min_time_ns,
                    .max_iterations = 50,
                },
                struct {
                    fn bench(a: std.mem.Allocator, vecs: [][]f32) !u64 {
                        return try benchInsertionThroughput(a, vecs);
                    }
                }.bench,
                .{ allocator, vectors },
            );

            std.debug.print("  {s}: {d:.0} vectors/sec\n", .{
                name,
                result.stats.opsPerSecond() * @as(f64, @floatFromInt(size)),
            });
        }
    }

    // Query latency benchmarks
    std.debug.print("\n[Query Latency (k-NN Search)]\n", .{});
    for (config.dataset_sizes[0..@min(2, config.dataset_sizes.len)]) |size| {
        const dim: usize = 256;
        const vectors = try core.vectors.clustered(allocator, size, dim, config.num_clusters);
        defer core.vectors.free(allocator, vectors);

        const queries = try core.vectors.normalized(allocator, @min(100, config.query_iterations), dim);
        defer core.vectors.free(allocator, queries);

        for (config.k_values[0..@min(4, config.k_values.len)]) |k| {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "query_{d}_k{d}", .{ size, k }) catch "query";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "database/query",
                    .warmup_iterations = 5,
                    .min_time_ns = config.min_time_ns,
                    .max_iterations = 50,
                },
                struct {
                    fn bench(a: std.mem.Allocator, vecs: [][]f32, qs: [][]f32, kval: usize) !u64 {
                        return try benchQueryLatency(a, vecs, qs, kval, 64);
                    }
                }.bench,
                .{ allocator, vectors, queries, k },
            );

            std.debug.print("  {s}: {d:.0} queries/sec, {d:.0}ns/query\n", .{
                name,
                result.stats.opsPerSecond(),
                result.stats.mean_ns,
            });
        }
    }

    // Batch insertion comparison
    std.debug.print("\n[Batch Insertion Comparison]\n", .{});
    {
        const size: usize = 5000;
        const dim: usize = 256;
        const vectors = try core.vectors.normalized(allocator, size, dim);
        defer core.vectors.free(allocator, vectors);

        for (config.batch_sizes[0..@min(4, config.batch_sizes.len)]) |batch| {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "batch_{d}", .{batch}) catch "batch";

            const result = try runner.run(
                .{
                    .name = name,
                    .category = "database/batch",
                    .warmup_iterations = 3,
                    .min_time_ns = config.min_time_ns,
                    .max_iterations = 20,
                },
                struct {
                    fn bench(a: std.mem.Allocator, vecs: [][]f32, bs: usize) !u64 {
                        return try benchBatchInsertion(a, vecs, bs);
                    }
                }.bench,
                .{ allocator, vectors, batch },
            );

            std.debug.print("  {s}: {d:.0} vectors/sec\n", .{
                name,
                result.stats.opsPerSecond() * @as(f64, @floatFromInt(size)),
            });
        }
    }

    // Brute force baseline
    std.debug.print("\n[Brute Force Baseline (for Recall Reference)]\n", .{});
    {
        const size: usize = 1000;
        const dim: usize = 128;
        const k: usize = 10;

        const vectors = try core.vectors.normalized(allocator, size, dim);
        defer core.vectors.free(allocator, vectors);

        const queries = try core.vectors.normalized(allocator, 100, dim);
        defer core.vectors.free(allocator, queries);

        const result = try runner.run(
            .{
                .name = "brute_force",
                .category = "database/baseline",
                .warmup_iterations = 5,
                .min_time_ns = 500_000_000,
            },
            struct {
                fn bench(a: std.mem.Allocator, vecs: [][]f32, qs: [][]f32, kval: usize) !u64 {
                    var count: u64 = 0;
                    for (qs) |q| {
                        const res = try bruteForceSearch(a, vecs, q, kval, .euclidean_sq);
                        defer a.free(res);
                        count += res.len;
                    }
                    return count;
                }
            }.bench,
            .{ allocator, vectors, queries, k },
        );

        std.debug.print("  brute_force_{d}x{d}_k{d}: {d:.0} queries/sec\n", .{
            size,
            dim,
            k,
            result.stats.opsPerSecond(),
        });
    }

    std.debug.print("\n", .{});
    runner.printSummaryDebug();
}

// ============================================================================
// Tests
// ============================================================================

test "brute force search" {
    const allocator = std.testing.allocator;

    const vectors = try core.vectors.normalized(allocator, 100, 32);
    defer core.vectors.free(allocator, vectors);

    const results = try bruteForceSearch(allocator, vectors, vectors[0], 10, .euclidean_sq);
    defer allocator.free(results);

    try std.testing.expectEqual(@as(usize, 10), results.len);
    try std.testing.expectEqual(@as(u64, 0), results[0]); // First result should be the query itself
}

test "recall calculation" {
    const approx = [_]hnsw.SearchResult{
        .{ .id = 1, .dist = 0.1 },
        .{ .id = 2, .dist = 0.2 },
        .{ .id = 3, .dist = 0.3 },
    };
    const exact = [_]u64{ 1, 2, 4 };

    const recall = calculateRecall(&approx, &exact, 3);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), recall, 0.001);
}
