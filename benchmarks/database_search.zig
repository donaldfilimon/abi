//! Database search benchmarks.
//! Tests search latency and throughput with various configurations.

const std = @import("std");
const Database = @import("../src/features/database/database.zig").Database;
const DatabaseConfig = @import("../src/features/database/database.zig").DatabaseConfig;
const HnswIndex = @import("../src/features/database/hnsw.zig").HnswIndex;
const index_mod = @import("../src/features/database/index.zig");

pub const TOP_K = 10;

/// Run all search benchmarks.
pub fn run(
    allocator: std.mem.Allocator,
    vectors: []const []const f32,
    queries: []const []const f32,
) !void {
    std.debug.print("\n=== Search Benchmarks ===\n\n", .{});

    try benchmarkLinearSearch(allocator, vectors, queries);
    try benchmarkHnswSearch(allocator, vectors, queries);
    try benchmarkConcurrentSearch(allocator, vectors, queries);
}

fn benchmarkLinearSearch(
    allocator: std.mem.Allocator,
    vectors: []const []const f32,
    queries: []const []const f32,
) !void {
    std.debug.print("--- Linear Search (top-{d}) ---\n", .{TOP_K});

    var db = try Database.initWithConfig(allocator, "bench_search", .{
        .cache_norms = true,
        .initial_capacity = vectors.len,
    });
    defer db.deinit();

    // Populate
    for (vectors, 0..) |vec, i| {
        try db.insert(@intCast(i), vec, null);
    }

    // Warm-up
    for (0..10) |i| {
        const results = try db.search(allocator, queries[i % queries.len], TOP_K);
        allocator.free(results);
    }

    // Benchmark
    var total_ns: u64 = 0;
    var min_ns: u64 = std.math.maxInt(u64);
    var max_ns: u64 = 0;

    for (queries) |query| {
        var timer = try std.time.Timer.start();
        const results = try db.search(allocator, query, TOP_K);
        const elapsed = timer.read();
        allocator.free(results);

        total_ns += elapsed;
        min_ns = @min(min_ns, elapsed);
        max_ns = @max(max_ns, elapsed);
    }

    const query_count = queries.len;
    const avg_us = @as(f64, @floatFromInt(total_ns / query_count)) / 1000.0;
    const min_us = @as(f64, @floatFromInt(min_ns)) / 1000.0;
    const max_us = @as(f64, @floatFromInt(max_ns)) / 1000.0;
    const qps = @as(f64, @floatFromInt(query_count)) / (@as(f64, @floatFromInt(total_ns)) / 1_000_000_000.0);

    std.debug.print("  Queries: {d}, Avg: {d:.1}us, Min: {d:.1}us, Max: {d:.1}us\n", .{
        query_count,
        avg_us,
        min_us,
        max_us,
    });
    std.debug.print("  Throughput: {d:.0} queries/sec\n", .{qps});
}

fn benchmarkHnswSearch(
    allocator: std.mem.Allocator,
    vectors: []const []const f32,
    queries: []const []const f32,
) !void {
    std.debug.print("\n--- HNSW Search (top-{d}) ---\n", .{TOP_K});

    // Create record views for HNSW
    const records = try allocator.alloc(index_mod.VectorRecordView, vectors.len);
    defer allocator.free(records);
    for (vectors, 0..) |vec, i| {
        records[i] = .{ .id = @intCast(i), .vector = vec };
    }

    // Build index
    var build_timer = try std.time.Timer.start();
    var index = try HnswIndex.buildWithConfig(allocator, records, .{
        .m = 16,
        .ef_construction = 100,
        .search_pool_size = 8,
        .distance_cache_size = 2048,
    });
    defer index.deinit(allocator);
    const build_ms = @as(f64, @floatFromInt(build_timer.read())) / 1_000_000.0;
    std.debug.print("  Built index in {d:.2}ms\n", .{build_ms});

    // Warm-up
    for (0..10) |i| {
        const results = try index.search(allocator, records, queries[i % queries.len], TOP_K);
        allocator.free(results);
    }

    // Benchmark
    var total_ns: u64 = 0;
    var min_ns: u64 = std.math.maxInt(u64);
    var max_ns: u64 = 0;

    for (queries) |query| {
        var timer = try std.time.Timer.start();
        const results = try index.search(allocator, records, query, TOP_K);
        const elapsed = timer.read();
        allocator.free(results);

        total_ns += elapsed;
        min_ns = @min(min_ns, elapsed);
        max_ns = @max(max_ns, elapsed);
    }

    const query_count = queries.len;
    const avg_us = @as(f64, @floatFromInt(total_ns / query_count)) / 1000.0;
    const min_us = @as(f64, @floatFromInt(min_ns)) / 1000.0;
    const max_us = @as(f64, @floatFromInt(max_ns)) / 1000.0;
    const qps = @as(f64, @floatFromInt(query_count)) / (@as(f64, @floatFromInt(total_ns)) / 1_000_000_000.0);

    std.debug.print("  Queries: {d}, Avg: {d:.1}us, Min: {d:.1}us, Max: {d:.1}us\n", .{
        query_count,
        avg_us,
        min_us,
        max_us,
    });
    std.debug.print("  Throughput: {d:.0} queries/sec\n", .{qps});

    // Report cache stats
    if (index.getCacheStats()) |stats| {
        std.debug.print("  Distance cache: {d:.1}% hit rate\n", .{stats.hit_rate * 100.0});
    }
}

fn benchmarkConcurrentSearch(
    allocator: std.mem.Allocator,
    vectors: []const []const f32,
    queries: []const []const f32,
) !void {
    std.debug.print("\n--- Concurrent Search (thread-safe) ---\n", .{});

    var db = try Database.initWithConfig(allocator, "bench_concurrent", .{
        .cache_norms = true,
        .initial_capacity = vectors.len,
        .thread_safe = true,
    });
    defer db.deinit();

    // Populate
    for (vectors, 0..) |vec, i| {
        try db.insert(@intCast(i), vec, null);
    }

    const num_threads = 4;
    const queries_per_thread = queries.len / num_threads;

    const WorkerState = struct {
        db: *Database,
        queries_slice: []const []const f32,
        alloc: std.mem.Allocator,
        total_ns: std.atomic.Value(u64),
        completed: std.atomic.Value(usize),
    };

    var state = WorkerState{
        .db = &db,
        .queries_slice = queries,
        .alloc = allocator,
        .total_ns = std.atomic.Value(u64).init(0),
        .completed = std.atomic.Value(usize).init(0),
    };

    const workerFn = struct {
        fn run(s: *WorkerState, thread_id: usize) void {
            const start = thread_id * queries_per_thread;
            const end = @min(start + queries_per_thread, s.queries_slice.len);

            for (s.queries_slice[start..end]) |query| {
                var timer = std.time.Timer.start() catch continue;
                const results = s.db.searchThreadSafe(s.alloc, query, TOP_K) catch continue;
                const elapsed = timer.read();
                s.alloc.free(results);

                _ = s.total_ns.fetchAdd(elapsed, .monotonic);
                _ = s.completed.fetchAdd(1, .monotonic);
            }
        }
    }.run;

    var threads: [num_threads]std.Thread = undefined;
    var timer = try std.time.Timer.start();

    for (0..num_threads) |i| {
        threads[i] = try std.Thread.spawn(.{}, workerFn, .{ &state, i });
    }

    for (&threads) |*t| t.join();

    const wall_time = timer.read();
    const completed = state.completed.load(.monotonic);
    const total_query_time = state.total_ns.load(.monotonic);
    const qps = @as(f64, @floatFromInt(completed)) / (@as(f64, @floatFromInt(wall_time)) / 1_000_000_000.0);
    const avg_us = @as(f64, @floatFromInt(total_query_time / completed)) / 1000.0;

    std.debug.print("  Threads: {d}, Queries: {d}, Avg latency: {d:.1}us\n", .{
        num_threads,
        completed,
        avg_us,
    });
    std.debug.print("  Concurrent throughput: {d:.0} queries/sec\n", .{qps});
}
