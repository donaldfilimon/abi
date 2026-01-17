//! Benchmark suite for database performance improvements.
//! Executes various workloads and reports timings.
//!
//! Run with: zig build run -- benchmark database
//!
//! This file provides the main entry point. Individual benchmarks are in:
//! - database_insert.zig: Insertion benchmarks
//! - database_search.zig: Search and HNSW benchmarks
//! - database_memory.zig: Memory and cache benchmarks

const std = @import("std");
const Database = @import("../src/features/database/database.zig").Database;
const DatabaseConfig = @import("../src/features/database/database.zig").DatabaseConfig;
const HotVectorData = @import("../src/features/database/database.zig").HotVectorData;
const HnswIndex = @import("../src/features/database/hnsw.zig").HnswIndex;
const index_mod = @import("../src/features/database/index.zig");
const storage = @import("../src/features/database/storage.zig");
const batch = @import("../src/features/database/batch.zig");

// Modular benchmark imports
const insert_bench = @import("database_insert.zig");
const search_bench = @import("database_search.zig");
const memory_bench = @import("database_memory.zig");

const VECTOR_DIM = 128;
const VECTOR_COUNT = 10_000;
const QUERY_COUNT = 100;
const TOP_K = 10;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n=== Database Performance Benchmarks ===\n\n", .{});

    // Generate test vectors
    std.debug.print("Generating {d} test vectors (dim={d})...\n", .{ VECTOR_COUNT, VECTOR_DIM });
    const vectors = try generateVectors(allocator, VECTOR_COUNT, VECTOR_DIM);
    defer {
        for (vectors) |vec| allocator.free(vec);
        allocator.free(vectors);
    }

    const queries = try generateVectors(allocator, QUERY_COUNT, VECTOR_DIM);
    defer {
        for (queries) |q| allocator.free(q);
        allocator.free(queries);
    }

    // Benchmark 1: Bulk Insertion (standard)
    std.debug.print("\n--- Benchmark 1: Bulk Insertion (standard) ---\n", .{});
    {
        var db = try Database.init(allocator, "bench_standard");
        defer db.deinit();

        var timer = try std.time.Timer.start();
        for (vectors, 0..) |vec, i| {
            try db.insert(@intCast(i), vec, null);
        }
        const elapsed_ns = timer.read();
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        const rate = @as(f64, @floatFromInt(VECTOR_COUNT)) / (elapsed_ms / 1000.0);
        std.debug.print("  Inserted {d} vectors in {d:.2}ms ({d:.0} vec/sec)\n", .{ VECTOR_COUNT, elapsed_ms, rate });
    }

    // Benchmark 2: Bulk Insertion (with cached norms)
    std.debug.print("\n--- Benchmark 2: Bulk Insertion (cached norms) ---\n", .{});
    {
        var db = try Database.initWithConfig(allocator, "bench_cached", .{
            .cache_norms = true,
            .initial_capacity = VECTOR_COUNT,
        });
        defer db.deinit();

        var timer = try std.time.Timer.start();
        for (vectors, 0..) |vec, i| {
            try db.insert(@intCast(i), vec, null);
        }
        const elapsed_ns = timer.read();
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        const rate = @as(f64, @floatFromInt(VECTOR_COUNT)) / (elapsed_ms / 1000.0);
        std.debug.print("  Inserted {d} vectors in {d:.2}ms ({d:.0} vec/sec)\n", .{ VECTOR_COUNT, elapsed_ms, rate });
    }

    // Benchmark 3: Search Latency
    std.debug.print("\n--- Benchmark 3: Search Latency (top-{d}) ---\n", .{TOP_K});
    {
        var db = try Database.initWithConfig(allocator, "bench_search", .{
            .cache_norms = true,
            .initial_capacity = VECTOR_COUNT,
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

        const avg_us = @as(f64, @floatFromInt(total_ns / QUERY_COUNT)) / 1000.0;
        const min_us = @as(f64, @floatFromInt(min_ns)) / 1000.0;
        const max_us = @as(f64, @floatFromInt(max_ns)) / 1000.0;
        const qps = @as(f64, @floatFromInt(QUERY_COUNT)) / (@as(f64, @floatFromInt(total_ns)) / 1_000_000_000.0);

        std.debug.print("  Queries: {d}, Avg: {d:.1}us, Min: {d:.1}us, Max: {d:.1}us\n", .{ QUERY_COUNT, avg_us, min_us, max_us });
        std.debug.print("  Throughput: {d:.0} queries/sec\n", .{qps});
    }

    // Benchmark 4: HNSW Index Build
    std.debug.print("\n--- Benchmark 4: HNSW Index Build ---\n", .{});
    {
        // Create record views for HNSW
        const records = try allocator.alloc(index_mod.VectorRecordView, vectors.len);
        defer allocator.free(records);
        for (vectors, 0..) |vec, i| {
            records[i] = .{ .id = @intCast(i), .vector = vec };
        }

        var timer = try std.time.Timer.start();
        var index = try HnswIndex.buildWithConfig(allocator, records, .{
            .m = 16,
            .ef_construction = 100,
            .search_pool_size = 4,
            .distance_cache_size = 1024,
        });
        defer index.deinit(allocator);
        const elapsed_ns = timer.read();
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;

        std.debug.print("  Built HNSW index for {d} vectors in {d:.2}ms\n", .{ VECTOR_COUNT, elapsed_ms });

        // Report cache stats
        if (index.getCacheStats()) |stats| {
            std.debug.print("  Distance cache: {d} hits, {d} misses, {d:.1}% hit rate\n", .{
                stats.hits,
                stats.misses,
                stats.hit_rate * 100.0,
            });
        }
    }

    // Benchmark 5: HNSW Search
    std.debug.print("\n--- Benchmark 5: HNSW Search (top-{d}) ---\n", .{TOP_K});
    {
        const records = try allocator.alloc(index_mod.VectorRecordView, vectors.len);
        defer allocator.free(records);
        for (vectors, 0..) |vec, i| {
            records[i] = .{ .id = @intCast(i), .vector = vec };
        }

        var index = try HnswIndex.buildWithConfig(allocator, records, .{
            .m = 16,
            .ef_construction = 100,
            .search_pool_size = 8,
            .distance_cache_size = 2048,
        });
        defer index.deinit(allocator);

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

        const avg_us = @as(f64, @floatFromInt(total_ns / QUERY_COUNT)) / 1000.0;
        const min_us = @as(f64, @floatFromInt(min_ns)) / 1000.0;
        const max_us = @as(f64, @floatFromInt(max_ns)) / 1000.0;
        const qps = @as(f64, @floatFromInt(QUERY_COUNT)) / (@as(f64, @floatFromInt(total_ns)) / 1_000_000_000.0);

        std.debug.print("  Queries: {d}, Avg: {d:.1}us, Min: {d:.1}us, Max: {d:.1}us\n", .{ QUERY_COUNT, avg_us, min_us, max_us });
        std.debug.print("  Throughput: {d:.0} queries/sec\n", .{qps});
    }

    // Benchmark 6: Concurrent Search (thread-safe mode)
    std.debug.print("\n--- Benchmark 6: Concurrent Search (thread-safe) ---\n", .{});
    {
        var db = try Database.initWithConfig(allocator, "bench_concurrent", .{
            .cache_norms = true,
            .initial_capacity = VECTOR_COUNT,
            .thread_safe = true,
        });
        defer db.deinit();

        // Populate
        for (vectors, 0..) |vec, i| {
            try db.insert(@intCast(i), vec, null);
        }

        const num_threads = 4;
        const queries_per_thread = QUERY_COUNT / num_threads;

        const WorkerState = struct {
            db: *Database,
            queries_slice: []const []f32,
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

        std.debug.print("  Threads: {d}, Queries: {d}, Avg latency: {d:.1}us\n", .{ num_threads, completed, avg_us });
        std.debug.print("  Concurrent throughput: {d:.0} queries/sec\n", .{qps});
    }

    // Benchmark 7: Batch Insert
    std.debug.print("\n--- Benchmark 7: Batch Insert ---\n", .{});
    {
        var processor = batch.BatchProcessor.init(allocator, .{
            .batch_size = 1000,
            .parallel_workers = 4,
            .prefetch_distance = 4,
        });
        defer processor.deinit();

        // Create batch records
        const records = try allocator.alloc(batch.BatchRecord, vectors.len);
        defer allocator.free(records);
        for (vectors, 0..) |vec, i| {
            records[i] = .{ .id = @intCast(i), .vector = vec };
        }

        var timer = try std.time.Timer.start();
        const result = try processor.insertBatch(records);
        const elapsed_ns = timer.read();
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;

        std.debug.print("  Sequential: {d} vectors in {d:.2}ms ({d:.0} vec/sec)\n", .{
            result.successful,
            elapsed_ms,
            result.throughput,
        });

        // Parallel batch
        var parallel_processor = batch.BatchProcessor.init(allocator, .{
            .batch_size = 1000,
            .parallel_workers = 4,
            .prefetch_distance = 4,
        });
        defer parallel_processor.deinit();

        timer = try std.time.Timer.start();
        const parallel_result = try parallel_processor.insertBatch(records);
        const parallel_elapsed = timer.read();
        const parallel_ms = @as(f64, @floatFromInt(parallel_elapsed)) / 1_000_000.0;

        std.debug.print("  Parallel: {d} vectors in {d:.2}ms ({d:.0} vec/sec)\n", .{
            parallel_result.successful,
            parallel_ms,
            parallel_result.throughput,
        });
    }

    // Benchmark 8: Cache-Aligned Hot Data
    std.debug.print("\n--- Benchmark 8: Cache-Aligned Hot Data Access ---\n", .{});
    {
        var hot_data = try HotVectorData.init(allocator, VECTOR_DIM, VECTOR_COUNT);
        defer hot_data.deinit(allocator);

        // Populate with vectors
        for (vectors) |vec| {
            var norm: f32 = 0.0;
            for (vec) |v| norm += v * v;
            norm = @sqrt(norm);
            try hot_data.append(vec, norm);
        }

        // Benchmark sequential access with prefetching
        var timer = try std.time.Timer.start();
        var sum: f32 = 0.0;
        for (0..hot_data.count) |i| {
            // Prefetch next vector
            if (i + 4 < hot_data.count) {
                hot_data.prefetch(i + 4);
            }
            const vec = hot_data.getVector(i);
            sum += vec[0];
        }
        const elapsed_ns = timer.read();
        const elapsed_us = @as(f64, @floatFromInt(elapsed_ns)) / 1000.0;

        std.debug.print("  Sequential access with prefetch: {d:.1}us for {d} vectors\n", .{ elapsed_us, hot_data.count });
        std.debug.print("  (Sum check: {d:.4})\n", .{sum});
    }

    // Benchmark 9: Memory Usage Comparison
    std.debug.print("\n--- Benchmark 9: Memory Usage ---\n", .{});
    {
        const stats = try Database.initWithConfig(allocator, "bench_memory", .{
            .cache_norms = true,
            .initial_capacity = VECTOR_COUNT,
        });
        defer @constCast(&stats).deinit();

        std.debug.print("  Database with cached norms:\n", .{});
        std.debug.print("    Vectors: {d} x {d} dims = {d} bytes\n", .{
            VECTOR_COUNT,
            VECTOR_DIM,
            VECTOR_COUNT * VECTOR_DIM * @sizeOf(f32),
        });
        std.debug.print("    Cached norms: {d} x {d} = {d} bytes\n", .{
            VECTOR_COUNT,
            @sizeOf(f32),
            VECTOR_COUNT * @sizeOf(f32),
        });
        std.debug.print("    Overhead for norms: {d:.2}%\n", .{
            @as(f64, 1.0) / @as(f64, @floatFromInt(VECTOR_DIM)) * 100.0,
        });
    }

    std.debug.print("\n=== Benchmarks Complete ===\n", .{});
}

fn generateVectors(allocator: std.mem.Allocator, count: usize, dim: usize) ![][]f32 {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const result = try allocator.alloc([]f32, count);
    errdefer {
        for (result) |vec| if (vec.len > 0) allocator.free(vec);
        allocator.free(result);
    }

    for (result) |*vec| {
        vec.* = try allocator.alloc(f32, dim);
        for (vec.*) |*v| {
            v.* = random.float(f32) * 2.0 - 1.0; // [-1, 1]
        }
        // Normalize to unit vector
        var norm: f32 = 0.0;
        for (vec.*) |v| norm += v * v;
        norm = @sqrt(norm);
        if (norm > 0) {
            for (vec.*) |*v| v.* /= norm;
        }
    }

    return result;
}
