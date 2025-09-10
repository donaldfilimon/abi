const std = @import("std");
const abi = @import("abi");
const database = abi.wdbx.database;

const BenchmarkResult = struct {
    operation: []const u8,
    duration_ns: u64,
    operations_per_sec: f64,
    memory_usage: usize,
    details: ?[]const u8,
};

const BenchmarkSuite = struct {
    allocator: std.mem.Allocator,
    results: std.ArrayListUnmanaged(BenchmarkResult),

    pub fn init(allocator: std.mem.Allocator) BenchmarkSuite {
        return .{
            .allocator = allocator,
            .results = std.ArrayListUnmanaged(BenchmarkResult){},
        };
    }

    pub fn deinit(self: *BenchmarkSuite) void {
        for (self.results.items) |result| {
            if (result.details) |details| {
                self.allocator.free(details);
            }
            // Operation is owned (duplicated on insert)
            self.allocator.free(result.operation);
        }
        self.results.deinit(self.allocator);
    }

    pub fn addResult(self: *BenchmarkSuite, result: BenchmarkResult) !void {
        var owned = result;
        owned.operation = try self.allocator.dupe(u8, result.operation);
        try self.results.append(self.allocator, owned);
    }

    pub fn printResults(self: *const BenchmarkSuite) void {
        std.debug.print("\n=== Database Performance Benchmark Results ===\n", .{});
        std.debug.print("{s:<20} {s:<15} {s:<20} {s:<15}\n", .{ "Operation", "Duration (ns)", "Ops/sec", "Memory (bytes)" });
        std.debug.print("{s:-<20} {s:-<15} {s:-<20} {s:-<15}\n", .{ "", "", "", "" });

        for (self.results.items) |result| {
            const duration_ms = @as(f64, @floatFromInt(result.duration_ns)) / 1_000_000.0;
            std.debug.print("{s:<20} {d:.3} {s} {d:.2} {d}\n", .{
                result.operation,
                duration_ms,
                "ms",
                result.operations_per_sec,
                result.memory_usage,
            });

            if (result.details) |details| {
                std.debug.print("  Details: {s}\n", .{details});
            }
        }
    }
};

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    var benchmark = BenchmarkSuite.init(allocator);
    defer benchmark.deinit();

    const test_file = "benchmark_db.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    std.debug.print("Starting Database Performance Benchmark...\n", .{});

    // Benchmark 1: Database initialization
    try benchmarkDatabaseInit(&benchmark, test_file);

    // Benchmark 2: Single embedding insertion
    try benchmarkSingleInsertion(&benchmark, test_file);

    // Benchmark 3: Batch insertion
    try benchmarkBatchInsertion(&benchmark, test_file);

    // Benchmark 4: Search performance
    try benchmarkSearch(&benchmark, test_file);

    // Benchmark 5: Memory efficiency
    try benchmarkMemoryEfficiency(&benchmark, test_file);

    // Benchmark 6: Concurrent operations (disabled for now)
    // try benchmarkConcurrentOperations(&benchmark, test_file);

    // Print all results
    benchmark.printResults();
}

fn benchmarkDatabaseInit(benchmark: *BenchmarkSuite, test_file: []const u8) !void {
    const allocator = benchmark.allocator;

    // Warm up
    _ = try database.Db.open(test_file, true);
    std.fs.cwd().deleteFile(test_file) catch {};

    const iterations = 100;
    var total_time: u64 = 0;
    var total_memory: usize = 0;

    for (0..iterations) |_| {
        const start_time = std.time.nanoTimestamp();
        var db = try database.Db.open(test_file, true);
        const end_time = std.time.nanoTimestamp();

        total_time += @as(u64, @intCast(end_time - start_time));
        total_memory += db.read_buffer.len;

        db.close();
        std.fs.cwd().deleteFile(test_file) catch {};
    }

    const avg_time = total_time / iterations;
    const avg_memory = total_memory / iterations;
    const ops_per_sec = 1_000_000_000.0 / @as(f64, @floatFromInt(avg_time));

    try benchmark.addResult(.{
        .operation = "Database Init",
        .duration_ns = avg_time,
        .operations_per_sec = ops_per_sec,
        .memory_usage = avg_memory,
        .details = try std.fmt.allocPrint(allocator, "{} iterations, avg time: {}ns", .{ iterations, avg_time }),
    });
}

fn benchmarkSingleInsertion(benchmark: *BenchmarkSuite, _: []const u8) !void {
    const allocator = benchmark.allocator;

    const single_file = "single_test.wdbx";
    defer std.fs.cwd().deleteFile(single_file) catch {};

    var db = try database.Db.open(single_file, true);
    defer db.close();
    try db.init(128);

    const iterations = 1000;
    var total_time: u64 = 0;

    for (0..iterations) |i| {
        var embedding = try allocator.alloc(f32, 128);
        defer allocator.free(embedding);

        for (0..128) |j| {
            embedding[j] = @as(f32, @floatFromInt(i * 128 + j)) * 0.001;
        }

        const start_time = std.time.nanoTimestamp();
        _ = try db.addEmbedding(embedding);
        const end_time = std.time.nanoTimestamp();

        total_time += @as(u64, @intCast(end_time - start_time));
    }

    const avg_time = total_time / iterations;
    const ops_per_sec = 1_000_000_000.0 / @as(f64, @floatFromInt(avg_time));

    try benchmark.addResult(.{
        .operation = "Single Insert",
        .duration_ns = avg_time,
        .operations_per_sec = ops_per_sec,
        .memory_usage = 128 * @sizeOf(f32),
        .details = try std.fmt.allocPrint(allocator, "{} iterations, 128-dim vectors", .{iterations}),
    });
}

fn benchmarkBatchInsertion(benchmark: *BenchmarkSuite, _: []const u8) !void {
    const allocator = benchmark.allocator;

    const batch_file = "batch_test.wdbx";
    defer std.fs.cwd().deleteFile(batch_file) catch {};

    var db = try database.Db.open(batch_file, true);
    defer db.close();
    try db.init(128);

    const batch_sizes = [_]usize{ 10, 50, 100, 500 };

    for (batch_sizes) |batch_size| {
        var batch_embeddings = try allocator.alloc([]f32, batch_size);
        defer {
            for (batch_embeddings) |emb| {
                allocator.free(emb);
            }
            allocator.free(batch_embeddings);
        }

        for (0..batch_size) |i| {
            var embedding = try allocator.alloc(f32, 128);
            for (0..128) |j| {
                embedding[j] = @as(f32, @floatFromInt(i * 128 + j)) * 0.001;
            }
            batch_embeddings[i] = embedding;
        }

        const start_time = std.time.nanoTimestamp();
        const indices = try db.addEmbeddingsBatch(batch_embeddings);
        const end_time = std.time.nanoTimestamp();

        const duration = @as(u64, @intCast(end_time - start_time));
        const ops_per_sec = 1_000_000_000.0 / @as(f64, @floatFromInt(duration));

        try benchmark.addResult(.{
            .operation = try std.fmt.allocPrint(allocator, "Batch Insert {}", .{batch_size}),
            .duration_ns = duration,
            .operations_per_sec = ops_per_sec,
            .memory_usage = batch_size * 128 * @sizeOf(f32),
            .details = try std.fmt.allocPrint(allocator, "128-dim vectors, {} vectors", .{batch_size}),
        });

        allocator.free(indices);
    }
}

fn benchmarkSearch(benchmark: *BenchmarkSuite, _: []const u8) !void {
    const allocator = benchmark.allocator;

    const search_file = "search_test.wdbx";
    defer std.fs.cwd().deleteFile(search_file) catch {};

    var db = try database.Db.open(search_file, true);
    defer db.close();
    try db.init(128);

    // Pre-populate with test data
    const num_vectors = 10000;
    for (0..num_vectors) |i| {
        var embedding = try allocator.alloc(f32, 128);
        defer allocator.free(embedding);

        for (0..128) |j| {
            embedding[j] = @as(f32, @floatFromInt(i * 128 + j)) * 0.001;
        }

        _ = try db.addEmbedding(embedding);
    }

    const search_queries = [_]usize{ 1, 10, 100, 1000 };

    for (search_queries) |top_k| {
        var query = try allocator.alloc(f32, 128);
        defer allocator.free(query);

        for (0..128) |i| {
            query[i] = @as(f32, @floatFromInt(i)) * 0.01;
        }

        const iterations = 100;
        var total_time: u64 = 0;

        for (0..iterations) |_| {
            const start_time = std.time.nanoTimestamp();
            const results = try db.search(query, top_k, allocator);
            const end_time = std.time.nanoTimestamp();

            total_time += @as(u64, @intCast(end_time - start_time));
            allocator.free(results);
        }

        const avg_time = total_time / iterations;
        const ops_per_sec = 1_000_000_000.0 / @as(f64, @floatFromInt(avg_time));

        try benchmark.addResult(.{
            .operation = try std.fmt.allocPrint(allocator, "Search Top-{}", .{top_k}),
            .duration_ns = avg_time,
            .operations_per_sec = ops_per_sec,
            .memory_usage = top_k * @sizeOf(database.Db.Result),
            .details = try std.fmt.allocPrint(allocator, "{} vectors, {} iterations", .{ num_vectors, iterations }),
        });
    }
}

fn benchmarkMemoryEfficiency(benchmark: *BenchmarkSuite, _: []const u8) !void {
    const allocator = benchmark.allocator;

    const mem_file = "mem_test.wdbx";
    defer std.fs.cwd().deleteFile(mem_file) catch {};

    var db = try database.Db.open(mem_file, true);
    defer db.close();
    try db.init(128);

    const initial_memory = db.read_buffer.len;

    // Add vectors and measure memory growth
    const num_vectors = 1000;
    for (0..num_vectors) |i| {
        var embedding = try allocator.alloc(f32, 128);
        defer allocator.free(embedding);

        for (0..128) |j| {
            embedding[j] = @as(f32, @floatFromInt(i * 128 + j)) * 0.001;
        }

        _ = try db.addEmbedding(embedding);
    }

    const final_memory = db.read_buffer.len;
    const memory_per_vector = @as(f64, @floatFromInt(final_memory - initial_memory)) / @as(f64, @floatFromInt(num_vectors));

    try benchmark.addResult(.{
        .operation = "Memory Efficiency",
        .duration_ns = 0,
        .operations_per_sec = 0,
        .memory_usage = final_memory,
        .details = try std.fmt.allocPrint(allocator, "{} vectors, {d:.2} bytes/vector", .{ num_vectors, memory_per_vector }),
    });
}

fn benchmarkConcurrentOperations(benchmark: *BenchmarkSuite, test_file: []const u8) !void {
    const allocator = benchmark.allocator;

    var db = try database.Db.open(test_file, true);
    defer db.close();
    try db.init(128);

    // Pre-populate with some data
    const num_vectors = 1000;
    for (0..num_vectors) |i| {
        var embedding = try allocator.alloc(f32, 128);
        defer allocator.free(embedding);

        for (0..128) |j| {
            embedding[j] = @as(f32, @floatFromInt(i * 128 + j)) * 0.001;
        }

        _ = try db.addEmbedding(embedding);
    }

    // Simulate concurrent search operations
    const num_threads = 4;
    const operations_per_thread = 100;

    var threads = try allocator.alloc(std.Thread, num_threads);
    defer allocator.free(threads);

    const start_time = std.time.nanoTimestamp();

    for (0..num_threads) |thread_id| {
        threads[thread_id] = try std.Thread.spawn(.{}, searchWorker, .{ &db, operations_per_thread, thread_id });
    }

    for (threads) |thread| {
        thread.join();
    }

    const end_time = std.time.nanoTimestamp();
    const total_duration = @as(u64, @intCast(end_time - start_time));
    const total_operations = num_threads * operations_per_thread;
    const ops_per_sec = 1_000_000_000.0 * @as(f64, @floatFromInt(total_operations)) / @as(f64, @floatFromInt(total_duration));

    try benchmark.addResult(.{
        .operation = "Concurrent Search",
        .duration_ns = total_duration,
        .operations_per_sec = ops_per_sec,
        .memory_usage = num_threads * @sizeOf(database.Db.Result) * 10,
        .details = try std.fmt.allocPrint(allocator, "{} threads, {} ops/thread", .{ num_threads, operations_per_thread }),
    });
}

fn searchWorker(db: *database.Db, operations: usize, thread_id: usize) void {
    const allocator = std.heap.page_allocator;

    for (0..operations) |i| {
        var query = try allocator.alloc(f32, 128);
        defer allocator.free(query);

        for (0..128) |j| {
            query[j] = @as(f32, @floatFromInt(thread_id * 1000 + i * 100 + j)) * 0.001;
        }

        const results = db.search(query, 10, allocator) catch continue;
        defer allocator.free(results);
    }
}

/// Benchmark HNSW index performance
fn benchmarkHNSWIndex(suite: *BenchmarkSuite, allocator: std.mem.Allocator) !void {
    const test_file = "hnsw_test.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    var db = try database.Db.open(test_file, true);
    defer db.close();

    try db.init(384);

    // Initialize HNSW index
    const hnsw_start = std.time.nanoTimestamp();
    try db.initHNSW();
    const hnsw_init_time = std.time.nanoTimestamp() - hnsw_start;

    // Add vectors to both database and HNSW index
    const num_vectors = 10_000;
    const start_time = std.time.nanoTimestamp();

    for (0..num_vectors) |i| {
        var embedding = try allocator.alloc(f32, 384);
        defer allocator.free(embedding);

        for (0..384) |j| {
            embedding[j] = @as(f32, @floatFromInt(i * 384 + j)) * 0.001;
        }

        _ = try db.addEmbedding(embedding);
    }

    const insertion_time = std.time.nanoTimestamp() - start_time;

    // Benchmark HNSW search
    var query = try allocator.alloc(f32, 384);
    defer allocator.free(query);
    for (0..384) |i| {
        query[i] = @as(f32, @floatFromInt(i)) * 0.01;
    }

    const search_start = std.time.nanoTimestamp();
    const results = try db.searchHNSW(query, 10, allocator);
    defer allocator.free(results);
    const search_time = std.time.nanoTimestamp() - search_start;

    // Benchmark brute force search for comparison
    const brute_start = std.time.nanoTimestamp();
    const brute_results = try db.search(query, 10, allocator);
    defer allocator.free(brute_results);
    const brute_time = std.time.nanoTimestamp() - brute_start;

    try suite.addResult(.{
        .operation = "HNSW Index Initialization",
        .duration_ns = hnsw_init_time,
        .operations_per_sec = 1,
        .memory_usage = 0,
        .details = try std.fmt.allocPrint(allocator, "HNSW index created in {d:.3} ms", .{@as(f64, @floatFromInt(hnsw_init_time)) / 1_000_000.0}),
    });

    try suite.addResult(.{
        .operation = "HNSW Vector Insertion",
        .duration_ns = insertion_time,
        .operations_per_sec = num_vectors,
        .memory_usage = 0,
        .details = try std.fmt.allocPrint(allocator, "Inserted {d} vectors in {d:.3} ms", .{ num_vectors, @as(f64, @floatFromInt(insertion_time)) / 1_000_000.0 }),
    });

    try suite.addResult(.{
        .operation = "HNSW Search (10K vectors)",
        .duration_ns = search_time,
        .operations_per_sec = 1,
        .memory_usage = 0,
        .details = try std.fmt.allocPrint(allocator, "HNSW: {d:.3} ms, Brute Force: {d:.3} ms, Speedup: {d:.1}x", .{ @as(f64, @floatFromInt(search_time)) / 1_000_000.0, @as(f64, @floatFromInt(brute_time)) / 1_000_000.0, @as(f64, @floatFromInt(brute_time)) / @as(f64, @floatFromInt(search_time)) }),
    });
}

/// Benchmark parallel search performance
fn benchmarkParallelSearch(suite: *BenchmarkSuite, allocator: std.mem.Allocator) !void {
    const test_file = "parallel_test.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    var db = try database.Db.open(test_file, true);
    defer db.close();

    try db.init(128);

    // Add vectors
    const num_vectors = 50_000;
    for (0..num_vectors) |i| {
        var embedding = try allocator.alloc(f32, 128);
        defer allocator.free(embedding);

        for (0..128) |j| {
            embedding[j] = @as(f32, @floatFromInt(i * 128 + j)) * 0.001;
        }

        _ = try db.addEmbedding(embedding);
    }

    // Benchmark single-threaded search
    var query = try allocator.alloc(f32, 128);
    defer allocator.free(query);
    for (0..128) |i| {
        query[i] = @as(f32, @floatFromInt(i)) * 0.01;
    }

    const single_start = std.time.nanoTimestamp();
    const single_results = try db.search(query, 10, allocator);
    defer allocator.free(single_results);
    const single_time = std.time.nanoTimestamp() - single_start;

    // Benchmark parallel search with different thread counts
    const thread_counts = [_]u32{ 2, 4, 8 };

    for (thread_counts) |thread_count| {
        const parallel_start = std.time.nanoTimestamp();
        const parallel_results = try db.searchParallel(query, 10, allocator, thread_count);
        defer allocator.free(parallel_results);
        const parallel_time = std.time.nanoTimestamp() - parallel_start;

        const speedup = @as(f64, @floatFromInt(single_time)) / @as(f64, @floatFromInt(parallel_time));

        try suite.addResult(.{
            .operation = try std.fmt.allocPrint(allocator, "Parallel Search ({d} threads)", .{thread_count}),
            .duration_ns = parallel_time,
            .operations_per_sec = 1,
            .memory_usage = 0,
            .details = try std.fmt.allocPrint(allocator, "Single: {d:.3} ms, Parallel: {d:.3} ms, Speedup: {d:.1}x", .{ @as(f64, @floatFromInt(single_time)) / 1_000_000.0, @as(f64, @floatFromInt(parallel_time)) / 1_000_000.0, speedup }),
        });
    }
}

/// Benchmark advanced SIMD optimizations
fn benchmarkAdvancedSIMD(suite: *BenchmarkSuite, allocator: std.mem.Allocator) !void {
    const test_file = "simd_test.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    var db = try database.Db.open(test_file, true);
    defer db.close();

    // Test different vector dimensions to showcase SIMD benefits
    const dimensions = [_]u16{ 64, 128, 256, 512 };

    for (dimensions) |dim| {
        try db.init(dim);

        // Add test vectors
        const num_vectors = 1_000;
        for (0..num_vectors) |i| {
            var embedding = try allocator.alloc(f32, dim);
            defer allocator.free(embedding);

            for (0..dim) |j| {
                embedding[j] = @as(f32, @floatFromInt(i * dim + j)) * 0.001;
            }

            _ = try db.addEmbedding(embedding);
        }

        // Benchmark search performance
        var query = try allocator.alloc(f32, dim);
        defer allocator.free(query);
        for (0..dim) |i| {
            query[i] = @as(f32, @floatFromInt(i)) * 0.01;
        }

        const start_time = std.time.nanoTimestamp();
        const results = try db.search(query, 10, allocator);
        defer allocator.free(results);
        const search_time = std.time.nanoTimestamp() - start_time;

        const ops_per_sec = @as(f64, @floatFromInt(num_vectors)) / (@as(f64, @floatFromInt(search_time)) / 1_000_000_000.0);

        try suite.addResult(.{
            .operation = try std.fmt.allocPrint(allocator, "SIMD Search ({d}D)", .{dim}),
            .duration_ns = search_time,
            .operations_per_sec = ops_per_sec,
            .memory_usage = 0,
            .details = try std.fmt.allocPrint(allocator, "{d}D vectors: {d:.3} ms, {d:.0} ops/sec", .{ dim, @as(f64, @floatFromInt(search_time)) / 1_000_000.0, ops_per_sec }),
        });
    }
}
