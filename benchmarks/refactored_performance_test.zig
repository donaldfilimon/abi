const std = @import("std");
const core = @import("core");
const testing = std.testing;
const Timer = std.time.Timer;

const BenchmarkConfig = struct {
    warmup_iterations: u32 = 100,
    test_iterations: u32 = 1000,
    vector_sizes: []const usize = &[_]usize{ 128, 256, 384, 512, 768, 1024 },
    batch_sizes: []const usize = &[_]usize{ 1, 10, 100, 1000 },
    thread_counts: []const usize = &[_]usize{ 1, 2, 4, 8, 16 },
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("🚀 WDBX Refactored Performance Test Suite\n", .{});
    std.debug.print("=" ** 80 ++ "\n", .{});

    const config = BenchmarkConfig{};
    
    // Run all benchmarks
    try benchmarkVectorOperations(allocator, config);
    try benchmarkDatabaseOperations(allocator, config);
    try benchmarkIndexPerformance(allocator, config);
    try benchmarkConcurrency(allocator, config);
    try benchmarkMemoryEfficiency(allocator, config);
    
    std.debug.print("\n✅ All performance tests completed successfully!\n", .{});
}

fn benchmarkVectorOperations(allocator: std.mem.Allocator, config: BenchmarkConfig) !void {
    std.debug.print("\n📊 Vector Operations Benchmark\n", .{});
    std.debug.print("-" ** 40 ++ "\n", .{});

    var vector_ops = try core.vector.VectorOps.init(allocator, true);
    defer vector_ops.deinit();

    for (config.vector_sizes) |size| {
        // Create test vectors
        const a = try allocator.alloc(f32, size);
        defer allocator.free(a);
        const b = try allocator.alloc(f32, size);
        defer allocator.free(b);
        
        // Initialize with random data
        var rng = std.rand.DefaultPrng.init(42);
        const random = rng.random();
        for (a) |*val| val.* = random.float(f32);
        for (b) |*val| val.* = random.float(f32);

        // Benchmark distance calculations
        var timer = try Timer.start();
        
        // Warmup
        for (0..config.warmup_iterations) |_| {
            _ = try vector_ops.distance(.euclidean, a, b);
        }
        
        // Actual benchmark
        timer.reset();
        for (0..config.test_iterations) |_| {
            _ = try vector_ops.distance(.euclidean, a, b);
        }
        const euclidean_time = timer.read();
        
        timer.reset();
        for (0..config.test_iterations) |_| {
            _ = try vector_ops.distance(.cosine, a, b);
        }
        const cosine_time = timer.read();
        
        const euclidean_ops = @as(f64, @floatFromInt(config.test_iterations)) * 1e9 / @as(f64, @floatFromInt(euclidean_time));
        const cosine_ops = @as(f64, @floatFromInt(config.test_iterations)) * 1e9 / @as(f64, @floatFromInt(cosine_time));
        
        std.debug.print("Vector size {d}:\n", .{size});
        std.debug.print("  Euclidean: {d:.0} ops/sec\n", .{euclidean_ops});
        std.debug.print("  Cosine:    {d:.0} ops/sec\n", .{cosine_ops});
    }
}

fn benchmarkDatabaseOperations(allocator: std.mem.Allocator, config: BenchmarkConfig) !void {
    std.debug.print("\n📊 Database Operations Benchmark\n", .{});
    std.debug.print("-" ** 40 ++ "\n", .{});

    const tmp_dir = testing.tmpDir(.{});
    defer tmp_dir.cleanup();
    
    const db_path = try std.fmt.allocPrint(allocator, "{s}/benchmark.wdbx", .{tmp_dir.sub_path});
    defer allocator.free(db_path);

    for (config.vector_sizes) |dimensions| {
        const db = try core.Database.open(allocator, db_path, true);
        defer db.close();

        try db.init(.{
            .dimensions = @intCast(dimensions),
            .index_type = .hnsw,
            .distance_metric = .euclidean,
            .enable_simd = true,
        });

        // Generate test vectors
        var vectors = try allocator.alloc([]f32, config.batch_sizes[config.batch_sizes.len - 1]);
        defer {
            for (vectors) |vec| allocator.free(vec);
            allocator.free(vectors);
        }

        var rng = std.rand.DefaultPrng.init(42);
        const random = rng.random();
        
        for (vectors) |*vec| {
            vec.* = try allocator.alloc(f32, dimensions);
            for (vec.*) |*val| val.* = random.float(f32);
        }

        // Benchmark batch insertions
        for (config.batch_sizes) |batch_size| {
            const batch = vectors[0..batch_size];
            
            var timer = try Timer.start();
            for (batch) |vec| {
                _ = try db.addVector(vec, null);
            }
            const insert_time = timer.read();
            
            const insert_rate = @as(f64, @floatFromInt(batch_size)) * 1e9 / @as(f64, @floatFromInt(insert_time));
            
            std.debug.print("Batch size {d}, dimensions {d}:\n", .{ batch_size, dimensions });
            std.debug.print("  Insert rate: {d:.0} vectors/sec\n", .{insert_rate});
        }

        // Benchmark searches
        const query = vectors[0];
        
        var timer = try Timer.start();
        for (0..config.test_iterations) |_| {
            const results = try db.search(query, 10, allocator);
            allocator.free(results);
        }
        const search_time = timer.read();
        
        const search_rate = @as(f64, @floatFromInt(config.test_iterations)) * 1e9 / @as(f64, @floatFromInt(search_time));
        std.debug.print("  Search rate: {d:.0} searches/sec\n", .{search_rate});
    }
}

fn benchmarkIndexPerformance(allocator: std.mem.Allocator, config: BenchmarkConfig) !void {
    std.debug.print("\n📊 Index Performance Benchmark\n", .{});
    std.debug.print("-" ** 40 ++ "\n", .{});

    const dimensions = 128;
    const vector_count = 10000;

    // Benchmark Flat Index
    {
        var flat_index = try core.index.flat.FlatIndex.init(allocator, dimensions, .euclidean);
        defer flat_index.deinit();

        var rng = std.rand.DefaultPrng.init(42);
        const random = rng.random();

        // Add vectors
        var timer = try Timer.start();
        for (0..vector_count) |i| {
            var vec = try allocator.alloc(f32, dimensions);
            defer allocator.free(vec);
            for (vec) |*val| val.* = random.float(f32);
            try flat_index.add(i, vec);
        }
        const build_time = timer.read();

        // Search benchmark
        var query = try allocator.alloc(f32, dimensions);
        defer allocator.free(query);
        for (query) |*val| val.* = random.float(f32);

        timer.reset();
        for (0..config.test_iterations) |_| {
            const results = try flat_index.search(query, 10, allocator);
            allocator.free(results);
        }
        const search_time = timer.read();

        const build_rate = @as(f64, @floatFromInt(vector_count)) * 1e9 / @as(f64, @floatFromInt(build_time));
        const search_rate = @as(f64, @floatFromInt(config.test_iterations)) * 1e9 / @as(f64, @floatFromInt(search_time));

        std.debug.print("Flat Index (vectors={d}):\n", .{vector_count});
        std.debug.print("  Build rate:  {d:.0} vectors/sec\n", .{build_rate});
        std.debug.print("  Search rate: {d:.0} searches/sec\n", .{search_rate});
    }

    // Benchmark HNSW Index
    {
        var hnsw_index = try core.index.hnsw.HnswIndex.init(allocator, .{
            .dimensions = dimensions,
            .metric = .euclidean,
            .m = 16,
            .ef_construction = 200,
        });
        defer hnsw_index.deinit();

        var rng = std.rand.DefaultPrng.init(42);
        const random = rng.random();

        // Add vectors
        var timer = try Timer.start();
        for (0..vector_count) |i| {
            var vec = try allocator.alloc(f32, dimensions);
            defer allocator.free(vec);
            for (vec) |*val| val.* = random.float(f32);
            try hnsw_index.add(i, vec);
        }
        const build_time = timer.read();

        // Search benchmark
        var query = try allocator.alloc(f32, dimensions);
        defer allocator.free(query);
        for (query) |*val| val.* = random.float(f32);

        timer.reset();
        for (0..config.test_iterations) |_| {
            const results = try hnsw_index.search(query, 10, allocator);
            allocator.free(results);
        }
        const search_time = timer.read();

        const build_rate = @as(f64, @floatFromInt(vector_count)) * 1e9 / @as(f64, @floatFromInt(build_time));
        const search_rate = @as(f64, @floatFromInt(config.test_iterations)) * 1e9 / @as(f64, @floatFromInt(search_time));

        std.debug.print("HNSW Index (vectors={d}):\n", .{vector_count});
        std.debug.print("  Build rate:  {d:.0} vectors/sec\n", .{build_rate});
        std.debug.print("  Search rate: {d:.0} searches/sec\n", .{search_rate});
    }
}

fn benchmarkConcurrency(allocator: std.mem.Allocator, config: BenchmarkConfig) !void {
    std.debug.print("\n📊 Concurrency Benchmark\n", .{});
    std.debug.print("-" ** 40 ++ "\n", .{});

    const tmp_dir = testing.tmpDir(.{});
    defer tmp_dir.cleanup();
    
    const db_path = try std.fmt.allocPrint(allocator, "{s}/concurrent.wdbx", .{tmp_dir.sub_path});
    defer allocator.free(db_path);

    const db = try core.Database.open(allocator, db_path, true);
    defer db.close();

    try db.init(.{
        .dimensions = 128,
        .index_type = .hnsw,
        .distance_metric = .euclidean,
    });

    // Add initial vectors
    var rng = std.rand.DefaultPrng.init(42);
    const random = rng.random();
    
    for (0..1000) |_| {
        var vec = try allocator.alloc(f32, 128);
        defer allocator.free(vec);
        for (vec) |*val| val.* = random.float(f32);
        _ = try db.addVector(vec, null);
    }

    // Benchmark concurrent operations
    for (config.thread_counts) |thread_count| {
        const ThreadContext = struct {
            db: *core.Database,
            operations: usize,
            allocator: std.mem.Allocator,
        };

        const thread_fn = struct {
            fn run(ctx: ThreadContext) !void {
                var local_rng = std.rand.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
                const local_random = local_rng.random();
                
                for (0..ctx.operations) |_| {
                    var vec = try ctx.allocator.alloc(f32, 128);
                    defer ctx.allocator.free(vec);
                    for (vec) |*val| val.* = local_random.float(f32);
                    
                    // Mix of operations
                    if (local_random.boolean()) {
                        _ = try ctx.db.addVector(vec, null);
                    } else {
                        const results = try ctx.db.search(vec, 5, ctx.allocator);
                        ctx.allocator.free(results);
                    }
                }
            }
        }.run;

        var timer = try Timer.start();
        var threads = try allocator.alloc(std.Thread, thread_count);
        defer allocator.free(threads);

        const ops_per_thread = 1000 / thread_count;
        for (threads) |*thread| {
            thread.* = try std.Thread.spawn(.{}, thread_fn, .{
                ThreadContext{
                    .db = db,
                    .operations = ops_per_thread,
                    .allocator = allocator,
                },
            });
        }

        for (threads) |thread| {
            thread.join();
        }

        const elapsed = timer.read();
        const total_ops = thread_count * ops_per_thread;
        const ops_per_sec = @as(f64, @floatFromInt(total_ops)) * 1e9 / @as(f64, @floatFromInt(elapsed));

        std.debug.print("Threads {d}:\n", .{thread_count});
        std.debug.print("  Operations/sec: {d:.0}\n", .{ops_per_sec});
        std.debug.print("  Speedup: {d:.2}x\n", .{ops_per_sec / (ops_per_sec / @as(f64, @floatFromInt(thread_count)))});
    }
}

fn benchmarkMemoryEfficiency(allocator: std.mem.Allocator, config: BenchmarkConfig) !void {
    std.debug.print("\n📊 Memory Efficiency Benchmark\n", .{});
    std.debug.print("-" ** 40 ++ "\n", .{});

    const dimensions = 384;
    const vector_counts = [_]usize{ 1000, 10000, 50000 };

    for (vector_counts) |count| {
        const tmp_dir = testing.tmpDir(.{});
        defer tmp_dir.cleanup();
        
        const db_path = try std.fmt.allocPrint(allocator, "{s}/memory_test.wdbx", .{tmp_dir.sub_path});
        defer allocator.free(db_path);

        // Track memory usage
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const arena_allocator = arena.allocator();

        const db = try core.Database.open(arena_allocator, db_path, true);
        defer db.close();

        try db.init(.{
            .dimensions = dimensions,
            .index_type = .hnsw,
            .distance_metric = .euclidean,
        });

        var rng = std.rand.DefaultPrng.init(42);
        const random = rng.random();

        // Add vectors and track memory
        const start_memory = arena.queryCapacity();
        
        for (0..count) |_| {
            var vec = try arena_allocator.alloc(f32, dimensions);
            for (vec) |*val| val.* = random.float(f32);
            _ = try db.addVector(vec, null);
        }

        const end_memory = arena.queryCapacity();
        const memory_per_vector = (end_memory - start_memory) / count;

        std.debug.print("Vectors {d} (dim={d}):\n", .{ count, dimensions });
        std.debug.print("  Memory per vector: {d} bytes\n", .{memory_per_vector});
        std.debug.print("  Total memory: {d:.2} MB\n", .{@as(f64, @floatFromInt(end_memory)) / (1024 * 1024)});
        
        // Get database stats
        const stats = db.getStats();
        std.debug.print("  Index memory: {d:.2} MB\n", .{@as(f64, @floatFromInt(stats.index_memory_bytes)) / (1024 * 1024)});
    }
}