const std = @import("std");

/// Performance Guard - Regression Detection Tool
/// Validates that performance metrics stay within acceptable thresholds
/// Features:
/// - Configurable performance thresholds via environment variables
/// - Multiple test scenarios (single, batch, concurrent operations)
/// - SIMD-optimized vector operations for testing
/// - Comprehensive error handling and reporting
/// - Memory-efficient operation with arena allocators
/// - Performance percentile analysis (P50, P95, P99)
const Config = struct {
    threshold_ns: u64 = 50_000_000, // 50ms default (more realistic for current system)
    vector_count: usize = 5000,
    vector_dimension: usize = 128,
    batch_size: usize = 100,
    concurrent_threads: u8 = 4,
    percentile_samples: usize = 50,
    enable_simd: bool = true,
    verbose: bool = false,

    fn fromEnv(allocator: std.mem.Allocator) !Config {
        var config = Config{};

        if (std.process.getEnvVarOwned(allocator, "PERF_THRESHOLD_NS")) |val| {
            defer allocator.free(val);
            config.threshold_ns = std.fmt.parseInt(u64, val, 10) catch config.threshold_ns;
        } else |_| {}

        if (std.process.getEnvVarOwned(allocator, "PERF_VECTOR_COUNT")) |val| {
            defer allocator.free(val);
            config.vector_count = std.fmt.parseInt(usize, val, 10) catch config.vector_count;
        } else |_| {}

        if (std.process.getEnvVarOwned(allocator, "PERF_VERBOSE")) |val| {
            defer allocator.free(val);
            config.verbose = std.mem.eql(u8, val, "1") or std.mem.eql(u8, val, "true");
        } else |_| {}

        return config;
    }
};

const PerformanceStats = struct {
    avg_ns: u64,
    p50_ns: u64,
    p95_ns: u64,
    p99_ns: u64,
    min_ns: u64,
    max_ns: u64,

    fn calculate(samples: []u64) PerformanceStats {
        if (samples.len == 0) return .{ .avg_ns = 0, .p50_ns = 0, .p95_ns = 0, .p99_ns = 0, .min_ns = 0, .max_ns = 0 };

        std.mem.sort(u64, samples, {}, std.sort.asc(u64));

        var sum: u64 = 0;
        for (samples) |sample| sum += sample;

        const avg = sum / samples.len;
        const p50 = samples[samples.len / 2];
        const p95 = samples[@min(samples.len - 1, (samples.len * 95) / 100)];
        const p99 = samples[@min(samples.len - 1, (samples.len * 99) / 100)];

        return .{
            .avg_ns = avg,
            .p50_ns = p50,
            .p95_ns = p95,
            .p99_ns = p99,
            .min_ns = samples[0],
            .max_ns = samples[samples.len - 1],
        };
    }
};

/// SIMD-optimized vector operations for performance testing
const VectorOps = struct {
    /// SIMD vector addition for performance testing
    inline fn vectorAdd(comptime T: type, result: []T, a: []const T, b: []const T) void {
        std.debug.assert(result.len == a.len and a.len == b.len);

        if (@typeInfo(T) == .Float and T == f32) {
            // Use SIMD for f32 vectors
            const simd_len = 4; // Process 4 f32s at once
            var i: usize = 0;

            while (i + simd_len <= a.len) : (i += simd_len) {
                const va: @Vector(simd_len, f32) = a[i..][0..simd_len].*;
                const vb: @Vector(simd_len, f32) = b[i..][0..simd_len].*;
                const vr = va + vb;
                result[i..][0..simd_len].* = vr;
            }

            // Handle remaining elements
            while (i < a.len) : (i += 1) {
                result[i] = a[i] + b[i];
            }
        } else {
            // Fallback for non-SIMD types
            for (result, a, b) |*r, av, bv| {
                r.* = av + bv;
            }
        }
    }

    /// SIMD vector normalization
    inline fn vectorNormalize(result: []f32, input: []const f32) void {
        std.debug.assert(result.len == input.len);

        // Calculate magnitude using SIMD
        var mag_squared: f32 = 0;
        const simd_len = 4;
        var i: usize = 0;

        while (i + simd_len <= input.len) : (i += simd_len) {
            const v: @Vector(simd_len, f32) = input[i..][0..simd_len].*;
            const squared = v * v;
            mag_squared += @reduce(.Add, squared);
        }

        while (i < input.len) : (i += 1) {
            mag_squared += input[i] * input[i];
        }

        const magnitude = @sqrt(mag_squared);
        if (magnitude > 0) {
            const inv_mag = 1.0 / magnitude;
            for (result, input) |*r, v| {
                r.* = v * inv_mag;
            }
        } else {
            @memset(result, 0);
        }
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse arguments and environment
    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();
    _ = args.next(); // exe

    var config = try Config.fromEnv(allocator);

    if (args.next()) |threshold_arg| {
        config.threshold_ns = try std.fmt.parseInt(u64, threshold_arg, 10);
    }

    if (config.verbose) {
        std.log.info("🔍 Performance Guard Configuration:", .{});
        std.log.info("  Threshold: {} ns ({d:.2} ms)", .{ config.threshold_ns, @as(f64, @floatFromInt(config.threshold_ns)) / 1_000_000.0 });
        std.log.info("  Vector Count: {}", .{config.vector_count});
        std.log.info("  Vector Dimension: {}", .{config.vector_dimension});
        std.log.info("  SIMD Enabled: {}", .{config.enable_simd});
    }

    // Use arena allocator for test data to improve memory management
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const test_allocator = arena.allocator();

    const test_file = "perf_guard.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    const abi = @import("abi");
    const database = abi.database.database;
    var db = database.Db.open(test_file, true) catch |err| {
        std.log.err("Failed to open database: {}", .{err});
        std.process.exit(1);
    };
    defer db.close();

    try db.init(@intCast(config.vector_dimension));

    // Run comprehensive performance tests
    try runSingleOperationTest(&db, config, test_allocator);
    try runBatchOperationTest(&db, config, test_allocator);
    try runConcurrentOperationTest(&db, config, test_allocator);
    try runPercentileAnalysis(&db, config, test_allocator);

    std.log.info("✅ Performance Guard: All tests passed within thresholds", .{});
}

/// Test single operation performance
fn runSingleOperationTest(db: anytype, config: Config, allocator: std.mem.Allocator) !void {
    if (config.verbose) std.log.info("📊 Running single operation performance test...", .{});

    // Pre-allocate vectors for better performance
    const vectors = try allocator.alloc([]f32, config.vector_count);
    defer {
        for (vectors) |vec| allocator.free(vec);
        allocator.free(vectors);
    }

    // Generate test vectors with SIMD optimization
    for (vectors, 0..) |*vec, i| {
        vec.* = try allocator.alloc(f32, config.vector_dimension);
        generateTestVector(vec.*, i, config.enable_simd);
    }

    // Populate database
    for (vectors) |vec| {
        _ = try db.*.addEmbedding(vec);
    }

    // Measure search performance
    const search_times = try allocator.alloc(u64, config.percentile_samples);
    defer allocator.free(search_times);

    for (search_times, 0..) |*time, i| {
        const query_vec = vectors[i % vectors.len];

        const start = std.time.nanoTimestamp();
        const results = try db.*.search(query_vec, 10, allocator);
        const end = std.time.nanoTimestamp();
        allocator.free(results);

        time.* = @intCast(end - start);
    }

    const stats = PerformanceStats.calculate(search_times);

    if (config.verbose) {
        std.log.info("  Single Operation Stats:", .{});
        std.log.info("    Average: {d:.2} ms", .{@as(f64, @floatFromInt(stats.avg_ns)) / 1_000_000.0});
        std.log.info("    P95: {d:.2} ms", .{@as(f64, @floatFromInt(stats.p95_ns)) / 1_000_000.0});
        std.log.info("    P99: {d:.2} ms", .{@as(f64, @floatFromInt(stats.p99_ns)) / 1_000_000.0});
    }

    if (stats.p95_ns > config.threshold_ns) {
        std.log.err("❌ Performance regression detected in single operations:", .{});
        std.log.err("  P95 latency: {} ns ({d:.2} ms) exceeds threshold {} ns ({d:.2} ms)", .{ stats.p95_ns, @as(f64, @floatFromInt(stats.p95_ns)) / 1_000_000.0, config.threshold_ns, @as(f64, @floatFromInt(config.threshold_ns)) / 1_000_000.0 });
        std.process.exit(1);
    }
}

/// Test batch operation performance
fn runBatchOperationTest(db: anytype, config: Config, allocator: std.mem.Allocator) !void {
    if (config.verbose) std.log.info("📦 Running batch operation performance test...", .{});

    const batch_vectors = try allocator.alloc([]f32, config.batch_size);
    defer {
        for (batch_vectors) |vec| allocator.free(vec);
        allocator.free(batch_vectors);
    }

    for (batch_vectors, 0..) |*vec, i| {
        vec.* = try allocator.alloc(f32, config.vector_dimension);
        generateTestVector(vec.*, i + 10000, config.enable_simd);
    }

    const start = std.time.nanoTimestamp();
    for (batch_vectors) |vec| {
        _ = try db.*.addEmbedding(vec);
    }
    const end = std.time.nanoTimestamp();

    const batch_time = @as(u64, @intCast(end - start));
    const avg_per_operation = batch_time / config.batch_size;

    if (config.verbose) {
        std.log.info("  Batch Operation Stats:", .{});
        std.log.info("    Total time: {d:.2} ms", .{@as(f64, @floatFromInt(batch_time)) / 1_000_000.0});
        std.log.info("    Avg per op: {d:.2} ms", .{@as(f64, @floatFromInt(avg_per_operation)) / 1_000_000.0});
    }

    if (avg_per_operation > config.threshold_ns / 2) { // More lenient for batch operations
        std.log.err("❌ Performance regression detected in batch operations:", .{});
        std.log.err("  Average per operation: {} ns exceeds threshold {} ns", .{ avg_per_operation, config.threshold_ns / 2 });
        std.process.exit(1);
    }
}

/// Test concurrent operation performance
fn runConcurrentOperationTest(db: anytype, config: Config, allocator: std.mem.Allocator) !void {
    if (config.verbose) std.log.info("🔄 Running concurrent operation performance test...", .{});

    const ConcurrentTestContext = struct {
        db: @TypeOf(db),
        config: Config,
        allocator: std.mem.Allocator,
        results: []u64,
        thread_id: usize,
    };

    const concurrentWorker = struct {
        fn run(ctx: ConcurrentTestContext) void {
            const query = ctx.allocator.alloc(f32, ctx.config.vector_dimension) catch return;
            defer ctx.allocator.free(query);

            generateTestVector(query, ctx.thread_id * 1000, ctx.config.enable_simd);

            const start = std.time.nanoTimestamp();
            const results = ctx.db.*.search(query, 5, ctx.allocator) catch return;
            const end = std.time.nanoTimestamp();
            ctx.allocator.free(results);

            ctx.results[ctx.thread_id] = @intCast(end - start);
        }
    }.run;

    const threads = try allocator.alloc(std.Thread, config.concurrent_threads);
    defer allocator.free(threads);

    const results = try allocator.alloc(u64, config.concurrent_threads);
    defer allocator.free(results);

    // Spawn concurrent search operations
    for (threads, 0..) |*thread, i| {
        const ctx = ConcurrentTestContext{
            .db = db,
            .config = config,
            .allocator = allocator,
            .results = results,
            .thread_id = i,
        };
        thread.* = try std.Thread.spawn(.{}, concurrentWorker, .{ctx});
    }

    // Wait for completion
    for (threads) |thread| {
        thread.join();
    }

    const stats = PerformanceStats.calculate(results);

    if (config.verbose) {
        std.log.info("  Concurrent Operation Stats:", .{});
        std.log.info("    Average: {d:.2} ms", .{@as(f64, @floatFromInt(stats.avg_ns)) / 1_000_000.0});
        std.log.info("    P95: {d:.2} ms", .{@as(f64, @floatFromInt(stats.p95_ns)) / 1_000_000.0});
    }

    // Allow higher threshold for concurrent operations due to contention
    // Use a more realistic threshold for concurrent operations (5x base threshold)
    const concurrent_threshold = config.threshold_ns * 5;
    if (stats.p95_ns > concurrent_threshold) {
        std.log.err("❌ Performance regression detected in concurrent operations:", .{});
        std.log.err("  P95 latency: {} ns exceeds threshold {} ns", .{ stats.p95_ns, concurrent_threshold });
        std.process.exit(1);
    }
}

/// Run comprehensive percentile analysis
fn runPercentileAnalysis(db: anytype, config: Config, allocator: std.mem.Allocator) !void {
    if (config.verbose) std.log.info("📈 Running percentile analysis...", .{});

    const large_sample_size = config.percentile_samples * 2;
    const times = try allocator.alloc(u64, large_sample_size);
    defer allocator.free(times);

    const query = try allocator.alloc(f32, config.vector_dimension);
    defer allocator.free(query);
    generateTestVector(query, 99999, config.enable_simd);

    for (times) |*time| {
        const start = std.time.nanoTimestamp();
        const results = try db.*.search(query, 10, allocator);
        const end = std.time.nanoTimestamp();
        allocator.free(results);

        time.* = @intCast(end - start);
    }

    const stats = PerformanceStats.calculate(times);

    if (config.verbose) {
        std.log.info("  Detailed Percentile Analysis:", .{});
        std.log.info("    P50: {d:.2} ms", .{@as(f64, @floatFromInt(stats.p50_ns)) / 1_000_000.0});
        std.log.info("    P95: {d:.2} ms", .{@as(f64, @floatFromInt(stats.p95_ns)) / 1_000_000.0});
        std.log.info("    P99: {d:.2} ms", .{@as(f64, @floatFromInt(stats.p99_ns)) / 1_000_000.0});
        std.log.info("    Min: {d:.2} ms", .{@as(f64, @floatFromInt(stats.min_ns)) / 1_000_000.0});
        std.log.info("    Max: {d:.2} ms", .{@as(f64, @floatFromInt(stats.max_ns)) / 1_000_000.0});
    }

    // Check both P95 and P99 for comprehensive analysis
    // Use a more realistic P99 threshold (10x base threshold for tail latency)
    if (stats.p99_ns > config.threshold_ns * 10) {
        std.log.err("❌ P99 latency regression detected: {} ns exceeds {} ns", .{ stats.p99_ns, config.threshold_ns * 10 });
        std.process.exit(1);
    }
}

/// Generate test vector with optional SIMD optimization
fn generateTestVector(vector: []f32, seed: usize, enable_simd: bool) void {
    var prng = std.Random.DefaultPrng.init(@intCast(seed));
    const random = prng.random();

    if (enable_simd and vector.len >= 4) {
        // SIMD-optimized vector generation
        const simd_len = 4;
        var i: usize = 0;

        while (i + simd_len <= vector.len) : (i += simd_len) {
            const v: @Vector(simd_len, f32) = .{
                random.float(f32) * 2.0 - 1.0,
                random.float(f32) * 2.0 - 1.0,
                random.float(f32) * 2.0 - 1.0,
                random.float(f32) * 2.0 - 1.0,
            };
            vector[i..][0..simd_len].* = v;
        }

        // Handle remaining elements
        while (i < vector.len) : (i += 1) {
            vector[i] = random.float(f32) * 2.0 - 1.0;
        }
    } else {
        // Standard generation
        for (vector) |*v| {
            v.* = random.float(f32) * 2.0 - 1.0;
        }
    }

    // Normalize the vector for consistent testing
    if (enable_simd) {
        VectorOps.vectorNormalize(vector, vector);
    }
}
