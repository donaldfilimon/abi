//! Integration tests for WDBX-AI

const std = @import("std");
const testing = std.testing;
const wdbx = @import("wdbx");
const test_framework = @import("test_framework.zig");

test "integration: full system initialization" {
    const allocator = testing.allocator;
    
    // Initialize WDBX system
    try wdbx.init(allocator);
    defer wdbx.deinit();
    
    // Verify system info
    const info = wdbx.getSystemInfo();
    try testing.expectEqualStrings("2.0.0", info.version);
    try testing.expect(info.features.len > 0);
}

test "integration: vector database workflow" {
    const allocator = testing.allocator;
    
    var ctx = try test_framework.TestContext.init(allocator);
    defer ctx.deinit();
    
    // Initialize system
    try wdbx.init(allocator);
    defer wdbx.deinit();
    
    // Create database
    const db_path = try ctx.getTempPath("integration.db");
    defer allocator.free(db_path);
    
    const db = try wdbx.database.create(allocator, db_path);
    defer db.deinit();
    
    // Generate test vectors
    const vectors = try test_framework.TestData.generateNormalizedVectors(allocator, 100, 128);
    defer test_framework.TestData.freeVectors(allocator, vectors);
    
    // Insert vectors with metadata
    var ids = std.ArrayList([]const u8).init(allocator);
    defer {
        for (ids.items) |id| {
            allocator.free(id);
        }
        ids.deinit();
    }
    
    for (vectors, 0..) |vector, i| {
        const id = try std.fmt.allocPrint(allocator, "vec_{d}", .{i});
        try ids.append(id);
        
        const data = std.mem.sliceAsBytes(vector);
        _ = try db.writeRecord(data);
    }
    
    // Perform similarity search
    const query = vectors[0];
    const results = try wdbx.database.BatchOps.readBatch(db, &[_]u64{ 1, 2, 3, 4, 5 }, allocator);
    defer allocator.free(results);
    
    // Verify results
    try testing.expect(results.len > 0);
    for (results) |result| {
        try testing.expect(result != null);
    }
}

test "integration: concurrent operations with thread pool" {
    const allocator = testing.allocator;
    
    var ctx = try test_framework.TestContext.init(allocator);
    defer ctx.deinit();
    
    // Initialize system
    try wdbx.init(allocator);
    defer wdbx.deinit();
    
    // Create database
    const db_path = try ctx.getTempPath("concurrent.db");
    defer allocator.free(db_path);
    
    const db = try wdbx.database.create(allocator, db_path);
    defer db.deinit();
    
    // Create thread pool
    var pool = try wdbx.core.threading.ThreadPool.init(allocator, null);
    defer pool.deinit();
    
    // Generate test data
    const vectors = try test_framework.TestData.generateClusteredVectors(allocator, 5, 20, 64);
    defer test_framework.TestData.freeVectors(allocator, vectors);
    
    // Parallel insert
    const InsertTask = struct {
        fn insert(args: *struct { db: *wdbx.database.Db, vector: []f32, index: usize }) void {
            const data = std.mem.sliceAsBytes(args.vector);
            _ = args.db.writeRecord(data) catch |err| {
                std.log.err("Insert failed: {}", .{err});
            };
        }
    };
    
    var tasks = try allocator.alloc(
        struct { db: *wdbx.database.Db, vector: []f32, index: usize },
        vectors.len
    );
    defer allocator.free(tasks);
    
    for (vectors, 0..) |vector, i| {
        tasks[i] = .{ .db = db, .vector = vector, .index = i };
        try pool.submit(InsertTask.insert, &tasks[i]);
    }
    
    pool.wait();
    
    // Verify all vectors were inserted
    var count: u64 = 0;
    var id: u64 = 1;
    while (id <= vectors.len) : (id += 1) {
        if (try db.readRecord(id)) |_| {
            count += 1;
        }
    }
    try testing.expectEqual(@as(u64, vectors.len), count);
}

test "integration: error handling and recovery" {
    const allocator = testing.allocator;
    
    var ctx = try test_framework.TestContext.init(allocator);
    defer ctx.deinit();
    
    // Initialize error handler
    var handler = wdbx.errors.ErrorHandler.init(allocator);
    defer handler.deinit();
    
    var error_count: usize = 0;
    const errorHandler = struct {
        fn handle(info: wdbx.errors.ErrorInfo) void {
            _ = info;
            error_count += 1;
        }
    }.handle;
    
    try handler.register(wdbx.errors.WdbxError.FileNotFound, errorHandler);
    
    // Test error aggregation
    var aggregator = wdbx.errors.ErrorAggregator.init(allocator);
    defer aggregator.deinit();
    
    // Simulate multiple operations with some failures
    const operations = [_]struct { name: []const u8, should_fail: bool }{
        .{ .name = "op1", .should_fail = false },
        .{ .name = "op2", .should_fail = true },
        .{ .name = "op3", .should_fail = false },
        .{ .name = "op4", .should_fail = true },
    };
    
    for (operations) |op| {
        if (op.should_fail) {
            try aggregator.addError(
                wdbx.errors.WdbxError.InternalError,
                op.name
            );
        }
    }
    
    try testing.expectEqual(@as(usize, 2), aggregator.count());
}

test "integration: configuration and logging" {
    const allocator = testing.allocator;
    
    var ctx = try test_framework.TestContext.init(allocator);
    defer ctx.deinit();
    
    // Create config file
    const config_path = try ctx.getTempPath("test_config.toml");
    defer allocator.free(config_path);
    
    var config = wdbx.core.config.Config.init(allocator);
    defer config.deinit();
    
    // Set configuration values
    try config.setString("app.name", "WDBX-Test");
    try config.setInt("database.cache_size", 50 * 1024 * 1024);
    try config.setBool("performance.use_simd", true);
    try config.setInt("performance.thread_count", 4);
    
    // Save config
    try config.save(config_path);
    
    // Load config in new instance
    var loaded_config = try wdbx.core.config.Config.load(allocator, config_path);
    defer loaded_config.deinit();
    
    // Verify values
    try testing.expectEqualStrings("WDBX-Test", loaded_config.getString("app.name").?);
    try testing.expectEqual(@as(i64, 50 * 1024 * 1024), loaded_config.getInt("database.cache_size").?);
    
    // Setup logging based on config
    const log_path = try ctx.getTempPath("test.log");
    defer allocator.free(log_path);
    
    var logger = try wdbx.core.logging.Logger.init(allocator, .{
        .level = .info,
        .output = .{ .file = log_path },
    });
    defer logger.deinit();
    
    // Test scoped logging
    const db_logger = wdbx.core.logging.ScopedLogger.init(&logger, "database");
    const ai_logger = wdbx.core.logging.ScopedLogger.init(&logger, "ai");
    
    db_logger.info(@src(), "Database initialized", .{});
    ai_logger.info(@src(), "AI module loaded", .{});
}

test "integration: memory tracking" {
    // Create tracked allocator
    var base_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = base_allocator.deinit();
    
    var tracked = wdbx.memory_tracker.TrackedAllocator.init(base_allocator.allocator());
    defer {
        const stats = tracked.getStats();
        std.log.info("Memory stats - Peak: {} bytes, Current: {} bytes, Allocations: {}", .{
            stats.peak_bytes,
            stats.current_bytes,
            stats.total_allocations,
        });
        tracked.deinit();
    }
    
    const allocator = tracked.allocator();
    
    // Perform operations with memory tracking
    const data1 = try allocator.alloc(u8, 1024);
    defer allocator.free(data1);
    
    const data2 = try allocator.alloc(u8, 2048);
    defer allocator.free(data2);
    
    // Verify tracking
    const stats = tracked.getStats();
    try testing.expect(stats.total_allocations >= 2);
    try testing.expect(stats.peak_bytes >= 3072);
}

test "integration: performance monitoring" {
    const allocator = testing.allocator;
    
    // Initialize performance monitor
    var monitor = try wdbx.performance.Monitor.init(allocator);
    defer monitor.deinit();
    
    // Simulate operations with timing
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const start = std.time.nanoTimestamp();
        
        // Simulate work
        std.time.sleep(100 * std.time.ns_per_us); // 100 microseconds
        
        const end = std.time.nanoTimestamp();
        const latency_ns = @as(u64, @intCast(end - start));
        
        try monitor.recordLatency("test_operation", @as(f64, @floatFromInt(latency_ns)) / 1_000_000.0);
    }
    
    // Get statistics
    const stats = try monitor.getStats("test_operation");
    try testing.expect(stats.count == 100);
    try testing.expect(stats.avg > 0);
    try testing.expect(stats.min <= stats.avg);
    try testing.expect(stats.max >= stats.avg);
}

test "integration: end-to-end workflow" {
    const allocator = testing.allocator;
    
    var ctx = try test_framework.TestContext.init(allocator);
    defer ctx.deinit();
    
    // Initialize system
    try wdbx.init(allocator);
    defer wdbx.deinit();
    
    // Load configuration
    var config = wdbx.core.config.Config.init(allocator);
    defer config.deinit();
    try config.setDefaults();
    
    // Create logger
    var logger = try wdbx.core.logging.Logger.init(allocator, .{
        .level = .info,
        .output = .stdout,
    });
    defer logger.deinit();
    
    logger.info(@src(), "Starting end-to-end test", .{});
    
    // Create database
    const db_path = try ctx.getTempPath("e2e.db");
    defer allocator.free(db_path);
    
    const db = try wdbx.database.create(allocator, db_path);
    defer db.deinit();
    
    // Create thread pool
    var pool = try wdbx.core.threading.ThreadPool.init(allocator, 4);
    defer pool.deinit();
    
    // Generate test data
    const vectors = try test_framework.TestData.generateNormalizedVectors(allocator, 50, 256);
    defer test_framework.TestData.freeVectors(allocator, vectors);
    
    // Performance monitoring
    var perf = wdbx.core.logging.PerfLogger.begin(&logger, "e2e_test");
    
    // Insert vectors
    for (vectors) |vector| {
        const data = std.mem.sliceAsBytes(vector);
        _ = try db.writeRecord(data);
    }
    
    perf.checkpoint("insert_complete");
    
    // Optimize database
    try wdbx.database.Optimizer.optimize(db);
    
    perf.checkpoint("optimization_complete");
    
    // Perform queries
    const query_count = 10;
    var query_i: usize = 0;
    while (query_i < query_count) : (query_i += 1) {
        const ids = [_]u64{ 1, 2, 3, 4, 5 };
        const results = try wdbx.database.BatchOps.readBatch(db, &ids, allocator);
        defer allocator.free(results);
        
        try testing.expect(results.len == ids.len);
    }
    
    perf.end();
    
    logger.info(@src(), "End-to-end test completed successfully", .{});
}