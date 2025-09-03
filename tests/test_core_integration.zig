//! Core Integration Tests
//!
//! This file contains comprehensive integration tests for the core database functionality.

const std = @import("std");
const testing = std.testing;
const core = @import("core");

test "Database lifecycle" {
    const allocator = testing.allocator;
    
    // Create temporary directory
    const tmp_dir = testing.tmpDir(.{});
    defer tmp_dir.cleanup();
    
    const db_path = try std.fmt.allocPrint(allocator, "{s}/lifecycle_test.wdbx", .{tmp_dir.sub_path});
    defer allocator.free(db_path);

    // Test database creation
    {
        const db = try core.Database.open(allocator, db_path, true);
        defer db.close();

        try testing.expect(!db.is_initialized);

        // Initialize with configuration
        try db.init(.{
            .dimensions = 128,
            .index_type = .hnsw,
            .distance_metric = .euclidean,
            .enable_simd = true,
            .hnsw_m = 16,
            .hnsw_ef_construction = 200,
        });

        try testing.expect(db.is_initialized);
        try testing.expectEqual(@as(u32, 128), db.config.dimensions);
    }

    // Test database reopening
    {
        const db = try core.Database.open(allocator, db_path, false);
        defer db.close();

        try testing.expect(db.is_initialized);
        try testing.expectEqual(@as(u32, 128), db.config.dimensions);
        try testing.expectEqual(core.index.IndexType.hnsw, db.config.index_type);
    }
}

test "Vector operations with different dimensions" {
    const allocator = testing.allocator;
    
    const dimensions = [_]u32{ 64, 128, 256, 384, 512, 768, 1024 };
    
    for (dimensions) |dim| {
        const tmp_dir = testing.tmpDir(.{});
        defer tmp_dir.cleanup();
        
        const db_path = try std.fmt.allocPrint(allocator, "{s}/dim_{d}.wdbx", .{ tmp_dir.sub_path, dim });
        defer allocator.free(db_path);

        const db = try core.Database.open(allocator, db_path, true);
        defer db.close();

        try db.init(.{
            .dimensions = dim,
            .index_type = .flat,
            .distance_metric = .euclidean,
        });

        // Create test vectors
        const v1 = try allocator.alloc(f32, dim);
        defer allocator.free(v1);
        const v2 = try allocator.alloc(f32, dim);
        defer allocator.free(v2);

        // Initialize with pattern
        for (v1, 0..) |*val, i| {
            val.* = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(dim));
        }
        for (v2, 0..) |*val, i| {
            val.* = 1.0 - @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(dim));
        }

        // Add vectors
        const id1 = try db.addVector(v1, null);
        const id2 = try db.addVector(v2, null);

        try testing.expect(id1 != id2);

        // Search
        const results = try db.search(v1, 2, allocator);
        defer allocator.free(results);

        try testing.expectEqual(@as(usize, 2), results.len);
        try testing.expectEqual(id1, results[0].index);
    }
}

test "Concurrent operations" {
    const allocator = testing.allocator;
    
    const tmp_dir = testing.tmpDir(.{});
    defer tmp_dir.cleanup();
    
    const db_path = try std.fmt.allocPrint(allocator, "{s}/concurrent_test.wdbx", .{tmp_dir.sub_path});
    defer allocator.free(db_path);

    const db = try core.Database.open(allocator, db_path, true);
    defer db.close();

    try db.init(.{
        .dimensions = 64,
        .index_type = .hnsw,
        .distance_metric = .euclidean,
    });

    // Spawn multiple threads
    const thread_count = 4;
    const vectors_per_thread = 100;
    
    const ThreadContext = struct {
        db: *core.Database,
        thread_id: u32,
        allocator: std.mem.Allocator,
    };

    const thread_fn = struct {
        fn run(ctx: ThreadContext) !void {
            var rng = std.rand.DefaultPrng.init(ctx.thread_id);
            const random = rng.random();

            var i: usize = 0;
            while (i < vectors_per_thread) : (i += 1) {
                const vec = try ctx.allocator.alloc(f32, 64);
                defer ctx.allocator.free(vec);

                for (vec) |*val| {
                    val.* = random.float(f32);
                }

                _ = try ctx.db.addVector(vec, null);
            }
        }
    }.run;

    var threads: [thread_count]std.Thread = undefined;
    
    for (&threads, 0..) |*thread, i| {
        thread.* = try std.Thread.spawn(.{}, thread_fn, .{
            ThreadContext{
                .db = db,
                .thread_id = @intCast(i),
                .allocator = allocator,
            },
        });
    }

    for (threads) |thread| {
        thread.join();
    }

    // Verify all vectors were added
    const stats = db.getStats();
    try testing.expectEqual(@as(u64, thread_count * vectors_per_thread), stats.total_vectors);
}

test "Distance metrics comparison" {
    const allocator = testing.allocator;
    
    const metrics = [_]core.vector.DistanceMetric{
        .euclidean,
        .cosine,
        .dot_product,
        .manhattan,
    };

    for (metrics) |metric| {
        const tmp_dir = testing.tmpDir(.{});
        defer tmp_dir.cleanup();
        
        const db_path = try std.fmt.allocPrint(allocator, "{s}/{s}_test.wdbx", .{
            tmp_dir.sub_path,
            @tagName(metric),
        });
        defer allocator.free(db_path);

        const db = try core.Database.open(allocator, db_path, true);
        defer db.close();

        try db.init(.{
            .dimensions = 3,
            .index_type = .flat,
            .distance_metric = metric,
        });

        // Add orthogonal vectors
        const v1 = [_]f32{ 1.0, 0.0, 0.0 };
        const v2 = [_]f32{ 0.0, 1.0, 0.0 };
        const v3 = [_]f32{ 0.0, 0.0, 1.0 };

        _ = try db.addVector(&v1, null);
        _ = try db.addVector(&v2, null);
        _ = try db.addVector(&v3, null);

        // Search with a query close to v1
        const query = [_]f32{ 0.9, 0.1, 0.0 };
        const results = try db.search(&query, 3, allocator);
        defer allocator.free(results);

        try testing.expectEqual(@as(usize, 3), results.len);
        
        // First result should always be closest to v1
        try testing.expectEqual(@as(u64, 0), results[0].index);
    }
}

test "Index type comparison" {
    const allocator = testing.allocator;
    
    const index_types = [_]core.index.IndexType{
        .flat,
        .hnsw,
    };

    // Generate test data
    const vector_count = 1000;
    const dimensions = 64;
    var rng = std.rand.DefaultPrng.init(42);
    const random = rng.random();

    var test_vectors = try allocator.alloc([]f32, vector_count);
    defer {
        for (test_vectors) |vec| {
            allocator.free(vec);
        }
        allocator.free(test_vectors);
    }

    for (test_vectors) |*vec| {
        vec.* = try allocator.alloc(f32, dimensions);
        for (vec.*) |*val| {
            val.* = random.float(f32);
        }
    }

    // Test each index type
    for (index_types) |index_type| {
        const tmp_dir = testing.tmpDir(.{});
        defer tmp_dir.cleanup();
        
        const db_path = try std.fmt.allocPrint(allocator, "{s}/{s}_index.wdbx", .{
            tmp_dir.sub_path,
            @tagName(index_type),
        });
        defer allocator.free(db_path);

        const db = try core.Database.open(allocator, db_path, true);
        defer db.close();

        try db.init(.{
            .dimensions = dimensions,
            .index_type = index_type,
            .distance_metric = .euclidean,
        });

        // Add all vectors
        for (test_vectors) |vec| {
            _ = try db.addVector(vec, null);
        }

        // Perform searches and measure time
        const query_count = 10;
        var total_time: i64 = 0;
        
        var i: usize = 0;
        while (i < query_count) : (i += 1) {
            const query_idx = random.intRangeAtMost(usize, 0, vector_count - 1);
            const query = test_vectors[query_idx];
            
            const start = std.time.nanoTimestamp();
            const results = try db.search(query, 10, allocator);
            const end = std.time.nanoTimestamp();
            defer allocator.free(results);
            
            total_time += end - start;
            
            // Verify self is found first
            try testing.expectEqual(@as(u64, query_idx), results[0].index);
            try testing.expect(results[0].score < 0.001); // Should be very close to 0
        }

        const avg_time_ms = @as(f64, @floatFromInt(total_time)) / @as(f64, query_count) / 1_000_000.0;
        std.debug.print("{s} index: avg query time = {d:.2} ms\n", .{ @tagName(index_type), avg_time_ms });
    }
}

test "Storage backend comparison" {
    const allocator = testing.allocator;
    
    const tmp_dir = testing.tmpDir(.{});
    defer tmp_dir.cleanup();
    
    // Test file storage
    {
        const db_path = try std.fmt.allocPrint(allocator, "{s}/file_storage.wdbx", .{tmp_dir.sub_path});
        defer allocator.free(db_path);

        const db = try core.Database.open(allocator, db_path, true);
        defer db.close();

        try db.init(.{
            .dimensions = 32,
            .index_type = .flat,
            .storage_type = .file,
        });

        const vec = [_]f32{1.0} ** 32;
        _ = try db.addVector(&vec, "metadata");

        const stats = db.getStats();
        try testing.expectEqual(@as(u64, 1), stats.total_vectors);
    }

    // Test memory storage
    {
        // Memory storage would be initialized differently
        // This is a placeholder for when memory storage is fully integrated
    }
}

test "Error handling and recovery" {
    const allocator = testing.allocator;
    
    const tmp_dir = testing.tmpDir(.{});
    defer tmp_dir.cleanup();
    
    const db_path = try std.fmt.allocPrint(allocator, "{s}/error_test.wdbx", .{tmp_dir.sub_path});
    defer allocator.free(db_path);

    // Test uninitialized database
    {
        const db = try core.Database.open(allocator, db_path, true);
        defer db.close();

        const vec = [_]f32{ 1.0, 2.0, 3.0 };
        try testing.expectError(error.DatabaseNotInitialized, db.addVector(&vec, null));
    }

    // Test dimension mismatch
    {
        const db = try core.Database.open(allocator, db_path, true);
        defer db.close();

        try db.init(.{
            .dimensions = 4,
            .index_type = .flat,
        });

        const vec = [_]f32{ 1.0, 2.0, 3.0 }; // Wrong dimension
        try testing.expectError(error.DimensionMismatch, db.addVector(&vec, null));
    }

    // Test read-only database
    {
        // First create and populate
        {
            const db = try core.Database.open(allocator, db_path, true);
            defer db.close();
            
            try db.init(.{
                .dimensions = 3,
                .index_type = .flat,
            });
            
            const vec = [_]f32{ 1.0, 2.0, 3.0 };
            _ = try db.addVector(&vec, null);
        }

        // Then try to write to read-only
        const db = try core.Database.open(allocator, db_path, false);
        defer db.close();

        const vec = [_]f32{ 4.0, 5.0, 6.0 };
        try testing.expectError(error.InvalidOperation, db.addVector(&vec, null));
    }
}

test "Database optimization" {
    const allocator = testing.allocator;
    
    const tmp_dir = testing.tmpDir(.{});
    defer tmp_dir.cleanup();
    
    const db_path = try std.fmt.allocPrint(allocator, "{s}/optimize_test.wdbx", .{tmp_dir.sub_path});
    defer allocator.free(db_path);

    const db = try core.Database.open(allocator, db_path, true);
    defer db.close();

    try db.init(.{
        .dimensions = 16,
        .index_type = .hnsw,
        .distance_metric = .euclidean,
    });

    // Add many vectors
    var rng = std.rand.DefaultPrng.init(42);
    const random = rng.random();
    
    var i: usize = 0;
    while (i < 500) : (i += 1) {
        var vec: [16]f32 = undefined;
        for (&vec) |*val| {
            val.* = random.float(f32);
        }
        _ = try db.addVector(&vec, null);
    }

    // Get stats before optimization
    const stats_before = db.getStats();

    // Run optimization
    try db.optimize();

    // Get stats after optimization
    const stats_after = db.getStats();

    // Verify vector count unchanged
    try testing.expectEqual(stats_before.total_vectors, stats_after.total_vectors);
    
    // Performance should be same or better
    try testing.expect(stats_after.avg_search_time_ns <= stats_before.avg_search_time_ns);
}