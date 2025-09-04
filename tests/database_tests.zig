//! Database module tests

const std = @import("std");
const testing = std.testing;
const wdbx = @import("wdbx");
const test_framework = @import("test_framework.zig");

test "database: create and basic operations" {
    const allocator = testing.allocator;
    
    var ctx = try test_framework.TestContext.init(allocator);
    defer ctx.deinit();
    
    const db_path = try ctx.getTempPath("test.db");
    defer allocator.free(db_path);
    
    // Create database
    const db = try wdbx.database.create(allocator, db_path);
    defer db.deinit();
    
    // Write record
    const data = "test data";
    const id = try db.writeRecord(data);
    try testing.expect(id > 0);
    
    // Read record
    if (try db.readRecord(id)) |record| {
        try testing.expectEqual(id, record.id);
        try testing.expectEqualSlices(u8, data, record.data);
    } else {
        return error.RecordNotFound;
    }
    
    // Delete record
    const deleted = try db.deleteRecord(id);
    try testing.expect(deleted);
    
    // Verify deletion
    try testing.expect(try db.readRecord(id) == null);
}

test "database: batch operations" {
    const allocator = testing.allocator;
    
    var ctx = try test_framework.TestContext.init(allocator);
    defer ctx.deinit();
    
    const db_path = try ctx.getTempPath("batch_test.db");
    defer allocator.free(db_path);
    
    const db = try wdbx.database.create(allocator, db_path);
    defer db.deinit();
    
    // Prepare test data
    const records = [_][]const u8{
        "record 1",
        "record 2",
        "record 3",
        "record 4",
        "record 5",
    };
    
    // Batch insert
    try wdbx.database.BatchOps.insertBatch(db, &records);
    
    // Batch read
    const ids = [_]u64{ 1, 2, 3, 4, 5 };
    const results = try wdbx.database.BatchOps.readBatch(db, &ids, allocator);
    defer allocator.free(results);
    
    try testing.expectEqual(records.len, results.len);
    for (results, records) |result, expected| {
        try testing.expect(result != null);
        try testing.expectEqualSlices(u8, expected, result.?.data);
    }
}

test "database: vector similarity metrics" {
    const vec1 = [_]f32{ 1, 0, 0, 0 };
    const vec2 = [_]f32{ 0, 1, 0, 0 };
    const vec3 = [_]f32{ 1, 1, 0, 0 };
    
    // Test Euclidean distance
    const euclidean_12 = wdbx.database.Metric.euclidean.distance(&vec1, &vec2);
    try testing.expectApproxEqAbs(@as(f32, @sqrt(2.0)), euclidean_12, 0.001);
    
    const euclidean_13 = wdbx.database.Metric.euclidean.distance(&vec1, &vec3);
    try testing.expectApproxEqAbs(@as(f32, 1.0), euclidean_13, 0.001);
    
    // Test cosine distance
    const cosine_12 = wdbx.database.Metric.cosine.distance(&vec1, &vec2);
    try testing.expectApproxEqAbs(@as(f32, 0.0), cosine_12, 0.001); // Orthogonal vectors
    
    // Test Manhattan distance
    const manhattan_13 = wdbx.database.Metric.manhattan.distance(&vec1, &vec3);
    try testing.expectEqual(@as(f32, 1.0), manhattan_13);
}

test "database: optimization" {
    const allocator = testing.allocator;
    
    var ctx = try test_framework.TestContext.init(allocator);
    defer ctx.deinit();
    
    const db_path = try ctx.getTempPath("optimize_test.db");
    defer allocator.free(db_path);
    
    const db = try wdbx.database.create(allocator, db_path);
    defer db.deinit();
    
    // Add some data
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const data = try std.fmt.allocPrint(allocator, "data_{d}", .{i});
        defer allocator.free(data);
        _ = try db.writeRecord(data);
    }
    
    // Delete some records
    i = 0;
    while (i < 50) : (i += 2) {
        _ = try db.deleteRecord(@as(u64, i + 1));
    }
    
    // Run optimization
    try wdbx.database.Optimizer.optimize(db);
    
    // Verify remaining records
    i = 1;
    while (i < 100) : (i += 2) {
        if (try db.readRecord(@as(u64, i + 1))) |record| {
            const expected = try std.fmt.allocPrint(allocator, "data_{d}", .{i});
            defer allocator.free(expected);
            try testing.expectEqualSlices(u8, expected, record.data);
        }
    }
}

test "database: export to JSON" {
    const allocator = testing.allocator;
    
    var ctx = try test_framework.TestContext.init(allocator);
    defer ctx.deinit();
    
    const db_path = try ctx.getTempPath("export_test.db");
    defer allocator.free(db_path);
    
    const db = try wdbx.database.create(allocator, db_path);
    defer db.deinit();
    
    // Add test data
    const records = [_][]const u8{
        "record 1",
        "record 2",
        "record 3",
    };
    
    var ids = std.ArrayList(u64).init(allocator);
    defer ids.deinit();
    
    for (records) |record| {
        const id = try db.writeRecord(record);
        try ids.append(id);
    }
    
    // Export to JSON
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();
    
    try wdbx.database.ImportExport.exportJson(db, buffer.writer(), ids.items);
    
    const json = buffer.items;
    try testing.expect(std.mem.indexOf(u8, json, "\"version\": \"2.0.0\"") != null);
    try testing.expect(std.mem.indexOf(u8, json, "\"records\"") != null);
}

test "database: concurrent access" {
    const allocator = testing.allocator;
    
    var ctx = try test_framework.TestContext.init(allocator);
    defer ctx.deinit();
    
    const db_path = try ctx.getTempPath("concurrent_test.db");
    defer allocator.free(db_path);
    
    const db = try wdbx.database.create(allocator, db_path);
    defer db.deinit();
    
    // Create thread pool
    var pool = try wdbx.core.threading.ThreadPool.init(allocator, 4);
    defer pool.deinit();
    
    // Concurrent writes
    const Writer = struct {
        fn write(context: *struct { db: *wdbx.database.Db, id: usize }) void {
            const data = std.fmt.allocPrint(
                std.heap.page_allocator,
                "thread_{d}_data",
                .{context.id}
            ) catch return;
            defer std.heap.page_allocator.free(data);
            
            _ = context.db.writeRecord(data) catch {};
        }
    };
    
    var contexts: [10]struct { db: *wdbx.database.Db, id: usize } = undefined;
    for (&contexts, 0..) |*ctx, i| {
        ctx.* = .{ .db = db, .id = i };
        try pool.submit(Writer.write, ctx);
    }
    
    pool.wait();
    
    // Verify all writes succeeded
    var i: u64 = 1;
    while (i <= 10) : (i += 1) {
        try testing.expect(try db.readRecord(i) != null);
    }
}

test "database: error recovery" {
    const allocator = testing.allocator;
    
    var ctx = try test_framework.TestContext.init(allocator);
    defer ctx.deinit();
    
    const db_path = try ctx.getTempPath("recovery_test.db");
    defer allocator.free(db_path);
    
    const db = try wdbx.database.create(allocator, db_path);
    defer db.deinit();
    
    // Test reading non-existent record
    try testing.expect(try db.readRecord(999) == null);
    
    // Test deleting non-existent record
    try testing.expect(!(try db.deleteRecord(999)));
    
    // Test empty batch operations
    const empty: [][]const u8 = &[_][]const u8{};
    try wdbx.database.BatchOps.insertBatch(db, empty);
    
    const empty_ids: []u64 = &[_]u64{};
    const results = try wdbx.database.BatchOps.readBatch(db, empty_ids, allocator);
    defer allocator.free(results);
    try testing.expectEqual(@as(usize, 0), results.len);
}

test "database: performance benchmark" {
    const allocator = testing.allocator;
    
    var ctx = try test_framework.TestContext.init(allocator);
    defer ctx.deinit();
    
    const db_path = try ctx.getTempPath("benchmark_test.db");
    defer allocator.free(db_path);
    
    const db = try wdbx.database.create(allocator, db_path);
    defer db.deinit();
    
    // Benchmark write performance
    var write_bench = test_framework.Benchmark.init(allocator, "Database Write");
    defer write_bench.deinit();
    
    write_bench.iterations = 100;
    write_bench.warmup_iterations = 10;
    
    const WriteOp = struct {
        fn run(database: *wdbx.database.Db) !void {
            _ = try database.writeRecord("benchmark data");
        }
    };
    
    try write_bench.run(WriteOp.run, .{db});
    
    // Benchmark read performance
    var read_bench = test_framework.Benchmark.init(allocator, "Database Read");
    defer read_bench.deinit();
    
    read_bench.iterations = 100;
    read_bench.warmup_iterations = 10;
    
    const ReadOp = struct {
        fn run(database: *wdbx.database.Db) !void {
            _ = try database.readRecord(1);
        }
    };
    
    try read_bench.run(ReadOp.run, .{db});
    
    // Print results
    const stdout = std.io.getStdOut().writer();
    try write_bench.report(stdout);
    try read_bench.report(stdout);
}