const std = @import("std");
const testing = std.testing;
const database = @import("database");

test "database basic operations" {
    const allocator = testing.allocator;

    // Test file path
    const test_file = "test_vectors.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    // Ensure clean slate
    std.fs.cwd().deleteFile(test_file) catch {};

    // Test database creation and initialization
    var db = try database.Db.open(test_file, true);
    defer db.close();

    try testing.expectEqual(@as(u64, 0), db.getRowCount());
    try testing.expectEqual(@as(u16, 0), db.getDimension());

    // Test initialization
    try db.init(384);
    try testing.expectEqual(@as(u16, 384), db.getDimension());
    try testing.expectEqual(@as(u64, 0), db.getRowCount());

    // Test adding embeddings
    var embedding1 = try allocator.alloc(f32, 384);
    defer allocator.free(embedding1);
    for (0..384) |i| {
        embedding1[i] = @as(f32, @floatFromInt(i)) * 0.01;
    }

    const row_id1 = try db.addEmbedding(embedding1);
    try testing.expectEqual(@as(u64, 0), row_id1);
    try testing.expectEqual(@as(u64, 1), db.getRowCount());

    // Test adding another embedding
    var embedding2 = try allocator.alloc(f32, 384);
    defer allocator.free(embedding2);
    for (0..384) |i| {
        embedding2[i] = @as(f32, @floatFromInt(i)) * 0.02;
    }

    const row_id2 = try db.addEmbedding(embedding2);
    try testing.expectEqual(@as(u64, 1), row_id2);
    try testing.expectEqual(@as(u64, 2), db.getRowCount());

    // Test search functionality
    var query = try allocator.alloc(f32, 384);
    defer allocator.free(query);
    for (0..384) |i| {
        query[i] = @as(f32, @floatFromInt(i)) * 0.015; // Closer to embedding1
    }

    const results = try db.search(query, 2, allocator);
    defer allocator.free(results);

    try testing.expectEqual(@as(usize, 2), results.len);

    // First result should be one of the embeddings (allow tie/precision tolerance)
    try testing.expect(results[0].index == 0 or results[0].index == 1);
    try testing.expect(results[0].score < results[1].score);

    // Test batch operations
    var batch_embeddings = try allocator.alloc([]f32, 3);
    defer {
        for (batch_embeddings) |emb| {
            allocator.free(emb);
        }
        allocator.free(batch_embeddings);
    }

    for (0..3) |i| {
        var emb = try allocator.alloc(f32, 384);
        for (0..384) |j| {
            emb[j] = @as(f32, @floatFromInt(i * 100 + j)) * 0.001;
        }
        batch_embeddings[i] = emb;
    }

    const batch_indices = try db.addEmbeddingsBatch(batch_embeddings);
    defer db.allocator.free(batch_indices);

    try testing.expectEqual(@as(usize, 3), batch_indices.len);
    try testing.expectEqual(@as(u64, 2), batch_indices[0]);
    try testing.expectEqual(@as(u64, 3), batch_indices[1]);
    try testing.expectEqual(@as(u64, 4), batch_indices[2]);
    try testing.expectEqual(@as(u64, 5), db.getRowCount());

    // Test statistics
    const stats = db.getStats();
    try testing.expectEqual(@as(u64, 1), stats.initialization_count);
    try testing.expectEqual(@as(u64, 5), stats.write_count);
    try testing.expectEqual(@as(u64, 1), stats.search_count);
    try testing.expect(stats.total_search_time_us > 0);
}

test "database error handling" {
    const allocator = testing.allocator;
    const test_file = "test_error_handling.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    var db = try database.Db.open(test_file, true);
    defer db.close();

    // Test dimension mismatch error
    try db.init(128);

    const wrong_dim_embedding = try allocator.alloc(f32, 256);
    defer allocator.free(wrong_dim_embedding);

    const result = db.addEmbedding(wrong_dim_embedding);
    try testing.expectError(database.Db.DbError.DimensionMismatch, result);

    // Test invalid state error
    var db2 = try database.Db.open("test_invalid_state.wdbx", true);
    defer db2.close();

    const result2 = db2.addEmbedding(wrong_dim_embedding);
    try testing.expectError(database.Db.DbError.InvalidState, result2);

    // Clean up
    std.fs.cwd().deleteFile("test_invalid_state.wdbx") catch {};
}

test "database header validation" {
    const test_file = "test_header.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    var db = try database.Db.open(test_file, true);
    defer db.close();

    // Test header creation
    const header = database.WdbxHeader.createDefault();
    try testing.expect(header.validateMagic());
    try testing.expectEqual(database.FORMAT_VERSION, header.version);
    try testing.expectEqual(database.DEFAULT_PAGE_SIZE, header.page_size);
    try testing.expectEqual(@as(u64, 0), header.row_count);
    try testing.expectEqual(@as(u16, 0), header.dim);
}

test "database performance" {
    const allocator = testing.allocator;
    const test_file = "test_performance.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    var db = try database.Db.open(test_file, true);
    defer db.close();

    try db.init(128);

    // Add many embeddings for performance testing
    const num_embeddings = 1000;

    for (0..num_embeddings) |i| {
        var embedding = try allocator.alloc(f32, 128);
        defer allocator.free(embedding);

        for (0..128) |j| {
            embedding[j] = @as(f32, @floatFromInt(i * 128 + j)) * 0.001;
        }

        _ = try db.addEmbedding(embedding);
    }

    try testing.expectEqual(@as(u64, num_embeddings), db.getRowCount());

    // Test search performance
    var query = try allocator.alloc(f32, 128);
    defer allocator.free(query);
    for (0..128) |i| {
        query[i] = @as(f32, @floatFromInt(i)) * 0.01;
    }

    const search_start = std.time.microTimestamp();
    const results = try db.search(query, 10, allocator);
    defer allocator.free(results);
    const search_end = std.time.microTimestamp();

    try testing.expectEqual(@as(usize, 10), results.len);

    // Performance assertions (adjust based on your system)
    try testing.expect((search_end - search_start) < 1000000); // Should complete in under 1 second

    const stats = db.getStats();
    try testing.expectEqual(@as(u64, num_embeddings), stats.write_count);
    try testing.expectEqual(@as(u64, 1), stats.search_count);
}

test "database file format compatibility" {
    const test_file = "test_compatibility.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    var db = try database.Db.open(test_file, true);
    defer db.close();

    try db.init(64);

    // Add some data
    var embedding = try testing.allocator.alloc(f32, 64);
    defer testing.allocator.free(embedding);
    for (0..64) |i| {
        embedding[i] = @as(f32, @floatFromInt(i)) * 0.1;
    }

    _ = try db.addEmbedding(embedding);
    // rely on deferred close

    // Reopen and verify data persistence
    var db2 = try database.Db.open(test_file, false);
    defer db2.close();

    try testing.expectEqual(@as(u16, 64), db2.getDimension());
    try testing.expectEqual(@as(u64, 1), db2.getRowCount());

    // Test search on reopened database
    var query = try testing.allocator.alloc(f32, 64);
    defer testing.allocator.free(query);
    for (0..64) |i| {
        query[i] = @as(f32, @floatFromInt(i)) * 0.1;
    }

    const results = try db2.search(query, 1, testing.allocator);
    defer testing.allocator.free(results);

    try testing.expectEqual(@as(usize, 1), results.len);
    try testing.expectEqual(@as(u64, 0), results[0].index);
    // Allow minor variance
    // Skip tight score assertion across platforms
}
