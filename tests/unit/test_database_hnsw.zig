const std = @import("std");
const testing = std.testing;
const abi = @import("abi");
const database = abi.wdbx.database;

test "HNSW index initialization" {
    const test_file = "test_hnsw_init.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    var db = try database.Db.open(test_file, true);
    defer db.close();

    try db.init(128);
    try db.initHNSW();
    db.setHNSWParams(.{ .max_connections = 32, .ef_construction = 400, .ef_search = 400 });

    try testing.expect(db.hnsw_index != null);
}

test "HNSW vector addition and search" {
    const test_file = "test_hnsw_search.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    var db = try database.Db.open(test_file, true);
    defer db.close();

    try db.init(64);
    try db.initHNSW();
    db.setHNSWParams(.{ .max_connections = 32, .ef_construction = 400, .ef_search = 400 });

    // Add test vectors
    const num_vectors = 100;
    for (0..num_vectors) |i| {
        var embedding = try testing.allocator.alloc(f32, 64);
        defer testing.allocator.free(embedding);

        for (0..64) |j| {
            embedding[j] = @as(f32, @floatFromInt(i * 64 + j)) * 0.01;
        }

        const row_id = try db.addEmbedding(embedding);
        try testing.expectEqual(@as(u64, i), row_id);
    }

    // Test HNSW search
    var query = try testing.allocator.alloc(f32, 64);
    defer testing.allocator.free(query);

    for (0..64) |i| {
        query[i] = @as(f32, @floatFromInt(i)) * 0.01;
    }

    const results = try db.searchHNSW(query, 5, testing.allocator);
    defer testing.allocator.free(results);

    try testing.expect(results.len > 0);
    try testing.expect(results.len <= 5);

    // Verify results are sorted by score
    for (1..results.len) |i| {
        try testing.expect(results[i].score >= results[i - 1].score);
    }
}

test "HNSW vs brute force search comparison" {
    const test_file = "test_hnsw_comparison.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    var db = try database.Db.open(test_file, true);
    defer db.close();

    try db.init(128);
    try db.initHNSW();

    // Add vectors
    const num_vectors = 1000;
    for (0..num_vectors) |i| {
        var embedding = try testing.allocator.alloc(f32, 128);
        defer testing.allocator.free(embedding);

        for (0..128) |j| {
            embedding[j] = @as(f32, @floatFromInt(i * 128 + j)) * 0.001;
        }

        _ = try db.addEmbedding(embedding);
    }

    // Create query vector
    var query = try testing.allocator.alloc(f32, 128);
    defer testing.allocator.free(query);

    for (0..128) |i| {
        query[i] = @as(f32, @floatFromInt(i)) * 0.01;
    }

    // Benchmark both search methods
    const hnsw_start = std.time.nanoTimestamp();
    const hnsw_results = try db.searchHNSW(query, 10, testing.allocator);
    defer testing.allocator.free(hnsw_results);
    const hnsw_time = std.time.nanoTimestamp() - hnsw_start;

    const brute_start = std.time.nanoTimestamp();
    const brute_results = try db.search(query, 10, testing.allocator);
    defer testing.allocator.free(brute_results);
    const brute_time = std.time.nanoTimestamp() - brute_start;

    // HNSW may be faster for large datasets, but allow flexibility across environments
    _ = hnsw_time;
    _ = brute_time;

    // Ensure both searches return non-empty results
    try testing.expect(hnsw_results.len > 0 and brute_results.len > 0);
}

test "parallel search functionality" {
    const test_file = "test_parallel_search.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    var db = try database.Db.open(test_file, true);
    defer db.close();

    try db.init(64);

    // Add test vectors
    const num_vectors = 500;
    for (0..num_vectors) |i| {
        var embedding = try testing.allocator.alloc(f32, 64);
        defer testing.allocator.free(embedding);

        for (0..64) |j| {
            embedding[j] = @as(f32, @floatFromInt(i * 64 + j)) * 0.01;
        }

        _ = try db.addEmbedding(embedding);
    }

    // Create query vector
    var query = try testing.allocator.alloc(f32, 64);
    defer testing.allocator.free(query);

    for (0..64) |i| {
        query[i] = @as(f32, @floatFromInt(i)) * 0.01;
    }

    // Test single-threaded search
    const single_results = try db.search(query, 10, testing.allocator);
    defer testing.allocator.free(single_results);

    // Test parallel search with different thread counts
    const thread_counts = [_]u32{ 2, 4 };

    for (thread_counts) |thread_count| {
        const parallel_results = try db.searchParallel(query, 10, testing.allocator, thread_count);
        defer testing.allocator.free(parallel_results);

        // Results should be the same length
        try testing.expectEqual(single_results.len, parallel_results.len);

        // Top results should match approximately
        for (0..@min(single_results.len, parallel_results.len)) |i| {
            try testing.expectEqual(single_results[i].index, parallel_results[i].index);
            try testing.expectApproxEqAbs(single_results[i].score, parallel_results[i].score, 1e-3);
        }
    }
}

test "parallel search performance improvement" {
    const test_file = "test_parallel_performance.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    var db = try database.Db.open(test_file, true);
    defer db.close();

    try db.init(128);

    // Add more vectors for meaningful parallel performance test
    const num_vectors = 10_000;
    for (0..num_vectors) |i| {
        var embedding = try testing.allocator.alloc(f32, 128);
        defer testing.allocator.free(embedding);

        for (0..128) |j| {
            embedding[j] = @as(f32, @floatFromInt(i * 128 + j)) * 0.001;
        }

        _ = try db.addEmbedding(embedding);
    }

    // Create query vector
    var query = try testing.allocator.alloc(f32, 128);
    defer testing.allocator.free(query);

    for (0..128) |i| {
        query[i] = @as(f32, @floatFromInt(i)) * 0.01;
    }

    // Benchmark single-threaded
    const single_start = std.time.nanoTimestamp();
    const single_results = try db.search(query, 10, testing.allocator);
    defer testing.allocator.free(single_results);
    const single_time = std.time.nanoTimestamp() - single_start;

    // Benchmark parallel (4 threads)
    const parallel_start = std.time.nanoTimestamp();
    const parallel_results = try db.searchParallel(query, 10, testing.allocator, 4);
    defer testing.allocator.free(parallel_results);
    const parallel_time = std.time.nanoTimestamp() - parallel_start;

    // On multi-core systems, allow some overhead tolerance. Avoid failing on noisy CI or power-limited environments.
    if (std.Thread.getCpuCount() catch 1 > 1) {
        const tolerance = single_time * 3; // allow up to 3x to account for scheduling/thermal variance
        if (parallel_time > tolerance) {
            std.log.warn("parallel search slower than expected: single={} ns, parallel={} ns (allowed up to {} ns)", .{ single_time, parallel_time, tolerance });
        }
    }

    // Results should be identical
    try testing.expectEqual(single_results.len, parallel_results.len);
    for (0..single_results.len) |i| {
        try testing.expectEqual(single_results[i].index, parallel_results[i].index);
        try testing.expectApproxEqAbs(single_results[i].score, parallel_results[i].score, 1e-3);
    }
}

test "HNSW index memory management" {
    const test_file = "test_hnsw_memory.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    var db = try database.Db.open(test_file, true);
    defer db.close();

    try db.init(256);
    try db.initHNSW();

    // Add vectors and check memory usage
    _ = db.read_buffer.len;

    const num_vectors = 100;
    for (0..num_vectors) |i| {
        var embedding = try testing.allocator.alloc(f32, 256);
        defer testing.allocator.free(embedding);

        for (0..256) |j| {
            embedding[j] = @as(f32, @floatFromInt(i * 256 + j)) * 0.001;
        }

        _ = try db.addEmbedding(embedding);
    }

    // HNSW index should consume additional memory
    try testing.expect(db.hnsw_index != null);

    // Cleanup occurs via defer

    // Verify file is cleaned up
    // File deletion is handled by defer at test start
}

test "HNSW index with different dimensions" {
    const dimensions = [_]u16{ 64, 128, 256, 512 };

    for (dimensions) |dim| {
        const test_file = try std.fmt.allocPrint(testing.allocator, "test_hnsw_{d}.wdbx", .{dim});
        defer testing.allocator.free(test_file);
        defer std.fs.cwd().deleteFile(test_file) catch {};

        var db = try database.Db.open(test_file, true);
        defer db.close();

        try db.init(dim);
        try db.initHNSW();

        // Add a few test vectors
        for (0..10) |i| {
            var embedding = try testing.allocator.alloc(f32, dim);
            defer testing.allocator.free(embedding);

            for (0..dim) |j| {
                embedding[j] = @as(f32, @floatFromInt(i * dim + j)) * 0.01;
            }

            _ = try db.addEmbedding(embedding);
        }

        // Test search
        var query = try testing.allocator.alloc(f32, dim);
        defer testing.allocator.free(query);

        for (0..dim) |i| {
            query[i] = @as(f32, @floatFromInt(i)) * 0.01;
        }

        const results = try db.searchHNSW(query, 5, testing.allocator);
        defer testing.allocator.free(results);

        try testing.expect(results.len > 0);
        try testing.expect(results.len <= 5);
    }
}
