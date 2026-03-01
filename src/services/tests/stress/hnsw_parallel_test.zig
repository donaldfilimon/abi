//! HNSW Parallel Batch Search Stress Tests
//!
//! Performance and correctness tests for the parallel batch search implementation
//! using work-stealing parallelism via Chase-Lev deque.
//!
//! ## Running Tests
//!
//! ```bash
//! zig test src/services/tests/stress/hnsw_parallel_test.zig --test-filter "parallel"
//! ```

const std = @import("std");
const build_options = @import("build_options");
const abi = @import("abi");
const db = abi.features.database;
const hnsw = db.hnsw;
const index_mod = db.index;
const profiles = @import("profiles.zig");
const helpers = @import("../helpers.zig");
const StressProfile = profiles.StressProfile;
const LatencyHistogram = profiles.LatencyHistogram;
const Timer = profiles.Timer;

// ============================================================================
// Configuration
// ============================================================================

/// Test vector dimensions
const TEST_DIM: usize = 128;

/// Get the active stress profile for tests
fn getTestProfile() StressProfile {
    return StressProfile.quick;
}

/// Generate a random normalized vector for testing
fn generateRandomVector(rng: *std.Random.DefaultPrng, buffer: []f32) void {
    helpers.generateRandomVector(rng, buffer);
    helpers.normalizeVector(buffer);
}

// ============================================================================
// Parallel Batch Search Tests
// ============================================================================

test "parallel batch search: basic functionality" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    // Create test records
    const record_count = 100;
    var vectors = try allocator.alloc([TEST_DIM]f32, record_count);
    defer allocator.free(vectors);

    var records = try allocator.alloc(index_mod.VectorRecordView, record_count);
    defer allocator.free(records);

    var rng = std.Random.DefaultPrng.init(42);
    for (0..record_count) |i| {
        generateRandomVector(&rng, &vectors[i]);
        records[i] = .{
            .id = @intCast(i),
            .vector = &vectors[i],
        };
    }

    // Build HNSW index
    var index = try hnsw.HnswIndex.buildWithConfig(allocator, records, .{
        .m = 16,
        .ef_construction = 100,
        .search_pool_size = 4,
        .distance_cache_size = 256,
    });
    defer index.deinit(allocator);

    // Create batch of queries
    const query_count = 10;
    var query_vectors = try allocator.alloc([TEST_DIM]f32, query_count);
    defer allocator.free(query_vectors);

    var queries = try allocator.alloc([]const f32, query_count);
    defer allocator.free(queries);

    for (0..query_count) |i| {
        generateRandomVector(&rng, &query_vectors[i]);
        queries[i] = &query_vectors[i];
    }

    // Perform batch search
    const results = try index.batchSearch(allocator, records, queries, 10);
    defer hnsw.HnswIndex.freeBatchSearchResults(allocator, results);

    // Verify results
    try std.testing.expectEqual(query_count, results.len);
    for (results, 0..) |result, i| {
        try std.testing.expectEqual(i, result.query_index);
        try std.testing.expect(result.results.len <= 10);
    }
}

test "parallel batch search: empty queries" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    // Create minimal test records
    const records = [_]index_mod.VectorRecordView{
        .{ .id = 1, .vector = &[_]f32{ 1.0, 0.0 } },
        .{ .id = 2, .vector = &[_]f32{ 0.0, 1.0 } },
    };

    var index = try hnsw.HnswIndex.build(allocator, &records, 2, 10);
    defer index.deinit(allocator);

    // Empty query batch
    const queries: []const []const f32 = &.{};
    const results = try index.batchSearch(allocator, &records, queries, 5);
    defer hnsw.HnswIndex.freeBatchSearchResults(allocator, results);

    try std.testing.expectEqual(@as(usize, 0), results.len);
}

test "parallel batch search: single query fallback" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    // Create test records
    const records = [_]index_mod.VectorRecordView{
        .{ .id = 1, .vector = &[_]f32{ 1.0, 0.0, 0.0, 0.0 } },
        .{ .id = 2, .vector = &[_]f32{ 0.0, 1.0, 0.0, 0.0 } },
        .{ .id = 3, .vector = &[_]f32{ 0.0, 0.0, 1.0, 0.0 } },
        .{ .id = 4, .vector = &[_]f32{ 0.5, 0.5, 0.5, 0.5 } },
    };

    var index = try hnsw.HnswIndex.buildWithConfig(allocator, &records, .{
        .m = 4,
        .ef_construction = 20,
        .search_pool_size = 2,
    });
    defer index.deinit(allocator);

    // Single query should use sequential path (< 4 queries)
    const query = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const queries = [_][]const f32{&query};

    const results = try index.batchSearch(allocator, &records, &queries, 2);
    defer hnsw.HnswIndex.freeBatchSearchResults(allocator, results);

    try std.testing.expectEqual(@as(usize, 1), results.len);
    try std.testing.expect(results[0].results.len > 0);
    // Top result should be id=1 (exact match)
    try std.testing.expectEqual(@as(u64, 1), results[0].results[0].id);
}

test "parallel batch search: result quality" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    // Create test records with known patterns
    const record_count = 50;
    var vectors = try allocator.alloc([TEST_DIM]f32, record_count);
    defer allocator.free(vectors);

    var records = try allocator.alloc(index_mod.VectorRecordView, record_count);
    defer allocator.free(records);

    // Create vectors with distinct patterns
    for (0..record_count) |i| {
        @memset(&vectors[i], 0.0);
        // Set one dimension to 1.0 based on index
        vectors[i][i % TEST_DIM] = 1.0;
        records[i] = .{
            .id = @intCast(i),
            .vector = &vectors[i],
        };
    }

    var index = try hnsw.HnswIndex.buildWithConfig(allocator, records, .{
        .m = 16,
        .ef_construction = 100,
    });
    defer index.deinit(allocator);

    // Create queries that should match specific records
    const query_count = 5;
    var query_vectors = try allocator.alloc([TEST_DIM]f32, query_count);
    defer allocator.free(query_vectors);

    var queries = try allocator.alloc([]const f32, query_count);
    defer allocator.free(queries);

    for (0..query_count) |i| {
        @memset(&query_vectors[i], 0.0);
        query_vectors[i][i] = 1.0; // Should match record i
        queries[i] = &query_vectors[i];
    }

    // Perform batch search
    const results = try index.batchSearch(allocator, records, queries, 5);
    defer hnsw.HnswIndex.freeBatchSearchResults(allocator, results);

    // Verify that top results match expected records
    for (results, 0..) |result, i| {
        try std.testing.expect(result.results.len > 0);
        // Top result should have high similarity score
        try std.testing.expect(result.results[0].score >= 0.5);
        // The exact match should be in top results
        var found_match = false;
        for (result.results) |r| {
            if (r.id == i) {
                found_match = true;
                break;
            }
        }
        try std.testing.expect(found_match);
    }
}

test "parallel batch search: performance" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const profile = getTestProfile();

    // Create larger dataset for performance testing
    const record_count = @min(profile.operations / 10, 1000);
    var vectors = try allocator.alloc([TEST_DIM]f32, record_count);
    defer allocator.free(vectors);

    var records = try allocator.alloc(index_mod.VectorRecordView, record_count);
    defer allocator.free(records);

    var rng = std.Random.DefaultPrng.init(profile.getEffectiveSeed());
    for (0..record_count) |i| {
        generateRandomVector(&rng, &vectors[i]);
        records[i] = .{
            .id = @intCast(i),
            .vector = &vectors[i],
        };
    }

    // Build HNSW index
    var index = try hnsw.HnswIndex.buildWithConfig(allocator, records, .{
        .m = 16,
        .ef_construction = 200,
        .search_pool_size = 8,
        .distance_cache_size = 1024,
    });
    defer index.deinit(allocator);

    // Create batch of queries
    const query_count = @min(profile.operations / 100, 100);
    var query_vectors = try allocator.alloc([TEST_DIM]f32, query_count);
    defer allocator.free(query_vectors);

    var queries = try allocator.alloc([]const f32, query_count);
    defer allocator.free(queries);

    for (0..query_count) |i| {
        generateRandomVector(&rng, &query_vectors[i]);
        queries[i] = &query_vectors[i];
    }

    var latency = LatencyHistogram.init(allocator);
    defer latency.deinit();

    // Warm-up run
    {
        const warmup_results = try index.batchSearch(allocator, records, queries, 10);
        hnsw.HnswIndex.freeBatchSearchResults(allocator, warmup_results);
    }

    // Timed runs (at least 1 iteration)
    const iterations = @max(@min(profile.operations / 1000, 10), 1);
    for (0..iterations) |_| {
        const timer = Timer.start();
        const results = try index.batchSearch(allocator, records, queries, 10);
        const elapsed = timer.read();
        hnsw.HnswIndex.freeBatchSearchResults(allocator, results);

        try latency.recordUnsafe(elapsed);
    }

    // Check that batch search completed in reasonable time
    const stats = latency.getStats();
    try std.testing.expect(stats.count >= 1);

    // Performance assertion: batch search of 100 queries should complete in < 100ms
    // This is a soft assertion - actual performance depends on hardware
    if (query_count >= 100) {
        // Average latency per batch should be under 100ms
        try std.testing.expect(stats.avg < 100_000_000);
    }
}

test "parallel batch search: concurrent correctness" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    // Create test records
    const record_count = 200;
    var vectors = try allocator.alloc([TEST_DIM]f32, record_count);
    defer allocator.free(vectors);

    var records = try allocator.alloc(index_mod.VectorRecordView, record_count);
    defer allocator.free(records);

    var rng = std.Random.DefaultPrng.init(42);
    for (0..record_count) |i| {
        generateRandomVector(&rng, &vectors[i]);
        records[i] = .{
            .id = @intCast(i),
            .vector = &vectors[i],
        };
    }

    var index = try hnsw.HnswIndex.buildWithConfig(allocator, records, .{
        .m = 16,
        .ef_construction = 100,
        .search_pool_size = 8,
    });
    defer index.deinit(allocator);

    // Create batch queries
    const query_count = 50;
    var query_vectors = try allocator.alloc([TEST_DIM]f32, query_count);
    defer allocator.free(query_vectors);

    var queries = try allocator.alloc([]const f32, query_count);
    defer allocator.free(queries);

    for (0..query_count) |i| {
        generateRandomVector(&rng, &query_vectors[i]);
        queries[i] = &query_vectors[i];
    }

    // Run batch search multiple times and verify consistency
    var first_results: ?[]hnsw.HnswIndex.BatchSearchResult = null;
    defer if (first_results) |r| hnsw.HnswIndex.freeBatchSearchResults(allocator, r);

    for (0..5) |iteration| {
        const results = try index.batchSearch(allocator, records, queries, 10);

        if (first_results == null) {
            first_results = results;
        } else {
            // Compare with first results
            try std.testing.expectEqual(first_results.?.len, results.len);

            for (results, 0..) |result, i| {
                const first = first_results.?[i];
                try std.testing.expectEqual(first.query_index, result.query_index);
                try std.testing.expectEqual(first.results.len, result.results.len);

                // Results should be identical for same queries
                for (result.results, 0..) |r, j| {
                    try std.testing.expectEqual(first.results[j].id, r.id);
                    try std.testing.expectApproxEqAbs(first.results[j].score, r.score, 1e-5);
                }
            }

            hnsw.HnswIndex.freeBatchSearchResults(allocator, results);
        }

        _ = iteration;
    }
}

test "parallel batch search: large batch" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const profile = getTestProfile();

    // Create dataset
    const record_count = @min(profile.operations / 5, 500);
    var vectors = try allocator.alloc([TEST_DIM]f32, record_count);
    defer allocator.free(vectors);

    var records = try allocator.alloc(index_mod.VectorRecordView, record_count);
    defer allocator.free(records);

    var rng = std.Random.DefaultPrng.init(profile.getEffectiveSeed());
    for (0..record_count) |i| {
        generateRandomVector(&rng, &vectors[i]);
        records[i] = .{
            .id = @intCast(i),
            .vector = &vectors[i],
        };
    }

    var index = try hnsw.HnswIndex.buildWithConfig(allocator, records, .{
        .m = 16,
        .ef_construction = 100,
        .search_pool_size = 8,
    });
    defer index.deinit(allocator);

    // Create large batch of queries
    const query_count = @min(profile.operations / 50, 200);
    var query_vectors = try allocator.alloc([TEST_DIM]f32, query_count);
    defer allocator.free(query_vectors);

    var queries = try allocator.alloc([]const f32, query_count);
    defer allocator.free(queries);

    for (0..query_count) |i| {
        generateRandomVector(&rng, &query_vectors[i]);
        queries[i] = &query_vectors[i];
    }

    const timer = Timer.start();
    const results = try index.batchSearch(allocator, records, queries, 10);
    const elapsed = timer.read();
    defer hnsw.HnswIndex.freeBatchSearchResults(allocator, results);

    // Verify all queries were processed
    try std.testing.expectEqual(query_count, results.len);

    // All queries should have results
    for (results) |result| {
        try std.testing.expect(result.results.len > 0);
    }

    // Log performance (for debugging)
    if (query_count > 0) {
        const avg_per_query = elapsed / query_count;
        _ = avg_per_query;
    }
}

test "parallel batch search: compare with sequential" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    // Create test records
    const record_count = 100;
    var vectors = try allocator.alloc([TEST_DIM]f32, record_count);
    defer allocator.free(vectors);

    var records = try allocator.alloc(index_mod.VectorRecordView, record_count);
    defer allocator.free(records);

    var rng = std.Random.DefaultPrng.init(42);
    for (0..record_count) |i| {
        generateRandomVector(&rng, &vectors[i]);
        records[i] = .{
            .id = @intCast(i),
            .vector = &vectors[i],
        };
    }

    var index = try hnsw.HnswIndex.buildWithConfig(allocator, records, .{
        .m = 16,
        .ef_construction = 100,
        .search_pool_size = 4,
    });
    defer index.deinit(allocator);

    // Create queries
    const query_count = 20;
    var query_vectors = try allocator.alloc([TEST_DIM]f32, query_count);
    defer allocator.free(query_vectors);

    var queries = try allocator.alloc([]const f32, query_count);
    defer allocator.free(queries);

    for (0..query_count) |i| {
        generateRandomVector(&rng, &query_vectors[i]);
        queries[i] = &query_vectors[i];
    }

    // Run batch search (parallel)
    const batch_results = try index.batchSearch(allocator, records, queries, 5);
    defer hnsw.HnswIndex.freeBatchSearchResults(allocator, batch_results);

    // Run individual searches
    for (queries, 0..) |query, i| {
        const single_results = try index.search(allocator, records, query, 5);
        defer allocator.free(single_results);

        const batch_result = batch_results[i];

        // Compare results
        try std.testing.expectEqual(single_results.len, batch_result.results.len);

        for (single_results, 0..) |sr, j| {
            try std.testing.expectEqual(sr.id, batch_result.results[j].id);
            try std.testing.expectApproxEqAbs(sr.score, batch_result.results[j].score, 1e-5);
        }
    }
}
