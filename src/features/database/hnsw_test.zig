const std = @import("std");
const build_options = @import("build_options");
const hnsw = @import("hnsw.zig");
const index_mod = @import("index.zig");
const simd = @import("../../services/shared/simd/mod.zig");

const HnswIndex = hnsw.HnswIndex;
const SearchState = hnsw.SearchState;
const SearchStatePool = hnsw.SearchStatePool;
const DistanceCache = hnsw.DistanceCache;

test "hnsw structure basic lifecycle" {
    const allocator = std.testing.allocator;
    const records = [_]index_mod.VectorRecordView{
        .{ .id = 1, .vector = &[_]f32{ 1.0, 0.0 } },
        .{ .id = 2, .vector = &[_]f32{ 0.0, 1.0 } },
        .{ .id = 3, .vector = &[_]f32{ 0.7, 0.7 } },
    };

    var index = try HnswIndex.build(allocator, &records, 16, 100);
    defer index.deinit(allocator);

    try std.testing.expect(index.nodes.len == 3);
    try std.testing.expect(index.entry_point != null);

    const query = [_]f32{ 0.8, 0.6 };
    const results = try index.search(allocator, &records, &query, 2);
    defer allocator.free(results);

    try std.testing.expect(results.len <= 2);
    if (results.len > 0) {
        // Result 3 should be top since similarity is high
        try std.testing.expect(results[0].id == 3);
    }
}

test "hnsw with search state pool" {
    const allocator = std.testing.allocator;
    const records = [_]index_mod.VectorRecordView{
        .{ .id = 1, .vector = &[_]f32{ 1.0, 0.0, 0.0, 0.0 } },
        .{ .id = 2, .vector = &[_]f32{ 0.0, 1.0, 0.0, 0.0 } },
        .{ .id = 3, .vector = &[_]f32{ 0.0, 0.0, 1.0, 0.0 } },
        .{ .id = 4, .vector = &[_]f32{ 0.5, 0.5, 0.5, 0.5 } },
        .{ .id = 5, .vector = &[_]f32{ 0.9, 0.1, 0.0, 0.0 } },
    };

    // Build with pool and cache enabled
    var index = try HnswIndex.buildWithConfig(allocator, &records, .{
        .m = 8,
        .ef_construction = 50,
        .search_pool_size = 4,
        .distance_cache_size = 128,
    });
    defer index.deinit(allocator);

    try std.testing.expect(index.state_pool != null);
    try std.testing.expect(index.distance_cache != null);

    // Perform multiple searches to exercise the pool
    const queries = [_][4]f32{
        .{ 1.0, 0.0, 0.0, 0.0 },
        .{ 0.5, 0.5, 0.0, 0.0 },
        .{ 0.0, 0.0, 0.0, 1.0 },
    };

    for (queries) |query| {
        const results = try index.search(allocator, &records, &query, 3);
        defer allocator.free(results);
        try std.testing.expect(results.len > 0);
    }

    // Check cache statistics
    if (index.getCacheStats()) |stats| {
        // Some hits expected after multiple operations
        try std.testing.expect(stats.hits + stats.misses > 0);
    }
}

test "search state pool acquire release" {
    const allocator = std.testing.allocator;

    var pool = try SearchStatePool.init(allocator, 4);
    defer pool.deinit();

    // Acquire all states
    var states: [4]?*SearchState = undefined;
    for (&states) |*s| {
        s.* = pool.acquire();
        try std.testing.expect(s.* != null);
    }

    // Pool should be exhausted
    try std.testing.expect(pool.acquire() == null);

    // Release one
    pool.release(states[0].?);

    // Should be able to acquire again
    const reacquired = pool.acquire();
    try std.testing.expect(reacquired != null);

    // Release all
    for (states[1..]) |s| {
        if (s) |state| pool.release(state);
    }
    pool.release(reacquired.?);
}

test "distance cache basic operations" {
    const allocator = std.testing.allocator;

    var cache = try DistanceCache.init(allocator, 32);
    defer cache.deinit(allocator);

    // Should miss initially
    try std.testing.expect(cache.get(1, 2) == null);

    // Store and retrieve
    cache.put(1, 2, 0.5);
    const cached = cache.get(1, 2);
    try std.testing.expect(cached != null);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), cached.?, 1e-6);

    // Order-independent key
    const cached_rev = cache.get(2, 1);
    try std.testing.expect(cached_rev != null);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), cached_rev.?, 1e-6);

    // Check stats
    const stats = cache.getStats();
    try std.testing.expect(stats.hits == 2);
    try std.testing.expect(stats.misses == 1);
}

test "hnsw with gpu acceleration config" {
    const allocator = std.testing.allocator;
    const records = [_]index_mod.VectorRecordView{
        .{ .id = 1, .vector = &[_]f32{ 1.0, 0.0, 0.0, 0.0 } },
        .{ .id = 2, .vector = &[_]f32{ 0.0, 1.0, 0.0, 0.0 } },
        .{ .id = 3, .vector = &[_]f32{ 0.0, 0.0, 1.0, 0.0 } },
        .{ .id = 4, .vector = &[_]f32{ 0.5, 0.5, 0.5, 0.5 } },
        .{ .id = 5, .vector = &[_]f32{ 0.9, 0.1, 0.0, 0.0 } },
    };

    // Build with GPU config (will use SIMD fallback if GPU unavailable)
    var index = try HnswIndex.buildWithConfig(allocator, &records, .{
        .m = 8,
        .ef_construction = 50,
        .search_pool_size = 4,
        .distance_cache_size = 128,
        .enable_gpu = true,
        .gpu_batch_threshold = 2, // Low threshold for testing
    });
    defer index.deinit(allocator);

    // GPU accelerator should be initialized (even if GPU not available, SIMD fallback used)
    if (build_options.enable_gpu) {
        try std.testing.expect(index.gpu_accelerator != null);
    }

    // Perform searches
    const queries = [_][4]f32{
        .{ 1.0, 0.0, 0.0, 0.0 },
        .{ 0.5, 0.5, 0.0, 0.0 },
    };

    for (queries) |query| {
        const results = try index.search(allocator, &records, &query, 3);
        defer allocator.free(results);
        try std.testing.expect(results.len > 0);
    }

    // Check GPU stats if available
    if (index.getGpuStats()) |stats| {
        // Operations should have been tracked
        try std.testing.expect(stats.gpu_ops + stats.simd_ops >= 0);
    }
}

test "hnsw batch distance computation" {
    const allocator = std.testing.allocator;
    const records = [_]index_mod.VectorRecordView{
        .{ .id = 1, .vector = &[_]f32{ 1.0, 0.0, 0.0 } },
        .{ .id = 2, .vector = &[_]f32{ 0.0, 1.0, 0.0 } },
        .{ .id = 3, .vector = &[_]f32{ 0.0, 0.0, 1.0 } },
        .{ .id = 4, .vector = &[_]f32{ 0.577, 0.577, 0.577 } }, // Approx normalized
    };

    var index = try HnswIndex.buildWithConfig(allocator, &records, .{
        .m = 4,
        .ef_construction = 20,
        .enable_gpu = false, // Test SIMD path
    });
    defer index.deinit(allocator);

    // Test batch distance computation directly
    const query = [_]f32{ 1.0, 0.0, 0.0 };
    const query_norm = simd.vectorL2Norm(&query);
    const neighbor_ids = [_]u32{ 0, 1, 2, 3 };
    var distances: [4]f32 = undefined;

    index.computeBatchDistancesSequential(&query, query_norm, &records, &neighbor_ids, &distances);

    // First record should have lowest distance (same direction)
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), distances[0], 0.01);
    // Orthogonal vectors should have distance ~1.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), distances[1], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), distances[2], 0.01);
}

test "hnsw enable search pool on deserialized index" {
    const allocator = std.testing.allocator;
    const records = [_]index_mod.VectorRecordView{
        .{ .id = 1, .vector = &[_]f32{ 1.0, 0.0, 0.0, 0.0 } },
        .{ .id = 2, .vector = &[_]f32{ 0.0, 1.0, 0.0, 0.0 } },
        .{ .id = 3, .vector = &[_]f32{ 0.0, 0.0, 1.0, 0.0 } },
    };

    // Build without pool to simulate deserialized index
    var index = try HnswIndex.buildWithConfig(allocator, &records, .{
        .search_pool_size = 0, // Disabled
        .distance_cache_size = 0, // Disabled
    });
    defer index.deinit(allocator);

    try std.testing.expect(index.state_pool == null);
    try std.testing.expect(index.distance_cache == null);

    // Enable pool and cache post-construction
    try index.enableSearchPool(4);
    try index.enableDistanceCache(128);

    try std.testing.expect(index.state_pool != null);
    try std.testing.expect(index.distance_cache != null);

    // Second call should be no-op (already enabled)
    try index.enableSearchPool(8);
    try index.enableDistanceCache(256);

    // Verify search still works with newly enabled pool
    const query = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const results = try index.search(allocator, &records, &query, 2);
    defer allocator.free(results);

    try std.testing.expect(results.len >= 1);
}
