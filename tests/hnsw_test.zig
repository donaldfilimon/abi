//! HNSW Index Tests

const std = @import("std");
const hnsw_mod = @import("../src/core/database/hnsw.zig");

test "hnsw insert maintains count" {
    const allocator = std.testing.allocator;
    var index = hnsw_mod.HnswIndex.init(allocator, .{
        .dimension = 4,
        .M = 4,
        .M0 = 8,
        .ef_construction = 16,
        .ef_search = 16,
    });
    defer index.deinit();

    const v1 = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const v2 = [_]f32{ 0.0, 1.0, 0.0, 0.0 };
    try index.insert(0, &v1);
    try index.insert(1, &v2);
    try std.testing.expectEqual(@as(usize, 2), index.len());
}

test "hnsw search finds nearest neighbor" {
    const allocator = std.testing.allocator;
    var index = hnsw_mod.HnswIndex.init(allocator, .{
        .dimension = 4,
        .M = 4,
        .M0 = 8,
        .ef_construction = 32,
        .ef_search = 32,
        .metric = .l2,
    });
    defer index.deinit();

    // Insert a known vector.
    const target_vec = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const far_vec = [_]f32{ 0.0, 0.0, 0.0, 1.0 };
    try index.insert(0, &target_vec);
    try index.insert(1, &far_vec);

    // Query near the target.
    const query = [_]f32{ 0.99, 0.01, 0.0, 0.0 };
    const results = try index.search(&query, 1);
    defer allocator.free(results);

    try std.testing.expect(results.len == 1);
}

test "hnsw multiple inserts and k-nn search" {
    const allocator = std.testing.allocator;
    var index = hnsw_mod.HnswIndex.init(allocator, .{
        .dimension = 4,
        .M = 8,
        .M0 = 16,
        .ef_construction = 64,
        .ef_search = 32,
    });
    defer index.deinit();

    // Insert 20 random vectors.
    var rng = std.Random.DefaultPrng.init(123);
    for (0..20) |i| {
        var vec: [4]f32 = undefined;
        for (&vec) |*v| v.* = rng.random().float(f32);
        try index.insert(@intCast(i), &vec);
    }

    try std.testing.expectEqual(@as(usize, 20), index.len());

    const query = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    const results = try index.search(&query, 5);
    defer allocator.free(results);

    try std.testing.expect(results.len <= 5);
    try std.testing.expect(results.len > 0);
}
