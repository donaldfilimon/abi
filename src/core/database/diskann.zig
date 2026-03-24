//! DiskANN: Disk-based Approximate Nearest Neighbor Search
//!
//! Implements a graph-based ANN index optimized for disk I/O, enabling
//! billion-scale vector search with memory efficiency. Based on the
//! Microsoft Research DiskANN paper.
//!
//! Key features:
//! - Graph-based index stored on disk with efficient I/O patterns
//! - Vamana graph construction for high recall
//! - PQ (Product Quantization) compressed vectors for memory efficiency
//! - Beam search with disk prefetching
//!
//! Performance targets:
//! - Billion-scale datasets with <100GB memory
//! - Sub-millisecond query latency with SSD
//! - >95% recall@10 on standard benchmarks

// Re-export sub-modules
pub const types = @import("diskann/types.zig");
pub const codebook = @import("diskann/codebook.zig");
pub const index = @import("diskann/index.zig");
pub const vamana = @import("diskann/vamana.zig");

// Re-export primary types at top level for backward compatibility
pub const DiskANNConfig = types.DiskANNConfig;
pub const DiskNode = types.DiskNode;
pub const SearchCandidate = types.SearchCandidate;
pub const IndexStats = types.IndexStats;
pub const PersistError = types.PersistError;

pub const PQCodebook = codebook.PQCodebook;

pub const DiskANNIndex = index.DiskANNIndex;

pub const VamanaConfig = vamana.VamanaConfig;
pub const VamanaSearchResult = vamana.VamanaSearchResult;
pub const VamanaIndex = vamana.VamanaIndex;

// Re-export helpers for test access
pub const computeL2DistanceSquared = types.computeL2DistanceSquared;

// Constants
pub const DISKANN_MAGIC = types.DISKANN_MAGIC;
pub const DISKANN_FORMAT_VERSION = types.DISKANN_FORMAT_VERSION;
pub const DISKANN_HEADER_SIZE = types.DISKANN_HEADER_SIZE;

// Tests
const std = @import("std");
const index_mod = @import("index.zig");

test "pq codebook basic" {
    const allocator = std.testing.allocator;

    var cb = try PQCodebook.init(allocator, 4, 2, 256);
    defer cb.deinit();

    try std.testing.expect(cb.num_subspaces == 4);
    try std.testing.expect(cb.centroids.len == 4 * 256 * 2);
}

test "pq encoding" {
    const allocator = std.testing.allocator;

    var cb = try PQCodebook.init(allocator, 2, 2, 4);
    defer cb.deinit();

    // Simple initialization for testing
    @memset(cb.centroids, 0);
    cb.centroids[0] = 1.0; // Centroid 0, subspace 0
    cb.centroids[4] = 2.0; // Centroid 1, subspace 0

    const vector = [_]f32{ 0.9, 0.0, 0.0, 0.0 };
    var codes: [2]u8 = undefined;
    cb.encode(&vector, &codes);

    // Should encode to centroid 0 for subspace 0 (closest to 1.0)
    try std.testing.expect(codes[0] == 0);
}

test "diskann index basic" {
    const allocator = std.testing.allocator;

    const config = DiskANNConfig{
        .dimensions = 4,
        .max_degree = 4,
        .build_list_size = 10,
        .search_list_size = 10,
        .pq_subspaces = 2,
    };

    var idx = try DiskANNIndex.init(allocator, config);
    defer idx.deinit();

    // Build with small dataset
    const vectors = [_]f32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
        1, 1, 0, 0,
    };

    try idx.build(&vectors);

    try std.testing.expect(idx.num_vectors == 5);
    try std.testing.expect(idx.stats.build_complete);
}

test "diskann search" {
    const allocator = std.testing.allocator;

    const config = DiskANNConfig{
        .dimensions = 4,
        .max_degree = 4,
        .build_list_size = 10,
        .search_list_size = 10,
        .pq_subspaces = 2,
    };

    var idx = try DiskANNIndex.init(allocator, config);
    defer idx.deinit();

    const vectors = [_]f32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };

    try idx.build(&vectors);

    // Search for vector similar to first
    const query = [_]f32{ 0.9, 0.1, 0, 0 };
    const results = try idx.search(&query, 2);
    defer allocator.free(results);

    try std.testing.expect(results.len >= 1);
    // First result should be closest to query
    try std.testing.expect(results[0].id == 0);
}

test "diskann save and load round-trip" {
    const allocator = std.testing.allocator;

    const config = DiskANNConfig{
        .dimensions = 4,
        .max_degree = 4,
        .build_list_size = 10,
        .search_list_size = 10,
        .pq_subspaces = 2,
    };

    // Build an index
    var idx = try DiskANNIndex.init(allocator, config);
    defer idx.deinit();

    const vectors = [_]f32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
        1, 1, 0, 0,
    };
    try idx.build(&vectors);

    // Save to a temp file
    const tmp_path = "/tmp/diskann_test_roundtrip.dann";
    try idx.save(tmp_path);

    // Load it back
    var loaded = try DiskANNIndex.load(allocator, tmp_path);
    defer loaded.deinit();

    // Verify metadata
    try std.testing.expectEqual(idx.num_vectors, loaded.num_vectors);
    try std.testing.expectEqual(idx.entry_point, loaded.entry_point);
    try std.testing.expectEqual(idx.config.dimensions, loaded.config.dimensions);
    try std.testing.expectEqual(idx.config.max_degree, loaded.config.max_degree);
    try std.testing.expectEqual(idx.config.pq_subspaces, loaded.config.pq_subspaces);
    try std.testing.expect(loaded.stats.build_complete);

    // Verify graph structure
    for (0..idx.num_vectors) |i| {
        try std.testing.expectEqual(idx.graph.items[i].items.len, loaded.graph.items[i].items.len);
        for (idx.graph.items[i].items, loaded.graph.items[i].items) |a, b| {
            try std.testing.expectEqual(a, b);
        }
    }

    // Verify PQ codes
    for (0..idx.num_vectors) |i| {
        try std.testing.expectEqualSlices(u8, idx.pq_codes.items[i], loaded.pq_codes.items[i]);
    }

    // Verify codebook centroids
    const orig_cb = idx.codebook.?;
    const loaded_cb = loaded.codebook.?;
    try std.testing.expectEqual(orig_cb.centroids.len, loaded_cb.centroids.len);
    for (orig_cb.centroids, loaded_cb.centroids) |a, b| {
        try std.testing.expectApproxEqAbs(a, b, 1e-6);
    }

    // Clean up temp file
    std.posix.unlink(tmp_path) catch {};
}

test "diskann save rejects unbuilt index" {
    const allocator = std.testing.allocator;
    var idx = try DiskANNIndex.init(allocator, .{});
    defer idx.deinit();

    const result = idx.save("/tmp/diskann_test_notbuilt.dann");
    try std.testing.expectError(error.NotBuilt, result);
}

test "diskann load rejects invalid magic" {
    const allocator = std.testing.allocator;

    // Write a file with bad magic
    const tmp_path = "/tmp/diskann_test_badmagic.dann";
    const fd = std.posix.openatZ(std.posix.AT.FDCWD, tmp_path, .{
        .ACCMODE = .WRONLY,
        .CREAT = true,
        .TRUNC = true,
    }, 0o644) catch return;
    defer std.posix.close(fd);

    var buf: [4096]u8 = [_]u8{0} ** 4096;
    @memcpy(buf[0..4], "NOPE");
    _ = std.posix.write(fd, &buf) catch return;

    const result = DiskANNIndex.load(allocator, tmp_path);
    try std.testing.expectError(error.InvalidMagic, result);

    std.posix.unlink(tmp_path) catch {};
}

test "vamana build 100 random vectors and search returns correct nearest" {
    const allocator = std.testing.allocator;
    const dim: u32 = 8;
    const n: u32 = 100;

    var prng = std.Random.DefaultPrng.init(999);
    const random = prng.random();

    // Generate random vectors
    var data: [n * dim]f32 = undefined;
    for (&data) |*v| {
        v.* = random.float(f32) * 2.0 - 1.0;
    }

    var idx = VamanaIndex.init(allocator, dim, .{
        .max_degree = 16,
        .alpha = 1.2,
        .build_list_size = 32,
        .search_list_size = 32,
    });
    defer idx.deinit();
    try idx.build(&data);

    try std.testing.expectEqual(@as(u32, n), idx.num_vectors);

    // Use the first vector as query; it should be its own nearest neighbor
    const query = data[0..dim];
    const results = try idx.search(query, 5);
    defer allocator.free(results);

    try std.testing.expect(results.len >= 1);
    // The exact vector should be returned with distance ~0
    try std.testing.expect(results[0].distance < 0.01);

    // Brute-force find true nearest neighbor
    var bf_best_id: u32 = 0;
    var bf_best_dist: f32 = std.math.inf(f32);
    for (0..n) |i| {
        const d = computeL2DistanceSquared(query, data[i * dim ..][0..dim]);
        if (d < bf_best_dist) {
            bf_best_dist = d;
            bf_best_id = @intCast(i);
        }
    }
    try std.testing.expectEqual(bf_best_id, results[0].id);
}

test "vamana max_degree constraint is maintained" {
    const allocator = std.testing.allocator;
    const dim: u32 = 4;
    const n: u32 = 50;
    const max_deg: u32 = 8;

    var prng = std.Random.DefaultPrng.init(7777);
    const random = prng.random();

    var data: [n * dim]f32 = undefined;
    for (&data) |*v| {
        v.* = random.float(f32);
    }

    var idx = VamanaIndex.init(allocator, dim, .{
        .max_degree = max_deg,
        .alpha = 1.2,
        .build_list_size = 24,
        .search_list_size = 24,
    });
    defer idx.deinit();
    try idx.build(&data);

    try std.testing.expect(idx.maxOutDegree() <= max_deg);
}

test "vamana different alpha values affect graph density" {
    const allocator = std.testing.allocator;
    const dim: u32 = 4;
    const n: u32 = 40;

    var prng = std.Random.DefaultPrng.init(1234);
    const random = prng.random();

    var data: [n * dim]f32 = undefined;
    for (&data) |*v| {
        v.* = random.float(f32);
    }

    // Low alpha (strict pruning) -> fewer edges
    var index_low = VamanaIndex.init(allocator, dim, .{
        .max_degree = 16,
        .alpha = 1.0,
        .build_list_size = 24,
        .search_list_size = 24,
    });
    defer index_low.deinit();
    try index_low.build(&data);

    // High alpha (relaxed pruning) -> more edges
    var index_high = VamanaIndex.init(allocator, dim, .{
        .max_degree = 16,
        .alpha = 2.0,
        .build_list_size = 24,
        .search_list_size = 24,
    });
    defer index_high.deinit();
    try index_high.build(&data);

    // Count total edges
    var edges_low: u64 = 0;
    for (index_low.graph.items) |adj| {
        edges_low += adj.items.len;
    }
    var edges_high: u64 = 0;
    for (index_high.graph.items) |adj| {
        edges_high += adj.items.len;
    }

    // Higher alpha should produce at least as many edges (more lenient pruning)
    try std.testing.expect(edges_high >= edges_low);

    // Both should still respect max_degree
    try std.testing.expect(index_low.maxOutDegree() <= 16);
    try std.testing.expect(index_high.maxOutDegree() <= 16);
}

// Pull in sub-modules for test discovery
comptime {
    if (@import("builtin").is_test) {
        _ = types;
        _ = codebook;
        _ = index;
        _ = vamana;
    }
}
