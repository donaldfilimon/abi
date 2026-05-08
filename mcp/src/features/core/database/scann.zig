//! ScaNN: Scalable Nearest Neighbors with Learned Quantization
//!
//! Implements Google's ScaNN algorithm for efficient approximate nearest
//! neighbor search with learned quantization. Provides better recall vs
//! compression tradeoffs than traditional quantization methods.
//!
//! Key features:
//! - Anisotropic Vector Quantization (AVQ) for direction-aware compression
//! - Score-aware quantization loss for improved ranking
//! - Two-phase search: coarse quantization + fine reranking
//! - Asymmetric distance computation
//!
//! Performance targets:
//! - 10-100x faster than brute force at >95% recall
//! - 4-16x compression with minimal accuracy loss
//! - Sub-millisecond query latency for million-scale datasets

// Re-export sub-modules
pub const types = @import("scann/types.zig");
pub const codebook = @import("scann/codebook.zig");
pub const index = @import("scann/index.zig");

// Re-export primary types at top level for backward compatibility
pub const ScaNNConfig = types.ScaNNConfig;
pub const QuantizationType = types.QuantizationType;
pub const ScalarQuantParams = types.ScalarQuantParams;
pub const Partition = types.Partition;
pub const IndexStats = types.IndexStats;

pub const AVQCodebook = codebook.AVQCodebook;

pub const ScaNNIndex = index.ScaNNIndex;

// Re-export helpers for test access
pub const computeL2DistanceSquared = types.computeL2DistanceSquared;
pub const computeWeightedL2 = types.computeWeightedL2;

// Tests
const std = @import("std");

test "scalar quantization" {
    const params = ScalarQuantParams{
        .min_val = 0,
        .max_val = 1,
        .scale = 255,
        .offset = 0,
    };

    const code = params.quantize(0.5);
    try std.testing.expect(code == 127 or code == 128);

    const dequant = params.dequantize(127);
    try std.testing.expectApproxEqAbs(@as(f32, 0.498), dequant, 0.01);
}

test "avq codebook basic" {
    const allocator = std.testing.allocator;

    var cb = try AVQCodebook.init(allocator, 4, 8);
    defer cb.deinit();

    try std.testing.expect(cb.dimensions == 4);
    try std.testing.expect(cb.num_centroids == 8);
    try std.testing.expect(cb.centroids.len == 32);
}

test "scann index basic" {
    const allocator = std.testing.allocator;

    const config = ScaNNConfig{
        .dimensions = 4,
        .num_partitions = 2,
        .partitions_to_search = 2,
        .quantization_type = .scalar,
    };

    var idx = try ScaNNIndex.init(allocator, config);
    defer idx.deinit();

    const vectors = [_]f32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };

    try idx.build(&vectors);

    try std.testing.expect(idx.num_vectors == 4);
    try std.testing.expect(idx.stats.build_complete);
}

test "scann search" {
    const allocator = std.testing.allocator;

    const config = ScaNNConfig{
        .dimensions = 4,
        .num_partitions = 2,
        .partitions_to_search = 2,
        .quantization_type = .scalar,
        .rerank_factor = 2,
    };

    var idx = try ScaNNIndex.init(allocator, config);
    defer idx.deinit();

    const vectors = [_]f32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };

    try idx.build(&vectors);

    const query = [_]f32{ 0.9, 0.1, 0, 0 };
    const results = try idx.search(&query, 2);
    defer allocator.free(results);

    try std.testing.expect(results.len >= 1);
    try std.testing.expect(results[0].id == 0);
}

test "scann partition search with 200 vectors and 10 partitions" {
    const allocator = std.testing.allocator;
    const dim: u32 = 8;
    const num_vectors: u32 = 200;
    const num_partitions: u32 = 10;

    var flat_vectors = try allocator.alloc(f32, num_vectors * dim);
    defer allocator.free(flat_vectors);

    var seed: u64 = 12345;
    for (flat_vectors) |*v| {
        seed = seed *% 6364136223846793005 +% 1442695040888963407;
        v.* = @as(f32, @floatFromInt(@as(i32, @truncate(@as(i64, @bitCast(seed >> 33)))))) / 2147483648.0;
    }

    const config = ScaNNConfig{
        .dimensions = dim,
        .num_partitions = num_partitions,
        .partitions_to_search = 10,
        .quantization_type = .scalar,
        .rerank_factor = 4,
        .seed = 99,
    };

    var idx = try ScaNNIndex.init(allocator, config);
    defer idx.deinit();

    try idx.build(flat_vectors);

    try std.testing.expect(idx.num_vectors == num_vectors);
    try std.testing.expect(idx.partitions.items.len == num_partitions);

    const query = flat_vectors[0..dim];
    const results = try idx.search(query, 5);
    defer allocator.free(results);

    try std.testing.expect(results.len >= 1);
    var found_self = false;
    for (results) |r| {
        if (r.id == 0) {
            found_self = true;
            try std.testing.expect(r.score < 0.01);
        }
    }
    try std.testing.expect(found_self);

    // Compute brute-force top-5 to check recall
    var brute_force_top: [5]struct { id: u32, dist: f32 } = undefined;
    for (&brute_force_top) |*bf| {
        bf.* = .{ .id = 0, .dist = std.math.inf(f32) };
    }

    for (0..num_vectors) |i| {
        const vec = flat_vectors[i * dim ..][0..dim];
        const dist = computeL2DistanceSquared(query, vec);
        var pos: usize = 5;
        while (pos > 0 and dist < brute_force_top[pos - 1].dist) : (pos -= 1) {}
        if (pos < 5) {
            var j: usize = 4;
            while (j > pos) : (j -= 1) {
                brute_force_top[j] = brute_force_top[j - 1];
            }
            brute_force_top[pos] = .{ .id = @intCast(i), .dist = dist };
        }
    }

    var recall_hits: u32 = 0;
    for (results) |r| {
        for (brute_force_top) |bf| {
            if (r.id == bf.id) {
                recall_hits += 1;
                break;
            }
        }
    }
    try std.testing.expect(recall_hits >= 3);
}

test "scann num_probes affects recall" {
    const allocator = std.testing.allocator;
    const dim: u32 = 8;
    const num_vectors: u32 = 200;

    var flat_vectors = try allocator.alloc(f32, num_vectors * dim);
    defer allocator.free(flat_vectors);

    var seed: u64 = 67890;
    for (flat_vectors) |*v| {
        seed = seed *% 6364136223846793005 +% 1442695040888963407;
        v.* = @as(f32, @floatFromInt(@as(i32, @truncate(@as(i64, @bitCast(seed >> 33)))))) / 2147483648.0;
    }

    // Compute brute-force top-10
    const query = flat_vectors[0..dim];
    var bf_top: [10]u32 = undefined;
    var bf_dists: [10]f32 = undefined;
    for (&bf_dists) |*d| d.* = std.math.inf(f32);

    for (0..num_vectors) |i| {
        const vec = flat_vectors[i * dim ..][0..dim];
        const dist = computeL2DistanceSquared(query, vec);
        var pos: usize = 10;
        while (pos > 0 and dist < bf_dists[pos - 1]) : (pos -= 1) {}
        if (pos < 10) {
            var j: usize = 9;
            while (j > pos) : (j -= 1) {
                bf_top[j] = bf_top[j - 1];
                bf_dists[j] = bf_dists[j - 1];
            }
            bf_top[pos] = @intCast(i);
            bf_dists[pos] = dist;
        }
    }

    // Search with num_probes=1
    const config_low = ScaNNConfig{
        .dimensions = dim,
        .num_partitions = 10,
        .partitions_to_search = 1,
        .quantization_type = .scalar,
        .rerank_factor = 4,
        .seed = 77,
    };

    var index_low = try ScaNNIndex.init(allocator, config_low);
    defer index_low.deinit();
    try index_low.build(flat_vectors);

    const results_low = try index_low.search(query, 10);
    defer allocator.free(results_low);

    var recall_low: u32 = 0;
    for (results_low) |r| {
        for (bf_top) |bf_id| {
            if (r.id == bf_id) {
                recall_low += 1;
                break;
            }
        }
    }

    // Search with num_probes=10 (all partitions)
    const config_high = ScaNNConfig{
        .dimensions = dim,
        .num_partitions = 10,
        .partitions_to_search = 10,
        .quantization_type = .scalar,
        .rerank_factor = 4,
        .seed = 77,
    };

    var index_high = try ScaNNIndex.init(allocator, config_high);
    defer index_high.deinit();
    try index_high.build(flat_vectors);

    const results_high = try index_high.search(query, 10);
    defer allocator.free(results_high);

    var recall_high: u32 = 0;
    for (results_high) |r| {
        for (bf_top) |bf_id| {
            if (r.id == bf_id) {
                recall_high += 1;
                break;
            }
        }
    }

    try std.testing.expect(recall_high >= recall_low);
    try std.testing.expect(recall_high >= 8);
}

test "scann search with clustered vectors" {
    const allocator = std.testing.allocator;
    const dim: u32 = 4;

    const num_vectors: u32 = 30;
    var flat_vectors: [num_vectors * dim]f32 = undefined;

    var seed: u64 = 11111;
    for (0..num_vectors) |i| {
        const cluster = i / 10;
        for (0..dim) |d| {
            seed = seed *% 6364136223846793005 +% 1442695040888963407;
            const noise = @as(f32, @floatFromInt(@as(i32, @truncate(@as(i64, @bitCast(seed >> 33)))))) / 2147483648.0 * 0.5;
            flat_vectors[i * dim + d] = if (d == cluster) 10.0 + noise else noise;
        }
    }

    const config = ScaNNConfig{
        .dimensions = dim,
        .num_partitions = 3,
        .partitions_to_search = 1,
        .quantization_type = .scalar,
        .rerank_factor = 4,
        .seed = 42,
    };

    var idx = try ScaNNIndex.init(allocator, config);
    defer idx.deinit();

    try idx.build(&flat_vectors);

    const query0 = [_]f32{ 10.0, 0.0, 0.0, 0.0 };
    const results0 = try idx.search(&query0, 5);
    defer allocator.free(results0);
    for (results0) |r| {
        try std.testing.expect(r.id < 10);
    }

    const query1 = [_]f32{ 0.0, 10.0, 0.0, 0.0 };
    const results1 = try idx.search(&query1, 5);
    defer allocator.free(results1);
    for (results1) |r| {
        try std.testing.expect(r.id >= 10 and r.id < 20);
    }

    const query2 = [_]f32{ 0.0, 0.0, 10.0, 0.0 };
    const results2 = try idx.search(&query2, 5);
    defer allocator.free(results2);
    for (results2) |r| {
        try std.testing.expect(r.id >= 20 and r.id < 30);
    }
}

test "scann index stats" {
    const allocator = std.testing.allocator;

    const config = ScaNNConfig{
        .dimensions = 4,
        .num_partitions = 2,
        .partitions_to_search = 2,
        .quantization_type = .scalar,
    };

    var idx = try ScaNNIndex.init(allocator, config);
    defer idx.deinit();

    const vectors = [_]f32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };

    try idx.build(&vectors);

    const stats = idx.getStats();
    try std.testing.expect(stats.num_vectors == 4);
    try std.testing.expect(stats.num_partitions == 2);
    try std.testing.expect(stats.build_complete);
    try std.testing.expect(stats.memory_bytes > 0);
}

// Pull in sub-modules for test discovery
comptime {
    if (@import("builtin").is_test) {
        _ = types;
        _ = codebook;
        _ = index;
    }
}
