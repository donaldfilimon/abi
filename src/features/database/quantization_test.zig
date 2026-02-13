//! Tests for Vector Quantization
//!
//! Covers ScalarQuantizer (8-bit, 4-bit) and ProductQuantizer
//! (basic encode/decode, distance tables).

const std = @import("std");
const testing = std.testing;
const quantization = @import("quantization.zig");
const product_quantizer = @import("product_quantizer.zig");

const ScalarQuantizer = quantization.ScalarQuantizer;
const ProductQuantizer = product_quantizer.ProductQuantizer;

test "scalar quantizer 8-bit basic" {
    const allocator = testing.allocator;

    var sq = try ScalarQuantizer.init(allocator, 4, .{ .bits = 8 });
    defer sq.deinit();

    const vectors = [_][]const f32{
        &[_]f32{ 0.0, 1.0, -1.0, 0.5 },
        &[_]f32{ 0.5, 0.0, 0.0, 1.0 },
        &[_]f32{ 1.0, -1.0, 1.0, 0.0 },
    };

    try sq.train(&vectors);

    const stats = sq.getStats();
    try testing.expectEqual(@as(usize, 4), stats.dimension);
    try testing.expectEqual(@as(u8, 8), stats.bits);
    try testing.expectEqual(@as(f32, 4.0), stats.compression_ratio);

    // Encode and decode
    var encoded: [4]u8 = undefined;
    const bytes_written = try sq.encode(&vectors[0], &encoded);
    try testing.expectEqual(@as(usize, 4), bytes_written);

    var decoded: [4]f32 = undefined;
    try sq.decode(&encoded, &decoded);

    // Check approximate reconstruction
    for (vectors[0], decoded) |orig, dec| {
        try testing.expect(@abs(orig - dec) < 0.02); // Allow small quantization error
    }
}

test "scalar quantizer 4-bit compression" {
    const allocator = testing.allocator;

    var sq = try ScalarQuantizer.init(allocator, 8, .{ .bits = 4 });
    defer sq.deinit();

    const vectors = [_][]const f32{
        &[_]f32{ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 },
        &[_]f32{ 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0 },
    };

    try sq.train(&vectors);

    const stats = sq.getStats();
    try testing.expectEqual(@as(usize, 4), stats.bytes_per_vector); // 8 dims / 2 = 4 bytes
    try testing.expectEqual(@as(f32, 8.0), stats.compression_ratio);

    var encoded: [4]u8 = undefined;
    _ = try sq.encode(&vectors[0], &encoded);

    var decoded: [8]f32 = undefined;
    try sq.decode(&encoded, &decoded);

    // 4-bit has larger quantization error
    for (vectors[0], decoded) |orig, dec| {
        try testing.expect(@abs(orig - dec) < 0.1);
    }
}

test "product quantizer basic" {
    const allocator = testing.allocator;

    // 8-dim vectors with 2 subvectors of 4 dims each
    var pq = try ProductQuantizer.init(allocator, 8, .{
        .num_subvectors = 2,
        .bits_per_code = 8,
        .kmeans_iterations = 5,
    });
    defer pq.deinit();

    const vectors = [_][]const f32{
        &[_]f32{ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 },
        &[_]f32{ 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0 },
        &[_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 },
        &[_]f32{ 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1 },
    };

    try pq.train(&vectors, .{ .num_subvectors = 2, .bits_per_code = 8, .kmeans_iterations = 5 });

    const stats = pq.getStats();
    try testing.expectEqual(@as(usize, 8), stats.dimension);
    try testing.expectEqual(@as(u8, 2), stats.num_subvectors);
    try testing.expectEqual(@as(usize, 2), stats.bytes_per_vector); // 2 subvectors * 1 byte each
    try testing.expectEqual(@as(f32, 16.0), stats.compression_ratio); // 32 bytes -> 2 bytes

    // Encode and decode
    var encoded: [2]u8 = undefined;
    _ = try pq.encode(&vectors[0], &encoded);

    var decoded: [8]f32 = undefined;
    try pq.decode(&encoded, &decoded);

    // PQ has larger reconstruction error but should be in reasonable range
    var total_error: f32 = 0.0;
    for (vectors[0], decoded) |orig, dec| {
        total_error += @abs(orig - dec);
    }
    try testing.expect(total_error / 8.0 < 0.5); // Average error < 0.5
}

test "product quantizer distance table" {
    const allocator = testing.allocator;

    var pq = try ProductQuantizer.init(allocator, 4, .{
        .num_subvectors = 2,
        .bits_per_code = 8,
        .kmeans_iterations = 3,
    });
    defer pq.deinit();

    const vectors = [_][]const f32{
        &[_]f32{ 0.0, 0.0, 1.0, 1.0 },
        &[_]f32{ 1.0, 1.0, 0.0, 0.0 },
    };

    try pq.train(&vectors, .{ .num_subvectors = 2, .bits_per_code = 8, .kmeans_iterations = 3 });

    const query = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    const dist_table = try pq.computeDistanceTable(&query);
    defer allocator.free(dist_table);

    var encoded: [2]u8 = undefined;
    _ = try pq.encode(&vectors[0], &encoded);

    const dist = pq.asymmetricDistanceWithTable(dist_table, &encoded);
    try testing.expect(dist >= 0.0); // Distance should be non-negative
}
