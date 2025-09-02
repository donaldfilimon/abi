const std = @import("std");
const wdbx = @import("../src/@wdbx.zig");

test "CLI Command: help" {
    // Test that help command works
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var cli = try wdbx.WdbxCLI.init(allocator, .{
        .command = .help,
    });
    defer cli.deinit();

    // This should not crash
    try cli.run();
}

test "CLI Command: version" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var cli = try wdbx.WdbxCLI.init(allocator, .{
        .command = .version,
    });
    defer cli.deinit();

    try cli.run();
}

test "CLI Command: add vector" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var cli = try wdbx.WdbxCLI.init(allocator, .{
        .command = .add,
        .vector = "1.0,2.0,3.0,4.0",
    });
    defer cli.deinit();

    try cli.run();
}

test "CLI Command: stats" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var cli = try wdbx.WdbxCLI.init(allocator, .{
        .command = .stats,
    });
    defer cli.deinit();

    try cli.run();
}

test "CLI Command: monitor" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var cli = try wdbx.WdbxCLI.init(allocator, .{
        .command = .monitor,
    });
    defer cli.deinit();

    try cli.run();
}

test "CLI Command: optimize" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var cli = try wdbx.WdbxCLI.init(allocator, .{
        .command = .optimize,
    });
    defer cli.deinit();

    try cli.run();
}

test "CLI Command: save" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var cli = try wdbx.WdbxCLI.init(allocator, .{
        .command = .save,
    });
    defer cli.deinit();

    try cli.run();
}

test "CLI Command: load" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var cli = try wdbx.WdbxCLI.init(allocator, .{
        .command = .load,
    });
    defer cli.deinit();

    try cli.run();
}

test "CLI Command: http" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var cli = try wdbx.WdbxCLI.init(allocator, .{
        .command = .http,
    });
    defer cli.deinit();

    try cli.run();
}

test "CLI Command: tcp" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var cli = try wdbx.WdbxCLI.init(allocator, .{
        .command = .tcp,
    });
    defer cli.deinit();

    try cli.run();
}

test "CLI Command: ws" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var cli = try wdbx.WdbxCLI.init(allocator, .{
        .command = .ws,
    });
    defer cli.deinit();

    try cli.run();
}

test "CLI Command: gen_token" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var cli = try wdbx.WdbxCLI.init(allocator, .{
        .command = .gen_token,
    });
    defer cli.deinit();

    try cli.run();
}

test "Vector operations: add and search" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create database
    var db = try wdbx.WdbxProduction.init(allocator, .{});
    defer db.close();

    // Add vectors
    const vector1 = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const vector2 = [_]f32{ 2.0, 3.0, 4.0, 5.0 };
    const vector3 = [_]f32{ 0.5, 1.5, 2.5, 3.5 };

    const id1 = try db.addEmbedding(&vector1);
    const id2 = try db.addEmbedding(&vector2);
    const id3 = try db.addEmbedding(&vector3);

    try std.testing.expectEqual(@as(u64, 0), id1);
    try std.testing.expectEqual(@as(u64, 1), id2);
    try std.testing.expectEqual(@as(u64, 2), id3);

    // Search for nearest neighbor
    const query_vector = [_]f32{ 1.1, 2.1, 3.1, 4.1 };
    const results = try db.search(&query_vector, 2, allocator);
    defer allocator.free(results);

    try std.testing.expectEqual(@as(usize, 2), results.len);

    // First result should be closest (vector1)
    try std.testing.expectEqual(@as(u64, 0), results[0].index);
    try std.testing.expect(results[0].score < results[1].score);
}

test "Compression: 8-bit quantization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const compressed = try wdbx.CompressionType.quantization_8bit.compress(&data, allocator);
    defer allocator.free(compressed);

    // Should have header (8 bytes) + data (5 bytes) = 13 bytes
    try std.testing.expectEqual(@as(usize, 13), compressed.len);
}

test "Compression: 4-bit quantization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const compressed = try wdbx.CompressionType.quantization_4bit.compress(&data, allocator);
    defer allocator.free(compressed);

    // Should have header (8 bytes) + packed data (2 bytes for 4 values) = 10 bytes
    try std.testing.expectEqual(@as(usize, 10), compressed.len);
}

test "Compression: product quantization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const compressed = try wdbx.CompressionType.pq_compression.compress(&data, allocator);
    defer allocator.free(compressed);

    // Should have metadata (4 bytes) + subvectors (2 subvectors of 4 elements each)
    try std.testing.expectEqual(@as(usize, 6), compressed.len);
}

test "Compression: delta encoding" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const compressed = try wdbx.CompressionType.delta_encoding.compress(&data, allocator);
    defer allocator.free(compressed);

    // Should have first value (4 bytes) + 3 deltas (12 bytes) = 16 bytes
    try std.testing.expectEqual(@as(usize, 16), compressed.len);
}

test "LSH Index: insert and query" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var lsh = try wdbx.LshIndex.init(allocator, 128, 8);
    defer lsh.deinit();

    // Insert some vectors
    try lsh.insert("hash1", 1);
    try lsh.insert("hash2", 2);
    try lsh.insert("hash1", 3); // Same hash as first

    // Query
    const results1 = lsh.query("hash1");
    try std.testing.expectEqual(@as(usize, 2), results1.len);

    const results2 = lsh.query("hash2");
    try std.testing.expectEqual(@as(usize, 1), results2.len);
    try std.testing.expectEqual(@as(u64, 2), results2[0]);
}

test "Metrics: initialization and export" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var metrics = try wdbx.Metrics.init(allocator, 4);
    defer allocator.free(metrics.shard_distribution);
    defer allocator.destroy(metrics);

    // Test metrics operations
    _ = metrics.operations_total.fetchAdd(1, .monotonic);
    _ = metrics.operations_failed.fetchAdd(0, .monotonic);

    try std.testing.expectEqual(@as(u64, 1), metrics.operations_total.load(.monotonic));
    try std.testing.expectEqual(@as(u64, 0), metrics.operations_failed.load(.monotonic));
}

test "SIMD Capabilities: detection" {
    const caps = wdbx.SimdCapabilities.detect();

    // Should detect at least some capabilities on modern systems
    try std.testing.expect(caps.has_sse or caps.has_avx or caps.has_neon);
}

test "HTTP Server: initialization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var server = try wdbx.WdbxHttpServer.init(allocator, .{});
    defer server.deinit();

    // Server should be initialized
    try std.testing.expect(server.config.port == 8080);
    try std.testing.expect(std.mem.eql(u8, server.config.host, "127.0.0.1"));
}

test "Production Config: defaults" {
    const config = wdbx.ProductionConfig{};

    try std.testing.expectEqual(@as(usize, 16), config.shard_count);
    try std.testing.expectEqual(@as(usize, 1_000_000), config.max_vectors_per_shard);
    try std.testing.expect(config.auto_rebalance == true);
    try std.testing.expect(config.enable_metrics == true);
    try std.testing.expect(config.enable_simd == true);
}
