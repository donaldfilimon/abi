//! Tests for the in-memory vector database.
//!
//! Covers search ordering, norm caching, update/delete cache
//! consistency, fast cosine similarity, and diagnostics.

const std = @import("std");
const simd = @import("../../services/shared/simd/mod.zig");
const database_mod = @import("database.zig");
const Database = database_mod.Database;
const computeCosineSimilarityFast = database_mod.computeCosineSimilarityFast;

test "search sorts by descending similarity and truncates" {
    var db = try Database.init(std.testing.allocator, "search-test");
    defer db.deinit();

    try db.insert(1, &.{ 1.0, 0.0 }, null);
    try db.insert(2, &.{ 0.0, 1.0 }, null);
    try db.insert(3, &.{ 1.0, 1.0 }, null);

    const results = try db.search(std.testing.allocator, &.{ 1.0, 0.0 }, 2);
    defer std.testing.allocator.free(results);

    try std.testing.expectEqual(@as(usize, 2), results.len);
    try std.testing.expectEqual(@as(u64, 1), results[0].id);
    try std.testing.expectEqual(@as(u64, 3), results[1].id);
}

test "searchInto respects provided buffer" {
    var db = try Database.init(std.testing.allocator, "search-into-test");
    defer db.deinit();

    try db.insert(1, &.{ 1.0, 0.0 }, null);
    try db.insert(2, &.{ 0.0, 1.0 }, null);
    try db.insert(3, &.{ 1.0, 1.0 }, null);

    var buf: [2]database_mod.SearchResult = undefined;
    const count = db.searchInto(&.{ 1.0, 0.0 }, 3, &buf);

    try std.testing.expectEqual(@as(usize, 2), count);
    try std.testing.expectEqual(@as(u64, 1), buf[0].id);
    try std.testing.expectEqual(@as(u64, 3), buf[1].id);
}

test "database with cached norms" {
    var db = try Database.initWithConfig(std.testing.allocator, "cached-norms-test", .{
        .cache_norms = true,
        .initial_capacity = 10,
    });
    defer db.deinit();

    try db.insert(1, &.{ 1.0, 0.0, 0.0, 0.0 }, null);
    try db.insert(2, &.{ 0.0, 1.0, 0.0, 0.0 }, null);
    try db.insert(3, &.{ 0.5, 0.5, 0.5, 0.5 }, null);

    // Check that norms are cached
    try std.testing.expectEqual(@as(usize, 3), db.cached_norms.items.len);

    // Verify stats include norm cache info
    const s = db.stats();
    try std.testing.expect(s.norm_cache_enabled);
    try std.testing.expect(s.memory_bytes > 0);

    // Search should use cached norms
    const results = try db.search(std.testing.allocator, &.{ 1.0, 0.0, 0.0, 0.0 }, 2);
    defer std.testing.allocator.free(results);

    try std.testing.expectEqual(@as(usize, 2), results.len);
    try std.testing.expectEqual(@as(u64, 1), results[0].id);
}

test "database update maintains norm cache" {
    var db = try Database.initWithConfig(std.testing.allocator, "update-cache-test", .{
        .cache_norms = true,
    });
    defer db.deinit();

    try db.insert(1, &.{ 1.0, 0.0 }, null);
    const original_norm = db.cached_norms.items[0];

    // Update vector
    _ = try db.update(1, &.{ 0.0, 1.0 });

    // Norm should be updated
    try std.testing.expectEqual(@as(usize, 1), db.cached_norms.items.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), db.cached_norms.items[0], 1e-6);
    try std.testing.expect(db.cached_norms.items[0] == original_norm); // Both are 1.0
}

test "database delete maintains norm cache consistency" {
    var db = try Database.initWithConfig(std.testing.allocator, "delete-cache-test", .{
        .cache_norms = true,
    });
    defer db.deinit();

    try db.insert(1, &.{ 1.0, 0.0 }, null);
    try db.insert(2, &.{ 0.0, 1.0 }, null);
    try db.insert(3, &.{ 0.5, 0.5 }, null);

    try std.testing.expectEqual(@as(usize, 3), db.cached_norms.items.len);

    // Delete middle element
    try std.testing.expect(db.delete(2));

    // Norm cache should remain consistent
    try std.testing.expectEqual(@as(usize, 2), db.cached_norms.items.len);
    try std.testing.expectEqual(@as(usize, 2), db.records.items.len);
}

test "fast cosine similarity matches regular" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 4.0, 3.0, 2.0, 1.0 };

    const a_norm = simd.vectorL2Norm(&a);
    const b_norm = simd.vectorL2Norm(&b);

    const fast_result = computeCosineSimilarityFast(&a, a_norm, &b, b_norm);
    const regular_result = simd.cosineSimilarity(&a, &b);

    try std.testing.expectApproxEqAbs(regular_result, fast_result, 1e-6);
}

test "database diagnostics" {
    var db = try Database.initWithConfig(std.testing.allocator, "diagnostics-test", .{
        .cache_norms = true,
        .initial_capacity = 100,
    });
    defer db.deinit();

    // Insert some test data with metadata
    try db.insert(1, &.{ 1.0, 0.0, 0.0, 0.0 }, "metadata1");
    try db.insert(2, &.{ 0.0, 1.0, 0.0, 0.0 }, "metadata2");
    try db.insert(3, &.{ 0.5, 0.5, 0.5, 0.5 }, null);

    // Get diagnostics
    const diag = db.diagnostics();

    // Verify basic info
    try std.testing.expectEqualStrings("diagnostics-test", diag.name);
    try std.testing.expectEqual(@as(usize, 3), diag.vector_count);
    try std.testing.expectEqual(@as(usize, 4), diag.dimension);

    // Verify memory stats
    try std.testing.expectEqual(@as(usize, 3 * 4 * @sizeOf(f32)), diag.memory.vector_bytes);
    try std.testing.expectEqual(@as(usize, 3 * @sizeOf(f32)), diag.memory.norm_cache_bytes);
    try std.testing.expect(diag.memory.metadata_bytes > 0);
    try std.testing.expect(diag.memory.total_bytes > 0);
    try std.testing.expect(diag.memory.efficiency > 0.0 and diag.memory.efficiency <= 1.0);

    // Verify config status
    try std.testing.expect(diag.config.norm_cache_enabled);
    try std.testing.expect(!diag.config.vector_pool_enabled);
    try std.testing.expect(!diag.config.thread_safe_enabled);

    // Verify health metrics
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), diag.index_health, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), diag.norm_cache_health, 0.01);
    try std.testing.expect(diag.isHealthy());

    // Verify formatting works
    const formatted = try diag.formatToString(std.testing.allocator);
    defer std.testing.allocator.free(formatted);
    try std.testing.expect(std.mem.indexOf(u8, formatted, "diagnostics-test") != null);
    try std.testing.expect(std.mem.indexOf(u8, formatted, "Vectors: 3") != null);
}

test "diagnostics on empty database" {
    var db = try Database.init(std.testing.allocator, "empty-test");
    defer db.deinit();

    const diag = db.diagnostics();

    try std.testing.expectEqual(@as(usize, 0), diag.vector_count);
    try std.testing.expectEqual(@as(usize, 0), diag.dimension);
    try std.testing.expectEqual(@as(usize, 0), diag.memory.vector_bytes);
    try std.testing.expect(diag.isHealthy());
}

test {
    std.testing.refAllDecls(@This());
}
