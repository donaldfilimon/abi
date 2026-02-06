//! End-to-End Database Integration Tests
//!
//! Comprehensive tests for the database module covering:
//! - CRUD operations (Create, Read, Update, Delete)
//! - Vector search workflows
//! - Metadata handling
//! - Batch operations
//! - Edge cases: empty database, dimension mismatches, duplicate IDs
//! - Error conditions and recovery
//! - Database statistics and diagnostics
//!
//! These tests use the in-memory vector database without requiring
//! external storage or file system access.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");
const e2e = @import("e2e/mod.zig");

// ============================================================================
// Basic CRUD Operations
// ============================================================================

// Test database creation and basic lifecycle.
// Verifies database can be created, used, and properly cleaned up.
test "database: lifecycle management" {
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    var handle = try abi.database.open(allocator, "test-lifecycle");
    defer abi.database.close(&handle);

    // Database should be created successfully
    const s = abi.database.stats(&handle);
    try std.testing.expectEqual(@as(usize, 0), s.count);
}

// Test inserting vectors into the database.
// Verifies insert operation stores vectors correctly.
test "database: insert vectors" {
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    var handle = try abi.database.open(allocator, "test-insert");
    defer abi.database.close(&handle);

    // Insert first vector
    try abi.database.insert(&handle, 1, &[_]f32{ 1.0, 0.0, 0.0, 0.0 }, null);

    // Verify count increased
    var s = abi.database.stats(&handle);
    try std.testing.expectEqual(@as(usize, 1), s.count);
    try std.testing.expectEqual(@as(usize, 4), s.dimension);

    // Insert second vector
    try abi.database.insert(&handle, 2, &[_]f32{ 0.0, 1.0, 0.0, 0.0 }, null);

    s = abi.database.stats(&handle);
    try std.testing.expectEqual(@as(usize, 2), s.count);
}

// Test inserting vectors with metadata.
// Verifies metadata is stored and retrievable.
test "database: insert with metadata" {
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    var handle = try abi.database.open(allocator, "test-metadata");
    defer abi.database.close(&handle);

    // Insert with metadata
    try abi.database.insert(&handle, 1, &[_]f32{ 1.0, 2.0, 3.0 }, "document-1");
    try abi.database.insert(&handle, 2, &[_]f32{ 4.0, 5.0, 6.0 }, "document-2");

    // Retrieve and verify metadata
    const view1 = abi.database.get(&handle, 1);
    try std.testing.expect(view1 != null);
    try std.testing.expectEqualStrings("document-1", view1.?.metadata.?);

    const view2 = abi.database.get(&handle, 2);
    try std.testing.expect(view2 != null);
    try std.testing.expectEqualStrings("document-2", view2.?.metadata.?);
}

// Test retrieving vectors by ID.
// Verifies get operation returns correct vector data.
test "database: get vectors" {
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    var handle = try abi.database.open(allocator, "test-get");
    defer abi.database.close(&handle);

    const vec = [_]f32{ 1.5, 2.5, 3.5, 4.5 };
    try abi.database.insert(&handle, 42, &vec, "test-doc");

    // Get existing vector
    const view = abi.database.get(&handle, 42);
    try std.testing.expect(view != null);
    try std.testing.expectEqual(@as(u64, 42), view.?.id);
    try std.testing.expectEqual(@as(usize, 4), view.?.vector.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), view.?.vector[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.5), view.?.vector[3], 0.001);

    // Get non-existent vector
    const missing = abi.database.get(&handle, 999);
    try std.testing.expect(missing == null);
}

// Test updating existing vectors.
// Verifies update operation modifies vector data in place.
test "database: update vectors" {
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    var handle = try abi.database.open(allocator, "test-update");
    defer abi.database.close(&handle);

    // Insert initial vector
    try abi.database.insert(&handle, 1, &[_]f32{ 1.0, 0.0, 0.0 }, null);

    // Update with new values
    const updated = try abi.database.update(&handle, 1, &[_]f32{ 0.0, 1.0, 0.0 });
    try std.testing.expect(updated);

    // Verify update
    const view = abi.database.get(&handle, 1);
    try std.testing.expect(view != null);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), view.?.vector[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), view.?.vector[1], 0.001);

    // Update non-existent returns false
    const not_updated = try abi.database.update(&handle, 999, &[_]f32{ 1.0, 1.0, 1.0 });
    try std.testing.expect(!not_updated);
}

// Test deleting vectors from the database.
// Verifies delete operation removes vectors correctly.
test "database: delete vectors" {
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    var handle = try abi.database.open(allocator, "test-delete");
    defer abi.database.close(&handle);

    // Insert vectors
    try abi.database.insert(&handle, 1, &[_]f32{ 1.0, 0.0 }, null);
    try abi.database.insert(&handle, 2, &[_]f32{ 0.0, 1.0 }, null);
    try abi.database.insert(&handle, 3, &[_]f32{ 1.0, 1.0 }, null);

    var s = abi.database.stats(&handle);
    try std.testing.expectEqual(@as(usize, 3), s.count);

    // Delete one
    const deleted = abi.database.remove(&handle, 2);
    try std.testing.expect(deleted);

    s = abi.database.stats(&handle);
    try std.testing.expectEqual(@as(usize, 2), s.count);

    // Verify deleted vector is gone
    try std.testing.expect(abi.database.get(&handle, 2) == null);

    // Other vectors still exist
    try std.testing.expect(abi.database.get(&handle, 1) != null);
    try std.testing.expect(abi.database.get(&handle, 3) != null);

    // Delete non-existent returns false
    const not_deleted = abi.database.remove(&handle, 999);
    try std.testing.expect(!not_deleted);
}

// ============================================================================
// Vector Search Tests
// ============================================================================

// Test basic vector similarity search.
// Verifies search returns results sorted by similarity.
test "database: vector search" {
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    var handle = try abi.database.open(allocator, "test-search");
    defer abi.database.close(&handle);

    // Insert test vectors
    try abi.database.insert(&handle, 1, &[_]f32{ 1.0, 0.0, 0.0, 0.0 }, null);
    try abi.database.insert(&handle, 2, &[_]f32{ 0.0, 1.0, 0.0, 0.0 }, null);
    try abi.database.insert(&handle, 3, &[_]f32{ 0.7, 0.7, 0.0, 0.0 }, null);

    // Search for vector similar to [1, 0, 0, 0]
    const query = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const results = try abi.database.search(&handle, allocator, &query, 2);
    defer allocator.free(results);

    // Should return 2 results
    try std.testing.expectEqual(@as(usize, 2), results.len);

    // First result should be exact match (id=1)
    try std.testing.expectEqual(@as(u64, 1), results[0].id);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), results[0].score, 0.001);

    // Second result should be partial match (id=3)
    try std.testing.expectEqual(@as(u64, 3), results[1].id);
}

// Test search with top_k larger than database size.
// Should return all available results without error.
test "database: search top_k exceeds count" {
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    var handle = try abi.database.open(allocator, "test-search-topk");
    defer abi.database.close(&handle);

    // Insert only 2 vectors
    try abi.database.insert(&handle, 1, &[_]f32{ 1.0, 0.0 }, null);
    try abi.database.insert(&handle, 2, &[_]f32{ 0.0, 1.0 }, null);

    // Request top 10, but only 2 exist
    const results = try abi.database.search(&handle, allocator, &[_]f32{ 1.0, 0.0 }, 10);
    defer allocator.free(results);

    // Should return only 2 results
    try std.testing.expectEqual(@as(usize, 2), results.len);
}

// Test search on empty database.
// Should return empty results without error.
test "database: search empty database" {
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    var handle = try abi.database.open(allocator, "test-search-empty");
    defer abi.database.close(&handle);

    const results = try abi.database.search(&handle, allocator, &[_]f32{ 1.0, 0.0 }, 5);
    defer allocator.free(results);

    try std.testing.expectEqual(@as(usize, 0), results.len);
}

// Test search with zero vector query.
// Zero vectors should be handled gracefully.
test "database: search with zero vector" {
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    var handle = try abi.database.open(allocator, "test-search-zero");
    defer abi.database.close(&handle);

    try abi.database.insert(&handle, 1, &[_]f32{ 1.0, 0.0, 0.0 }, null);

    // Search with zero vector
    const results = try abi.database.search(&handle, allocator, &[_]f32{ 0.0, 0.0, 0.0 }, 1);
    defer allocator.free(results);

    // Should return empty (zero vector has no meaningful similarity)
    try std.testing.expectEqual(@as(usize, 0), results.len);
}

// ============================================================================
// List Operations
// ============================================================================

// Test listing vectors from database.
// Verifies list operation returns correct subset of vectors.
test "database: list vectors" {
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    var handle = try abi.database.open(allocator, "test-list");
    defer abi.database.close(&handle);

    // Insert multiple vectors
    try abi.database.insert(&handle, 1, &[_]f32{ 1.0, 0.0 }, "a");
    try abi.database.insert(&handle, 2, &[_]f32{ 0.0, 1.0 }, "b");
    try abi.database.insert(&handle, 3, &[_]f32{ 1.0, 1.0 }, "c");

    // List with limit
    const list2 = try abi.database.list(&handle, allocator, 2);
    defer allocator.free(list2);
    try std.testing.expectEqual(@as(usize, 2), list2.len);

    // List all
    const list_all = try abi.database.list(&handle, allocator, 100);
    defer allocator.free(list_all);
    try std.testing.expectEqual(@as(usize, 3), list_all.len);
}

// ============================================================================
// Database Statistics and Diagnostics
// ============================================================================

// Test database statistics.
// Verifies stats accurately reflect database state.
test "database: statistics" {
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    var handle = try abi.database.open(allocator, "test-stats");
    defer abi.database.close(&handle);

    // Empty database stats
    var s = abi.database.stats(&handle);
    try std.testing.expectEqual(@as(usize, 0), s.count);
    try std.testing.expectEqual(@as(usize, 0), s.dimension);
    try std.testing.expectEqual(@as(usize, 0), s.memory_bytes);

    // Add vectors
    try abi.database.insert(&handle, 1, &[_]f32{ 1.0, 2.0, 3.0, 4.0 }, null);
    try abi.database.insert(&handle, 2, &[_]f32{ 5.0, 6.0, 7.0, 8.0 }, null);

    s = abi.database.stats(&handle);
    try std.testing.expectEqual(@as(usize, 2), s.count);
    try std.testing.expectEqual(@as(usize, 4), s.dimension);
    try std.testing.expect(s.memory_bytes > 0);
}

// Test database optimize operation.
// Verifies optimize doesn't corrupt data.
test "database: optimize" {
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    var handle = try abi.database.open(allocator, "test-optimize");
    defer abi.database.close(&handle);

    // Add and delete some vectors to create fragmentation
    try abi.database.insert(&handle, 1, &[_]f32{ 1.0, 0.0 }, null);
    try abi.database.insert(&handle, 2, &[_]f32{ 0.0, 1.0 }, null);
    try abi.database.insert(&handle, 3, &[_]f32{ 1.0, 1.0 }, null);
    _ = abi.database.remove(&handle, 2);

    // Optimize should not crash
    try abi.database.optimize(&handle);

    // Data should still be intact
    const s = abi.database.stats(&handle);
    try std.testing.expectEqual(@as(usize, 2), s.count);

    // Remaining vectors should be accessible
    try std.testing.expect(abi.database.get(&handle, 1) != null);
    try std.testing.expect(abi.database.get(&handle, 3) != null);
}

// ============================================================================
// Error Condition Tests
// ============================================================================

// Test duplicate ID insertion.
// Should return error when inserting same ID twice.
test "database: duplicate id error" {
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    var handle = try abi.database.open(allocator, "test-duplicate");
    defer abi.database.close(&handle);

    // First insert succeeds
    try abi.database.insert(&handle, 1, &[_]f32{ 1.0, 0.0 }, null);

    // Second insert with same ID should fail
    const result = abi.database.insert(&handle, 1, &[_]f32{ 0.0, 1.0 }, null);
    try std.testing.expectError(abi.database.database.DatabaseError.DuplicateId, result);

    // Count should still be 1
    const s = abi.database.stats(&handle);
    try std.testing.expectEqual(@as(usize, 1), s.count);
}

// Test dimension mismatch error.
// Should return error when inserting vector with wrong dimension.
test "database: dimension mismatch error" {
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    var handle = try abi.database.open(allocator, "test-dimension");
    defer abi.database.close(&handle);

    // First insert with dimension 3
    try abi.database.insert(&handle, 1, &[_]f32{ 1.0, 0.0, 0.0 }, null);

    // Second insert with dimension 4 should fail
    const result = abi.database.insert(&handle, 2, &[_]f32{ 1.0, 0.0, 0.0, 0.0 }, null);
    try std.testing.expectError(abi.database.database.DatabaseError.InvalidDimension, result);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

// Test with single-dimension vectors.
// Minimum dimension should work correctly.
test "edge case: single dimension vectors" {
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    var handle = try abi.database.open(allocator, "test-1d");
    defer abi.database.close(&handle);

    try abi.database.insert(&handle, 1, &[_]f32{1.0}, null);
    try abi.database.insert(&handle, 2, &[_]f32{-1.0}, null);
    try abi.database.insert(&handle, 3, &[_]f32{0.5}, null);

    const results = try abi.database.search(&handle, allocator, &[_]f32{1.0}, 2);
    defer allocator.free(results);

    try std.testing.expectEqual(@as(usize, 2), results.len);
    try std.testing.expectEqual(@as(u64, 1), results[0].id);
}

// Test with high-dimension vectors.
// Large dimensions should work correctly.
test "edge case: high dimension vectors" {
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    var handle = try abi.database.open(allocator, "test-high-dim");
    defer abi.database.close(&handle);

    // Create 1024-dimension vectors
    var vec1: [1024]f32 = undefined;
    var vec2: [1024]f32 = undefined;
    var query: [1024]f32 = undefined;

    for (0..1024) |i| {
        vec1[i] = 1.0;
        vec2[i] = 0.0;
        query[i] = 1.0;
    }

    try abi.database.insert(&handle, 1, &vec1, null);
    try abi.database.insert(&handle, 2, &vec2, null);

    const results = try abi.database.search(&handle, allocator, &query, 1);
    defer allocator.free(results);

    try std.testing.expectEqual(@as(u64, 1), results[0].id);
}

// Test with empty metadata.
// Null and empty metadata should be distinguished.
test "edge case: empty vs null metadata" {
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    var handle = try abi.database.open(allocator, "test-empty-meta");
    defer abi.database.close(&handle);

    // Insert with null metadata
    try abi.database.insert(&handle, 1, &[_]f32{ 1.0, 0.0 }, null);

    // Insert with empty string metadata
    try abi.database.insert(&handle, 2, &[_]f32{ 0.0, 1.0 }, "");

    // Insert with actual metadata
    try abi.database.insert(&handle, 3, &[_]f32{ 1.0, 1.0 }, "data");

    const view1 = abi.database.get(&handle, 1);
    try std.testing.expect(view1.?.metadata == null);

    const view2 = abi.database.get(&handle, 2);
    try std.testing.expectEqualStrings("", view2.?.metadata.?);

    const view3 = abi.database.get(&handle, 3);
    try std.testing.expectEqualStrings("data", view3.?.metadata.?);
}

// Test with unicode metadata.
// Unicode in metadata should be preserved.
test "edge case: unicode metadata" {
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    var handle = try abi.database.open(allocator, "test-unicode-meta");
    defer abi.database.close(&handle);

    const unicode_meta = "Hello \xe4\xb8\x96\xe7\x95\x8c \xf0\x9f\x98\x80";
    try abi.database.insert(&handle, 1, &[_]f32{ 1.0, 0.0 }, unicode_meta);

    const view = abi.database.get(&handle, 1);
    try std.testing.expect(view != null);
    try std.testing.expectEqualStrings(unicode_meta, view.?.metadata.?);
}

// Test with special float values.
// NaN and infinity should be handled appropriately.
test "edge case: special float values" {
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    var handle = try abi.database.open(allocator, "test-special-floats");
    defer abi.database.close(&handle);

    // Normal vector
    try abi.database.insert(&handle, 1, &[_]f32{ 1.0, 0.0 }, null);

    // Vector with very small values
    try abi.database.insert(&handle, 2, &[_]f32{ 1e-38, 1e-38 }, null);

    // Vector with very large values
    try abi.database.insert(&handle, 3, &[_]f32{ 1e38, 1e38 }, null);

    // Should not crash when searching
    const results = try abi.database.search(&handle, allocator, &[_]f32{ 1.0, 0.0 }, 3);
    defer allocator.free(results);

    // Results may vary but should not crash
    try std.testing.expect(results.len > 0);
}

// Test with large number of vectors.
// Database should handle many vectors without issues.
test "edge case: many vectors" {
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    var handle = try abi.database.open(allocator, "test-many");
    defer abi.database.close(&handle);

    // Insert 100 vectors
    for (0..100) |i| {
        const id: u64 = @intCast(i);
        const val: f32 = @floatFromInt(i);
        try abi.database.insert(&handle, id, &[_]f32{ val, 100.0 - val }, null);
    }

    const s = abi.database.stats(&handle);
    try std.testing.expectEqual(@as(usize, 100), s.count);

    // Search should still work
    const results = try abi.database.search(&handle, allocator, &[_]f32{ 50.0, 50.0 }, 5);
    defer allocator.free(results);

    try std.testing.expectEqual(@as(usize, 5), results.len);
}

// ============================================================================
// Database Module Feature Tests
// ============================================================================

// Test database module feature detection.
// Verifies isEnabled() returns correct value.
test "database feature: detection" {
    const enabled = abi.database.isEnabled();

    if (build_options.enable_database) {
        try std.testing.expect(enabled);
    } else {
        try std.testing.expect(!enabled);
    }
}

// Test database module initialization.
// Verifies init/deinit cycle works correctly.
test "database feature: init cycle" {
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    try abi.database.init(allocator);
    try std.testing.expect(abi.database.isInitialized());

    abi.database.deinit();
    try std.testing.expect(!abi.database.isInitialized());
}

// ============================================================================
// Delete Consistency Tests
// ============================================================================

// Test that delete operations maintain index consistency.
// Swap Removeshould update index correctly for moved elements.
test "database: delete maintains index consistency" {
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    var handle = try abi.database.open(allocator, "test-delete-consistency");
    defer abi.database.close(&handle);

    // Insert vectors with known IDs
    try abi.database.insert(&handle, 10, &[_]f32{ 1.0, 0.0 }, "first");
    try abi.database.insert(&handle, 20, &[_]f32{ 0.0, 1.0 }, "middle");
    try abi.database.insert(&handle, 30, &[_]f32{ 1.0, 1.0 }, "last");

    // Delete middle element - should swap with last
    _ = abi.database.remove(&handle, 20);

    // All remaining should be accessible
    try std.testing.expect(abi.database.get(&handle, 10) != null);
    try std.testing.expect(abi.database.get(&handle, 20) == null);
    try std.testing.expect(abi.database.get(&handle, 30) != null);

    // Operations on remaining should work
    const view = abi.database.get(&handle, 30);
    try std.testing.expectEqualStrings("last", view.?.metadata.?);
}

// Test multiple sequential deletes.
// Ensures consistency is maintained across multiple deletions.
test "database: sequential deletes" {
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    var handle = try abi.database.open(allocator, "test-seq-delete");
    defer abi.database.close(&handle);

    // Insert 5 vectors
    for (1..6) |i| {
        const id: u64 = @intCast(i);
        const val: f32 = @floatFromInt(i);
        try abi.database.insert(&handle, id, &[_]f32{ val, 0.0 }, null);
    }

    // Delete in various orders
    _ = abi.database.remove(&handle, 2);
    _ = abi.database.remove(&handle, 4);
    _ = abi.database.remove(&handle, 1);

    // Check remaining
    try std.testing.expect(abi.database.get(&handle, 1) == null);
    try std.testing.expect(abi.database.get(&handle, 2) == null);
    try std.testing.expect(abi.database.get(&handle, 3) != null);
    try std.testing.expect(abi.database.get(&handle, 4) == null);
    try std.testing.expect(abi.database.get(&handle, 5) != null);

    const s = abi.database.stats(&handle);
    try std.testing.expectEqual(@as(usize, 2), s.count);
}
