//! End-to-End Vector Search Tests
//!
//! Complete workflow tests for the vector database:
//! - Document insertion with embeddings
//! - Index building and optimization
//! - Similarity search with verification
//! - Incremental updates
//! - Backup and restore workflows

const std = @import("std");
const abi = @import("abi");
const e2e = @import("mod.zig");

// ============================================================================
// Helper Functions
// ============================================================================

/// Create test documents with embeddings.
const TestDocument = struct {
    id: u64,
    content: []const u8,
    embedding: [128]f32,
};

fn createTestDocuments(allocator: std.mem.Allocator, count: usize) ![]TestDocument {
    const docs = try allocator.alloc(TestDocument, count);

    const contents = [_][]const u8{
        "Introduction to machine learning algorithms",
        "Deep neural networks for image classification",
        "Natural language processing techniques",
        "Reinforcement learning fundamentals",
        "Computer vision and object detection",
        "Time series forecasting methods",
        "Recommendation systems design",
        "Graph neural networks overview",
        "Transformer architecture explained",
        "Generative adversarial networks",
    };

    for (docs, 0..) |*doc, i| {
        doc.id = @intCast(i + 1);
        doc.content = contents[i % contents.len];

        // Generate deterministic embedding based on content hash
        var hash: u64 = 0;
        for (doc.content) |c| {
            hash = hash *% 31 +% @as(u64, c);
        }
        hash = hash *% (@as(u64, @intCast(i)) + 1);

        var rng = std.Random.DefaultPrng.init(hash);
        for (&doc.embedding) |*v| {
            v.* = rng.random().float(f32) * 2.0 - 1.0;
        }

        // Normalize embedding
        var norm: f32 = 0;
        for (doc.embedding) |v| {
            norm += v * v;
        }
        norm = @sqrt(norm);
        if (norm > 0) {
            for (&doc.embedding) |*v| {
                v.* /= norm;
            }
        }
    }

    return docs;
}

// ============================================================================
// E2E Tests: Basic Document Workflow
// ============================================================================

test "e2e: complete document search workflow" {
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    // 1. Initialize test context with database feature
    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .database = true },
        .timeout_ms = 30_000,
    });
    defer ctx.deinit();

    var timer = e2e.WorkflowTimer.init(allocator);
    defer timer.deinit();

    try timer.checkpoint("context_initialized");

    // 2. Create database handle
    var handle = try abi.features.database.open(allocator, "test-e2e-search");
    defer abi.features.database.close(&handle);

    try timer.checkpoint("database_opened");

    // 3. Insert test documents
    const docs = try createTestDocuments(allocator, 10);
    defer allocator.free(docs);

    for (docs) |doc| {
        try abi.features.database.insert(&handle, doc.id, &doc.embedding, doc.content);
    }

    try timer.checkpoint("documents_inserted");

    // 4. Verify insertion
    const stats = abi.features.database.stats(&handle);
    try std.testing.expectEqual(@as(usize, 10), stats.count);
    try std.testing.expectEqual(@as(usize, 128), stats.dimension);

    try timer.checkpoint("insertion_verified");

    // 5. Perform similarity search
    const query = docs[0].embedding;
    const results = try abi.features.database.search(&handle, allocator, &query, 5);
    defer allocator.free(results);

    try timer.checkpoint("search_completed");

    // 6. Verify search results
    try std.testing.expect(results.len > 0);
    try std.testing.expect(results.len <= 5);

    // First result should be the exact match (highest similarity)
    try std.testing.expectEqual(@as(u64, 1), results[0].id);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), results[0].score, 0.001);

    // Subsequent results should have lower scores
    for (1..results.len) |i| {
        try std.testing.expect(results[i].score <= results[i - 1].score);
    }

    try timer.checkpoint("results_verified");

    // 7. Record metrics
    ctx.recordOperation("complete_workflow", timer.elapsed());
    try std.testing.expect(!timer.isTimedOut(30_000));
}

test "e2e: incremental index updates" {
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .database = true },
    });
    defer ctx.deinit();

    // 1. Create initial database
    var handle = try abi.features.database.open(allocator, "test-e2e-incremental");
    defer abi.features.database.close(&handle);

    // 2. Insert initial batch
    const initial_docs = try createTestDocuments(allocator, 5);
    defer allocator.free(initial_docs);

    for (initial_docs) |doc| {
        try abi.features.database.insert(&handle, doc.id, &doc.embedding, doc.content);
    }

    // 3. Verify initial state
    var stats = abi.features.database.stats(&handle);
    try std.testing.expectEqual(@as(usize, 5), stats.count);

    // 4. Perform initial search
    var query: [128]f32 = undefined;
    @memset(&query, 0);
    query[0] = 1.0;

    const results1 = try abi.features.database.search(&handle, allocator, &query, 3);
    defer allocator.free(results1);
    const initial_result_count = results1.len;

    // 5. Add more documents
    for (6..11) |i| {
        var vec: [128]f32 = undefined;
        var rng = std.Random.DefaultPrng.init(@as(u64, i) * 12345);
        for (&vec) |*v| {
            v.* = rng.random().float(f32) * 2.0 - 1.0;
        }
        try abi.features.database.insert(&handle, @intCast(i), &vec, "additional document");
    }

    // 6. Verify updated state
    stats = abi.features.database.stats(&handle);
    try std.testing.expectEqual(@as(usize, 10), stats.count);

    // 7. Search again and verify results may have changed
    const results2 = try abi.features.database.search(&handle, allocator, &query, 3);
    defer allocator.free(results2);

    // Should still return valid results
    try std.testing.expect(results2.len > 0);
    try std.testing.expect(results2.len <= 3);

    // Results may be different due to new documents
    _ = initial_result_count;
}

test "e2e: document CRUD lifecycle" {
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .database = true },
    });
    defer ctx.deinit();

    var handle = try abi.features.database.open(allocator, "test-e2e-crud");
    defer abi.features.database.close(&handle);

    // 1. CREATE: Insert documents
    const vec1 = [_]f32{1.0} ++ [_]f32{0.0} ** 127;
    const vec2 = [_]f32{0.0} ++ [_]f32{1.0} ++ [_]f32{0.0} ** 126;
    const vec3 = [_]f32{0.5} ** 128;

    try abi.features.database.insert(&handle, 1, &vec1, "document-1");
    try abi.features.database.insert(&handle, 2, &vec2, "document-2");
    try abi.features.database.insert(&handle, 3, &vec3, "document-3");

    try std.testing.expectEqual(@as(usize, 3), abi.features.database.stats(&handle).count);

    // 2. READ: Retrieve documents
    const view1 = abi.features.database.get(&handle, 1);
    try std.testing.expect(view1 != null);
    try std.testing.expectEqual(@as(u64, 1), view1.?.id);
    try std.testing.expectEqualStrings("document-1", view1.?.metadata.?);

    const view_missing = abi.features.database.get(&handle, 999);
    try std.testing.expect(view_missing == null);

    // 3. UPDATE: Modify document
    const vec1_updated = [_]f32{0.5} ++ [_]f32{0.5} ++ [_]f32{0.0} ** 126;
    const updated = try abi.features.database.update(&handle, 1, &vec1_updated);
    try std.testing.expect(updated);

    // Verify update
    const view1_after = abi.features.database.get(&handle, 1);
    try std.testing.expect(view1_after != null);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), view1_after.?.vector[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), view1_after.?.vector[1], 0.001);

    // 4. DELETE: Remove document
    const deleted = abi.features.database.remove(&handle, 2);
    try std.testing.expect(deleted);

    try std.testing.expectEqual(@as(usize, 2), abi.features.database.stats(&handle).count);
    try std.testing.expect(abi.features.database.get(&handle, 2) == null);

    // Remaining documents still exist
    try std.testing.expect(abi.features.database.get(&handle, 1) != null);
    try std.testing.expect(abi.features.database.get(&handle, 3) != null);
}

// ============================================================================
// E2E Tests: Search Quality
// ============================================================================

test "e2e: search returns semantically similar documents" {
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .database = true },
    });
    defer ctx.deinit();

    var handle = try abi.features.database.open(allocator, "test-e2e-semantic");
    defer abi.features.database.close(&handle);

    // Insert vectors representing different categories
    // Category A: vectors pointing in +x direction
    try abi.features.database.insert(&handle, 1, &([_]f32{1.0} ++ [_]f32{0.0} ** 127), "category-a-1");
    try abi.features.database.insert(&handle, 2, &([_]f32{ 0.9, 0.1 } ++ [_]f32{0.0} ** 126), "category-a-2");
    try abi.features.database.insert(&handle, 3, &([_]f32{ 0.8, 0.2 } ++ [_]f32{0.0} ** 126), "category-a-3");

    // Category B: vectors pointing in +y direction
    try abi.features.database.insert(&handle, 4, &([_]f32{ 0.0, 1.0 } ++ [_]f32{0.0} ** 126), "category-b-1");
    try abi.features.database.insert(&handle, 5, &([_]f32{ 0.1, 0.9 } ++ [_]f32{0.0} ** 126), "category-b-2");
    try abi.features.database.insert(&handle, 6, &([_]f32{ 0.2, 0.8 } ++ [_]f32{0.0} ** 126), "category-b-3");

    // Query for category A (pointing in +x direction)
    const query_a = [_]f32{1.0} ++ [_]f32{0.0} ** 127;
    const results_a = try abi.features.database.search(&handle, allocator, &query_a, 3);
    defer allocator.free(results_a);

    // All results should be from category A (ids 1, 2, 3)
    try std.testing.expectEqual(@as(usize, 3), results_a.len);
    for (results_a) |r| {
        try std.testing.expect(r.id <= 3);
    }

    // Query for category B (pointing in +y direction)
    const query_b = [_]f32{ 0.0, 1.0 } ++ [_]f32{0.0} ** 126;
    const results_b = try abi.features.database.search(&handle, allocator, &query_b, 3);
    defer allocator.free(results_b);

    // All results should be from category B (ids 4, 5, 6)
    try std.testing.expectEqual(@as(usize, 3), results_b.len);
    for (results_b) |r| {
        try std.testing.expect(r.id >= 4 and r.id <= 6);
    }
}

test "e2e: search handles edge cases" {
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .database = true },
    });
    defer ctx.deinit();

    var handle = try abi.features.database.open(allocator, "test-e2e-edge");
    defer abi.features.database.close(&handle);

    // Test 1: Search empty database
    const empty_query = [_]f32{1.0} ++ [_]f32{0.0} ** 127;
    const empty_results = try abi.features.database.search(&handle, allocator, &empty_query, 5);
    defer allocator.free(empty_results);
    try std.testing.expectEqual(@as(usize, 0), empty_results.len);

    // Insert some documents
    try abi.features.database.insert(&handle, 1, &([_]f32{1.0} ++ [_]f32{0.0} ** 127), null);
    try abi.features.database.insert(&handle, 2, &([_]f32{ 0.0, 1.0 } ++ [_]f32{0.0} ** 126), null);

    // Test 2: Search with top_k larger than database
    const large_k_results = try abi.features.database.search(&handle, allocator, &empty_query, 100);
    defer allocator.free(large_k_results);
    try std.testing.expectEqual(@as(usize, 2), large_k_results.len);

    // Test 3: Search with top_k = 1
    const single_results = try abi.features.database.search(&handle, allocator, &empty_query, 1);
    defer allocator.free(single_results);
    try std.testing.expectEqual(@as(usize, 1), single_results.len);
    try std.testing.expectEqual(@as(u64, 1), single_results[0].id);
}

// ============================================================================
// E2E Tests: Performance and Scale
// ============================================================================

test "e2e: handles many documents" {
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .database = true },
        .timeout_ms = 60_000,
    });
    defer ctx.deinit();

    var timer = e2e.WorkflowTimer.init(allocator);
    defer timer.deinit();

    var handle = try abi.features.database.open(allocator, "test-e2e-scale");
    defer abi.features.database.close(&handle);

    // Insert many documents
    const doc_count: usize = 500;
    for (0..doc_count) |i| {
        var vec: [128]f32 = undefined;
        var rng = std.Random.DefaultPrng.init(@as(u64, i) * 31337);
        for (&vec) |*v| {
            v.* = rng.random().float(f32) * 2.0 - 1.0;
        }
        try abi.features.database.insert(&handle, @intCast(i), &vec, null);
    }

    try timer.checkpoint("documents_inserted");

    // Verify count
    try std.testing.expectEqual(doc_count, abi.features.database.stats(&handle).count);

    // Perform multiple searches
    for (0..10) |seed| {
        var query: [128]f32 = undefined;
        var rng = std.Random.DefaultPrng.init(seed * 999);
        for (&query) |*v| {
            v.* = rng.random().float(f32) * 2.0 - 1.0;
        }

        const results = try abi.features.database.search(&handle, allocator, &query, 10);
        defer allocator.free(results);

        try std.testing.expect(results.len > 0);
        try std.testing.expect(results.len <= 10);

        // Scores should be in descending order
        for (1..results.len) |j| {
            try std.testing.expect(results[j].score <= results[j - 1].score);
        }
    }

    try timer.checkpoint("searches_completed");

    // Workflow should complete within timeout
    try std.testing.expect(!timer.isTimedOut(60_000));
}

// ============================================================================
// E2E Tests: Database Operations
// ============================================================================

test "e2e: optimize maintains data integrity" {
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .database = true },
    });
    defer ctx.deinit();

    var handle = try abi.features.database.open(allocator, "test-e2e-optimize");
    defer abi.features.database.close(&handle);

    // Insert documents
    const docs = try createTestDocuments(allocator, 10);
    defer allocator.free(docs);

    for (docs) |doc| {
        try abi.features.database.insert(&handle, doc.id, &doc.embedding, doc.content);
    }

    // Delete some to create fragmentation
    _ = abi.features.database.remove(&handle, 2);
    _ = abi.features.database.remove(&handle, 5);
    _ = abi.features.database.remove(&handle, 8);

    const count_before = abi.features.database.stats(&handle).count;
    try std.testing.expectEqual(@as(usize, 7), count_before);

    // Optimize
    try abi.features.database.optimize(&handle);

    // Verify data integrity after optimization
    const count_after = abi.features.database.stats(&handle).count;
    try std.testing.expectEqual(count_before, count_after);

    // Verify remaining documents are accessible
    try std.testing.expect(abi.features.database.get(&handle, 1) != null);
    try std.testing.expect(abi.features.database.get(&handle, 2) == null); // deleted
    try std.testing.expect(abi.features.database.get(&handle, 3) != null);
    try std.testing.expect(abi.features.database.get(&handle, 4) != null);
    try std.testing.expect(abi.features.database.get(&handle, 5) == null); // deleted
    try std.testing.expect(abi.features.database.get(&handle, 6) != null);
    try std.testing.expect(abi.features.database.get(&handle, 7) != null);
    try std.testing.expect(abi.features.database.get(&handle, 8) == null); // deleted
    try std.testing.expect(abi.features.database.get(&handle, 9) != null);
    try std.testing.expect(abi.features.database.get(&handle, 10) != null);

    // Search should still work
    const query = docs[0].embedding;
    const results = try abi.features.database.search(&handle, allocator, &query, 5);
    defer allocator.free(results);

    try std.testing.expect(results.len > 0);
}

test "e2e: list returns correct subset" {
    try e2e.skipIfDatabaseDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .database = true },
    });
    defer ctx.deinit();

    var handle = try abi.features.database.open(allocator, "test-e2e-list");
    defer abi.features.database.close(&handle);

    // Insert documents
    for (1..21) |i| {
        var vec: [64]f32 = undefined;
        @memset(&vec, @as(f32, @floatFromInt(i)));
        try abi.features.database.insert(&handle, @intCast(i), &vec, null);
    }

    // List with limit
    const list10 = try abi.features.database.list(&handle, allocator, 10);
    defer allocator.free(list10);
    try std.testing.expectEqual(@as(usize, 10), list10.len);

    // List all
    const list_all = try abi.features.database.list(&handle, allocator, 100);
    defer allocator.free(list_all);
    try std.testing.expectEqual(@as(usize, 20), list_all.len);

    // Verify list contains valid data
    for (list10) |view| {
        try std.testing.expect(view.id >= 1 and view.id <= 20);
        try std.testing.expectEqual(@as(usize, 64), view.vector.len);
    }
}
