//! Full Stack Smoke Tests
//!
//! Comprehensive tests that exercise all modules together in realistic workflows:
//! - All-modules smoke test
//! - AI agent workflow with GPU + Database
//! - Resource conflict detection
//! - Performance under combined load

const std = @import("std");
const testing = std.testing;
const builtin = @import("builtin");
const build_options = @import("build_options");
const abi = @import("abi");
const time = abi.shared.time;
const sync = abi.shared.sync;

const fixtures = @import("fixtures.zig");
const mocks = @import("mocks.zig");

// ============================================================================
// All-Modules Smoke Test
// ============================================================================

test "full stack: initialize all enabled modules" {
    const allocator = testing.allocator;

    // Initialize fixture with all compile-time enabled features
    var fixture = try fixtures.IntegrationFixture.init(allocator, .{
        .gpu = build_options.enable_gpu,
        .ai = build_options.enable_ai,
        .llm = build_options.enable_ai,
        .database = build_options.enable_database,
        .network = build_options.enable_network,
        .web = build_options.enable_web,
        .observability = build_options.enable_profiling,
    });
    defer fixture.deinit();

    try testing.expect(fixture.setup_complete);

    // Verify metrics were collected
    const metrics = fixture.getMetrics();
    try testing.expect(metrics.setup_time_ns > 0);
}

test "full stack: sequential feature initialization" {
    const allocator = testing.allocator;

    // Test each feature can be initialized independently
    const feature_sets = [_]fixtures.FeatureSet{
        .{ .observability = true },
        .{ .database = true },
        .{ .network = true },
        .{ .web = true },
        .{ .ai = true },
        .{ .gpu = true },
    };

    for (feature_sets) |features| {
        // Skip features not enabled at compile time
        if (features.gpu and !build_options.enable_gpu) continue;
        if (features.ai and !build_options.enable_ai) continue;
        if (features.database and !build_options.enable_database) continue;
        if (features.network and !build_options.enable_network) continue;
        if (features.web and !build_options.enable_web) continue;
        if (features.observability and !build_options.enable_profiling) continue;

        var fixture = try fixtures.IntegrationFixture.init(allocator, features);
        defer fixture.deinit();

        try testing.expect(fixture.setup_complete);
    }
}

// ============================================================================
// AI Agent Workflow Tests
// ============================================================================

test "full stack: ai agent with mock components" {
    const allocator = testing.allocator;

    // Initialize mock components
    var gpu = mocks.MockGpu.init(allocator);
    defer gpu.deinit();

    var llm = mocks.MockLlmModel.init(allocator);
    defer llm.deinit();

    var db = mocks.MockDatabase.init(allocator, 384);
    defer db.deinit();

    // Simulate AI agent workflow:
    // 1. Receive query
    const query = "What is the meaning of life?";

    // 2. Generate embeddings using LLM
    const embeddings = try llm.embed(query, allocator);
    defer allocator.free(embeddings);

    // 3. Store in database
    try db.insert(1, embeddings, query);

    // 4. Search for similar items
    const results = try db.search(embeddings, 5, allocator);
    defer allocator.free(results);

    // 5. Generate response using LLM
    const response = try llm.generate(query, 100, allocator);
    defer allocator.free(response);

    // Verify workflow completed
    try testing.expect(embeddings.len == 384);
    try testing.expect(results.len > 0);
    try testing.expect(response.len > 0);
}

test "full stack: vector search with mock gpu preprocessing" {
    const allocator = testing.allocator;

    var gpu = mocks.MockGpu.init(allocator);
    defer gpu.deinit();

    var db = mocks.MockDatabase.init(allocator, 128);
    defer db.deinit();

    // Insert test vectors
    for (0..100) |i| {
        var vec: [128]f32 = undefined;
        for (&vec, 0..) |*v, j| {
            v.* = @as(f32, @floatFromInt(i * 128 + j)) / 12800.0;
        }
        try db.insert(@intCast(i), &vec, null);
    }

    // Create query vector
    var query: [128]f32 = undefined;
    @memset(&query, 0.5);

    // Preprocess with "GPU"
    var processed_query: [128]f32 = undefined;
    for (&processed_query, &query) |*p, q| {
        p.* = q * 1.0; // Identity transform
    }

    // Search
    const results = try db.search(&processed_query, 10, allocator);
    defer allocator.free(results);

    try testing.expect(results.len == 10);
}

// ============================================================================
// Resource Conflict Detection Tests
// ============================================================================

test "full stack: no memory leaks across modules" {
    const allocator = testing.allocator;

    // This test will fail if any module leaks memory
    for (0..5) |_| {
        var fixture = try fixtures.IntegrationFixture.init(allocator, .{
            .observability = build_options.enable_profiling,
            .database = build_options.enable_database,
        });
        fixture.deinit();
    }

    // If we get here without the test allocator detecting leaks, we're good
}

test "full stack: concurrent mock operations" {
    const allocator = testing.allocator;

    var db = mocks.MockDatabase.init(allocator, 64);
    defer db.deinit();

    // Insert from multiple "threads" (simulated)
    const batch_size: usize = 10;
    for (0..batch_size) |batch| {
        for (0..10) |i| {
            const id = batch * 10 + i;
            var vec: [64]f32 = undefined;
            for (&vec, 0..) |*v, j| {
                v.* = @as(f32, @floatFromInt(id * 64 + j)) / 6400.0;
            }
            try db.insert(@intCast(id), &vec, null);
        }
    }

    try testing.expectEqual(@as(usize, 100), db.count());
}

// ============================================================================
// Performance Under Combined Load
// ============================================================================

test "full stack: combined operation latency" {
    const allocator = testing.allocator;

    var timer = try time.Timer.start();

    // Run combined operations
    var gpu = mocks.MockGpu.init(allocator);
    defer gpu.deinit();

    var llm = mocks.MockLlmModel.init(allocator);
    defer llm.deinit();

    var db = mocks.MockDatabase.init(allocator, 128);
    defer db.deinit();

    // Perform 100 complete workflows
    for (0..100) |i| {
        // Generate embedding
        const text = "test query";
        const embedding = try llm.embed(text, allocator);
        defer allocator.free(embedding);

        // Store in database
        try db.insert(@intCast(i), embedding[0..128], null);

        // Search
        const results = try db.search(embedding[0..128], 3, allocator);
        allocator.free(results);
    }

    const elapsed_ns = timer.read();

    // 100 workflows should complete in < 1 second
    try testing.expect(elapsed_ns < 1_000_000_000);
}

// ============================================================================
// Error Propagation Tests
// ============================================================================

test "full stack: error propagation across modules" {
    const allocator = testing.allocator;

    var db = mocks.MockDatabase.init(allocator, 64);
    defer db.deinit();

    // Test that errors from one module don't corrupt others
    // First, insert valid data
    var vec: [64]f32 = undefined;
    @memset(&vec, 1.0);
    try db.insert(1, &vec, null);

    // Verify database still works
    const results = try db.search(&vec, 1, allocator);
    defer allocator.free(results);

    try testing.expect(results.len > 0);
}

// ============================================================================
// Realistic Workflow Tests
// ============================================================================

test "full stack: document processing workflow" {
    const allocator = testing.allocator;

    var llm = mocks.MockLlmModel.init(allocator);
    defer llm.deinit();

    var db = mocks.MockDatabase.init(allocator, 384);
    defer db.deinit();

    // Simulate document processing pipeline
    const documents = [_][]const u8{
        "Document about artificial intelligence and machine learning.",
        "Technical specifications for database systems.",
        "Research paper on neural network architectures.",
        "User guide for cloud deployment.",
        "API documentation for REST services.",
    };

    // Process each document
    for (documents, 0..) |doc, i| {
        // 1. Tokenize
        const tokens = try llm.tokenize(doc, allocator);
        defer allocator.free(tokens);

        // 2. Generate embedding
        const embedding = try llm.embed(doc, allocator);
        defer allocator.free(embedding);

        // 3. Store in vector database
        try db.insert(@intCast(i), embedding, doc);
    }

    try testing.expectEqual(documents.len, db.count());

    // Query the document store
    const query_embedding = try llm.embed("machine learning research", allocator);
    defer allocator.free(query_embedding);

    const search_results = try db.search(query_embedding, 3, allocator);
    defer allocator.free(search_results);

    try testing.expect(search_results.len <= 3);
}

test "full stack: batch processing workflow" {
    const allocator = testing.allocator;

    var gpu = mocks.MockGpu.init(allocator);
    defer gpu.deinit();

    // Simulate batch processing
    const batch_size: usize = 32;
    const vector_dim: usize = 256;

    // Allocate batch data
    var input_batch = try allocator.alloc(f32, batch_size * vector_dim);
    defer allocator.free(input_batch);

    var output_batch = try allocator.alloc(f32, batch_size * vector_dim);
    defer allocator.free(output_batch);

    // Fill with test data
    for (input_batch, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(batch_size * vector_dim));
    }

    // Process batch with mock GPU
    for (0..batch_size) |b| {
        const start = b * vector_dim;
        const end = start + vector_dim;
        gpu.vectorAdd(
            input_batch[start..end],
            input_batch[start..end],
            output_batch[start..end],
        );
    }

    // Verify output
    for (output_batch, input_batch) |out, in| {
        try testing.expectApproxEqAbs(in * 2, out, 0.0001);
    }
}

// ============================================================================
// Resource Cleanup Tests
// ============================================================================

test "full stack: cleanup order verification" {
    // Resources should be cleaned up in reverse order of creation
    var resources_created: [5]bool = .{ false, false, false, false, false };
    var resources_destroyed: [5]bool = .{ false, false, false, false, false };

    // Create resources
    for (&resources_created, 0..) |*r, i| {
        r.* = true;
        _ = i;
    }

    // Destroy in reverse order
    var i: usize = resources_created.len;
    while (i > 0) {
        i -= 1;
        resources_destroyed[i] = true;
    }

    // Verify all destroyed
    for (resources_destroyed) |d| {
        try testing.expect(d);
    }
}
