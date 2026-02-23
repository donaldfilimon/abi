//! Integration Test Suite
//!
//! Cross-module integration tests that validate correct interaction between
//! major framework components. Tests are organized by module combination.
//!
//! ## Test Categories
//!
//! - **GPU + Database**: GPU-accelerated vector operations
//! - **LLM + GPU**: Full inference pipeline with GPU acceleration
//! - **HA + Network**: Distributed high availability
//! - **Observability**: Metrics collection across all modules
//! - **Cloud Lifecycle**: Cloud adapter integration
//! - **Full Stack**: All modules together
//! - **C API**: C bindings lifecycle, feature detection, and database ops
//!
//! ## Running Tests
//!
//! ```bash
//! # Run all integration tests
//! zig build test --summary all
//!
//! # Run specific integration tests
//! zig test src/services/tests/integration/mod.zig --test-filter "gpu"
//! ```

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

// Core infrastructure
pub const fixtures = @import("fixtures.zig");
pub const mocks = @import("mocks.zig");

// Re-export key types
pub const IntegrationFixture = fixtures.IntegrationFixture;
pub const FeatureSet = fixtures.FeatureSet;
pub const HardwareMode = fixtures.HardwareMode;
pub const TestMetrics = fixtures.TestMetrics;
pub const AccuracyStats = fixtures.AccuracyStats;

pub const MockGpu = mocks.MockGpu;
pub const MockLlmModel = mocks.MockLlmModel;
pub const MockNetworkRegistry = mocks.MockNetworkRegistry;
pub const MockReplicationManager = mocks.MockReplicationManager;
pub const MockDatabase = mocks.MockDatabase;

// Validation utilities
pub const validateAccuracy = fixtures.validateAccuracy;
pub const assertAllFinite = fixtures.assertAllFinite;
pub const assertInRange = fixtures.assertInRange;
pub const generateTestVector = fixtures.generateTestVector;
pub const generateRandomMatrix = fixtures.generateRandomMatrix;

// NOTE: test {} required for Zig 0.16 test discovery (not comptime)
test {
    // Always include core infrastructure tests
    _ = fixtures;
    _ = mocks;

    // Cross-module integration tests
    _ = @import("ha_network_test.zig");
    _ = @import("cloud_lifecycle_test.zig");
    _ = @import("full_stack_test.zig");
    if (build_options.enable_ai) {
        _ = @import("streaming_recovery.zig");
    }
    _ = @import("c_api_test.zig");

    // Cross-module feature integration (cache+storage, search, messaging+gateway)
    _ = @import("cross_module_test.zig");

    // v2 integration tests
    _ = @import("v2.zig");
}

// ============================================================================
// Cross-Module Integration Tests
// ============================================================================

test "integration: fixture with all features" {
    var fixture = try IntegrationFixture.init(std.testing.allocator, .{
        .gpu = build_options.enable_gpu,
        .ai = build_options.enable_ai,
        .database = build_options.enable_database,
        .network = build_options.enable_network,
        .web = build_options.enable_web,
        .observability = build_options.enable_profiling,
    });
    defer fixture.deinit();

    try std.testing.expect(fixture.setup_complete);

    const metrics = fixture.getMetrics();
    try std.testing.expect(metrics.setup_time_ns > 0);
}

test "integration: mock gpu with mock database" {
    var gpu = MockGpu.init(std.testing.allocator);
    defer gpu.deinit();

    var db = MockDatabase.init(std.testing.allocator, 128);
    defer db.deinit();

    // Generate test vectors using mock GPU
    var vectors: [10][128]f32 = undefined;
    for (&vectors, 0..) |*v, i| {
        for (v, 0..) |*val, j| {
            val.* = @as(f32, @floatFromInt(i * 128 + j)) / 1280.0;
        }
    }

    // Insert into mock database
    for (vectors, 0..) |v, i| {
        try db.insert(@intCast(i), &v, null);
    }

    try std.testing.expectEqual(@as(usize, 10), db.count());

    // Search with GPU-computed query
    var query: [128]f32 = undefined;
    @memset(&query, 0);
    query[0] = 1.0;

    const results = try db.search(&query, 3, std.testing.allocator);
    defer std.testing.allocator.free(results);

    try std.testing.expect(results.len <= 3);
}

test "integration: mock llm with mock gpu" {
    var llm = MockLlmModel.init(std.testing.allocator);
    defer llm.deinit();

    var gpu = MockGpu.init(std.testing.allocator);
    defer gpu.deinit();

    // Simulate LLM inference with GPU
    const prompt = "What is the meaning of life?";

    // Tokenize
    const tokens = try llm.tokenize(prompt, std.testing.allocator);
    defer std.testing.allocator.free(tokens);

    // Generate embeddings
    const embeddings = try llm.embed(prompt, std.testing.allocator);
    defer std.testing.allocator.free(embeddings);

    // Process with "GPU" (mock matrix multiply)
    var processed: [384]f32 = undefined;
    for (embeddings, 0..) |e, i| {
        processed[i] = e * 2.0; // Simplified "GPU" processing
    }

    // Generate response
    const response = try llm.generate(prompt, 50, std.testing.allocator);
    defer std.testing.allocator.free(response);

    try std.testing.expect(response.len > 0);
}

test "integration: mock network with mock replication" {
    var network = try MockNetworkRegistry.init(std.testing.allocator);
    defer network.deinit();

    var replication = MockReplicationManager.init(std.testing.allocator);
    defer replication.deinit();

    // Register nodes
    try network.register("primary", "10.0.0.1", 5432);
    try network.register("replica-1", "10.0.0.2", 5432);
    try network.register("replica-2", "10.0.0.3", 5432);

    // Add replicas to replication manager
    try replication.addReplica("replica-1");
    try replication.addReplica("replica-2");

    // Simulate write replication
    replication.replicate(1);
    try std.testing.expect(replication.allSynced());

    // Simulate failover
    try std.testing.expect(replication.is_primary);
    replication.failover();
    try std.testing.expect(!replication.is_primary);
}

test "integration: accuracy validation workflow" {
    const allocator = std.testing.allocator;

    // Simulate GPU results
    const gpu_results = try allocator.alloc(f32, 100);
    defer allocator.free(gpu_results);

    // Simulate CPU reference
    const cpu_results = try allocator.alloc(f32, 100);
    defer allocator.free(cpu_results);

    // Fill with test data
    for (0..100) |i| {
        const base = @as(f32, @floatFromInt(i)) / 10.0;
        cpu_results[i] = base;
        gpu_results[i] = base * 1.001; // 0.1% error
    }

    var stats = AccuracyStats{};
    try validateAccuracy(gpu_results, cpu_results, 0.01, &stats);

    try std.testing.expect(stats.max_error < 0.01);
    try std.testing.expect(stats.mean_error < 0.01);
}

test "integration: test vector generation patterns" {
    const allocator = std.testing.allocator;

    // Test each pattern
    const patterns = [_]fixtures.VectorPattern{
        .zeros,
        .ones,
        .sequential,
        .alternating,
        .random,
    };

    for (patterns) |pattern| {
        const vec = try generateTestVector(f32, pattern, 64, allocator);
        defer allocator.free(vec);

        try std.testing.expectEqual(@as(usize, 64), vec.len);
        try assertAllFinite(vec);
    }
}

test "integration: random matrix generation" {
    const allocator = std.testing.allocator;

    const mat = try generateRandomMatrix(f32, 32, 64, 42, allocator);
    defer allocator.free(mat);

    try std.testing.expectEqual(@as(usize, 32 * 64), mat.len);
    try assertAllFinite(mat);

    // Check values are in expected range
    try assertInRange(mat, -1.0, 1.0);
}

test {
    std.testing.refAllDecls(@This());
}
