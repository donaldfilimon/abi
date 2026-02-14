//! Error Handling Integration Tests
//!
//! Comprehensive tests for error handling across all modules covering:
//! - Error type definitions and categorization
//! - Error propagation through call chains
//! - Graceful degradation when features are disabled
//! - Recovery from error conditions
//! - Resource cleanup on errors (defer/errdefer patterns)
//! - Error message quality and debugging support
//!
//! These tests verify the system handles errors correctly without
//! leaking resources or entering undefined states.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

// ============================================================================
// Framework Initialization Error Tests
// ============================================================================

// Testframework initialization with valid options.
// Verifiessuccessful initialization path.
test "framework: successful initialization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var framework = try abi.initDefault(gpa.allocator());
    defer framework.deinit();

    try std.testing.expect(framework.isRunning());
}

// Testmultiple framework init/deinit cycles.
// Verifiesno resource leaks across cycles.
test "framework: multiple init cycles" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const check = gpa.deinit();
        // No leak should be detected
        if (check == .leak) @panic("Memory leak detected in framework cycles");
    }

    // Multiple cycles
    for (0..5) |_| {
        var framework = try abi.initDefault(gpa.allocator());
        defer framework.deinit();
        try std.testing.expect(framework.isRunning());
    }
}

// Testframework with feature build options.
// Verifiesframework respects build-time feature flags.
test "framework: feature flag consistency" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var framework = try abi.initDefault(gpa.allocator());
    defer framework.deinit();

    // Feature states should match what was requested (limited by build options)
    if (build_options.enable_gpu) {
        try std.testing.expectEqual(true, framework.isEnabled(.gpu));
    }
    if (build_options.enable_ai) {
        try std.testing.expectEqual(true, framework.isEnabled(.ai));
    }
}

// ============================================================================
// Database Error Handling Tests
// ============================================================================

// Testdatabase error type definitions.
// Verifiesall expected error types exist.
test "database errors: type definitions" {
    if (!build_options.enable_database) return error.SkipZigTest;

    // Verify error types can be used
    const errors = [_]abi.database.database.DatabaseError{
        abi.database.database.DatabaseError.DuplicateId,
        abi.database.database.DatabaseError.VectorNotFound,
        abi.database.database.DatabaseError.InvalidDimension,
        abi.database.database.DatabaseError.PoolExhausted,
        abi.database.database.DatabaseError.PersistenceError,
        abi.database.database.DatabaseError.ConcurrencyError,
    };

    // Each error should have unique value
    for (errors, 0..) |e1, i| {
        for (errors[i + 1 ..]) |e2| {
            try std.testing.expect(@intFromError(e1) != @intFromError(e2));
        }
    }
}

// Test database handles duplicate ID gracefully.
// Should return error without corrupting state.
test "database errors: duplicate id recovery" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var handle = try abi.database.open(allocator, "test-dup-recovery");
    defer abi.database.close(&handle);

    // First insert succeeds
    try abi.database.insert(&handle, 1, &[_]f32{ 1.0, 0.0 }, null);

    // Duplicate should fail
    const result = abi.database.insert(&handle, 1, &[_]f32{ 0.0, 1.0 }, null);
    try std.testing.expectError(abi.database.database.DatabaseError.DuplicateId, result);

    // Database should still be usable
    const view = abi.database.get(&handle, 1);
    try std.testing.expect(view != null);

    // Original data should be intact
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), view.?.vector[0], 0.001);

    // New inserts should still work
    try abi.database.insert(&handle, 2, &[_]f32{ 0.0, 1.0 }, null);

    const s = abi.database.stats(&handle);
    try std.testing.expectEqual(@as(usize, 2), s.count);
}

// Test database handles dimension mismatch gracefully.
// Should return error without corrupting state.
test "database errors: dimension mismatch recovery" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var handle = try abi.database.open(allocator, "test-dim-recovery");
    defer abi.database.close(&handle);

    // First insert establishes dimension
    try abi.database.insert(&handle, 1, &[_]f32{ 1.0, 2.0, 3.0 }, null);

    // Different dimension should fail
    const result = abi.database.insert(&handle, 2, &[_]f32{ 1.0, 2.0 }, null);
    try std.testing.expectError(abi.database.database.DatabaseError.InvalidDimension, result);

    // Database should still be usable
    const s = abi.database.stats(&handle);
    try std.testing.expectEqual(@as(usize, 1), s.count);
    try std.testing.expectEqual(@as(usize, 3), s.dimension);

    // Correct dimension inserts should work
    try abi.database.insert(&handle, 2, &[_]f32{ 4.0, 5.0, 6.0 }, null);
}

// Test database cleanup on error paths.
// Verifies no leaks when operations fail.
test "database errors: cleanup on failure" {
    if (!build_options.enable_database) return error.SkipZigTest;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const check = gpa.deinit();
        if (check == .leak) @panic("Memory leak on database error path");
    }

    var handle = try abi.database.open(gpa.allocator(), "test-cleanup");
    defer abi.database.close(&handle);

    // Successful insert
    try abi.database.insert(&handle, 1, &[_]f32{ 1.0, 0.0 }, "metadata");

    // Failed insert (duplicate) should not leak
    _ = abi.database.insert(&handle, 1, &[_]f32{ 0.0, 1.0 }, "other metadata") catch {};

    // Failed insert (wrong dimension) should not leak
    _ = abi.database.insert(&handle, 2, &[_]f32{1.0}, "more metadata") catch {};
}

// ============================================================================
// LLM Error Handling Tests
// ============================================================================

// Test LLM error type definitions.
// Verifies all expected error types exist.
test "llm errors: type definitions" {
    if (!build_options.enable_llm) return error.SkipZigTest;

    // Verify LlmError types exist and are distinct
    const llm_errors = [_]abi.ai.llm.LlmError{
        abi.ai.llm.LlmError.InvalidModelFormat,
        abi.ai.llm.LlmError.UnsupportedQuantization,
        abi.ai.llm.LlmError.ModelTooLarge,
        abi.ai.llm.LlmError.ContextLengthExceeded,
        abi.ai.llm.LlmError.TokenizationFailed,
        abi.ai.llm.LlmError.InferenceError,
        abi.ai.llm.LlmError.OutOfMemory,
        abi.ai.llm.LlmError.GpuUnavailable,
    };

    for (llm_errors, 0..) |e1, i| {
        for (llm_errors[i + 1 ..]) |e2| {
            try std.testing.expect(@intFromError(e1) != @intFromError(e2));
        }
    }
}

// Test LLM engine without model returns appropriate error.
// Should fail gracefully when no model loaded.
test "llm errors: no model loaded" {
    if (!build_options.enable_llm) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var engine = abi.ai.llm.Engine.init(allocator, .{});
    defer engine.deinit();

    // Generate without model should fail
    const gen_result = engine.generate(allocator, "test");
    try std.testing.expectError(abi.ai.llm.LlmError.InvalidModelFormat, gen_result);

    // Tokenize without model should fail
    const tok_result = engine.tokenize(allocator, "test");
    try std.testing.expectError(abi.ai.llm.LlmError.InvalidModelFormat, tok_result);

    // Detokenize without model should fail
    const detok_result = engine.detokenize(allocator, &[_]u32{1});
    try std.testing.expectError(abi.ai.llm.LlmError.InvalidModelFormat, detok_result);
}

// Test LLM tokenizer error handling.
// Verifies tokenizer errors are properly defined.
test "llm errors: tokenizer errors" {
    if (!build_options.enable_llm) return error.SkipZigTest;

    const tok_errors = [_]abi.ai.llm.tokenizer.TokenizerError{
        abi.ai.llm.tokenizer.TokenizerError.InvalidUtf8,
        abi.ai.llm.tokenizer.TokenizerError.VocabNotLoaded,
        abi.ai.llm.tokenizer.TokenizerError.UnknownToken,
        abi.ai.llm.tokenizer.TokenizerError.EncodingError,
        abi.ai.llm.tokenizer.TokenizerError.DecodingError,
        abi.ai.llm.tokenizer.TokenizerError.OutOfMemory,
    };

    for (tok_errors, 0..) |e1, i| {
        for (tok_errors[i + 1 ..]) |e2| {
            try std.testing.expect(@intFromError(e1) != @intFromError(e2));
        }
    }
}

// ============================================================================
// Memory Safety Tests
// ============================================================================

// Test allocator stress with error conditions.
// Verifies no leaks under allocation pressure.
test "memory: allocator stress" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const check = gpa.deinit();
        if (check == .leak) @panic("Memory leak under stress");
    }

    // Many small allocations and frees
    for (0..100) |i| {
        const buf = try gpa.allocator().alloc(u8, (i % 10 + 1) * 100);
        defer gpa.allocator().free(buf);
        @memset(buf, @intCast(i % 256));
    }
}

// Test framework allocator is properly tracked.
// Verifies framework doesn't leak on normal usage.
test "memory: framework lifecycle" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const check = gpa.deinit();
        if (check == .leak) @panic("Memory leak in framework lifecycle");
    }

    var framework = try abi.initDefault(gpa.allocator());
    framework.deinit();
}

// Testdatabase operations don't leak memory.
// VerifiesCRUD operations are leak-free.
test "memory: database operations" {
    if (!build_options.enable_database) return error.SkipZigTest;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const check = gpa.deinit();
        if (check == .leak) @panic("Memory leak in database operations");
    }

    var handle = try abi.database.open(gpa.allocator(), "test-mem");
    defer abi.database.close(&handle);

    // Insert
    try abi.database.insert(&handle, 1, &[_]f32{ 1.0, 0.0 }, "meta");

    // Search
    const results = try abi.database.search(&handle, gpa.allocator(), &[_]f32{ 1.0, 0.0 }, 1);
    gpa.allocator().free(results);

    // Update
    _ = try abi.database.update(&handle, 1, &[_]f32{ 0.0, 1.0 });

    // List
    const list = try abi.database.list(&handle, gpa.allocator(), 10);
    gpa.allocator().free(list);

    // Delete
    _ = abi.database.remove(&handle, 1);
}

// ============================================================================
// Feature Disabled Error Tests
// ============================================================================

// Test database when disabled at build time.
// Should provide clear error when feature not available.
test "feature disabled: database" {
    if (build_options.enable_database) return error.SkipZigTest;

    // When database is disabled, isEnabled should return false
    try std.testing.expect(!abi.database.isEnabled());
}

// Test LLM when disabled at build time.
// Should provide clear error when feature not available.
test "feature disabled: llm" {
    if (build_options.enable_llm) return error.SkipZigTest;

    // When LLM is disabled, isEnabled should return false
    try std.testing.expect(!abi.ai.llm.isEnabled());
}

// Test GPU when disabled at build time.
// Should provide clear indication when GPU unavailable.
test "feature disabled: gpu" {
    if (build_options.enable_gpu) return error.SkipZigTest;

    // When GPU is disabled at build time, module should handle gracefully
    // This test just verifies compilation succeeds with GPU disabled
    try std.testing.expect(!build_options.enable_gpu);
}

// ============================================================================
// Error Propagation Tests
// ============================================================================

// Test error propagation through function calls.
// Verifies errors bubble up correctly.
test "propagation: database error chain" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var handle = try abi.database.open(allocator, "test-propagation");
    defer abi.database.close(&handle);

    try abi.database.insert(&handle, 1, &[_]f32{ 1.0, 0.0 }, null);

    // Error from duplicate insert should propagate
    const err = abi.database.insert(&handle, 1, &[_]f32{ 0.0, 1.0 }, null);

    // Should get specific error type, not generic
    try std.testing.expectError(abi.database.database.DatabaseError.DuplicateId, err);
}

// Test multiple errors in sequence.
// Verifies error handling doesn't break subsequent operations.
test "propagation: sequential errors" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var handle = try abi.database.open(allocator, "test-seq-errors");
    defer abi.database.close(&handle);

    try abi.database.insert(&handle, 1, &[_]f32{ 1.0, 0.0 }, null);

    // Multiple errors in sequence
    _ = abi.database.insert(&handle, 1, &[_]f32{ 0.0, 1.0 }, null) catch {};
    _ = abi.database.insert(&handle, 1, &[_]f32{ 0.5, 0.5 }, null) catch {};
    _ = abi.database.insert(&handle, 2, &[_]f32{1.0}, null) catch {}; // wrong dim

    // Database should still work
    const s = abi.database.stats(&handle);
    try std.testing.expectEqual(@as(usize, 1), s.count);

    // Can still do valid operations
    try abi.database.insert(&handle, 2, &[_]f32{ 0.0, 1.0 }, null);
}

// ============================================================================
// Boundary Condition Tests
// ============================================================================

// Test with maximum u64 ID value.
// Verifies no overflow issues with large IDs.
test "boundary: max u64 id" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var handle = try abi.database.open(allocator, "test-max-id");
    defer abi.database.close(&handle);

    const max_id: u64 = std.math.maxInt(u64);
    try abi.database.insert(&handle, max_id, &[_]f32{ 1.0, 0.0 }, null);

    const view = abi.database.get(&handle, max_id);
    try std.testing.expect(view != null);
    try std.testing.expectEqual(max_id, view.?.id);
}

// Test with zero ID value.
// Zero should be valid ID.
test "boundary: zero id" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var handle = try abi.database.open(allocator, "test-zero-id");
    defer abi.database.close(&handle);

    try abi.database.insert(&handle, 0, &[_]f32{ 1.0, 0.0 }, null);

    const view = abi.database.get(&handle, 0);
    try std.testing.expect(view != null);
    try std.testing.expectEqual(@as(u64, 0), view.?.id);
}

// Test with empty search results.
// Should return empty slice, not error.
test "boundary: empty results" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var handle = try abi.database.open(allocator, "test-empty-results");
    defer abi.database.close(&handle);

    // Search empty database
    const results = try abi.database.search(&handle, allocator, &[_]f32{ 1.0, 0.0 }, 10);
    defer allocator.free(results);

    try std.testing.expectEqual(@as(usize, 0), results.len);
}

// Test with top_k of zero.
// Should return empty results.
test "boundary: top_k zero" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var handle = try abi.database.open(allocator, "test-topk-zero");
    defer abi.database.close(&handle);

    try abi.database.insert(&handle, 1, &[_]f32{ 1.0, 0.0 }, null);

    const results = try abi.database.search(&handle, allocator, &[_]f32{ 1.0, 0.0 }, 0);
    defer allocator.free(results);

    try std.testing.expectEqual(@as(usize, 0), results.len);
}

// ============================================================================
// Concurrent Access Simulation Tests
// ============================================================================

// Test rapid state transitions.
// Simulates concurrent-like access patterns.
test "concurrency: rapid operations" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var handle = try abi.database.open(allocator, "test-rapid");
    defer abi.database.close(&handle);

    // Rapid insert/delete/search cycle
    for (0..50) |i| {
        const id: u64 = @intCast(i);
        const val: f32 = @floatFromInt(i);

        // Insert
        try abi.database.insert(&handle, id, &[_]f32{ val, 0.0 }, null);

        // Search
        const results = try abi.database.search(&handle, allocator, &[_]f32{ val, 0.0 }, 1);
        allocator.free(results);

        // Delete every other
        if (i % 2 == 0) {
            _ = abi.database.remove(&handle, id);
        }
    }

    // Final count should be ~25
    const s = abi.database.stats(&handle);
    try std.testing.expectEqual(@as(usize, 25), s.count);
}

// ============================================================================
// Version and Build Info Tests
// ============================================================================

// Testversion string is valid.
// Verifiesversion returns non-empty semantic version.
test "version: format validity" {
    const version = abi.version();

    try std.testing.expect(version.len > 0);

    // Should contain at least one dot (semantic versioning)
    try std.testing.expect(std.mem.indexOf(u8, version, ".") != null);
}

// Testbuild options are accessible.
// Verifiesbuild configuration can be queried.
test "build options: accessibility" {
    // These should all be accessible booleans
    _ = build_options.enable_ai;
    _ = build_options.enable_gpu;
    _ = build_options.enable_database;
    _ = build_options.enable_network;
    _ = build_options.enable_web;
    _ = build_options.enable_profiling;
}
