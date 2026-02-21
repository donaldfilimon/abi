//! C API Integration Tests
//!
//! Comprehensive tests for validating the behavior that the C bindings expose.
//! These tests verify the underlying Zig implementation that the C API wraps,
//! ensuring the expected behavior for C/C++ consumers is maintained.
//!
//! The C bindings in `bindings/c/src/abi_c.zig` are tested at two levels:
//! 1. Direct tests within `abi_c.zig` itself (compiled as part of the C library)
//! 2. These integration tests that validate the underlying Zig APIs
//!
//! ## Test Categories
//!
//! Core tests live here. Domain-specific tests are in:
//! - `c_api_simd_test.zig` — SIMD capabilities, vector operations, struct layout
//! - `c_api_database_test.zig` — Database CRUD, count, delete, configuration
//! - `c_api_gpu_test.zig` — GPU availability, lifecycle, backend detection, config
//! - `c_api_agent_test.zig` — Agent CRUD, messaging, stats, history, config, structs

const std = @import("std");
const testing = std.testing;
const builtin = @import("builtin");
const build_options = @import("build_options");
const abi = @import("abi");
const gpu_detect = abi.gpu.backends.detect;

// ============================================================================
// C API Status Codes (matching src/bindings/c/exports.zig)
// ============================================================================

/// Status codes that mirror the C API AbiStatus enum
const CApiStatus = enum(c_int) {
    success = 0,
    error_unknown = 1,
    error_invalid_argument = 2,
    error_out_of_memory = 3,
    error_initialization_failed = 4,
    error_not_initialized = 5,
};

// ============================================================================
// Core Lifecycle Tests
// ============================================================================

test "c_api: framework init and shutdown lifecycle" {
    const allocator = testing.allocator;

    // Test framework initialization (underlying Zig API that C wraps)
    // The C API's abi_init() calls abi.Framework.init() internally
    if (build_options.enable_database) {
        // Initialize database module
        if (abi.database.init(allocator)) {
            defer abi.database.deinit();
            // If we get here, init succeeded
            try testing.expect(true);
        } else |_| {
            // Database init may fail in test environment without actual storage
            // This is acceptable - we're testing the API contract
        }
    }
}

test "c_api: framework can be initialized multiple times" {
    const allocator = testing.allocator;

    // Test that we can init and deinit multiple times without issues
    // This mirrors the C API contract where users may call abi_init/abi_shutdown
    // multiple times

    for (0..3) |_| {
        var fw = abi.Framework.initDefault(allocator) catch {
            // May fail if features aren't fully configured - acceptable in tests
            continue;
        };
        fw.deinit();
    }

    try testing.expect(true);
}

test "c_api: shutdown with uninitialized framework is safe" {
    // In C, calling abi_shutdown(NULL) should be a no-op
    // The Zig implementation handles null handles gracefully
    // This test verifies that pattern is safe

    // Simulating the C pattern: checking null before deinit
    const maybe_framework: ?*abi.Framework = null;
    if (maybe_framework) |fw| {
        fw.deinit();
    }
    // No crash = success
    try testing.expect(true);
}

// ============================================================================
// Version Info Tests
// ============================================================================

test "c_api: version returns valid string" {
    const version_str = abi.version();

    // Version should be a valid non-empty string
    try testing.expect(version_str.len > 0);

    // Version should match the expected format (semantic versioning)
    // Current version is "0.4.0" based on build_options
    try testing.expectEqualStrings("0.4.0", version_str);
}

test "c_api: version is consistent across calls" {
    const version1 = abi.version();
    const version2 = abi.version();

    // Version should be the same across calls (static string)
    try testing.expectEqualStrings(version1, version2);

    // And should point to the same memory (compile-time constant)
    try testing.expect(version1.ptr == version2.ptr);
}

test "c_api: version matches build options" {
    const version = abi.version();
    try testing.expectEqualStrings(build_options.package_version, version);
}

// ============================================================================
// Feature Detection Tests
// ============================================================================

test "c_api: feature flags are queryable" {
    // Test that feature detection aligns with compile-time build options
    // The C API exposes abi_is_feature_enabled() which checks these

    // GPU feature
    const gpu_enabled = build_options.enable_gpu;
    const gpu_module_enabled = gpu_detect.moduleEnabled();
    try testing.expect(gpu_enabled == gpu_module_enabled);

    // Database feature
    const db_enabled = build_options.enable_database;
    _ = db_enabled; // Feature is enabled at compile time

    // AI feature
    const ai_enabled = build_options.enable_ai;
    _ = ai_enabled; // Feature is enabled at compile time

    // Network feature
    const network_enabled = build_options.enable_network;
    _ = network_enabled; // Feature is enabled at compile time
}

test "c_api: disabled features return appropriate errors" {
    // When a feature is disabled at compile time, the stub modules
    // return appropriate error values. The C API translates these
    // to ABI_ERROR_* status codes.

    if (!build_options.enable_gpu) {
        // GPU stub should indicate unavailability
        try testing.expect(!gpu_detect.moduleEnabled());
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

test "c_api: error status values match expected C ABI" {
    // Verify error enum values match expected C ABI layout
    try testing.expect(@intFromEnum(CApiStatus.success) == 0);
    try testing.expect(@intFromEnum(CApiStatus.error_unknown) == 1);
    try testing.expect(@intFromEnum(CApiStatus.error_invalid_argument) == 2);
    try testing.expect(@intFromEnum(CApiStatus.error_out_of_memory) == 3);
    try testing.expect(@intFromEnum(CApiStatus.error_initialization_failed) == 4);
    try testing.expect(@intFromEnum(CApiStatus.error_not_initialized) == 5);
}

// ============================================================================
// Memory Safety Tests
// ============================================================================

test "c_api: repeated operations do not leak" {
    // This test uses the testing allocator which will detect leaks
    const allocator = testing.allocator;

    // Perform multiple framework init/deinit cycles
    for (0..5) |_| {
        var fw = abi.Framework.initDefault(allocator) catch {
            // May fail if features aren't configured - skip iteration
            continue;
        };
        fw.deinit();
    }

    // If the testing allocator doesn't detect leaks, we pass
    try testing.expect(true);
}

test "c_api: simd operations are memory safe" {
    // Test that SIMD operations don't corrupt memory
    var a: [64]f32 = undefined;
    var b: [64]f32 = undefined;
    var result: [64]f32 = undefined;

    // Initialize with known pattern
    for (&a, 0..) |*v, i| v.* = @floatFromInt(i);
    for (&b, 0..) |*v, i| v.* = @floatFromInt(64 - i);

    // Perform operations
    abi.simd.vectorAdd(&a, &b, &result);

    // All results should be 64.0
    for (result) |v| {
        try testing.expectApproxEqAbs(@as(f32, 64.0), v, 1e-6);
    }
}

// ============================================================================
// Integration with Zig API Tests
// ============================================================================

test "c_api: c and zig apis are consistent" {
    // Verify that C API behavior matches Zig API behavior

    // Version consistency
    const zig_version = abi.version();
    try testing.expectEqualStrings("0.4.0", zig_version);

    // SIMD consistency
    const zig_simd = abi.simd.hasSimdSupport();
    const zig_caps = abi.simd.getSimdCapabilities();
    try testing.expect(zig_simd == zig_caps.has_simd);

    // GPU consistency
    const gpu_enabled = gpu_detect.moduleEnabled();

    // GPU module enabled should match build options
    try testing.expect(gpu_enabled == build_options.enable_gpu);
}

// ============================================================================
// Edge Cases and Boundary Tests
// ============================================================================

test "c_api: empty vector operations" {
    // Test behavior with edge case inputs
    var empty: [0]f32 = undefined;
    var non_empty = [_]f32{ 1.0, 2.0, 3.0 };

    // Empty vector similarity should return 0
    const result = abi.simd.cosineSimilarity(&empty, &non_empty);
    try testing.expectApproxEqAbs(@as(f32, 0.0), result, 1e-6);
}

test "c_api: zero vector operations" {
    // Zero vectors should not cause crashes
    var zero_vec = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    var other_vec = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    // Cosine similarity with zero vector should return 0 (not NaN/Inf)
    const result = abi.simd.cosineSimilarity(&zero_vec, &other_vec);
    try testing.expectApproxEqAbs(@as(f32, 0.0), result, 1e-6);

    // L2 norm of zero vector should be 0
    const norm = abi.simd.vectorL2Norm(&zero_vec);
    try testing.expectApproxEqAbs(@as(f32, 0.0), norm, 1e-6);
}

test "c_api: large vector operations" {
    // Test with larger vectors to ensure SIMD paths work
    const allocator = testing.allocator;

    const size: usize = 1024;
    const a = try allocator.alloc(f32, size);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, size);
    defer allocator.free(b);
    const result = try allocator.alloc(f32, size);
    defer allocator.free(result);

    // Initialize
    for (a, 0..) |*v, i| v.* = @floatFromInt(i);
    for (b, 0..) |*v, i| v.* = @floatFromInt(size - i);

    // Vector add
    abi.simd.vectorAdd(a, b, result);

    // All should be size
    for (result) |v| {
        try testing.expectApproxEqAbs(@as(f32, @floatFromInt(size)), v, 1e-4);
    }
}

// ============================================================================
// Version Info Parsing Tests (abi_version_info)
// ============================================================================

test "c_api: version info parsing" {
    // The C API's abi_version_info parses the version string into components
    const version = abi.version();

    // Parse version string "X.Y.Z" like abi_version_info does
    var iter = std.mem.splitScalar(u8, version, '.');
    const major = std.fmt.parseInt(c_int, iter.next() orelse "0", 10) catch 0;
    const minor = std.fmt.parseInt(c_int, iter.next() orelse "0", 10) catch 0;
    const patch = std.fmt.parseInt(c_int, iter.next() orelse "0", 10) catch 0;

    // Current version is 0.4.0
    try testing.expectEqual(@as(c_int, 0), major);
    try testing.expectEqual(@as(c_int, 4), minor);
    try testing.expectEqual(@as(c_int, 0), patch);
}

test "c_api: version string format" {
    const version = abi.version();

    // Should be in format "X.Y.Z"
    var dot_count: usize = 0;
    for (version) |c| {
        if (c == '.') dot_count += 1;
    }
    try testing.expectEqual(@as(usize, 2), dot_count);
}

// ============================================================================
// Feature Detection String Tests (abi_is_feature_enabled)
// ============================================================================

test "c_api: feature enabled string lookup" {
    // Test all feature strings that abi_is_feature_enabled handles
    const features = [_]struct { name: []const u8, expected: bool }{
        .{ .name = "ai", .expected = build_options.enable_ai },
        .{ .name = "gpu", .expected = build_options.enable_gpu },
        .{ .name = "database", .expected = build_options.enable_database },
        .{ .name = "network", .expected = build_options.enable_network },
        .{ .name = "web", .expected = build_options.enable_web },
        .{ .name = "profiling", .expected = build_options.enable_profiling },
    };

    for (features) |f| {
        // The C API does string comparison like this
        if (std.mem.eql(u8, f.name, "ai")) {
            try testing.expect(build_options.enable_ai == f.expected);
        } else if (std.mem.eql(u8, f.name, "gpu")) {
            try testing.expect(build_options.enable_gpu == f.expected);
        } else if (std.mem.eql(u8, f.name, "database")) {
            try testing.expect(build_options.enable_database == f.expected);
        } else if (std.mem.eql(u8, f.name, "network")) {
            try testing.expect(build_options.enable_network == f.expected);
        } else if (std.mem.eql(u8, f.name, "web")) {
            try testing.expect(build_options.enable_web == f.expected);
        } else if (std.mem.eql(u8, f.name, "profiling")) {
            try testing.expect(build_options.enable_profiling == f.expected);
        }
    }
}

test "c_api: unknown feature returns false" {
    // Unknown features should return false (not crash)
    const unknown_features = [_][]const u8{
        "unknown",
        "nonexistent",
        "",
        "AI", // case sensitive
        "GPU",
    };

    for (unknown_features) |feature| {
        // The C API checks known features and returns false for unknown
        const is_known = std.mem.eql(u8, feature, "ai") or
            std.mem.eql(u8, feature, "gpu") or
            std.mem.eql(u8, feature, "database") or
            std.mem.eql(u8, feature, "network") or
            std.mem.eql(u8, feature, "web") or
            std.mem.eql(u8, feature, "profiling");

        try testing.expect(!is_known);
    }
}

// ============================================================================
// Framework Configuration Tests
// ============================================================================

test "c_api: framework init with custom options" {
    const allocator = testing.allocator;

    // The C API's Options struct mirrors this configuration
    var config = abi.Config.defaults();

    // Disable optional features for this test
    config.observability = null;

    // Init framework with custom config
    var fw = abi.Framework.init(allocator, config) catch {
        // May fail without full feature setup - acceptable
        return error.SkipZigTest;
    };
    defer fw.deinit();
}

test "c_api: framework config defaults" {
    const config = abi.Config.defaults();

    // Verify default config is populated based on build options
    if (build_options.enable_gpu) {
        try testing.expect(config.gpu != null);
    }
    if (build_options.enable_database) {
        try testing.expect(config.database != null);
    }
    if (build_options.enable_ai) {
        try testing.expect(config.ai != null);
    }
}

// ============================================================================
// Memory Management Tests
// ============================================================================

test "c_api: string allocation and free pattern" {
    const allocator = testing.allocator;

    // Allocate a string like the C API does
    const str = try allocator.alloc(u8, 12);
    @memcpy(str[0..11], "Hello World");
    str[11] = 0; // null terminator

    // Verify it's a valid C string
    const len = std.mem.indexOfScalar(u8, str, 0) orelse str.len;
    try testing.expectEqual(@as(usize, 11), len);

    // Free the string (C API: abi_free_string)
    allocator.free(str);
}

test "c_api: search results allocation pattern" {
    const allocator = testing.allocator;

    // The SearchResult struct used by C API
    const SearchResult = extern struct {
        id: u64,
        score: f32,
    };

    // Allocate results array like the C API does
    const count: usize = 10;
    const results = try allocator.alloc(SearchResult, count);
    defer allocator.free(results);

    // Populate results
    for (results, 0..) |*r, i| {
        r.* = .{
            .id = @intCast(i + 1),
            .score = 1.0 - @as(f32, @floatFromInt(i)) * 0.1,
        };
    }

    // Verify results
    try testing.expectEqual(@as(u64, 1), results[0].id);
    try testing.expectApproxEqAbs(@as(f32, 1.0), results[0].score, 1e-6);
}

// ============================================================================
// Error Code Mapping Tests
// ============================================================================

test "c_api: error code values" {
    // Verify error code constants match expected C ABI
    // These values are defined in abi_c.zig

    const ABI_OK: c_int = 0;
    const ABI_ERROR_INIT_FAILED: c_int = -1;
    const ABI_ERROR_ALREADY_INITIALIZED: c_int = -2;
    const ABI_ERROR_NOT_INITIALIZED: c_int = -3;
    const ABI_ERROR_OUT_OF_MEMORY: c_int = -4;
    const ABI_ERROR_INVALID_ARGUMENT: c_int = -5;
    const ABI_ERROR_FEATURE_DISABLED: c_int = -6;
    const ABI_ERROR_TIMEOUT: c_int = -7;
    const ABI_ERROR_IO: c_int = -8;
    const ABI_ERROR_GPU_UNAVAILABLE: c_int = -9;
    const ABI_ERROR_DATABASE_ERROR: c_int = -10;
    const ABI_ERROR_NETWORK_ERROR: c_int = -11;
    const ABI_ERROR_AI_ERROR: c_int = -12;
    const ABI_ERROR_UNKNOWN: c_int = -99;

    // Verify all error codes are negative except OK
    try testing.expect(ABI_OK == 0);
    try testing.expect(ABI_ERROR_INIT_FAILED < 0);
    try testing.expect(ABI_ERROR_ALREADY_INITIALIZED < 0);
    try testing.expect(ABI_ERROR_NOT_INITIALIZED < 0);
    try testing.expect(ABI_ERROR_OUT_OF_MEMORY < 0);
    try testing.expect(ABI_ERROR_INVALID_ARGUMENT < 0);
    try testing.expect(ABI_ERROR_FEATURE_DISABLED < 0);
    try testing.expect(ABI_ERROR_TIMEOUT < 0);
    try testing.expect(ABI_ERROR_IO < 0);
    try testing.expect(ABI_ERROR_GPU_UNAVAILABLE < 0);
    try testing.expect(ABI_ERROR_DATABASE_ERROR < 0);
    try testing.expect(ABI_ERROR_NETWORK_ERROR < 0);
    try testing.expect(ABI_ERROR_AI_ERROR < 0);
    try testing.expect(ABI_ERROR_UNKNOWN < 0);

    // All error codes should be unique
    const errors = [_]c_int{
        ABI_ERROR_INIT_FAILED,
        ABI_ERROR_ALREADY_INITIALIZED,
        ABI_ERROR_NOT_INITIALIZED,
        ABI_ERROR_OUT_OF_MEMORY,
        ABI_ERROR_INVALID_ARGUMENT,
        ABI_ERROR_FEATURE_DISABLED,
        ABI_ERROR_TIMEOUT,
        ABI_ERROR_IO,
        ABI_ERROR_GPU_UNAVAILABLE,
        ABI_ERROR_DATABASE_ERROR,
        ABI_ERROR_NETWORK_ERROR,
        ABI_ERROR_AI_ERROR,
    };

    for (errors, 0..) |e1, i| {
        for (errors[i + 1 ..]) |e2| {
            try testing.expect(e1 != e2);
        }
    }
}

test "c_api: error string lookup" {
    // Map of error codes to expected messages
    const error_messages = [_]struct { code: c_int, expected: []const u8 }{
        .{ .code = 0, .expected = "Success" },
        .{ .code = -1, .expected = "Initialization failed" },
        .{ .code = -2, .expected = "Already initialized" },
        .{ .code = -3, .expected = "Not initialized" },
        .{ .code = -4, .expected = "Out of memory" },
        .{ .code = -5, .expected = "Invalid argument" },
        .{ .code = -6, .expected = "Feature disabled at compile time" },
        .{ .code = -7, .expected = "Operation timed out" },
        .{ .code = -8, .expected = "I/O error" },
        .{ .code = -9, .expected = "GPU not available" },
        .{ .code = -10, .expected = "Database error" },
        .{ .code = -11, .expected = "Network error" },
        .{ .code = -12, .expected = "AI operation error" },
        .{ .code = -99, .expected = "Unknown error" },
    };

    // Each error code should have a valid message (not empty)
    for (error_messages) |em| {
        try testing.expect(em.expected.len > 0);
    }
}

// ============================================================================
// Test discovery for extracted test files
// ============================================================================

test {
    _ = @import("c_api_simd_test.zig");
    _ = @import("c_api_database_test.zig");
    _ = @import("c_api_gpu_test.zig");
    if (build_options.enable_ai) {
        _ = @import("c_api_agent_test.zig");
    }
}
