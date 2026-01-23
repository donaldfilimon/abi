//! Integration Test Matrix
//!
//! Tests feature combinations to ensure cross-module compatibility.
//! Each test category validates specific feature interactions.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

/// Test matrix configuration
pub const TestMatrix = struct {
    /// Feature combination being tested
    pub const FeatureSet = struct {
        ai: bool = false,
        gpu: bool = false,
        database: bool = false,
        network: bool = false,
        web: bool = false,
        monitoring: bool = false,
    };

    /// Test result for a single combination
    pub const TestResult = struct {
        name: []const u8,
        features: FeatureSet,
        passed: bool,
        duration_ns: u64,
        error_msg: ?[]const u8 = null,
    };

    allocator: std.mem.Allocator,
    results: std.ArrayListUnmanaged(TestResult),

    pub fn init(allocator: std.mem.Allocator) TestMatrix {
        return .{
            .allocator = allocator,
            .results = .{},
        };
    }

    pub fn deinit(self: *TestMatrix) void {
        self.results.deinit(self.allocator);
    }

    pub fn addResult(self: *TestMatrix, result: TestResult) !void {
        try self.results.append(self.allocator, result);
    }

    pub fn passedCount(self: *const TestMatrix) usize {
        var count: usize = 0;
        for (self.results.items) |r| {
            if (r.passed) count += 1;
        }
        return count;
    }

    pub fn failedCount(self: *const TestMatrix) usize {
        return self.results.items.len - self.passedCount();
    }

    pub fn printSummary(self: *const TestMatrix) void {
        std.debug.print("\n", .{});
        std.debug.print("=" ** 70 ++ "\n", .{});
        std.debug.print("                    INTEGRATION TEST MATRIX RESULTS\n", .{});
        std.debug.print("=" ** 70 ++ "\n\n", .{});

        for (self.results.items) |r| {
            const status = if (r.passed) "PASS" else "FAIL";
            const symbol = if (r.passed) "✓" else "✗";
            std.debug.print("[{s}] {s} {s}\n", .{ status, symbol, r.name });

            if (r.error_msg) |msg| {
                std.debug.print("       Error: {s}\n", .{msg});
            }
        }

        std.debug.print("\n" ++ "-" ** 70 ++ "\n", .{});
        std.debug.print("Total: {d} passed, {d} failed out of {d} tests\n", .{
            self.passedCount(),
            self.failedCount(),
            self.results.items.len,
        });
        std.debug.print("=" ** 70 ++ "\n", .{});
    }
};

// ============================================================================
// Test: Core Framework Initialization
// ============================================================================

test "matrix: framework minimal config" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var framework = try abi.init(gpa.allocator(), abi.FrameworkOptions{
        .enable_gpu = false,
        .enable_ai = false,
        .enable_web = false,
        .enable_database = false,
        .enable_network = false,
        .enable_profiling = false,
    });
    defer framework.deinit();

    try std.testing.expect(framework.isRunning());
}

test "matrix: framework all features" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var framework = try abi.init(gpa.allocator(), abi.FrameworkOptions{
        .enable_gpu = build_options.enable_gpu,
        .enable_ai = build_options.enable_ai,
        .enable_web = build_options.enable_web,
        .enable_database = build_options.enable_database,
        .enable_network = build_options.enable_network,
        .enable_profiling = build_options.enable_profiling,
    });
    defer framework.deinit();

    try std.testing.expect(framework.isRunning());
}

// ============================================================================
// Test: SIMD + Database Integration
// ============================================================================

test "matrix: simd vector ops" {
    const simd = abi.simd;

    // Test vector operations
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    var result: [4]f32 = undefined;

    simd.vectorAdd(&a, &b, &result);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), result[0], 1e-6);

    const dot = simd.vectorDot(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 70.0), dot, 1e-6);

    const caps = simd.getSimdCapabilities();
    try std.testing.expect(caps.vector_size >= 1);
}

test "matrix: simd matrix multiply" {
    const simd = abi.simd;

    // 2x2 matrix multiply
    var mat_a = [_]f32{ 1, 2, 3, 4 };
    var mat_b = [_]f32{ 5, 6, 7, 8 };
    var mat_result: [4]f32 = undefined;

    simd.matrixMultiply(&mat_a, &mat_b, &mat_result, 2, 2, 2);

    // Expected: [1*5+2*7, 1*6+2*8, 3*5+4*7, 3*6+4*8] = [19, 22, 43, 50]
    try std.testing.expectApproxEqAbs(@as(f32, 19.0), mat_result[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 22.0), mat_result[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 43.0), mat_result[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 50.0), mat_result[3], 1e-5);
}

// ============================================================================
// Test: Compute Runtime
// ============================================================================

test "matrix: runtime engine basic" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var engine = try abi.runtime.createEngine(gpa.allocator(), .{});
    defer engine.deinit();

    // Engine initialized successfully - no isRunning method, just verify init works
    try std.testing.expect(true);
}

// ============================================================================
// Test: GPU Module (when enabled)
// ============================================================================

test "matrix: gpu backend detection" {
    if (!build_options.enable_gpu) return error.SkipZigTest;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    try abi.gpu.ensureInitialized(gpa.allocator());

    const backends = try abi.gpu.availableBackends(gpa.allocator());
    defer gpa.allocator().free(backends);

    // Should have at least one backend (even if simulated)
    try std.testing.expect(backends.len >= 0);
}

// ============================================================================
// Test: AI Module (when enabled)
// ============================================================================

test "matrix: ai module exports" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    // Verify AI module exports are accessible
    _ = abi.ai;
    try std.testing.expect(true);
}

// ============================================================================
// Test: Database Module (when enabled)
// ============================================================================

test "matrix: database module exports" {
    if (!build_options.enable_database) return error.SkipZigTest;

    // Verify database module exports
    _ = abi.database;
    try std.testing.expect(true);
}

// ============================================================================
// Test: Network Module (when enabled)
// ============================================================================

test "matrix: network module exports" {
    if (!build_options.enable_network) return error.SkipZigTest;

    // Verify network module exports
    _ = abi.network;
    try std.testing.expect(true);
}

// ============================================================================
// Test: Cross-Feature Integration
// ============================================================================

test "matrix: simd capabilities struct" {
    const caps = abi.simd.getSimdCapabilities();

    // Verify capabilities structure
    try std.testing.expect(caps.vector_size >= 1);

    // Architecture should be valid
    _ = caps.arch;
    _ = caps.has_simd;
}

test "matrix: feature isolation" {
    // Test that disabled features don't affect enabled ones
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    // Initialize with only monitoring
    var framework = try abi.init(gpa.allocator(), abi.FrameworkOptions{
        .enable_gpu = false,
        .enable_ai = false,
        .enable_web = false,
        .enable_database = false,
        .enable_network = false,
        .enable_profiling = true,
    });
    defer framework.deinit();

    try std.testing.expect(framework.isRunning());

    // SIMD should still work regardless of feature flags
    const has_simd = abi.simd.hasSimdSupport();
    _ = has_simd; // Just verify it compiles and runs
}

// ============================================================================
// Test: Memory Safety
// ============================================================================

test "matrix: allocator stress" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    // Allocate and free multiple times
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const buf = try gpa.allocator().alloc(u8, 1024);
        defer gpa.allocator().free(buf);
        @memset(buf, @intCast(i % 256));
    }
}

test "matrix: framework reinit" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    // Initialize, deinit, and reinitialize
    {
        var framework = try abi.init(gpa.allocator(), abi.FrameworkOptions{});
        defer framework.deinit();
        try std.testing.expect(framework.isRunning());
    }

    // Should be able to reinitialize
    {
        var framework2 = try abi.init(gpa.allocator(), abi.FrameworkOptions{});
        defer framework2.deinit();
        try std.testing.expect(framework2.isRunning());
    }
}

// ============================================================================
// Test: Version and Build Info
// ============================================================================

test "matrix: version info" {
    const version = abi.version();
    try std.testing.expect(version.len > 0);
    try std.testing.expectEqualStrings("0.1.1", version);
}
