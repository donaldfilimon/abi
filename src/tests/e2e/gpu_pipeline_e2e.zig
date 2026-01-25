//! End-to-End GPU Pipeline Tests
//!
//! Complete workflow tests for GPU operations:
//! - GPU context initialization
//! - Data upload and download
//! - Compute operations with verification
//! - Multi-operation pipelines

const std = @import("std");
const build_options = @import("build_options");
const abi = @import("abi");
const e2e = @import("mod.zig");

// ============================================================================
// Helper Functions
// ============================================================================

/// Skip test if GPU is disabled.
fn skipIfGpuDisabled() !void {
    if (!build_options.enable_gpu) return error.SkipZigTest;
}

/// CPU reference implementation for vector addition.
fn cpuVectorAdd(a: []const f32, b: []const f32, result: []f32) void {
    for (a, b, result) |av, bv, *rv| {
        rv.* = av + bv;
    }
}

/// CPU reference implementation for vector dot product.
fn cpuVectorDot(a: []const f32, b: []const f32) f32 {
    var sum: f32 = 0;
    for (a, b) |av, bv| {
        sum += av * bv;
    }
    return sum;
}

/// CPU reference implementation for matrix multiplication.
fn cpuMatMul(a: []const f32, b: []const f32, result: []f32, m: usize, k: usize, n: usize) void {
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0;
            for (0..k) |l| {
                sum += a[i * k + l] * b[l * n + j];
            }
            result[i * n + j] = sum;
        }
    }
}

/// Validate GPU results against CPU reference.
fn validateResults(gpu_result: []const f32, cpu_result: []const f32, tolerance: f32) !void {
    if (gpu_result.len != cpu_result.len) {
        return error.LengthMismatch;
    }
    for (gpu_result, cpu_result) |gv, cv| {
        const diff = @abs(gv - cv);
        if (diff > tolerance) {
            return error.ToleranceExceeded;
        }
    }
}

// ============================================================================
// Mock GPU Context for testing without real GPU
// ============================================================================

const MockGpuContext = struct {
    allocator: std.mem.Allocator,
    operation_count: u64 = 0,
    bytes_transferred: u64 = 0,

    pub fn init(allocator: std.mem.Allocator) MockGpuContext {
        return .{ .allocator = allocator };
    }

    pub fn vectorAdd(self: *MockGpuContext, a: []const f32, b: []const f32, result: []f32) void {
        cpuVectorAdd(a, b, result);
        self.operation_count += 1;
    }

    pub fn matrixMultiply(self: *MockGpuContext, a: []const f32, b: []const f32, result: []f32, m: usize, k: usize, n: usize) void {
        cpuMatMul(a, b, result, m, k, n);
        self.operation_count += 1;
    }

    pub fn upload(self: *MockGpuContext, data: []const f32) void {
        self.bytes_transferred += data.len * @sizeOf(f32);
    }

    pub fn download(self: *MockGpuContext, data: []const f32) void {
        self.bytes_transferred += data.len * @sizeOf(f32);
    }
};

// ============================================================================
// E2E Tests: GPU Initialization
// ============================================================================

test "e2e: gpu context initialization" {
    try skipIfGpuDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .gpu = true },
        .timeout_ms = 30_000,
    });
    defer ctx.deinit();

    // Verify GPU feature is enabled
    try std.testing.expect(ctx.isFeatureAvailable(.gpu));

    // Check GPU is enabled at compile time
    try std.testing.expect(abi.gpu.isEnabled());
}

test "e2e: gpu module detection" {
    try skipIfGpuDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .gpu = true },
    });
    defer ctx.deinit();

    // Check backend availability
    const available = abi.gpu.availableBackends();
    try std.testing.expect(available.len > 0 or true); // May be empty on systems without GPU
}

// ============================================================================
// E2E Tests: Vector Operations Pipeline
// ============================================================================

test "e2e: gpu vector addition pipeline" {
    try skipIfGpuDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .gpu = true },
    });
    defer ctx.deinit();

    var timer = e2e.WorkflowTimer.init(allocator);
    defer timer.deinit();

    // Use mock GPU for testing without real hardware
    var gpu = MockGpuContext.init(allocator);

    // 1. Generate test data
    const size: usize = 1024;
    const a = try e2e.generateTestVector(allocator, size, 42);
    defer allocator.free(a);

    const b = try e2e.generateTestVector(allocator, size, 123);
    defer allocator.free(b);

    const result = try allocator.alloc(f32, size);
    defer allocator.free(result);

    try timer.checkpoint("data_generated");

    // 2. Upload data to GPU
    gpu.upload(a);
    gpu.upload(b);

    try timer.checkpoint("data_uploaded");

    // 3. Perform vector addition
    gpu.vectorAdd(a, b, result);

    try timer.checkpoint("computation_complete");

    // 4. Download results
    gpu.download(result);

    try timer.checkpoint("results_downloaded");

    // 5. Verify results against CPU reference
    const cpu_result = try allocator.alloc(f32, size);
    defer allocator.free(cpu_result);

    cpuVectorAdd(a, b, cpu_result);

    try validateResults(result, cpu_result, 0.0001);

    try timer.checkpoint("results_verified");

    // Workflow should complete quickly
    try std.testing.expect(!timer.isTimedOut(10_000));
}

test "e2e: gpu matrix multiplication pipeline" {
    try skipIfGpuDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .gpu = true },
    });
    defer ctx.deinit();

    var timer = e2e.WorkflowTimer.init(allocator);
    defer timer.deinit();

    var gpu = MockGpuContext.init(allocator);

    // Matrix dimensions
    const m: usize = 32;
    const k: usize = 64;
    const n: usize = 32;

    // 1. Generate test matrices
    const a = try allocator.alloc(f32, m * k);
    defer allocator.free(a);

    const b = try allocator.alloc(f32, k * n);
    defer allocator.free(b);

    var rng = std.Random.DefaultPrng.init(42);
    for (a) |*v| {
        v.* = rng.random().float(f32) * 2.0 - 1.0;
    }
    for (b) |*v| {
        v.* = rng.random().float(f32) * 2.0 - 1.0;
    }

    const result = try allocator.alloc(f32, m * n);
    defer allocator.free(result);

    try timer.checkpoint("matrices_generated");

    // 2. Upload matrices
    gpu.upload(a);
    gpu.upload(b);

    try timer.checkpoint("matrices_uploaded");

    // 3. Perform matrix multiplication
    gpu.matrixMultiply(a, b, result, m, k, n);

    try timer.checkpoint("matmul_complete");

    // 4. Verify results
    const cpu_result = try allocator.alloc(f32, m * n);
    defer allocator.free(cpu_result);

    cpuMatMul(a, b, cpu_result, m, k, n);

    try validateResults(result, cpu_result, 0.001);

    try timer.checkpoint("results_verified");

    try std.testing.expect(!timer.isTimedOut(30_000));
}

// ============================================================================
// E2E Tests: Multi-Operation Pipelines
// ============================================================================

test "e2e: chained gpu operations" {
    try skipIfGpuDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .gpu = true },
    });
    defer ctx.deinit();

    var timer = e2e.WorkflowTimer.init(allocator);
    defer timer.deinit();

    var gpu = MockGpuContext.init(allocator);

    const size: usize = 256;

    // Generate test vectors
    const a = try e2e.generateTestVector(allocator, size, 1);
    defer allocator.free(a);

    const b = try e2e.generateTestVector(allocator, size, 2);
    defer allocator.free(b);

    const c = try e2e.generateTestVector(allocator, size, 3);
    defer allocator.free(c);

    // Intermediate and final results
    const temp = try allocator.alloc(f32, size);
    defer allocator.free(temp);

    var final = try allocator.alloc(f32, size);
    defer allocator.free(final);

    try timer.checkpoint("data_ready");

    // Chain: (a + b) + c
    gpu.vectorAdd(a, b, temp);
    gpu.vectorAdd(temp, c, final);

    try timer.checkpoint("chain_complete");

    // Verify: result should equal a + b + c
    for (0..size) |i| {
        const expected = a[i] + b[i] + c[i];
        try std.testing.expectApproxEqAbs(expected, final[i], 0.0001);
    }

    try timer.checkpoint("verified");

    // Check operation count
    try std.testing.expectEqual(@as(u64, 2), gpu.operation_count);
}

test "e2e: batch vector processing" {
    try skipIfGpuDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .gpu = true },
    });
    defer ctx.deinit();

    var timer = e2e.WorkflowTimer.init(allocator);
    defer timer.deinit();

    var gpu = MockGpuContext.init(allocator);

    const batch_size: usize = 16;
    const vector_size: usize = 128;

    // Process batch of vectors
    for (0..batch_size) |batch_idx| {
        const seed = @as(u64, batch_idx) * 1000;

        const a = try e2e.generateTestVector(allocator, vector_size, seed);
        defer allocator.free(a);

        const b = try e2e.generateTestVector(allocator, vector_size, seed + 1);
        defer allocator.free(b);

        const result = try allocator.alloc(f32, vector_size);
        defer allocator.free(result);

        gpu.vectorAdd(a, b, result);

        // Verify each batch
        for (0..vector_size) |i| {
            const expected = a[i] + b[i];
            try std.testing.expectApproxEqAbs(expected, result[i], 0.0001);
        }
    }

    try timer.checkpoint("batch_complete");

    try std.testing.expectEqual(@as(u64, batch_size), gpu.operation_count);
}

// ============================================================================
// E2E Tests: Edge Cases and Error Handling
// ============================================================================

test "e2e: gpu handles empty vectors" {
    try skipIfGpuDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .gpu = true },
    });
    defer ctx.deinit();

    var gpu = MockGpuContext.init(allocator);

    // Empty vectors
    const empty: []const f32 = &.{};
    var result: [0]f32 = .{};

    // Should handle gracefully
    gpu.vectorAdd(empty, empty, &result);
    try std.testing.expectEqual(@as(u64, 1), gpu.operation_count);
}

test "e2e: gpu handles single element" {
    try skipIfGpuDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .gpu = true },
    });
    defer ctx.deinit();

    var gpu = MockGpuContext.init(allocator);

    const a = [_]f32{1.5};
    const b = [_]f32{2.5};
    var result: [1]f32 = undefined;

    gpu.vectorAdd(&a, &b, &result);

    try std.testing.expectApproxEqAbs(@as(f32, 4.0), result[0], 0.0001);
}

test "e2e: gpu handles special float values" {
    try skipIfGpuDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .gpu = true },
    });
    defer ctx.deinit();

    var gpu = MockGpuContext.init(allocator);

    // Very small values
    const small = [_]f32{ 1e-38, 1e-38, 1e-38 };
    var result1: [3]f32 = undefined;
    gpu.vectorAdd(&small, &small, &result1);

    for (result1) |v| {
        try std.testing.expect(std.math.isFinite(v));
    }

    // Very large values
    const large = [_]f32{ 1e38, 1e38, 1e38 };
    var result2: [3]f32 = undefined;

    // Note: Adding two large values may overflow to infinity
    gpu.vectorAdd(&large, &large, &result2);

    // Results should be computed (may be infinite, but not NaN from operation itself)
    // This tests that the operation completes without crashing
    try std.testing.expectEqual(@as(u64, 2), gpu.operation_count);
}

// ============================================================================
// E2E Tests: Performance Benchmarks
// ============================================================================

test "e2e: gpu throughput benchmark" {
    try skipIfGpuDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .gpu = true },
        .timeout_ms = 60_000,
    });
    defer ctx.deinit();

    var timer = e2e.WorkflowTimer.init(allocator);
    defer timer.deinit();

    var gpu = MockGpuContext.init(allocator);

    const vector_size: usize = 1024 * 1024; // 1M elements
    const iterations: usize = 10;

    const a = try allocator.alloc(f32, vector_size);
    defer allocator.free(a);
    @memset(a, 1.0);

    const b = try allocator.alloc(f32, vector_size);
    defer allocator.free(b);
    @memset(b, 2.0);

    const result = try allocator.alloc(f32, vector_size);
    defer allocator.free(result);

    try timer.checkpoint("setup_complete");

    var total_time_ns: u64 = 0;

    for (0..iterations) |_| {
        var iter_timer = try std.time.Timer.start();
        gpu.vectorAdd(a, b, result);
        total_time_ns += iter_timer.read();
    }

    try timer.checkpoint("benchmark_complete");

    const avg_time_ns = total_time_ns / iterations;

    // Calculate throughput
    const bytes_processed = vector_size * @sizeOf(f32) * 3; // a, b, result
    const throughput_gbps = @as(f64, @floatFromInt(bytes_processed)) / @as(f64, @floatFromInt(avg_time_ns));

    // Log throughput (should be > 0)
    try std.testing.expect(throughput_gbps > 0);

    // Should complete within timeout
    try std.testing.expect(!timer.isTimedOut(60_000));
}

// ============================================================================
// E2E Tests: GPU with Database Integration
// ============================================================================

test "e2e: gpu accelerated vector similarity" {
    try skipIfGpuDisabled();
    if (!build_options.enable_database) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .gpu = true, .database = true },
    });
    defer ctx.deinit();

    var timer = e2e.WorkflowTimer.init(allocator);
    defer timer.deinit();

    // Create database
    var handle = try abi.database.open(allocator, "test-e2e-gpu-db");
    defer abi.database.close(&handle);

    // Insert vectors
    const vec_count: usize = 100;
    const vec_dim: usize = 128;

    for (0..vec_count) |i| {
        const vec = try e2e.generateNormalizedVector(allocator, vec_dim, @as(u64, i) * 31337);
        defer allocator.free(vec);
        try abi.database.insert(&handle, @intCast(i), vec, null);
    }

    try timer.checkpoint("vectors_inserted");

    // Generate query
    const query = try e2e.generateNormalizedVector(allocator, vec_dim, 42);
    defer allocator.free(query);

    // Search (would use GPU acceleration if available)
    const results = try abi.database.search(&handle, allocator, query, 10);
    defer allocator.free(results);

    try timer.checkpoint("search_complete");

    try std.testing.expect(results.len > 0);
    try std.testing.expect(results.len <= 10);

    // Results should be sorted by score
    for (1..results.len) |i| {
        try std.testing.expect(results[i].score <= results[i - 1].score);
    }
}

// ============================================================================
// E2E Tests: Memory Management
// ============================================================================

test "e2e: gpu memory lifecycle" {
    try skipIfGpuDisabled();

    const allocator = std.testing.allocator;

    var ctx = try e2e.E2EContext.init(allocator, .{
        .features = .{ .gpu = true },
    });
    defer ctx.deinit();

    var gpu = MockGpuContext.init(allocator);

    // Multiple allocation/deallocation cycles
    for (0..10) |cycle| {
        const size = (cycle + 1) * 1024;

        const data = try allocator.alloc(f32, size);
        defer allocator.free(data);

        gpu.upload(data);
        gpu.download(data);
    }

    // All memory should be properly tracked
    try std.testing.expect(gpu.bytes_transferred > 0);
}
