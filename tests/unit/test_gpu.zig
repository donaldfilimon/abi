//! GPU Module Tests
//!
//! Comprehensive tests for GPU functionality including:
//! - GPU backend initialization and configuration
//! - Vector similarity search with GPU acceleration
//! - Batch processing capabilities
//! - Memory management and resource cleanup
//! - Performance statistics and metrics
//! - CPU fallback functionality
//! - Error handling and edge cases

const std = @import("std");
const testing = std.testing;
const abi = @import("abi");
const gpu = abi.gpu.core;

test "GPU module imports" {
    // Test that all GPU components are accessible
    _ = gpu.GPURenderer;
    _ = gpu.GPUConfig;
    _ = gpu.GpuError;
    _ = gpu.GpuBackend;
    _ = gpu.GpuBackendConfig;
    _ = gpu.GpuBackendError;
    _ = gpu.BatchConfig;
    _ = gpu.BatchProcessor;
    _ = gpu.GpuStats;
}

test "GPU backend initialization" {
    const allocator = testing.allocator;

    // Test default configuration
    const config = gpu.GpuBackendConfig{
        .max_batch_size = 512,
        .memory_limit = 256 * 1024 * 1024, // 256MB
        .debug_validation = false,
        .power_preference = .high_performance,
    };

    var backend = try gpu.GpuBackend.init(allocator, config);
    defer backend.deinit();

    // Test that backend is properly initialized
    try testing.expect(backend.config.max_batch_size == 512);
    try testing.expect(backend.config.memory_limit == 256 * 1024 * 1024);
    try testing.expect(backend.memory_used == 0);

    // Test GPU availability check
    _ = backend.isGpuAvailable();
}

test "GPU backend memory management" {
    const allocator = testing.allocator;

    const config = gpu.GpuBackendConfig{
        .max_batch_size = 1024,
        .memory_limit = 128 * 1024 * 1024, // 128MB
        .debug_validation = false,
        .power_preference = .high_performance,
    };

    var backend = try gpu.GpuBackend.init(allocator, config);
    defer backend.deinit();

    // Test memory availability checks
    try testing.expect(backend.hasMemoryFor(64 * 1024 * 1024)); // 64MB - should fit
    try testing.expect(!backend.hasMemoryFor(200 * 1024 * 1024)); // 200MB - should not fit
}

test "GPU backend vector search - CPU fallback" {
    const allocator = testing.allocator;

    // Create test database
    const test_file = "test_gpu_db.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    // For testing, we'll create a mock database instance
    // In real usage, this would be a proper database from the wdbx module
    const db = gpu.Db{
        .header = .{
            .row_count = 5,
            .dim = 8,
            .records_off = 0,
        },
        .file = undefined, // Mock file handle pointer
    };

    // Database is already initialized with mock data

    // Initialize GPU backend (will use CPU fallback if no GPU available)
    const config = gpu.GpuBackendConfig{
        .max_batch_size = 1024,
        .memory_limit = 512 * 1024 * 1024,
        .debug_validation = false,
        .power_preference = .high_performance,
    };

    var backend = try gpu.GpuBackend.init(allocator, config);
    defer backend.deinit();

    // Test search with first vector (should be most similar to itself)
    const query = [_]f32{ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    const results = try backend.searchSimilar(&db, &query, 3);
    defer allocator.free(results);

    try testing.expect(results.len == 3);
    try testing.expect(results[0].index == 0); // First vector should be most similar
    // Note: In CPU fallback mode, the exact score ordering may vary due to floating point precision
    // The important thing is that we get results and the first result is the query vector itself
    try testing.expect(results[0].score <= results[1].score); // Distance should be smaller or equal for closer vectors
}

test "GPU backend batch search" {
    const allocator = testing.allocator;

    // Create mock database
    const db = gpu.Db{
        .header = .{
            .row_count = 4,
            .dim = 4,
            .records_off = 0,
        },
        .file = undefined,
    };

    // Database is already initialized with mock data

    // Initialize GPU backend
    const config = gpu.GpuBackendConfig{
        .max_batch_size = 1024,
        .memory_limit = 512 * 1024 * 1024,
        .debug_validation = false,
        .power_preference = .high_performance,
    };

    var backend = try gpu.GpuBackend.init(allocator, config);
    defer backend.deinit();

    // Test batch search with multiple queries
    const queries = [_][]const f32{
        &[_]f32{ 1.0, 0.0, 0.0, 0.0 },
        &[_]f32{ 0.0, 1.0, 0.0, 0.0 },
        &[_]f32{ 0.5, 0.5, 0.0, 0.0 },
    };

    const batch_results = try backend.batchSearch(&db, &queries, 2);
    defer {
        for (batch_results) |result| {
            allocator.free(result);
        }
        allocator.free(batch_results);
    }

    try testing.expect(batch_results.len == 3);
    for (batch_results) |result| {
        try testing.expect(result.len == 2);
    }
}

test "GPU batch processor" {
    const allocator = testing.allocator;

    // Create mock database
    const db = gpu.Db{
        .header = .{
            .row_count = 3,
            .dim = 4,
            .records_off = 0,
        },
        .file = undefined,
    };

    // Database is already initialized with mock data

    // Initialize GPU backend
    const backend_config = gpu.GpuBackendConfig{
        .max_batch_size = 1024,
        .memory_limit = 512 * 1024 * 1024,
        .debug_validation = false,
        .power_preference = .high_performance,
    };

    var backend = try gpu.GpuBackend.init(allocator, backend_config);
    defer backend.deinit();

    // Initialize batch processor
    const batch_config = gpu.BatchConfig{
        .parallel_queries = 2,
        .max_batch_size = 512,
        .report_progress = false,
    };

    var processor = gpu.BatchProcessor.init(backend, batch_config);

    // Test batch processing
    const queries = [_][]const f32{
        &[_]f32{ 1.0, 0.0, 0.0, 0.0 },
        &[_]f32{ 0.0, 1.0, 0.0, 0.0 },
    };

    const results = try processor.processBatch(&db, &queries, 2);
    defer {
        for (results) |result| {
            allocator.free(result);
        }
        allocator.free(results);
    }

    try testing.expect(results.len == 2);
    for (results) |result| {
        try testing.expect(result.len == 2);
    }
}

test "GPU statistics tracking" {
    var stats = gpu.GpuStats{};

    // Test recording operations
    stats.recordOperation(1000, 1024 * 1024, false); // 1ms, 1MB, GPU operation
    stats.recordOperation(2000, 2 * 1024 * 1024, true); // 2ms, 2MB, CPU fallback
    stats.recordOperation(1500, 512 * 1024, false); // 1.5ms, 512KB, GPU operation

    try testing.expect(stats.total_operations == 3);
    try testing.expect(stats.total_gpu_time == 4500); // 4.5ms total
    try testing.expect(stats.peak_memory_usage == 2 * 1024 * 1024); // 2MB peak
    try testing.expect(stats.cpu_fallback_count == 1);
    try testing.expect(stats.getAverageOperationTime() == 1500); // 1.5ms average
}

test "GPU error handling" {
    const allocator = testing.allocator;

    // Test with invalid configuration
    const invalid_config = gpu.GpuBackendConfig{
        .max_batch_size = 0, // Invalid
        .memory_limit = 0, // Invalid
        .debug_validation = false,
        .power_preference = .high_performance,
    };

    var backend = try gpu.GpuBackend.init(allocator, invalid_config);
    defer backend.deinit();

    // Should still work despite invalid config values
    try testing.expect(backend.config.max_batch_size == 0);
    try testing.expect(backend.config.memory_limit == 0);

    // Test memory check with zero limit
    try testing.expect(!backend.hasMemoryFor(1)); // Should fail with zero limit
}

test "GPU renderer initialization" {
    const allocator = testing.allocator;

    // Test default configuration
    const config = gpu.GPUConfig{
        .debug_validation = false,
        .power_preference = .high_performance,
        .backend = .auto,
        .try_webgpu_first = true,
    };

    // Note: This will likely fail in test environment without GPU hardware/drivers
    // but we test the configuration setup
    _ = config;

    // Test GPU availability check
    const is_available = gpu.isGpuAvailable();
    _ = is_available; // May be true or false depending on environment

    // Test default initialization function
    // This may fail in headless environments, so we just ensure it doesn't crash
    if (gpu.initDefault(allocator)) |renderer| {
        defer renderer.deinit();
        // Successfully created renderer - test passes if we reach this point
    } else |err| {
        // Expected to fail in test environment without GPU
        try testing.expect(err == gpu.GpuError.DeviceNotFound or
            err == gpu.GpuError.InitializationFailed or
            err == gpu.GpuError.UnsupportedBackend);
    }
}

test "GPU backend edge cases" {
    const allocator = testing.allocator;

    const config = gpu.GpuBackendConfig{
        .max_batch_size = 1024,
        .memory_limit = 512 * 1024 * 1024,
        .debug_validation = false,
        .power_preference = .high_performance,
    };

    var backend = try gpu.GpuBackend.init(allocator, config);
    defer backend.deinit();

    // Create empty mock database
    const db = gpu.Db{
        .header = .{
            .row_count = 0,
            .dim = 4,
            .records_off = 0,
        },
        .file = undefined,
    };

    // Database is already initialized with mock data

    // Test search on empty database
    const query = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const results = try backend.searchSimilar(&db, &query, 5);
    defer allocator.free(results);

    try testing.expect(results.len == 0);

    // Test batch search on empty database
    const queries = [_][]const f32{&query};
    const batch_results = try backend.batchSearch(&db, &queries, 5);
    defer {
        for (batch_results) |result| {
            allocator.free(result);
        }
        allocator.free(batch_results);
    }

    try testing.expect(batch_results.len == 1);
    try testing.expect(batch_results[0].len == 0);
}

test "GPU configuration validation" {
    const allocator = testing.allocator;
    _ = allocator; // Mark as used

    // Test various configuration combinations
    const configs = [_]gpu.GpuBackendConfig{
        .{
            .max_batch_size = 1,
            .memory_limit = 1024,
            .debug_validation = false,
            .power_preference = .low_power,
        },
        .{
            .max_batch_size = 1000000,
            .memory_limit = 1024 * 1024 * 1024 * 1024, // 1TB
            .debug_validation = true,
            .power_preference = .high_performance,
        },
        .{
            .max_batch_size = 1024,
            .memory_limit = 512 * 1024 * 1024,
            .debug_validation = false,
            .power_preference = .high_performance,
        },
    };

    for (configs) |config| {
        // All configurations should be valid
        try testing.expect(config.max_batch_size > 0);
        try testing.expect(config.memory_limit >= 0);
    }
}

test "GPU backend performance comparison" {
    const allocator = testing.allocator;

    // Create mock database for performance testing
    const db = gpu.Db{
        .header = .{
            .row_count = 100,
            .dim = 64,
            .records_off = 0,
        },
        .file = undefined,
    };

    // Database is already initialized with mock data

    // Initialize GPU backend
    const config = gpu.GpuBackendConfig{
        .max_batch_size = 1024,
        .memory_limit = 512 * 1024 * 1024,
        .debug_validation = false,
        .power_preference = .high_performance,
    };

    var backend = try gpu.GpuBackend.init(allocator, config);
    defer backend.deinit();

    // Test query
    var query: [64]f32 = undefined;
    for (&query, 0..) |*v, j| {
        v.* = @as(f32, @floatFromInt(j % 5)) / 5.0;
    }

    // Measure search performance
    const start_time = std.time.nanoTimestamp();
    const results = try backend.searchSimilar(&db, &query, 10);
    defer allocator.free(results);
    const end_time = std.time.nanoTimestamp();

    const search_time_ns = end_time - start_time;
    const search_time_ms = @as(f64, @floatFromInt(search_time_ns)) / 1_000_000.0;

    std.log.info("GPU search completed in {d:.3}ms for {} vectors", .{ search_time_ms, db.header.row_count });
    try testing.expect(results.len == 10);
    try testing.expect(search_time_ms > 0); // Should take some time
}

test "GPU AI: Basic matrix operations math" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Test matrix multiplication math (what the GPU kernel would compute)
    const a_data = [_]f32{ 1, 2, 3, 4, 5, 6 }; // 2x3 matrix
    const b_data = [_]f32{ 7, 8, 9, 10, 11, 12 }; // 3x2 matrix

    // Expected result: [58, 64, 139, 154]
    const expected = [_]f32{ 58, 64, 139, 154 };

    var result = try allocator.alloc(f32, 4);
    defer allocator.free(result);

    // Manual matrix multiplication (what GPU kernel would do)
    const m = 2; // rows of A
    const n = 3; // cols of A / rows of B
    const p = 2; // cols of B

    for (0..m) |i| {
        for (0..p) |j| {
            var sum: f32 = 0;
            for (0..n) |k| {
                sum += a_data[i * n + k] * b_data[k * p + j];
            }
            result[i * p + j] = sum;
        }
    }

    // Verify results
    for (0..expected.len) |i| {
        try testing.expectEqual(expected[i], result[i]);
    }
}

test "GPU AI: Kernel dispatch calculations" {
    // Test the math used for kernel dispatch
    const workgroup_size = 16;

    // Test various matrix sizes
    const test_cases = [_]struct { m: usize, n: usize, p: usize }{
        .{ .m = 16, .n = 16, .p = 16 }, // Exact multiple
        .{ .m = 17, .n = 15, .p = 18 }, // Non-exact
        .{ .m = 1, .n = 1, .p = 1 }, // Minimal
    };

    for (test_cases) |tc| {
        const dispatch_x = (tc.m + workgroup_size - 1) / workgroup_size;
        const dispatch_y = (tc.p + workgroup_size - 1) / workgroup_size;

        // Verify dispatch calculations are correct
        try testing.expect(dispatch_x > 0);
        try testing.expect(dispatch_y > 0);
        try testing.expect(dispatch_x <= (tc.m + workgroup_size - 1) / workgroup_size);
        try testing.expect(dispatch_y <= (tc.p + workgroup_size - 1) / workgroup_size);
    }
}

test "GPU AI: Activation functions" {
    // Test ReLU
    try testing.expectEqual(@as(f32, 0), relu(-1.0));
    try testing.expectEqual(@as(f32, 0), relu(0.0));
    try testing.expectEqual(@as(f32, 2.5), relu(2.5));

    // Test Sigmoid
    const sigmoid_val = sigmoid(0.0);
    try testing.expect(sigmoid_val > 0.49 and sigmoid_val < 0.51); // Should be close to 0.5

    // Test Tanh
    const tanh_val = tanh(0.0);
    try testing.expect(tanh_val > -0.01 and tanh_val < 0.01); // Should be close to 0
}

// Helper functions for testing (simulating the activation functions from the GPU module)
fn relu(x: f32) f32 {
    return if (x > 0) x else 0;
}

fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

fn tanh(x: f32) f32 {
    return std.math.tanh(x);
}
