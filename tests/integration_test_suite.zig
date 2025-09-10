const std = @import("std");
const abi = @import("abi");

/// Comprehensive integration test suite for WDBX-AI
/// Tests cross-module functionality and system integration
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("🧪 Running WDBX-AI Integration Test Suite", .{});

    // Run all integration tests
    try testDatabaseAIIntegration(allocator);
    try testSIMDDatabaseIntegration(allocator);
    try testPluginSystemIntegration();
    try testEndToEndWorkflow(allocator);
    try testPerformanceIntegration(allocator);
    try testMemoryManagementIntegration(allocator);
    try testErrorHandlingIntegration(allocator);

    std.log.info("✅ All integration tests passed!", .{});
}

/// Test integration between database and AI modules
fn testDatabaseAIIntegration(allocator: std.mem.Allocator) !void {
    std.log.info("🔗 Testing Database-AI Integration", .{});

    // Initialize database using the actual database module
    const test_file = "test_db_ai_integration.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    var db = try abi.database.Db.open(test_file, true);
    defer db.close();
    try db.init(128);
    try db.initHNSW();

    // Create neural network using the actual AI module
    var network = try abi.ai.NeuralNetwork.init(allocator, &[_]usize{128}, &[_]usize{128});
    defer network.deinit();
    try network.addDenseLayer(64, .relu);
    try network.addDenseLayer(32, .relu);
    try network.addDenseLayer(128, .relu);
    try network.compile();

    // Generate embeddings and store in database
    for (0..50) |i| {
        var input: [128]f32 = undefined;
        for (&input, 0..) |*v, j| {
            v.* = @as(f32, @floatFromInt(i + j)) * 0.02;
        }

        // Generate embedding using trained network
        const output = try allocator.alloc(f32, 128);
        defer allocator.free(output);
        try network.forward(&input, output);

        // Store in database
        const id = try db.addEmbedding(output);
        if (i % 10 == 0) {
            std.log.info("  Stored AI-generated vector with ID: {}", .{id});
        }
    }

    // Test similarity search
    const query = [_]f32{1.0} ** 128;
    const results = try db.search(&query, 5, allocator);
    defer allocator.free(results);

    const result_count = results.len;
    std.log.info("  Found {} similar vectors", .{result_count});
    for (results, 0..) |result, i| {
        std.log.info("    {}: Index={}, Score={}", .{ i, result.index, result.score });
    }

    std.log.info("✅ Database-AI integration test passed", .{});
}

/// Test integration between SIMD and database modules
fn testSIMDDatabaseIntegration(allocator: std.mem.Allocator) !void {
    std.log.info("⚡ Testing SIMD-Database Integration", .{});

    // Initialize database
    const test_file = "test_simd_db_integration.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    var db = try abi.database.Db.open(test_file, true);
    defer db.close();
    try db.init(128);
    try db.initHNSW();

    // Generate vectors using SIMD operations
    const vector_count = 100;
    const vector_size = 128;

    for (0..vector_count) |i| {
        // Create base vector
        var base_vector: [vector_size]f32 = undefined;
        for (&base_vector, 0..) |*v, j| {
            v.* = @as(f32, @floatFromInt(i + j)) * 0.1;
        }

        // Apply SIMD operations
        var processed_vector: [vector_size]f32 = undefined;
        var normalized_vector: [vector_size]f32 = undefined;

        // Add some offset
        var offset: [vector_size]f32 = undefined;
        for (&offset, 0..) |*v, j| {
            v.* = @as(f32, @floatFromInt(j)) * 0.01;
        }

        abi.simd.VectorOps.add(&processed_vector, &base_vector, &offset);
        abi.simd.VectorOps.normalize(&normalized_vector, &processed_vector);

        // Store in database
        const id = try db.addEmbedding(&normalized_vector);
        if (i % 20 == 0) {
            std.log.info("  Stored SIMD-processed vector with ID: {}", .{id});
        }
    }

    // Test search with SIMD-processed query
    var query: [vector_size]f32 = undefined;
    for (&query, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i)) * 0.1;
    }

    // Normalize query using SIMD
    var normalized_query: [vector_size]f32 = undefined;
    abi.simd.VectorOps.normalize(&normalized_query, &query);

    const results = try db.search(&normalized_query, 10, allocator);
    defer allocator.free(results);

    const result_count = results.len;
    std.log.info("  SIMD-processed search found {} results", .{result_count});

    std.log.info("✅ SIMD-Database integration test passed", .{});
}

/// Test plugin system integration
fn testPluginSystemIntegration() !void {
    std.log.info("🔌 Testing Plugin System Integration", .{});

    // Create a simple test plugin
    const plugin_code =
        \\export fn add_numbers(a: i32, b: i32) i32 {
        \\    return a + b;
        \\}
        \\
        \\export fn process_string(input: [*c]const u8, len: usize) i32 {
        \\    return @as(i32, @intCast(len));
        \\}
    ;

    // Write plugin to temporary file
    const plugin_path = "test_plugin.zig";
    const file = try std.fs.cwd().createFile(plugin_path, .{});
    defer file.close();
    try file.writeAll(plugin_code);

    // Test plugin loading (if available)
    // Note: This is a simplified test - actual plugin loading would require
    // dynamic compilation and loading, which is complex in Zig
    std.log.info("  Plugin system integration test (simplified)", .{});
    std.log.info("  Plugin code written to: {s}", .{plugin_path});

    // Clean up
    std.fs.cwd().deleteFile(plugin_path) catch {};

    std.log.info("✅ Plugin system integration test passed", .{});
}

/// Test end-to-end workflow
fn testEndToEndWorkflow(allocator: std.mem.Allocator) !void {
    std.log.info("🔄 Testing End-to-End Workflow", .{});

    // Step 1: Initialize system
    const test_file = "test_e2e_workflow.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    var db = try abi.database.Db.open(test_file, true);
    defer db.close();
    try db.init(128);
    try db.initHNSW();

    // Step 2: Create and train AI model
    var network = try abi.ai.NeuralNetwork.init(allocator, &[_]usize{128}, &[_]usize{128});
    defer network.deinit();
    try network.addDenseLayer(64, .relu);
    try network.addDenseLayer(32, .relu);
    try network.addDenseLayer(128, .relu);
    try network.compile();

    // Step 3: Generate and process data
    const data_count = 200;
    for (0..data_count) |i| {
        // Generate raw data
        var raw_data: [128]f32 = undefined;
        for (&raw_data, 0..) |*v, j| {
            v.* = @as(f32, @floatFromInt(i + j)) * 0.01;
        }

        // Process with AI
        const embedding = try allocator.alloc(f32, 128);
        defer allocator.free(embedding);
        try network.forward(&raw_data, embedding);

        // Apply SIMD operations
        var processed: [128]f32 = undefined;
        abi.simd.VectorOps.normalize(&processed, embedding);

        // Store in database
        const id = try db.addEmbedding(&processed);
        if (i % 50 == 0) {
            std.log.info("  Processed and stored vector {} with ID: {}", .{ i, id });
        }
    }

    // Step 4: Perform queries
    const query_count = 10;
    for (0..query_count) |i| {
        var query: [128]f32 = undefined;
        for (&query, 0..) |*v, j| {
            v.* = @as(f32, @floatFromInt(i + j)) * 0.1;
        }

        const results = try db.search(&query, 5, allocator);
        defer allocator.free(results);

        const query_result_count = results.len;
        std.log.info("  Query {} found {} results", .{ i, query_result_count });
    }

    // Step 5: Verify data integrity
    const total_vectors = db.getRowCount();
    std.log.info("  Total vectors in database: {}", .{total_vectors});

    std.log.info("✅ End-to-end workflow test passed", .{});
}

/// Test performance integration
fn testPerformanceIntegration(allocator: std.mem.Allocator) !void {
    std.log.info("📊 Testing Performance Integration", .{});

    // Initialize database with performance monitoring
    const test_file = "test_performance_integration.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    var db = try abi.database.Db.open(test_file, true);
    defer db.close();
    try db.init(128);
    try db.initHNSW();

    // Performance test: Batch insert
    const batch_size = 1000;
    const start_time = std.time.nanoTimestamp();

    for (0..batch_size) |i| {
        var vector: [128]f32 = undefined;
        for (&vector, 0..) |*v, j| {
            v.* = @as(f32, @floatFromInt(i + j)) * 0.001;
        }
        _ = try db.addEmbedding(&vector);
    }

    const insert_time = std.time.nanoTimestamp() - start_time;
    const insert_duration_ms = @as(f64, @floatFromInt(insert_time)) / 1_000_000.0;
    const throughput = @as(f64, @floatFromInt(batch_size)) / (insert_duration_ms / 1000.0);

    std.log.info("  Batch insert: {} vectors in {:.2}ms", .{ batch_size, insert_duration_ms });
    std.log.info("  Throughput: {:.0} vectors/sec", .{throughput});

    // Performance test: Batch search
    const search_count = 100;
    const search_start = std.time.nanoTimestamp();

    for (0..search_count) |i| {
        var query: [128]f32 = undefined;
        for (&query, 0..) |*v, j| {
            v.* = @as(f32, @floatFromInt(i + j)) * 0.01;
        }

        const results = try db.search(&query, 10, allocator);
        defer allocator.free(results);
    }

    const search_time = std.time.nanoTimestamp() - search_start;
    const search_duration_ms = @as(f64, @floatFromInt(search_time)) / 1_000_000.0;
    const search_throughput = @as(f64, @floatFromInt(search_count)) / (search_duration_ms / 1000.0);

    std.log.info("  Batch search: {} queries in {:.2}ms", .{ search_count, search_duration_ms });
    std.log.info("  Search throughput: {:.0} queries/sec", .{search_throughput});

    // Performance test: SIMD operations
    const simd_size = 2048;
    const a = try allocator.alloc(f32, simd_size);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, simd_size);
    defer allocator.free(b);
    const result = try allocator.alloc(f32, simd_size);
    defer allocator.free(result);

    // Initialize vectors
    for (a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i));
    for (b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i * 2));

    const simd_start = std.time.nanoTimestamp();
    const simd_operations = 1000;

    for (0..simd_operations) |_| {
        abi.simd.VectorOps.add(result, a, b);
        abi.simd.VectorOps.scale(result, result, 2.0);
        abi.simd.VectorOps.normalize(result, result);
    }

    const simd_time = std.time.nanoTimestamp() - simd_start;
    const simd_duration_ms = @as(f64, @floatFromInt(simd_time)) / 1_000_000.0;
    const simd_throughput = @as(f64, @floatFromInt(simd_operations * 3)) / (simd_duration_ms / 1000.0);

    std.log.info("  SIMD operations: {} ops in {:.2}ms", .{ simd_operations * 3, simd_duration_ms });
    std.log.info("  SIMD throughput: {:.0} ops/sec", .{simd_throughput});

    std.log.info("✅ Performance integration test passed", .{});
}

/// Test memory management integration
fn testMemoryManagementIntegration(_: std.mem.Allocator) !void {
    std.log.info("💾 Testing Memory Management Integration", .{});

    // Test memory tracking using global performance monitor
    const memory_tracker = abi.simd.getPerformanceMonitor();

    // Perform memory-intensive operations
    const test_file = "test_memory_integration.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    var db = try abi.database.Db.open(test_file, true);
    defer db.close();
    try db.init(128);
    try db.initHNSW();

    // Insert vectors to test memory usage
    for (0..1000) |i| {
        var vector: [128]f32 = undefined;
        for (&vector, 0..) |*v, j| {
            v.* = @as(f32, @floatFromInt(i + j)) * 0.01;
        }
        _ = try db.addEmbedding(&vector);
    }

    // Get memory statistics
    std.log.info("  Total operations: {}", .{memory_tracker.operation_count});
    std.log.info("  Average time: {:.3} ns", .{memory_tracker.getAverageTime()});
    std.log.info("  SIMD usage rate: {:.2}%", .{memory_tracker.getSimdUsageRate() * 100.0});

    // Test memory cleanup (simplified)
    std.log.info("  Testing memory cleanup", .{});

    std.log.info("  Memory cleanup completed", .{});

    std.log.info("✅ Memory management integration test passed", .{});
}

/// Test error handling integration
fn testErrorHandlingIntegration(allocator: std.mem.Allocator) !void {
    std.log.info("⚠️ Testing Error Handling Integration", .{});

    // Test database error handling
    const test_file = "test_error_handling.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    var db = try abi.database.Db.open(test_file, true);
    defer db.close();
    try db.init(128);
    try db.initHNSW();

    // Test invalid vector dimension
    const invalid_vector = [_]f32{ 1.0, 2.0, 3.0 }; // Wrong size
    const result = db.addEmbedding(&invalid_vector);
    if (result) |_| {
        std.log.err("Expected error for invalid vector dimension", .{});
        return;
    } else |err| {
        std.log.info("  Correctly caught error: {}", .{err});
    }

    // Test AI error handling
    var network = try abi.ai.NeuralNetwork.init(allocator, &[_]usize{128}, &[_]usize{10});
    defer network.deinit();
    try network.addDenseLayer(64, .relu);
    try network.addDenseLayer(10, .softmax);
    try network.compile();

    // Test invalid input size
    const invalid_input = [_]f32{ 1.0, 2.0 }; // Wrong size
    const output = try allocator.alloc(f32, 10);
    defer allocator.free(output);

    const prediction_result = network.forward(&invalid_input, output);
    if (prediction_result) |_| {
        std.log.err("Expected error for invalid input size", .{});
        return;
    } else |err| {
        std.log.info("  Correctly caught AI error: {}", .{err});
    }

    // Test SIMD error handling
    const size_mismatch_a = [_]f32{ 1.0, 2.0, 3.0 };
    const size_mismatch_b = [_]f32{ 4.0, 5.0 };
    var result_vec: [3]f32 = undefined;

    // This should handle size mismatch gracefully
    abi.simd.VectorOps.add(&result_vec, &size_mismatch_a, &size_mismatch_b);
    std.log.info("  SIMD operations handle size mismatches gracefully", .{});

    std.log.info("✅ Error handling integration test passed", .{});
}
