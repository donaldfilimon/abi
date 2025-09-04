//! WDBX-AI Vector Database - Unified Root Module
//!
//! This module provides a unified interface to all WDBX-AI functionality:
//! - Vector database operations (HNSW, parallel search, SIMD)
//! - AI and machine learning capabilities
//! - Core utilities and data structures
//! - Performance monitoring and optimization

const std = @import("std");

// Import consolidated modules
pub const database = @import("database/mod.zig");
pub const simd = @import("simd/mod.zig");
pub const ai = @import("ai/mod.zig");
pub const core = @import("core/mod.zig");
pub const wdbx = @import("wdbx/mod.zig");
pub const plugins = @import("plugins/mod.zig");

// Re-export commonly used types and functions
pub const Db = database.Db;
pub const DbError = database.DbError;
pub const Result = database.Result;
pub const WdbxHeader = database.WdbxHeader;

// Re-export SIMD operations
pub const Vector = simd.Vector;
pub const VectorOps = simd.VectorOps;
pub const MatrixOps = simd.MatrixOps;
pub const PerformanceMonitor = simd.PerformanceMonitor;

// Re-export AI capabilities
pub const NeuralNetwork = ai.NeuralNetwork;
pub const EmbeddingGenerator = ai.EmbeddingGenerator;
pub const ModelTrainer = ai.ModelTrainer;
pub const Layer = ai.Layer;
pub const Activation = ai.Activation;

// Re-export core utilities
pub const Allocator = core.Allocator;
pub const ArrayList = core.ArrayList;
pub const HashMap = core.HashMap;
pub const random = core.random;
pub const string = core.string;
pub const time = core.time;
pub const log = core.log;
pub const perf = core.performance;

// Re-export WDBX utilities
pub const Command = wdbx.Command;
pub const OutputFormat = wdbx.OutputFormat;
pub const LogLevel = wdbx.LogLevel;

/// Main application entry point
pub fn main() !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.print("WDBX-AI Vector Database v1.0.0\n", .{});
    try stdout.print("Unified high-performance vector database with AI capabilities\n", .{});
    try stdout.print("Use 'zig build test' to run tests\n", .{});
    try stdout.print("Use 'zig build benchmark' to run benchmarks\n", .{});
}

/// Initialize the WDBX-AI system
pub fn init(allocator: std.mem.Allocator) !void {
    // Initialize core systems
    _ = allocator;

    // Log system initialization
    log.info("WDBX-AI system initialized", .{});
}

/// Cleanup the WDBX-AI system
pub fn deinit() void {
    // Cleanup resources
    log.info("WDBX-AI system shutdown complete", .{});
}

/// Get system information
pub fn getSystemInfo() struct {
    version: []const u8,
    features: []const []const u8,
    simd_support: struct {
        f32x4: bool,
        f32x8: bool,
        f32x16: bool,
    },
} {
    return .{
        .version = "1.0.0",
        .features = &[_][]const u8{
            "HNSW Indexing",
            "Parallel Search",
            "SIMD Optimization",
            "Neural Networks",
            "Embedding Generation",
            "Performance Monitoring",
        },
        .simd_support = .{
            .f32x4 = simd.Vector.isSimdAvailable(4),
            .f32x8 = simd.Vector.isSimdAvailable(8),
            .f32x16 = simd.Vector.isSimdAvailable(16),
        },
    };
}

/// Run a quick system test
pub fn runSystemTest() !void {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Initialize core system
    try core.init(allocator);
    defer core.deinit();

    // Test database operations
    const test_file = "test_system.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    var db = try database.createStandard(test_file, true);
    defer db.close();
    try db.init(64);

    // Test HNSW indexing
    try db.initHNSW();

    // Test vector operations
    const test_vector = [_]f32{0.1} ** 64;
    _ = try db.addEmbedding(&test_vector);

    // Test search
    const results = try db.search(&test_vector, 5, allocator);
    defer allocator.free(results);
    try testing.expect(results.len > 0);

    // Test SIMD operations
    const vector_a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const vector_b = [_]f32{ 2.0, 3.0, 4.0, 5.0 };
    const distance = simd.VectorOps.distance(&vector_a, &vector_b);
    try testing.expect(distance > 0.0);

    // Test AI operations
    var network = try ai.NeuralNetwork.init(allocator, &[_]usize{4}, &[_]usize{2});
    defer network.deinit();
    try network.addDenseLayer(8, .relu);
    try network.addDenseLayer(2, .softmax);
    try network.compile();

    const output = try allocator.alloc(f32, 2);
    defer allocator.free(output);
    try network.forward(&vector_a, output);

    // Test core utilities
    const random_val = core.random.int(u32, 1, 100);
    try testing.expect(random_val >= 1 and random_val <= 100);

    const trimmed = core.string.trim("  test  ");
    try testing.expectEqualStrings("test", trimmed);

    core.log.info("System test completed successfully", .{});
}

test "Root module functionality" {
    const testing = std.testing;

    // Test system info
    const info = getSystemInfo();
    try testing.expectEqualStrings("1.0.0", info.version);
    try testing.expect(info.features.len > 0);

    // Test system initialization
    try init(testing.allocator);
    defer deinit();

    // Test system test
    try runSystemTest();
}

test "Module integration" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Initialize core system
    try core.init(allocator);
    defer core.deinit();

    // Test that all modules can be imported and used together
    try testing.expect(@TypeOf(database.Db) == @TypeOf(Db));
    try testing.expect(@TypeOf(simd.Vector) == @TypeOf(Vector));
    try testing.expect(@TypeOf(ai.NeuralNetwork) == @TypeOf(NeuralNetwork));
    try testing.expect(@TypeOf(core.Allocator) == @TypeOf(Allocator));

    // Test cross-module functionality
    const test_file = "test_integration.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    var db = try database.createStandard(test_file, true);
    defer db.close();
    try db.init(32);
    try db.initHNSW();

    // Use SIMD for vector operations
    const vector = [_]f32{0.5} ** 32;
    _ = try db.addEmbedding(&vector);

    // Use AI for processing
    var network = try ai.NeuralNetwork.init(allocator, &[_]usize{32}, &[_]usize{16});
    defer network.deinit();
    try network.addDenseLayer(16, .relu);
    try network.compile();

    const output = try allocator.alloc(f32, 16);
    defer allocator.free(output);
    try network.forward(&vector, output);

    // Use core utilities
    const random_val = core.random.int(u32, 1, 10);
    try testing.expect(random_val >= 1 and random_val <= 10);

<<<<<<< Current (Your changes)
    log.info("Module integration test completed", .{});
=======
    core.log.info("Module integration test completed", .{});
>>>>>>> Incoming (Background Agent changes)
}
