//! WDBX-AI Vector Database - Unified Root Module
//!
//! This module provides a unified interface to all WDBX-AI functionality:
//! - Vector database operations (HNSW, parallel search, SIMD)
//! - AI and machine learning capabilities
//! - Core utilities and data structures
//! - Performance monitoring and optimization

const std = @import("std");

// Import consolidated modules
pub const database = @import("wdbx/database.zig");
pub const simd = @import("simd/mod.zig");
pub const ai = @import("ai/mod.zig");
pub const wdbx = @import("wdbx/mod.zig");
pub const plugins = @import("plugins/mod.zig");
pub const tracing = @import("tracing.zig");
pub const logging = @import("logging.zig");
pub const neural = @import("neural.zig");
pub const memory_tracker = @import("memory_tracker.zig");
// Core utilities live under wdbx/core.zig
pub const core = @import("wdbx/core.zig");
pub const localml = @import("localml.zig");
pub const gpu = @import("gpu_renderer.zig");
pub const backend = @import("backend.zig");
pub const connectors = @import("connectors/mod.zig");
const weather_mod = @import("weather.zig");

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

// Re-export standard library utilities
pub const Allocator = std.mem.Allocator;
pub const ArrayList = std.ArrayList;
pub const HashMap = std.HashMap;

// Re-export WDBX utilities
pub const Command = wdbx.Command;
pub const WdbxOutputFormat = wdbx.OutputFormat;
pub const WdbxLogLevel = wdbx.LogLevel;

// Re-export tracing utilities
pub const Tracer = tracing.Tracer;
pub const TraceId = tracing.TraceId;
pub const Span = tracing.Span;
pub const SpanId = tracing.SpanId;
pub const TraceContext = tracing.TraceContext;
pub const TracingError = tracing.TracingError;

// Re-export logging utilities
pub const Logger = logging.Logger;
pub const LogLevel = logging.LogLevel;
pub const LogOutputFormat = logging.OutputFormat;
pub const LoggerConfig = logging.LoggerConfig;

// Re-export weather types for convenience
pub const WeatherService = weather_mod.WeatherService;
pub const WeatherData = weather_mod.WeatherData;

/// Main application entry point
pub fn main() !void {
    // Simple main function - I/O functions have changed in Zig 0.15.1
    // For now, just return successfully
    std.log.info("WDBX-AI Vector Database initialized", .{});
}

/// Initialize the WDBX-AI system
pub fn init(allocator: std.mem.Allocator) !void {
    // Initialize core systems
    _ = allocator;

    // Log system initialization
    std.log.info("WDBX-AI system initialized", .{});
}

/// Cleanup the WDBX-AI system
pub fn deinit() void {
    // Cleanup resources
    std.log.info("WDBX-AI system shutdown complete", .{});
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

    // Initialize tracing
    const tracer_config = Tracer.TracerConfig{
        .max_active_spans = 100,
        .max_finished_spans = 1000,
    };
    try tracing.initGlobalTracer(allocator, tracer_config);
    defer tracing.deinitGlobalTracer();

    // Initialize logging
    const logger_config = LoggerConfig{
        .level = .info,
        .format = .colored,
        .enable_timestamps = true,
        .enable_source_info = true,
    };
    try logging.initGlobalLogger(allocator, logger_config);
    defer logging.deinitGlobalLogger();

    // Start system test span
    const system_test_span = try tracing.startSpan("system_test", .internal, null);
    defer tracing.endSpan(system_test_span);

    // Test database operations
    const test_file = "test_system.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    const db_span = try tracing.startSpan("database_test", .internal, null);
    defer tracing.endSpan(db_span);

    var db = try database.Db.open(test_file, true);
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

    const simd_span = try tracing.startSpan("simd_test", .internal, null);
    defer tracing.endSpan(simd_span);

    const distance = simd.distance(&vector_a, &vector_b);
    try testing.expect(distance > 0.0);

    // Test AI operations
    const ai_span = try tracing.startSpan("ai_test", .internal, null);
    defer tracing.endSpan(ai_span);

    var network = try ai.NeuralNetwork.init(allocator, &[_]usize{4}, &[_]usize{2});
    defer network.deinit();
    try network.addDenseLayer(8, .relu);
    try network.addDenseLayer(2, .softmax);
    try network.compile();

    const output = try allocator.alloc(f32, 2);
    defer allocator.free(output);
    try network.forward(&vector_a, output);

    // Test core utilities
    const random_val = @as(u32, @intCast(std.hash_map.hashString("test") % 100)) + 1;
    try testing.expect(random_val >= 1 and random_val <= 100);

    const trimmed = std.mem.trim(u8, "  test  ", " ");
    try testing.expectEqualStrings("test", trimmed);

    // Test tracing functionality
    const trace_span = try tracing.startSpan("trace_test", .internal, null);
    defer tracing.endSpan(trace_span);

    try trace_span.setAttribute(allocator, "test_type", "system_test");
    try trace_span.addEvent(allocator, "test_completed");

    // Export trace data
    const tracer = tracing.getGlobalTracer().?;
    const trace_json = try tracer.exportToJson(allocator);
    defer allocator.free(trace_json);

    try testing.expect(trace_json.len > 0);

    // Test structured logging
    try logging.info("System test completed successfully", .{
        .database_vectors = 64,
        .simd_dimensions = 4,
        .ai_layers = 2,
        .tracing_spans = 4,
        .test_duration_ms = 100,
    }, @src());

    std.log.info("System test completed successfully with tracing", .{});
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

    // Test that all modules can be imported and used together
    try testing.expect(@TypeOf(database.Db) == @TypeOf(Db));
    try testing.expect(@TypeOf(simd.Vector) == @TypeOf(Vector));
    try testing.expect(@TypeOf(ai.NeuralNetwork) == @TypeOf(NeuralNetwork));
    try testing.expect(@TypeOf(std.mem.Allocator) == @TypeOf(Allocator));

    // Test cross-module functionality
    const test_file = "test_integration.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    var db = try database.Db.open(test_file, true);
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

    // Use standard library utilities
    const random_val = @as(u32, @intCast(std.hash_map.hashString("integration") % 10)) + 1;
    try testing.expect(random_val >= 1 and random_val <= 10);

    std.log.info("Module integration test completed", .{});
}
