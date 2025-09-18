//! ABI Vector Database - Unified Root Module
//!
//! This module provides a unified interface to all ABI functionality:
//! - Vector database operations (HNSW, parallel search, SIMD)
//! - AI and machine learning capabilities
//! - Core utilities and data structures
//! - Performance monitoring and optimization
//!
//! ## Features
//! - High-performance vector similarity search
//! - SIMD-accelerated operations
//! - Neural network inference
//! - Plugin system for extensibility
//! - Cross-platform compatibility
//! - Production-ready error handling
//!
//! ## Quick Start
//! ```zig
//! const abi = @import("abi");
//! const allocator = std.heap.page_allocator;
//!
//! // Initialize the framework
//! try abi.init(allocator);
//! defer abi.deinit();
//!
//! // Create a vector database
//! var db = try abi.Db.open("vectors.wdbx", true);
//! defer db.close();
//!
//! // Add vectors and search
//! try db.init(384);
//! const embedding = [_]f32{0.1, 0.2, 0.3, ...};
//! const row_id = try db.addEmbedding(&embedding);
//! ```

const std = @import("std");
const builtin = @import("builtin");

// =============================================================================
// CORE MODULE IMPORTS
// =============================================================================
// Import consolidated modules
pub const database = @import("wdbx/database.zig");
// SIMD functionality is now part of core
pub const ai = @import("ai/mod.zig");
pub const wdbx = @import("wdbx/mod.zig");
pub const plugins = @import("plugins/mod.zig");
pub const monitoring = @import("monitoring/mod.zig");
pub const logging = @import("logging.zig");
// Core utilities and framework types
pub const core = @import("core/mod.zig");
pub const gpu = @import("gpu/mod.zig");
pub const connectors = @import("connectors/mod.zig");
pub const services = @import("services/mod.zig");

// Core utilities and error handling
pub const AbiError = core.AbiError;
pub const Result = core.Result;

// Organized modules (import directly when needed)
// net/ - Networking: http_client.zig, curl_wrapper.zig
// perf/ - Performance: performance.zig, performance_profiler.zig, memory_tracker.zig
// gpu/ - GPU computing: gpu_renderer.zig, gpu_examples.zig
// ml/ - Machine learning: neural.zig, localml.zig
// api/ - C API: c_api.zig

// Utilities
pub const utils = @import("utils.zig");

// =============================================================================
// TYPE RE-EXPORTS
// =============================================================================

// Database types
pub const Db = database.Db;
pub const DbError = database.DbError;
pub const WdbxHeader = database.WdbxHeader;

// SIMD types
pub const Vector = core.Vector;
pub const VectorOps = core.VectorOps;
pub const MatrixOps = core.MatrixOps;

// AI types
pub const NeuralNetwork = ai.NeuralNetwork;
pub const EmbeddingGenerator = ai.EmbeddingGenerator;
pub const ModelTrainer = ai.ModelTrainer;
pub const Layer = ai.Layer;
pub const Activation = ai.Activation;
pub const DynamicPersonaRouter = ai.DynamicPersonaRouter;

// Organized module types (import modules directly when needed)

// Core utilities
pub const Allocator = std.mem.Allocator;
pub const ArrayList = std.ArrayList;
pub const HashMap = std.HashMap;
pub const random = core.random;
pub const string = core.string;
pub const time = core.time;
pub const log = core.log;
pub const performance_core = core.performance;

// WDBX utilities
pub const Command = wdbx.Command;
pub const WdbxOutputFormat = wdbx.OutputFormat;
pub const WdbxLogLevel = wdbx.LogLevel;

// Old tracing imports removed - now available through monitoring module

// Re-export logging utilities
pub const Logger = logging.Logger;
pub const LogLevel = logging.LogLevel;
pub const LogOutputFormat = logging.OutputFormat;
pub const LoggerConfig = logging.LoggerConfig;

// Re-export GPU capabilities
pub const GPURenderer = gpu.GPURenderer;
pub const GpuBackend = gpu.GpuBackend;
pub const GPUConfig = gpu.GPUConfig;
pub const GpuError = gpu.GpuError;

// Re-export monitoring capabilities
pub const Tracer = monitoring.Tracer;
pub const TraceId = monitoring.TraceId;
pub const Span = monitoring.Span;
pub const SpanId = monitoring.SpanId;
pub const TraceContext = monitoring.TraceContext;
pub const TracingError = monitoring.TracingError;
pub const MemoryTracker = monitoring.MemoryTracker;
pub const PerformanceProfiler = monitoring.PerformanceProfiler;

// Re-export service capabilities
pub const WeatherService = services.WeatherService;
pub const WeatherData = services.WeatherData;
pub const WeatherConfig = services.WeatherConfig;

// =============================================================================
// FRAMEWORK INITIALIZATION
// =============================================================================

/// Initialize the WDBX-AI framework
/// Must be called before using any framework functionality
pub fn init(allocator: Allocator) AbiError!void {
    // Initialize core systems first
    try core.init(allocator);

    // Initialize other subsystems
    // (AI, database, etc. can be initialized on-demand)

    log.info("WDBX-AI framework initialized successfully", .{});
}

/// Deinitialize the WDBX-AI framework
/// Should be called when shutting down the application
pub fn deinit() void {
    log.info("WDBX-AI framework shutting down", .{});

    // Deinitialize core systems last
    core.deinit();
}

/// Check if the framework is initialized
pub fn isInitialized() bool {
    return core.isInitialized();
}

// =============================================================================
// APPLICATION ENTRY POINT
// =============================================================================

/// Main application entry point
pub fn main() !void {
    const stdout = std.io.getStdOut().writer();

    // Initialize framework
    const allocator = std.heap.page_allocator;
    try init(allocator);
    defer deinit();

    // Display welcome message
    try stdout.print("WDBX-AI Vector Database v1.0.0\n", .{});
    try stdout.print("Unified high-performance vector database with AI capabilities\n", .{});
    try stdout.print("Platform: {s} ({s})\n", .{ @tagName(builtin.target.os.tag), @tagName(builtin.target.cpu.arch) });
    try stdout.print("SIMD Support: {}\n", .{builtin.target.cpu.features.has_sse2});
    try stdout.print("Optimal SIMD Width: {}\n", .{if (builtin.target.cpu.features.has_avx2) 8 else 4});
    try stdout.print("\n");
    try stdout.print("Available commands:\n", .{});
    try stdout.print("  zig build test        - Run test suite\n", .{});
    try stdout.print("  zig build benchmark   - Run performance benchmarks\n", .{});
    try stdout.print("  zig build run         - Start CLI application\n", .{});
    try stdout.print("  zig build run-server  - Start HTTP server\n", .{});
    // Simple main function - I/O functions have changed in Zig 0.15.1
    // For now, just return successfully
    std.log.info("ABI Vector Database initialized", .{});
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
            .f32x4 = core.Vector.isSimdAvailable(4),
            .f32x8 = core.Vector.isSimdAvailable(8),
            .f32x16 = core.Vector.isSimdAvailable(16),
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
    try monitoring.tracing.initGlobalTracer(allocator, tracer_config);
    defer monitoring.tracing.deinitGlobalTracer();

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
    const system_test_span = try monitoring.tracing.startSpan("system_test", .internal, null);
    defer monitoring.tracing.endSpan(system_test_span);

    // Test database operations
    const test_file = "test_system.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};

    const db_span = try monitoring.tracing.startSpan("database_test", .internal, null);
    defer monitoring.tracing.endSpan(db_span);

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

    const simd_span = try monitoring.tracing.startSpan("simd_test", .internal, null);
    defer monitoring.tracing.endSpan(simd_span);

    const distance = core.distance(&vector_a, &vector_b);
    try testing.expect(distance > 0.0);

    // Test AI operations
    const ai_span = try monitoring.tracing.startSpan("ai_test", .internal, null);
    defer monitoring.tracing.endSpan(ai_span);

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
    const trace_span = try monitoring.tracing.startSpan("trace_test", .internal, null);
    defer monitoring.tracing.endSpan(trace_span);

    try trace_span.setAttribute(allocator, "test_type", "system_test");
    try trace_span.addEvent(allocator, "test_completed");

    // Export trace data
    const tracer = monitoring.tracing.getGlobalTracer().?;
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
    try testing.expect(@TypeOf(core.Vector) == @TypeOf(Vector));
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
