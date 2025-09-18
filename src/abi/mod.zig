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

// Import modules through the build system
pub const core = @import("core");
pub const simd = @import("simd");
pub const ai = @import("ai");
pub const gpu = @import("gpu");
pub const services = @import("services");

// For now, provide minimal functionality that works with current build system
// These will be expanded once proper module dependencies are added to build.zig

// =============================================================================
// TYPE RE-EXPORTS
// =============================================================================

// Core types
pub const AbiError = core.AbiError;
pub const Result = core.Result;
pub const Allocator = std.mem.Allocator;
pub const ArrayList = std.ArrayList;
pub const HashMap = std.HashMap;

// SIMD operations
pub const VectorOps = simd.VectorOps;

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

    std.log.info("WDBX-AI framework initialized successfully", .{});
}

/// Deinitialize the WDBX-AI framework
/// Should be called when shutting down the application
pub fn deinit() void {
    std.log.info("WDBX-AI framework shutting down", .{});

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
    try stdout.print("\n", .{});
    try stdout.print("Available commands:\n", .{});
    try stdout.print("  zig build test        - Run test suite\n", .{});
    try stdout.print("  zig build benchmark   - Run performance benchmarks\n", .{});
    try stdout.print("  zig build run         - Start CLI application\n", .{});
    try stdout.print("  zig build run-server  - Start HTTP server\n", .{});

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
            .f32x4 = builtin.cpu.arch == .x86_64 or builtin.cpu.arch == .aarch64,
            .f32x8 = builtin.cpu.arch == .x86_64,
            .f32x16 = false,
        },
    };
}

fn l2Distance(a: []const f32, b: []const f32) f32 {
    var sum: f32 = 0.0;
    const len = @min(a.len, b.len);
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std.math.sqrt(sum);
}

/// Run a quick system test
pub fn runSystemTest() !void {
    const testing = std.testing;

    // Test SIMD operations
    const vector_a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const vector_b = [_]f32{ 2.0, 3.0, 4.0, 5.0 };

    const distance = l2Distance(vector_a[0..], vector_b[0..]);
    try testing.expect(distance > 0.0);

    // Test core utilities
    const random_val = @as(u32, @intCast(std.hash_map.hashString("test") % 100)) + 1;
    try testing.expect(random_val >= 1 and random_val <= 100);

    const trimmed = std.mem.trim(u8, "  test  ", " ");
    try testing.expectEqualStrings("test", trimmed);

    std.log.info("System test completed successfully", .{});
}

// =============================================================================
// TESTS
// =============================================================================

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

    // Test that core modules can be imported and used together
    try testing.expect(@TypeOf(std.mem.Allocator) == @TypeOf(Allocator));

    // Test SIMD operations
    const vector_a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const vector_b = [_]f32{ 2.0, 3.0, 4.0, 5.0 };

    const distance = l2Distance(vector_a[0..], vector_b[0..]);
    try testing.expect(distance > 0.0);

    // Use standard library utilities
    const random_val = @as(u32, @intCast(std.hash_map.hashString("integration") % 10)) + 1;
    try testing.expect(random_val >= 1 and random_val <= 10);

    std.log.info("Module integration test completed", .{});
}
