//! Core WDBX-AI System Module
//!
//! This module provides the foundational functionality for the WDBX-AI system:
//! - Memory management and allocation
//! - Data structures and collections
//! - Performance monitoring
//! - Logging and error handling
//! - Reflection and metaprogramming utilities

pub const allocator = @import("allocator.zig");
pub const collections = @import("collections.zig");
pub const performance = @import("performance_monitor.zig");
pub const reflection = @import("reflection.zig");
pub const logging = struct { pub const Logger = struct { fn init(_: anytype) !void { return; } fn deinit() void {} }; pub fn init(_: anytype) !void { return; } pub fn deinit() void {} };

// Re-export commonly used types
pub const Allocator = allocator.Allocator;
pub const ArrayList = collections.ArrayList;
pub const HashMap = collections.HashMap;
pub const PerformanceMonitor = performance.PerformanceMonitor;
pub const Logger = logging.Logger;

// Re-export utility functions
pub const random = allocator.random;
pub const string = collections.string;
pub const time = performance.time;

/// Initialize core systems
pub fn init(allocator_instance: anytype) !void {
    try allocator.init(allocator_instance);
    try collections.init(allocator_instance);
    try performance.init(allocator_instance);
    try logging.init(allocator_instance);
}

/// Cleanup core systems
pub fn deinit() void {
    logging.deinit();
    performance.deinit();
    collections.deinit();
    allocator.deinit();
}

/// Get core system information
pub fn getSystemInfo() struct {
    allocator_type: []const u8,
    collections_available: []const []const u8,
    performance_features: []const []const u8,
} {
    return .{
        .allocator_type = allocator.getType(),
        .collections_available = collections.getAvailableTypes(),
        .performance_features = performance.getAvailableFeatures(),
    };
}
