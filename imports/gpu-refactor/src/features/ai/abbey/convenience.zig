//! Abbey Convenience Functions
//!
//! Helper functions for creating Abbey engines, configurations,
//! and advanced cognition systems.

const std = @import("std");
const reexports = @import("reexports.zig");

/// Create a new Abbey engine with default configuration
pub fn createEngine(allocator: std.mem.Allocator) !reexports.AbbeyEngine {
    return reexports.AbbeyEngine.init(allocator, .{});
}

/// Create an Abbey engine with custom configuration
pub fn createEngineWithConfig(
    allocator: std.mem.Allocator,
    config: reexports.AbbeyConfig,
) !reexports.AbbeyEngine {
    return reexports.AbbeyEngine.init(allocator, config);
}

/// Create configuration using builder pattern
pub fn builder() reexports.ConfigBuilder {
    return reexports.ConfigBuilder.init();
}

/// Create an advanced cognition system with default configuration
pub fn createAdvancedCognition(
    allocator: std.mem.Allocator,
) !reexports.AdvancedCognition {
    return reexports.AdvancedCognition.init(allocator, .{});
}

/// Create an advanced cognition system with custom configuration
pub fn createAdvancedCognitionWithConfig(
    allocator: std.mem.Allocator,
    config: reexports.AdvancedCognition.Config,
) !reexports.AdvancedCognition {
    return reexports.AdvancedCognition.init(allocator, config);
}
