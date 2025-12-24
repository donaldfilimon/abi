//! Shared Utilities and Plugin System Module
//!
//! This module provides shared utilities, observability tools, and a plugin architecture
//! for extending ABI functionality across different domains.

const std = @import("std");
const builtin = @import("builtin");
const core = @import("../core/mod.zig");

// Plugin system
pub const plugins = @import("plugins/mod.zig");

// Observability and monitoring
pub const observability = @import("observability/mod.zig");
pub const logging = @import("logging/mod.zig");
pub const platform = @import("platform/mod.zig");

// Utilities
pub const utils = @import("utils/mod.zig");
pub const simd = @import("simd.zig");
pub const tui = @import("tui.zig");

// Re-export commonly used types for convenience
pub const PluginInterface = plugins.PluginInterface;
pub const PluginRegistry = plugins.PluginRegistry;
pub const PluginLoader = plugins.PluginLoader;
pub const PluginInfo = plugins.PluginInfo;
pub const PluginError = plugins.PluginError;

/// Initialize the plugin system
pub fn initPluginSystem(allocator: std.mem.Allocator) !plugins.PluginRegistry {
    core.log.info("Initializing plugin system", .{});
    return try plugins.PluginRegistry.init(allocator);
}

/// Plugin system version
pub const VERSION = struct {
    pub const MAJOR = 1;
    pub const MINOR = 0;
    pub const PATCH = 0;

    pub fn string() []const u8 {
        return "1.0.0";
    }

    pub fn isCompatible(major: u32, minor: u32) bool {
        return major == MAJOR and minor <= MINOR;
    }
};

test "plugin system initialization" {
    var plugin_registry = try initPluginSystem(std.testing.allocator);
    defer plugin_registry.deinit();

    try std.testing.expectEqual(@as(usize, 0), plugin_registry.getPluginCount());
}
