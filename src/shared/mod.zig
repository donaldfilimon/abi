//! Plugin System Module
//!
//! This module provides a comprehensive plugin architecture for the Abi AI Framework,
//! enabling dynamic loading of extensions, custom algorithms, and third-party integrations.

const std = @import("std");
const builtin = @import("builtin");
const core = @import("core");

pub const interface = @import("interface.zig");
pub const loader = @import("loader.zig");
pub const registry = @import("registry.zig");
pub const types = @import("types.zig");

// Re-export main types
pub const Plugin = interface.Plugin;
pub const PluginInterface = interface.PluginInterface;
pub const PluginLoader = loader.PluginLoader;
pub const PluginRegistry = registry.PluginRegistry;
pub const PluginError = types.PluginError;
pub const PluginType = types.PluginType;
pub const PluginInfo = types.PluginInfo;
pub const PluginConfig = types.PluginConfig;

// Re-export main functions
pub const createLoader = loader.createLoader;
pub const createRegistry = registry.createRegistry;
pub const registerBuiltinInterface = registry.registerBuiltinInterface;

/// Initialize the plugin system
pub fn init(allocator: std.mem.Allocator) !PluginRegistry {
    core.log.info("Initializing plugin system", .{});
    return try PluginRegistry.init(allocator);
}

/// Plugin system version
pub const VERSION = struct {
    pub const MAJOR = 0;
    pub const MINOR = 1;
    pub const PATCH = 0;
    pub const PRE_RELEASE = "0.1.0a";

    pub fn string() []const u8 {
        return "0.1.0a";
    }

    pub fn isCompatible(major: u32, minor: u32) bool {
        return major == MAJOR and minor <= MINOR;
    }
};

test "plugin system initialization" {
    var plugin_registry = try init(std.testing.allocator);
    defer plugin_registry.deinit();

    try std.testing.expectEqual(@as(usize, 0), plugin_registry.getPluginCount());
}
