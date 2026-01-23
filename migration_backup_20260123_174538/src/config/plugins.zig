//! Plugin Configuration
//!
//! Configuration for plugin loading and discovery.

const std = @import("std");

/// Plugin configuration.
pub const PluginConfig = struct {
    /// Paths to search for plugins.
    paths: []const []const u8 = &.{},

    /// Auto-discover plugins in paths.
    auto_discover: bool = false,

    /// Plugins to load by name.
    load: []const []const u8 = &.{},
};
