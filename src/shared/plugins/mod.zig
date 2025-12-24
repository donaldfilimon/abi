//! Plugin Architecture Module
//!
//! Provides a flexible plugin system for extending ABI functionality.
//! Supports dynamic loading, registration, and lifecycle management.

const std = @import("std");

pub const Interface = @import("interface.zig");
pub const Loader = @import("loader.zig");
pub const Registry = @import("registry.zig");
pub const Types = @import("types.zig");

// Re-export commonly used types for convenience
pub const PluginInterface = Interface.PluginInterface;
pub const PluginRegistry = Registry.PluginRegistry;
pub const PluginLoader = Loader.PluginLoader;
pub const PluginInfo = Types.PluginInfo;
pub const PluginError = Types.PluginError;
