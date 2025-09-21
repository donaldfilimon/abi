//! Shared runtime state container holding logging handles, plugin registries,
//! and initialization options consumed by feature callbacks.

const std = @import("std");
const logging = @import("../shared/logging/logging.zig");
const plugin = @import("../shared/mod.zig");
const feature_manager = @import("feature_manager.zig");

/// Runtime configuration toggles supplied by callers when bootstrapping the framework.
pub const RuntimeOptions = struct {
    ensure_core: bool = true,
    ensure_logging: bool = true,
    ensure_plugin_system: bool = true,
    enable_features: []const []const u8 = &.{},
    enable_categories: []const feature_manager.FeatureCategory = &.{},
    enable_all_features: bool = false,
    logging: LoggingOptions = .{},
};

/// Structured logging configuration.
pub const LoggingOptions = struct {
    enabled: bool = true,
    config: logging.LoggerConfig = .{},
};

/// Mutable state shared across feature initialization callbacks.
pub const RuntimeState = struct {
    allocator: std.mem.Allocator,
    options: RuntimeOptions,
    logger: ?*logging.Logger = null,
    plugin_registry: ?plugin.PluginRegistry = null,

    pub fn init(allocator: std.mem.Allocator, options: RuntimeOptions) RuntimeState {
        return .{ .allocator = allocator, .options = options };
    }

    pub fn setLogger(self: *RuntimeState, logger: ?*logging.Logger) void {
        self.logger = logger;
    }

    pub fn setPluginRegistry(self: *RuntimeState, registry: plugin.PluginRegistry) void {
        if (self.plugin_registry) |*existing| {
            existing.deinit();
            self.plugin_registry = null;
        }
        self.plugin_registry = registry;
    }

    pub fn clearLogger(self: *RuntimeState) void {
        self.logger = null;
    }

    pub fn deinit(self: *RuntimeState) void {
        if (self.plugin_registry) |*registry| {
            registry.deinit();
            self.plugin_registry = null;
        }
    }
};
