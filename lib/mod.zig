//! ABI Framework - Main Library Interface
//!
//! High level entrypoints and curated re-exports for the modernized framework
//! runtime. The framework module exposes the orchestration layer that
//! coordinates feature toggles, plugin discovery, and lifecycle management.

const std = @import("std");
const build_options = @import("build_options");

const compat = @import("compat.zig");
/// Core utilities and fundamental types
pub const core = @import("core/mod.zig");
/// Feature modules grouped for discoverability
pub const features = @import("features/mod.zig");
/// Individual feature namespaces re-exported at the root for ergonomic
/// imports (`abi.ai`, `abi.database`, etc.).
pub const ai = features.ai;
pub const gpu = features.gpu;
pub const database = features.database;
pub const web = features.web;
pub const monitoring = features.monitoring;
pub const connectors = features.connectors;
/// Framework orchestration layer that coordinates features and plugins.
pub const framework = @import("framework/mod.zig");
pub const Feature = framework.Feature;
pub const Framework = framework.Framework;
pub const FrameworkOptions = framework.FrameworkOptions;
pub const RuntimeConfig = framework.RuntimeConfig;
pub const logging = @import("shared/logging/mod.zig");
pub const plugins = @import("shared/mod.zig");
pub const observability = @import("shared/observability/mod.zig");
pub const platform = @import("shared/platform/mod.zig");
pub const simd = @import("shared/simd.zig");
pub const VectorOps = simd.VectorOps;
pub const utils = @import("shared/utils/mod.zig");

comptime {
    _ = compat;
}

// =============================================================================
// CORE MODULES
// =============================================================================

// =============================================================================
// FEATURE MODULES
// =============================================================================

/// Compatibility namespace for the WDBX tooling. Older call sites referenced
/// `abi.wdbx.*` directly, so we surface the unified helpers alongside the
/// underlying database module.
pub const wdbx = struct {
    // Explicit exports instead of usingnamespace
    pub const database = features.database.database;
    pub const helpers = features.database.db_helpers;
    pub const cli = features.database.cli;
    pub const http = features.database.http;

    // Re-export unified functions explicitly
    pub const createDatabase = features.database.unified.createDatabase;
    pub const connectDatabase = features.database.unified.connectDatabase;
    pub const closeDatabase = features.database.unified.closeDatabase;
    pub const insertVector = features.database.unified.insertVector;
    pub const searchVectors = features.database.unified.searchVectors;
    pub const deleteVector = features.database.unified.deleteVector;
    pub const updateVector = features.database.unified.updateVector;
    pub const getVector = features.database.unified.getVector;
    pub const listVectors = features.database.unified.listVectors;
    pub const getStats = features.database.unified.getStats;
    pub const optimize = features.database.unified.optimize;
    pub const backup = features.database.unified.backup;
    pub const restore = features.database.unified.restore;
};

// =============================================================================
// FRAMEWORK MODULE
// =============================================================================

// =============================================================================
// SHARED MODULES
// =============================================================================

// =============================================================================
// PUBLIC API
// =============================================================================

/// Convert high level framework options into a runtime configuration.
pub fn runtimeConfigFromOptions(
    allocator: std.mem.Allocator,
    options: FrameworkOptions,
) !RuntimeConfig {
    var config = try framework.runtimeConfigFromOptions(allocator, options);
    config.plugin_paths = options.plugin_paths;
    config.auto_discover_plugins = options.auto_discover_plugins;
    config.auto_register_plugins = options.auto_register_plugins;
    config.auto_start_plugins = options.auto_start_plugins;
    return config;
}

/// Initialise the ABI framework and return the orchestration handle. Call
/// `Framework.deinit` (or `abi.shutdown`) when finished. Accepts either a
/// `RuntimeConfig` or `FrameworkOptions` which will be converted automatically.
pub fn init(allocator: std.mem.Allocator, config_or_options: anytype) !Framework {
    const runtime_config = try resolveRuntimeConfig(allocator, config_or_options);
    return try framework.runtime.Framework.init(allocator, runtime_config);
}

/// Convenience wrapper around `Framework.deinit` for callers that prefer the
/// legacy function-style shutdown.
pub fn shutdown(instance: *Framework) void {
    instance.deinit();
}

/// Get framework version information.
pub fn version() []const u8 {
    return build_options.package_version;
}

/// Create a framework with default configuration
pub fn createDefaultFramework(allocator: std.mem.Allocator) !Framework {
    return try init(allocator, FrameworkOptions{});
}

/// Create a framework with custom configuration
pub fn createFramework(
    allocator: std.mem.Allocator,
    config_or_options: anytype,
) !Framework {
    const runtime_config = try resolveRuntimeConfig(allocator, config_or_options);
    return try framework.createFramework(allocator, runtime_config);
}

fn resolveRuntimeConfig(
    allocator: std.mem.Allocator,
    config_or_options: anytype,
) !RuntimeConfig {
    return switch (@TypeOf(config_or_options)) {
        RuntimeConfig => config_or_options,
        FrameworkOptions => try runtimeConfigFromOptions(allocator, config_or_options),
        else => @compileError("Unsupported configuration type for abi.init"),
    };
}

test {
    std.testing.refAllDecls(@This());
}

test "abi.version returns build package version" {
    try std.testing.expectEqualStrings("0.1.0", version());
}

test "framework initialization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var framework_instance = try createDefaultFramework(gpa.allocator());
    defer framework_instance.deinit();

    try std.testing.expect(!framework_instance.isRunning());
    try std.testing.expect(framework_instance.isFeatureEnabled(.ai));
    try std.testing.expect(framework_instance.isFeatureEnabled(.database));
}

test "framework options convert to runtime config" {
    const options = FrameworkOptions{
        .enable_ai = false,
        .enable_gpu = true,
        .disabled_features = &.{.gpu},
        .plugin_paths = &.{"/opt/abi/plugins"},
        .auto_discover_plugins = true,
    };

    var config = try runtimeConfigFromOptions(std.testing.allocator, options);
    defer std.testing.allocator.free(config.enabled_features);
    defer std.testing.allocator.free(config.disabled_features);

    try std.testing.expect(std.mem.indexOfScalar(features.FeatureTag, config.enabled_features, .ai) == null);
    try std.testing.expect(std.mem.indexOfScalar(features.FeatureTag, config.enabled_features, .gpu) != null);
    try std.testing.expect(std.mem.indexOfScalar(features.FeatureTag, config.disabled_features, .gpu) != null);
    try std.testing.expectEqualStrings("/opt/abi/plugins", config.plugin_paths[0]);
    try std.testing.expect(config.auto_discover_plugins);
}
