//! ABI Framework - Main Library Interface
//!
//! High level entrypoints and curated re-exports for the modernized framework
//! runtime. The framework module exposes the orchestration layer that
//! coordinates feature toggles, plugin discovery, and lifecycle management.

const std = @import("std");

// =============================================================================
// CORE MODULES
// =============================================================================

/// Core utilities and fundamental types
pub const core = @import("core/mod.zig");

// =============================================================================
// FEATURE MODULES
// =============================================================================

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

/// Compatibility namespace for the WDBX tooling. Older call sites referenced
/// `abi.wdbx.*` directly, so we surface the unified helpers alongside the
/// underlying database module.
pub const wdbx = struct {
    pub usingnamespace features.database.unified;
    pub const database = features.database.database;
    pub const helpers = features.database.db_helpers;
    pub const cli = features.database.cli;
    pub const http = features.database.http;
};

// =============================================================================
// FRAMEWORK MODULE
// =============================================================================

/// Framework orchestration layer that coordinates features and plugins.
pub const framework = @import("framework/mod.zig");

// =============================================================================
// SHARED MODULES
// =============================================================================

pub const utils = @import("shared/utils/mod.zig");
pub const platform = @import("shared/platform/mod.zig");
pub const logging = @import("shared/logging/mod.zig");
pub const observability = @import("shared/observability/mod.zig");
pub const plugins = @import("shared/mod.zig");
pub const simd = @import("shared/simd.zig");
pub const VectorOps = simd.VectorOps;

// =============================================================================
// PUBLIC API
// =============================================================================

pub const Feature = framework.Feature;
pub const Framework = framework.Framework;
pub const FrameworkOptions = framework.FrameworkOptions;
pub const RuntimeConfig = framework.RuntimeConfig;

/// Initialise the ABI framework and return the orchestration handle. Call
/// `Framework.deinit` (or `abi.shutdown`) when finished.
pub fn init(allocator: std.mem.Allocator, options: FrameworkOptions) !Framework {
    return try framework.runtime.Framework.init(allocator, options);
}

/// Convenience wrapper around `Framework.deinit` for callers that prefer the
/// legacy function-style shutdown.
pub fn shutdown(instance: *Framework) void {
    instance.deinit();
}

/// Get framework version information.
pub fn version() []const u8 {
    return "0.1.0a";
}

/// Create a framework with default configuration
pub fn createDefaultFramework(allocator: std.mem.Allocator) !Framework {
    return try init(allocator, framework.defaultConfig());
}

/// Create a framework with custom configuration
pub fn createFramework(allocator: std.mem.Allocator, config: RuntimeConfig) !Framework {
    return try framework.createFramework(allocator, config);
}

test {
    std.testing.refAllDecls(@This());
}

test "abi.version returns build package version" {
    try std.testing.expectEqualStrings("0.1.0a", version());
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