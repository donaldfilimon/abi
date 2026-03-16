//! Registry Types
//!
//! Core type definitions for the feature registry system.

const std = @import("std");
const build_options = @import("build_options");
const config_module = @import("../config/mod.zig");
const feature_catalog = @import("../feature_catalog.zig");

pub const Feature = config_module.Feature;

/// Registration mode determines how features are discovered and managed.
pub const RegistrationMode = enum {
    /// Features resolved at compile time only. Zero overhead.
    /// Enabled features are statically known and cannot be toggled.
    comptime_only,

    /// Features compiled in but can be enabled/disabled at runtime.
    /// Small overhead for state checking and conditional initialization.
    runtime_toggle,

    /// Features can be dynamically loaded from shared libraries at runtime.
    /// Requires platform support (dlopen/LoadLibrary). Most flexible but highest overhead.
    dynamic,
};

/// Plugin API structure expected from dynamic shared libraries.
pub const PluginApi = struct {
    /// Semantic version of the plugin API.
    pub const Version = struct {
        major: u32,
        minor: u32,
        patch: u32,

        pub fn format(
            self: Version,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = fmt;
            _ = options;
            try writer.print("{d}.{d}.{d}", .{ self.major, self.minor, self.patch });
        }
    };

    /// Metadata provided by the plugin.
    pub const Metadata = struct {
        name: []const u8,
        description: []const u8,
        version: Version,
        author: []const u8,
    };

    /// Function pointers for plugin lifecycle.
    pub const Interface = struct {
        /// Initialize the plugin.
        init: *const fn (std.mem.Allocator) anyerror!*anyopaque,
        /// Deinitialize the plugin.
        deinit: *const fn (*anyopaque) void,
        /// Get plugin metadata.
        getMetadata: *const fn () Metadata,
    };

    /// Current supported plugin API version.
    pub const current_version = Version{ .major = 1, .minor = 0, .patch = 0 };
};

/// Feature registration entry with lifecycle management.
pub const FeatureRegistration = struct {
    feature: Feature,
    mode: RegistrationMode,

    // For comptime_only and runtime_toggle
    context_ptr: ?*anyopaque = null,
    config_ptr: ?*const anyopaque = null,
    init_fn: ?*const fn (std.mem.Allocator, *const anyopaque) anyerror!*anyopaque = null,
    deinit_fn: ?*const fn (*anyopaque) void = null,

    // For dynamic mode
    dyn_lib: ?std.DynLib = null,
    library_path: ?[]const u8 = null,
    plugin_interface: ?PluginApi.Interface = null,

    // Runtime state
    enabled: bool = false,
    initialized: bool = false,
};

/// Registry error set
pub const Error = error{
    FeatureNotRegistered,
    FeatureAlreadyRegistered,
    FeatureNotCompiled,
    FeatureDisabled,
    InitializationFailed,
    AlreadyInitialized,
    NotInitialized,
    DynamicLoadingNotSupported,
    LibraryLoadFailed,
    SymbolNotFound,
    InvalidMode,
    InvalidPluginApi,
    PluginIncompatible,
} || std.mem.Allocator.Error;

// ============================================================================
// Compile-time Feature Checking
// ============================================================================

/// Check if a feature is compiled in via build_options.
pub fn isFeatureCompiledIn(comptime feature: Feature) bool {
    const field_name = feature_catalog.compileFlagFieldFromEnum(feature);
    return @field(build_options, field_name);
}

/// Get parent feature for sub-features.
pub fn getParentFeature(feature: Feature) ?Feature {
    return feature_catalog.parentAsEnum(Feature, feature);
}

// ============================================================================
// Tests
// ============================================================================

test "isFeatureCompiledIn matches build_options" {
    try std.testing.expectEqual(build_options.feat_gpu, comptime isFeatureCompiledIn(.gpu));
    try std.testing.expectEqual(build_options.feat_ai, comptime isFeatureCompiledIn(.ai));
    try std.testing.expectEqual(build_options.feat_database, comptime isFeatureCompiledIn(.database));
}

test "getParentFeature returns correct parent" {
    try std.testing.expectEqual(Feature.ai, getParentFeature(.llm).?);
    try std.testing.expectEqual(Feature.ai, getParentFeature(.embeddings).?);
    try std.testing.expectEqual(Feature.ai, getParentFeature(.agents).?);
    try std.testing.expectEqual(Feature.ai, getParentFeature(.training).?);
    try std.testing.expectEqual(Feature.ai, getParentFeature(.constitution).?);
    try std.testing.expect(getParentFeature(.gpu) == null);
    try std.testing.expect(getParentFeature(.ai) == null);
}

test {
    std.testing.refAllDecls(@This());
}
