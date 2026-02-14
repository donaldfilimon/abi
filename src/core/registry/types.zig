//! Registry Types
//!
//! Core type definitions for the feature registry system.

const std = @import("std");
const build_options = @import("build_options");
const config_module = @import("../config/mod.zig");

pub const Feature = config_module.Feature;

/// Registration mode determines how features are discovered and managed.
pub const RegistrationMode = enum {
    /// Features resolved at compile time only. Zero runtime overhead.
    /// Enabled features are statically known and cannot be toggled.
    comptime_only,

    /// Features compiled in but can be enabled/disabled at runtime.
    /// Small overhead for state checking and conditional initialization.
    runtime_toggle,

    /// Features can be dynamically loaded from shared libraries at runtime.
    /// Requires platform support (dlopen/LoadLibrary). Most flexible but highest overhead.
    dynamic,
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

    // For dynamic mode (future)
    library_handle: ?*anyopaque = null,
    library_path: ?[]const u8 = null,

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
} || std.mem.Allocator.Error;

// ============================================================================
// Compile-time Feature Checking
// ============================================================================

/// Check if a feature is compiled in via build_options.
pub fn isFeatureCompiledIn(comptime feature: Feature) bool {
    return switch (feature) {
        .gpu => build_options.enable_gpu,
        .ai => build_options.enable_ai,
        .llm => build_options.enable_llm,
        .embeddings => build_options.enable_ai,
        .agents => build_options.enable_ai,
        .training => build_options.enable_training,
        .personas => build_options.enable_ai,
        .database => build_options.enable_database,
        .network => build_options.enable_network,
        .observability => build_options.enable_profiling,
        .web => build_options.enable_web,
        .cloud => build_options.enable_cloud,
        .analytics => build_options.enable_analytics,
        .auth => build_options.enable_auth,
        .messaging => build_options.enable_messaging,
        .cache => build_options.enable_cache,
        .storage => build_options.enable_storage,
        .search => build_options.enable_search,
        .mobile => build_options.enable_mobile,
        .reasoning => build_options.enable_reasoning,
    };
}

/// Get parent feature for sub-features.
pub fn getParentFeature(feature: Feature) ?Feature {
    return switch (feature) {
        .llm, .embeddings, .agents, .training, .personas => .ai,
        else => null,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "isFeatureCompiledIn matches build_options" {
    try std.testing.expectEqual(build_options.enable_gpu, comptime isFeatureCompiledIn(.gpu));
    try std.testing.expectEqual(build_options.enable_ai, comptime isFeatureCompiledIn(.ai));
    try std.testing.expectEqual(build_options.enable_database, comptime isFeatureCompiledIn(.database));
}

test "getParentFeature returns correct parent" {
    try std.testing.expectEqual(Feature.ai, getParentFeature(.llm).?);
    try std.testing.expectEqual(Feature.ai, getParentFeature(.embeddings).?);
    try std.testing.expectEqual(Feature.ai, getParentFeature(.agents).?);
    try std.testing.expectEqual(Feature.ai, getParentFeature(.training).?);
    try std.testing.expect(getParentFeature(.gpu) == null);
    try std.testing.expect(getParentFeature(.ai) == null);
}
