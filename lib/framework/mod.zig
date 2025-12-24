//! Framework Module
//!
//! Core framework orchestration and runtime management

const std = @import("std");
pub const runtime = @import("runtime.zig");
pub const config = @import("config.zig");
pub const catalog = @import("catalog.zig");
pub const feature_manager = @import("feature_manager.zig");
pub const state = @import("state.zig");

// Re-export main types for convenience
pub const Framework = runtime.Framework;
pub const RuntimeConfig = runtime.RuntimeConfig;
pub const Component = runtime.Component;
pub const RuntimeStats = runtime.RuntimeStats;

// Re-export utility functions
pub const createFramework = runtime.createFramework;
pub const defaultConfig = runtime.defaultConfig;

// Re-export configuration types
pub const Feature = config.Feature;
pub const FeatureToggles = config.FeatureToggles;
pub const FrameworkOptions = config.FrameworkOptions;

pub const featureLabel = config.featureLabel;
pub const featureDescription = config.featureDescription;
pub const deriveFeatureToggles = config.deriveFeatureToggles;
pub const runtimeConfigFromOptions = config.runtimeConfigFromOptions;

test "framework module refAllDecls" {
    std.testing.refAllDecls(@This());
}

fn containsFeature(haystack: []const Feature, needle: Feature) bool {
    for (haystack) |feature| {
        if (feature == needle) return true;
    }
    return false;
}

test "runtimeConfigFromOptions maps feature toggles" {
    const testing = std.testing;

    const options = FrameworkOptions{
        .enable_ai = false,
        .enable_gpu = true,
        .disabled_features = &.{.web},
    };

    const runtime_config = try runtimeConfigFromOptions(testing.allocator, options);

    try testing.expect(!containsFeature(runtime_config.enabled_features, .ai));
    try testing.expect(containsFeature(runtime_config.enabled_features, .gpu));
    try testing.expect(containsFeature(runtime_config.enabled_features, .database));
    try testing.expect(containsFeature(runtime_config.enabled_features, .monitoring));
    try testing.expect(!containsFeature(runtime_config.enabled_features, .web));
    try testing.expect(containsFeature(runtime_config.disabled_features, .web));
}
