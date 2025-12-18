//! Framework Module
//!
//! Core framework orchestration and runtime management

const std = @import("std");
const features = @import("../features/mod.zig");

pub const runtime = @import("runtime.zig");
pub const config = @import("config.zig");

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

test "framework module refAllDecls" {
    std.testing.refAllDecls(@This());
}

fn featureToTag(feature: Feature) ?features.FeatureTag {
    return switch (feature) {
        .ai => .ai,
        .database => .database,
        .web => .web,
        .monitoring => .monitoring,
        .gpu => .gpu,
        .connectors => .connectors,
        .simd => null,
    };
}

/// Convert public framework options into a runtime configuration.
/// The returned slices are allocator-owned and should be freed by the caller.
pub fn runtimeConfigFromOptions(allocator: std.mem.Allocator, options: FrameworkOptions) !RuntimeConfig {
    var toggles = deriveFeatureToggles(options);

    var enabled_tags = std.ArrayList(features.FeatureTag).init(allocator);
    defer enabled_tags.deinit();

    var toggle_iter = toggles.iterator();
    while (toggle_iter.next()) |feature| {
        if (featureToTag(feature)) |tag| {
            try enabled_tags.append(tag);
        }
    }

    var disabled_tags = std.ArrayList(features.FeatureTag).init(allocator);
    defer disabled_tags.deinit();
    for (options.disabled_features) |feature| {
        if (featureToTag(feature)) |tag| {
            try disabled_tags.append(tag);
        }
    }

    const enabled_features = try enabled_tags.toOwnedSlice();
    errdefer allocator.free(enabled_features);

    const disabled_features = try disabled_tags.toOwnedSlice();
    errdefer allocator.free(disabled_features);

    return RuntimeConfig{
        .enabled_features = enabled_features,
        .disabled_features = disabled_features,
    };
}

fn containsFeature(haystack: []const features.FeatureTag, needle: features.FeatureTag) bool {
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

    var runtime_config = try runtimeConfigFromOptions(testing.allocator, options);
    defer testing.allocator.free(runtime_config.enabled_features);
    defer testing.allocator.free(runtime_config.disabled_features);

    try testing.expect(!containsFeature(runtime_config.enabled_features, .ai));
    try testing.expect(containsFeature(runtime_config.enabled_features, .gpu));
    try testing.expect(containsFeature(runtime_config.enabled_features, .database));
    try testing.expect(containsFeature(runtime_config.enabled_features, .monitoring));
    try testing.expect(!containsFeature(runtime_config.enabled_features, .web));
    try testing.expect(containsFeature(runtime_config.disabled_features, .web));
}
