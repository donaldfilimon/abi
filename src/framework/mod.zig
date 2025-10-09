const std = @import("std");

pub const config = @import("config.zig");
pub const runtime = @import("runtime.zig");

// Re-export main types
pub const Feature = config.Feature;
pub const FeatureToggles = config.FeatureToggles;
pub const FrameworkOptions = config.FrameworkOptions;
pub const Framework = runtime.Framework;

// Re-export utility functions
pub const featureLabel = config.featureLabel;
pub const featureDescription = config.featureDescription;
pub const deriveFeatureToggles = config.deriveFeatureToggles;

test "framework module refAllDecls" {
    std.testing.refAllDecls(@This());
}
