const std = @import("std");

pub const config = @import("config.zig");
pub const runtime = @import("runtime.zig");

pub const Feature = config.Feature;
pub const FeatureToggles = config.FeatureToggles;
pub const FrameworkOptions = config.FrameworkOptions;
pub const Framework = runtime.Framework;

pub const featureLabel = config.featureLabel;
pub const featureDescription = config.featureDescription;

pub fn deriveFeatureToggles(options: FrameworkOptions) config.FeatureToggles {
    return config.deriveFeatureToggles(options);
}

test "framework module refAllDecls" {
    std.testing.refAllDecls(@This());
}
