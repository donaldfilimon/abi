//! Framework Module
//!
//! Core framework orchestration and runtime management

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

const std = @import("std");