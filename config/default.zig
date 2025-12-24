//! Default Configuration
//!
//! Default configuration for the ABI framework

const abi = @import("../src/mod.zig");

pub const default_config = abi.FrameworkConfiguration{
    .max_plugins = 128,
    .enable_hot_reload = false,
    .enable_profiling = false,
    .memory_limit_mb = null,
    .log_level = .info,
    .enabled_features = &[_]abi.features.FeatureTag{ .ai, .database, .web, .monitoring, .simd },
    .disabled_features = &[_]abi.features.FeatureTag{.gpu},
};

pub const development_config = abi.FrameworkConfiguration{
    .max_plugins = 256,
    .enable_hot_reload = true,
    .enable_profiling = true,
    .memory_limit_mb = 1024,
    .log_level = .debug,
    .enabled_features = &[_]abi.features.FeatureTag{ .ai, .gpu, .database, .web, .monitoring, .connectors, .simd },
    .disabled_features = &[_]abi.features.FeatureTag{},
};

pub const production_config = abi.FrameworkConfiguration{
    .max_plugins = 64,
    .enable_hot_reload = false,
    .enable_profiling = false,
    .memory_limit_mb = 512,
    .log_level = .warn,
    .enabled_features = &[_]abi.features.FeatureTag{ .ai, .database, .web, .monitoring, .simd },
    .disabled_features = &[_]abi.features.FeatureTag{ .gpu, .connectors },
};
