//! Benchmarks Stub
//!
//! Placeholder when benchmarks module is disabled via build flags.

const std = @import("std");
const core_config = @import("../../core/config/benchmarks.zig");
const stub_context = @import("../../core/stub_context.zig");

pub const Config = core_config.BenchmarksConfig;
pub const BenchmarksError = error{
    FeatureDisabled,
    OutOfMemory,
    InvalidConfig,
    BenchmarkFailed,
};

pub const Context = stub_context.StubContextWithConfig(Config);

pub fn isEnabled() bool {
    return false;
}
