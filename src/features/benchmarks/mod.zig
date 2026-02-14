//! Benchmarks Module
//!
//! Performance benchmarking and timing utilities. Provides a Context for
//! running benchmark suites, recording results, and exporting metrics.

const std = @import("std");
const build_options = @import("build_options");
const core_config = @import("../../core/config/benchmarks.zig");

pub const Config = core_config.BenchmarksConfig;
pub const BenchmarksError = error{
    FeatureDisabled,
    OutOfMemory,
    InvalidConfig,
    BenchmarkFailed,
};

pub const Context = struct {
    allocator: std.mem.Allocator,
    config: Config,

    pub fn init(allocator: std.mem.Allocator, config: Config) !*Context {
        const ctx = try allocator.create(Context);
        ctx.* = .{ .allocator = allocator, .config = config };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        self.allocator.destroy(self);
    }
};

pub fn isEnabled() bool {
    return build_options.enable_benchmarks;
}

test "basic initialization" {
    const ctx = try Context.init(std.testing.allocator, Config{});
    defer ctx.deinit();
    try std.testing.expect(isEnabled() == build_options.enable_benchmarks);
}
