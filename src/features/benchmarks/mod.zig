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

test "Config default values" {
    const config = Config{};
    try std.testing.expectEqual(@as(u32, 3), config.warmup_iterations);
    try std.testing.expectEqual(@as(u32, 10), config.sample_iterations);
    try std.testing.expect(!config.export_json);
    try std.testing.expect(config.output_path == null);
}

test "Config.defaults returns same as zero-init" {
    const a = Config{};
    const b = Config.defaults();
    try std.testing.expectEqual(a.warmup_iterations, b.warmup_iterations);
    try std.testing.expectEqual(a.sample_iterations, b.sample_iterations);
    try std.testing.expectEqual(a.export_json, b.export_json);
    try std.testing.expect(b.output_path == null);
}

test "Config custom values" {
    const config = Config{
        .warmup_iterations = 100,
        .sample_iterations = 500,
        .export_json = true,
        .output_path = "/tmp/bench.json",
    };
    try std.testing.expectEqual(@as(u32, 100), config.warmup_iterations);
    try std.testing.expectEqual(@as(u32, 500), config.sample_iterations);
    try std.testing.expect(config.export_json);
    try std.testing.expectEqualStrings("/tmp/bench.json", config.output_path.?);
}

test "Context stores config correctly" {
    const config = Config{
        .warmup_iterations = 7,
        .sample_iterations = 20,
        .export_json = true,
    };
    const ctx = try Context.init(std.testing.allocator, config);
    defer ctx.deinit();
    try std.testing.expectEqual(@as(u32, 7), ctx.config.warmup_iterations);
    try std.testing.expectEqual(@as(u32, 20), ctx.config.sample_iterations);
    try std.testing.expect(ctx.config.export_json);
}

test "Context multiple init and deinit" {
    const ctx1 = try Context.init(std.testing.allocator, Config{});
    const ctx2 = try Context.init(std.testing.allocator, Config{ .warmup_iterations = 1 });
    ctx1.deinit();
    ctx2.deinit();
}

test "BenchmarksError error set" {
    // Verify all error variants exist in the error set
    const errors = [_]BenchmarksError{
        error.FeatureDisabled,
        error.OutOfMemory,
        error.InvalidConfig,
        error.BenchmarkFailed,
    };
    try std.testing.expectEqual(@as(usize, 4), errors.len);
}
