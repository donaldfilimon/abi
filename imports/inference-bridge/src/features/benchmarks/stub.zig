//! Benchmarks Stub
//!
//! Placeholder when benchmarks module is disabled via build flags.

const std = @import("std");
const stub_context = @import("../core/stub_helpers.zig");
pub const types = @import("types.zig");

pub const Config = types.Config;
pub const BenchmarksError = types.BenchmarksError;
pub const Error = BenchmarksError;

pub const Context = stub_context.StubContextWithConfig(Config);

pub fn isEnabled() bool {
    return false;
}

pub fn isInitialized() bool {
    return false;
}

pub const BenchmarkFn = types.BenchmarkFn;
pub const BenchmarkState = types.BenchmarkState;
pub const BenchmarkResult = types.BenchmarkResult;

/// Stub benchmark suite — all operations return `error.FeatureDisabled`.
pub const BenchmarkSuite = struct {
    name: []const u8,
    benchmarks: std.ArrayListUnmanaged(Entry) = .empty,
    results: std.ArrayListUnmanaged(BenchmarkResult) = .empty,
    config: Config,

    const Entry = struct {
        name: []const u8,
        func: BenchmarkFn,
    };

    pub fn init(name: []const u8, config: Config) BenchmarkSuite {
        return .{
            .name = name,
            .config = config,
        };
    }

    pub fn deinit(self: *BenchmarkSuite, allocator: std.mem.Allocator) void {
        self.benchmarks.deinit(allocator);
        self.results.deinit(allocator);
    }

    pub fn addBenchmark(
        _: *BenchmarkSuite,
        _: std.mem.Allocator,
        _: []const u8,
        _: BenchmarkFn,
    ) !void {
        return error.FeatureDisabled;
    }

    pub fn run(_: *BenchmarkSuite, _: std.mem.Allocator) !void {
        return error.FeatureDisabled;
    }

    pub fn formatReport(_: *const BenchmarkSuite, _: std.mem.Allocator) ![]u8 {
        return error.FeatureDisabled;
    }

    pub fn formatJson(_: *const BenchmarkSuite, _: std.mem.Allocator) ![]u8 {
        return error.FeatureDisabled;
    }
};

test {
    std.testing.refAllDecls(@This());
}
