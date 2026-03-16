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

/// Stub benchmark function type.
pub const BenchmarkFn = *const fn (state: *BenchmarkState) void;

/// Stub benchmark state.
pub const BenchmarkState = struct {
    iteration: usize = 0,
    total_iterations: usize = 0,
    allocator: std.mem.Allocator,

    pub fn doNotOptimize(_: *BenchmarkState, _: anytype) void {}
};

/// Stub benchmark result.
pub const BenchmarkResult = struct {
    name: []const u8,
    iterations: usize,
    total_ns: u64,
    min_ns: u64,
    max_ns: u64,
    mean_ns: u64,
    median_ns: u64,

    pub fn opsPerSecond(_: BenchmarkResult) f64 {
        return 0.0;
    }
};

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
