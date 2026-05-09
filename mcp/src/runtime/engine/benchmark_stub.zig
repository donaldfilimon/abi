//! Benchmark stub for platforms without thread support.

const std = @import("std");

pub const BenchmarkConfig = struct {
    iterations: u32 = 1000,
    warmup: u32 = 10,
};

pub const Benchmark = struct {
    pub fn init(_: std.mem.Allocator, _: BenchmarkConfig) !Benchmark {
        return error.ThreadsNotSupported;
    }
    pub fn deinit(_: *Benchmark) void {}
    pub fn run(_: *Benchmark, _: anytype) !void {}
};
