//! GPU Benchmark Module
//!
//! GPU performance benchmarking and testing

const std = @import("std");

pub const benchmarks = @import("benchmarks.zig");

test {
    std.testing.refAllDecls(@This());
}
