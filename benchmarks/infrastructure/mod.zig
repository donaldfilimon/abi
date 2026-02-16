//! Infrastructure Benchmarks Module
//!
//! Infrastructure-level performance testing for:
//! - Concurrency and parallelism
//! - Cryptographic operations
//! - Memory management
//! - SIMD/vectorization
//! - Network operations

const std = @import("std");

// Infrastructure benchmark suites
pub const suites = struct {
    pub const concurrency = @import("concurrency.zig");
    pub const crypto = @import("crypto.zig");
    pub const memory = @import("memory.zig");
    pub const simd = @import("simd.zig");
    pub const network = @import("network/mod.zig");
    pub const v2_modules = @import("v2_modules.zig");
};

pub fn runAll(allocator: std.mem.Allocator) !void {
    std.debug.print("Running Infrastructure Benchmarks\n", .{});
    std.debug.print("==============================\n\n", .{});

    try suites.concurrency.run(allocator);
    try suites.crypto.run(allocator);
    try suites.memory.run(allocator);
    try suites.simd.run(allocator);
    try suites.network.run(allocator);
    try suites.v2_modules.run(allocator);

    std.debug.print("\nInfrastructure Benchmarks Complete\n", .{});
    std.debug.print("==============================\n", .{});
}

test {
    _ = suites.concurrency;
    _ = suites.crypto;
    _ = suites.memory;
    _ = suites.simd;
    _ = suites.network;
    _ = suites.v2_modules;
}
