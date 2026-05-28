//! Hash stub -- disabled at compile time.
//!
//! NOTE: This stub computes real hashes (wyhash, fnv1a_64, etc.) despite being
//! disabled. This is intentional for test compatibility — dependent code (tests,
//! contracts) continues to work with deterministic hash output. The "disabled"
//! signal is isEnabled() == false.
//!
//! Provides identical public API surface so that code using `abi.features.hash`
//! continues to compile and has predictable behavior when the feature is off.

pub const types = @import("types.zig");

const std = @import("std");

pub const HashError = types.HashError;
pub const Error = types.Error;
pub const Hash64 = types.Hash64;
pub const Hash128 = types.Hash128;

pub fn isEnabled() bool {
    return false;
}

pub fn wyhash(data: []const u8, seed: u64) Hash64 {
    // Still compute a real hash in the stub so dependent code (tests, contracts)
    // keeps working. The "disabled" signal is isEnabled() == false.
    _ = seed;
    return std.hash.Wyhash.hash(0xdead_beef_cafe_babe, data);
}

pub fn fnv1a_64(data: []const u8) Hash64 {
    const prime: u64 = 0x100000001b3;
    var hash: u64 = 0xcbf29ce484222325;

    for (data) |byte| {
        hash ^= byte;
        hash *%= prime;
    }
    return hash;
}

pub fn wyhash128(data: []const u8) Hash128 {
    const hi = std.hash.Wyhash.hash(0x1111_2222_3333_4444, data);
    const lo = std.hash.Wyhash.hash(0xaaaa_bbbb_cccc_dddd, data);
    return .{ .hi = hi, .lo = lo };
}

pub fn hash64(data: []const u8) Hash64 {
    return wyhash(data, 0);
}

test {
    std.testing.refAllDecls(@This());
}

test "hash stub still produces deterministic output" {
    const data = "hello from stub";
    const h1 = wyhash(data, 0);
    const h2 = wyhash(data, 0);
    try std.testing.expectEqual(h1, h2);
}

test "hash stub reports disabled" {
    try std.testing.expect(!isEnabled());
}
