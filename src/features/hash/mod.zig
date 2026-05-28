//! Hash Feature
//!
//! Stable, portable 64-bit and 128-bit hashing utilities with feature flag
//! for minimal static builds. Provides Wyhash (fast, good distribution) and
//! FNV-1a fallbacks.

pub const types = @import("types.zig");

const std = @import("std");

pub const HashError = types.HashError;
pub const Error = types.Error;
pub const Hash64 = types.Hash64;
pub const Hash128 = types.Hash128;

pub fn isEnabled() bool {
    return true;
}

/// Fast, high-quality 64-bit hash (Wyhash).
/// Seed allows different hash domains.
pub fn wyhash(data: []const u8, seed: u64) Hash64 {
    return std.hash.Wyhash.hash(seed, data);
}

/// Simple, portable FNV-1a 64-bit hash.
/// Good for short keys and when you want stable cross-platform results with seed=0.
pub fn fnv1a_64(data: []const u8) Hash64 {
    const prime: u64 = 0x100000001b3;
    var hash: u64 = 0xcbf29ce484222325; // FNV offset basis

    for (data) |byte| {
        hash ^= byte;
        hash *%= prime;
    }
    return hash;
}

/// 128-bit hash (hi/lo) built from two Wyhash passes with different seeds.
/// Useful when collision resistance matters more than speed.
pub fn wyhash128(data: []const u8) Hash128 {
    const hi = std.hash.Wyhash.hash(0x1234_5678_9abc_def0, data);
    const lo = std.hash.Wyhash.hash(0xfedc_ba98_7654_3210, data);
    return .{ .hi = hi, .lo = lo };
}

/// Convenience: 64-bit hash of a string using the default Wyhash seed (0).
pub fn hash64(data: []const u8) Hash64 {
    return wyhash(data, 0);
}

test {
    std.testing.refAllDecls(@This());
}

test "wyhash is deterministic and seed-sensitive" {
    const data = "hello feature scaffolding";
    const h1 = wyhash(data, 0);
    const h2 = wyhash(data, 0);
    const h3 = wyhash(data, 1);

    try std.testing.expectEqual(h1, h2);
    try std.testing.expect(h1 != h3);
    try std.testing.expect(h1 != 0);
}

test "fnv1a_64 produces stable portable results" {
    const data = "abi";
    const h1 = fnv1a_64(data);
    const h2 = fnv1a_64(data);
    try std.testing.expectEqual(h1, h2);
    try std.testing.expect(h1 != 0);
}

test "wyhash128 returns distinct hi/lo for non-empty input" {
    const h = wyhash128("distinct");
    try std.testing.expect(h.hi != h.lo);
    try std.testing.expect(h.hi != 0);
    try std.testing.expect(h.lo != 0);
}
