//! Cryptographic utilities including hashing, hex encoding, and constant-time comparisons.
//!
//! Provides FNV hash functions, SHA-256, hex encoding/decoding, and constant-time
//! string comparison for security-sensitive operations.

const std = @import("std");

pub const HexError = error{
    InvalidHex,
};

pub fn fnv1a64(bytes: []const u8) u64 {
    var hash: u64 = 0xcbf29ce484222325;
    for (bytes) |byte| {
        hash ^= byte;
        hash *%= 0x100000001b3;
    }
    return hash;
}

pub fn fnv1a32(bytes: []const u8) u32 {
    var hash: u32 = 0x811c9dc5;
    for (bytes) |byte| {
        hash ^= byte;
        hash *%= 0x01000193;
    }
    return hash;
}

pub fn sha256(bytes: []const u8) [32]u8 {
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hasher.update(bytes);
    var digest: [32]u8 = undefined;
    hasher.final(&digest);
    return digest;
}

pub fn toHexLower(allocator: std.mem.Allocator, bytes: []const u8) ![]u8 {
    const out = try allocator.alloc(u8, bytes.len * 2);
    for (bytes, 0..) |byte, i| {
        out[i * 2] = nibbleToHex(@intCast(byte >> 4));
        out[i * 2 + 1] = nibbleToHex(@intCast(byte & 0x0f));
    }
    return out;
}

pub fn fromHex(allocator: std.mem.Allocator, text: []const u8) (HexError || error{OutOfMemory})![]u8 {
    if (text.len % 2 != 0) return HexError.InvalidHex;
    const out = try allocator.alloc(u8, text.len / 2);
    errdefer allocator.free(out);
    var i: usize = 0;
    while (i < out.len) : (i += 1) {
        const hi = try hexToNibble(text[i * 2]);
        const lo = try hexToNibble(text[i * 2 + 1]);
        out[i] = (hi << 4) | lo;
    }
    return out;
}

pub fn toHexLowerU64(allocator: std.mem.Allocator, value: u64) ![]u8 {
    var buffer: [16]u8 = undefined;
    var i: usize = 0;
    while (i < buffer.len) : (i += 1) {
        const shift: u6 = @intCast((buffer.len - 1 - i) * 4);
        const nibble: u8 = @intCast((value >> shift) & 0x0f);
        buffer[i] = nibbleToHex(nibble);
    }
    return allocator.dupe(u8, &buffer);
}

pub fn toHexLowerU32(allocator: std.mem.Allocator, value: u32) ![]u8 {
    var buffer: [8]u8 = undefined;
    var i: usize = 0;
    while (i < buffer.len) : (i += 1) {
        const shift: u5 = @intCast((buffer.len - 1 - i) * 4);
        const nibble: u8 = @intCast((value >> shift) & 0x0f);
        buffer[i] = nibbleToHex(nibble);
    }
    return allocator.dupe(u8, &buffer);
}

pub fn fnv1a64HexLower(allocator: std.mem.Allocator, bytes: []const u8) ![]u8 {
    return toHexLowerU64(allocator, fnv1a64(bytes));
}

pub fn fnv1a32HexLower(allocator: std.mem.Allocator, bytes: []const u8) ![]u8 {
    return toHexLowerU32(allocator, fnv1a32(bytes));
}

pub fn sha256HexLower(allocator: std.mem.Allocator, bytes: []const u8) ![]u8 {
    const digest = sha256(bytes);
    return toHexLower(allocator, &digest);
}

pub fn constantTimeEqual(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    var diff: u8 = 0;
    for (a, 0..) |value, i| {
        diff |= value ^ b[i];
    }
    return diff == 0;
}

fn nibbleToHex(nibble: u8) u8 {
    return if (nibble < 10) nibble + '0' else nibble - 10 + 'a';
}

fn hexToNibble(byte: u8) HexError!u8 {
    if (byte >= '0' and byte <= '9') return byte - '0';
    if (byte >= 'a' and byte <= 'f') return byte - 'a' + 10;
    if (byte >= 'A' and byte <= 'F') return byte - 'A' + 10;
    return HexError.InvalidHex;
}

test "hex formatting utilities" {
    const allocator = std.testing.allocator;
    const hex64 = try toHexLowerU64(allocator, 0x0123456789abcdef);
    defer allocator.free(hex64);
    try std.testing.expectEqualStrings("0123456789abcdef", hex64);

    const hex32 = try toHexLowerU32(allocator, 0x89abcdef);
    defer allocator.free(hex32);
    try std.testing.expectEqualStrings("89abcdef", hex32);
}

test "hex encoding and decoding" {
    const allocator = std.testing.allocator;
    const encoded = try toHexLower(allocator, "hi");
    defer allocator.free(encoded);
    try std.testing.expectEqualStrings("6869", encoded);

    const decoded = try fromHex(allocator, "6869");
    defer allocator.free(decoded);
    try std.testing.expectEqualStrings("hi", decoded);

    const decoded_upper = try fromHex(allocator, "FF00");
    defer allocator.free(decoded_upper);
    try std.testing.expectEqualSlices(u8, &.{ 0xff, 0x00 }, decoded_upper);
}

test "sha256 and constant time compare" {
    const allocator = std.testing.allocator;
    const digest_text = try sha256HexLower(allocator, "abc");
    defer allocator.free(digest_text);
    try std.testing.expectEqualStrings(
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        digest_text,
    );

    try std.testing.expect(constantTimeEqual("same", "same"));
    try std.testing.expect(!constantTimeEqual("same", "diff"));
}

test {
    std.testing.refAllDecls(@This());
}
