//! Crypto Utilities Module
//!
//! Cryptographic utilities and functions

const std = @import("std");

/// Generate a cryptographically secure random bytes
pub fn secureRandomBytes(allocator: std.mem.Allocator, len: usize) ![]u8 {
    const bytes = try allocator.alloc(u8, len);
    std.crypto.random.bytes(bytes);
    return bytes;
}

/// Generate a secure random string using alphanumeric characters
pub fn secureRandomString(allocator: std.mem.Allocator, len: usize) ![]u8 {
    const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    var result = try allocator.alloc(u8, len);

    for (0..len) |i| {
        const random_index = std.crypto.random.uintLessThan(u8, charset.len);
        result[i] = charset[random_index];
    }

    return result;
}

/// Simple hash function using Wyhash (not cryptographically secure)
pub fn hash(data: []const u8) u64 {
    return std.hash.Wyhash.hash(0, data);
}

/// Constant-time comparison to prevent timing attacks
pub fn secureCompare(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;

    var result: u8 = 0;
    var i: usize = 0;
    while (i < a.len) : (i += 1) {
        result |= a[i] ^ b[i];
    }

    return result == 0;
}

test "secure random bytes" {
    const allocator = std.testing.allocator;
    const bytes = try secureRandomBytes(allocator, 32);
    defer allocator.free(bytes);

    try std.testing.expectEqual(@as(usize, 32), bytes.len);
    // Check that not all bytes are the same (very unlikely for random data)
    var all_same = true;
    for (bytes[1..]) |b| {
        if (b != bytes[0]) {
            all_same = false;
            break;
        }
    }
    try std.testing.expect(!all_same);
}

test "secure random string" {
    const allocator = std.testing.allocator;
    const str = try secureRandomString(allocator, 16);
    defer allocator.free(str);

    try std.testing.expectEqual(@as(usize, 16), str.len);

    // Check all characters are valid
    const valid_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    for (str) |c| {
        try std.testing.expect(std.mem.indexOfScalar(u8, valid_chars, c) != null);
    }
}

test "hash function" {
    const data1 = "hello";
    const data2 = "world";
    const data3 = "hello";

    const hash1 = hash(data1);
    const hash2 = hash(data2);
    const hash3 = hash(data3);

    try std.testing.expect(hash1 != hash2);
    try std.testing.expectEqual(hash1, hash3);
}

test "secure compare" {
    const str1 = "password123";
    const str2 = "password123";
    const str3 = "password124";

    try std.testing.expect(secureCompare(str1, str2));
    try std.testing.expect(!secureCompare(str1, str3));
    try std.testing.expect(!secureCompare(str1, "different_length"));
}

test {
    std.testing.refAllDecls(@This());
}
