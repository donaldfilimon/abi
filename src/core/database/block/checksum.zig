//! Integrity validation.

const std = @import("std");

pub fn calculateChecksum(data: []const u8) [32]u8 {
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(data, &hash, .{});
    return hash;
}

pub fn verifyChecksum(data: []const u8, expected: [32]u8) bool {
    const actual = calculateChecksum(data);
    return std.mem.eql(u8, &actual, &expected);
}
