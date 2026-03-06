//! Integrity validation.

const std = @import("std");

pub fn calculateChecksum(data: []const u8) [32]u8 {
    _ = data;
    unreachable; // TODO
}

pub fn verifyChecksum(data: []const u8, expected: [32]u8) bool {
    _ = data;
    _ = expected;
    unreachable; // TODO
}
