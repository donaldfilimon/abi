//! Server Utility Functions
//!
//! Shared helper functions for HTTP handling, header parsing,
//! and security utilities.

const std = @import("std");

/// Split an HTTP target into path and query components.
pub fn splitTarget(target: []const u8) struct { path: []const u8, query: []const u8 } {
    if (std.mem.indexOfScalar(u8, target, '?')) |idx| {
        return .{ .path = target[0..idx], .query = target[idx + 1 ..] };
    }
    return .{ .path = target, .query = "" };
}

/// Find a header value in a raw HTTP header buffer.
pub fn findHeaderInBuffer(buffer: []const u8, header_name: []const u8) ?[]const u8 {
    var it = std.mem.splitSequence(u8, buffer, "\r\n");
    _ = it.next(); // Skip request line

    while (it.next()) |line| {
        if (line.len == 0) break;
        const colon_idx = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        const name = line[0..colon_idx];

        if (std.ascii.eqlIgnoreCase(name, header_name)) {
            const value_start = colon_idx + 1;
            if (value_start >= line.len) return "";
            return std.mem.trim(u8, line[value_start..], " \t");
        }
    }
    return null;
}

/// Constant-time byte comparison to prevent timing attacks.
pub fn timingSafeEqual(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    var diff: u8 = 0;
    for (a, b) |x, y| {
        diff |= x ^ y;
    }
    return diff == 0;
}


test {
    std.testing.refAllDecls(@This());
}
