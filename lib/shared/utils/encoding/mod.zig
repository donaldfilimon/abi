//! Encoding Utilities Module
//!
//! Data encoding and decoding utilities

const std = @import("std");

/// Base64 encode data
pub fn base64Encode(allocator: std.mem.Allocator, data: []const u8) ![]u8 {
    return std.base64.standard.Encoder.encode(allocator, data);
}

/// Base64 decode data
pub fn base64Decode(allocator: std.mem.Allocator, encoded: []const u8) ![]u8 {
    return std.base64.standard.Decoder.decode(allocator, encoded);
}

/// Hex encode data
pub fn hexEncode(allocator: std.mem.Allocator, data: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "{x}", .{std.fmt.fmtSliceHexLower(data)});
}

test {
    std.testing.refAllDecls(@This());
}

test "base64Encode and base64Decode" {
    const allocator = std.testing.allocator;
    const data = "Hello, World!";
    const encoded = try base64Encode(allocator, data);
    defer allocator.free(encoded);
    const decoded = try base64Decode(allocator, encoded);
    defer allocator.free(decoded);
    try std.testing.expectEqualStrings(data, decoded);
}
