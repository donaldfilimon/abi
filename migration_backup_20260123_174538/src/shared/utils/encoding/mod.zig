const std = @import("std");

pub const EncodingError = error{
    InvalidHex,
};

pub fn hexEncodeLower(allocator: std.mem.Allocator, bytes: []const u8) ![]u8 {
    const output = try allocator.alloc(u8, bytes.len * 2);
    for (bytes, 0..) |byte, i| {
        const hi = byte >> 4;
        const lo = byte & 0x0f;
        output[i * 2] = nibbleToHexLower(hi);
        output[i * 2 + 1] = nibbleToHexLower(lo);
    }
    return output;
}

pub fn hexDecode(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    if (text.len % 2 != 0) return EncodingError.InvalidHex;
    const output = try allocator.alloc(u8, text.len / 2);
    errdefer allocator.free(output);
    var i: usize = 0;
    while (i < text.len) : (i += 2) {
        const hi = try hexToNibble(text[i]);
        const lo = try hexToNibble(text[i + 1]);
        output[i / 2] = (hi << 4) | lo;
    }
    return output;
}

fn nibbleToHexLower(nibble: u8) u8 {
    return if (nibble < 10) nibble + '0' else nibble - 10 + 'a';
}

fn hexToNibble(char: u8) EncodingError!u8 {
    return switch (char) {
        '0'...'9' => char - '0',
        'a'...'f' => char - 'a' + 10,
        'A'...'F' => char - 'A' + 10,
        else => EncodingError.InvalidHex,
    };
}

test "hex encoding roundtrip" {
    const allocator = std.testing.allocator;
    const bytes = &.{ 0xde, 0xad, 0xbe, 0xef };
    const encoded = try hexEncodeLower(allocator, bytes);
    defer allocator.free(encoded);
    try std.testing.expectEqualStrings("deadbeef", encoded);

    const decoded = try hexDecode(allocator, encoded);
    defer allocator.free(decoded);
    try std.testing.expectEqualSlices(u8, bytes, decoded);
}

test "hex decode rejects odd length" {
    try std.testing.expectError(EncodingError.InvalidHex, hexDecode(std.testing.allocator, "abc"));
}
