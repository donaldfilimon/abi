//! Binary encode and decode logic.
//!
//! Uses direct memory writes instead of stream APIs for Zig 0.16 compatibility.

const std = @import("std");
const block = @import("block.zig");

pub fn encodeBlock(allocator: std.mem.Allocator, b: block.StoredBlock) ![]u8 {
    const header_size = 32 + 4 + 4 + 32 + 8 + 4 + 2 + 1; // [32]u8, u32, u32, [32]u8, u64, u32, u16, u8
    const total_size = header_size + b.payload.len;
    const data = try allocator.alloc(u8, total_size);
    errdefer allocator.free(data);

    var offset: usize = 0;

    // id: [32]u8
    @memcpy(data[offset..][0..32], &b.header.id.id);
    offset += 32;

    // kind: u32 LE
    std.mem.writeInt(u32, data[offset..][0..4], @intFromEnum(b.header.kind), .little);
    offset += 4;

    // version: u32 LE
    std.mem.writeInt(u32, data[offset..][0..4], b.header.version, .little);
    offset += 4;

    // content_hash: [32]u8
    @memcpy(data[offset..][0..32], &b.header.content_hash);
    offset += 32;

    // timestamp counter: u64 LE
    std.mem.writeInt(u64, data[offset..][0..8], b.header.timestamp.counter, .little);
    offset += 8;

    // size: u32 LE
    std.mem.writeInt(u32, data[offset..][0..4], b.header.size, .little);
    offset += 4;

    // flags: u16 LE
    std.mem.writeInt(u16, data[offset..][0..2], b.header.flags, .little);
    offset += 2;

    // compression_marker: u8
    data[offset] = b.header.compression_marker;
    offset += 1;

    // payload
    @memcpy(data[offset..][0..b.payload.len], b.payload);

    return data;
}

pub fn decodeBlock(allocator: std.mem.Allocator, data: []const u8) !block.StoredBlock {
    const header_size = 32 + 4 + 4 + 32 + 8 + 4 + 2 + 1;
    if (data.len < header_size) return error.MalformedBlock;

    var b: block.StoredBlock = undefined;
    var offset: usize = 0;

    // id: [32]u8
    @memcpy(&b.header.id.id, data[offset..][0..32]);
    offset += 32;

    // kind: u32 LE
    b.header.kind = @enumFromInt(std.mem.readInt(u32, data[offset..][0..4], .little));
    offset += 4;

    // version: u32 LE
    b.header.version = std.mem.readInt(u32, data[offset..][0..4], .little);
    offset += 4;

    // content_hash: [32]u8
    @memcpy(&b.header.content_hash, data[offset..][0..32]);
    offset += 32;

    // timestamp counter: u64 LE
    b.header.timestamp.counter = std.mem.readInt(u64, data[offset..][0..8], .little);
    offset += 8;

    // size: u32 LE
    b.header.size = std.mem.readInt(u32, data[offset..][0..4], .little);
    offset += 4;

    // flags: u16 LE
    b.header.flags = std.mem.readInt(u16, data[offset..][0..2], .little);
    offset += 2;

    // compression_marker: u8
    b.header.compression_marker = data[offset];
    offset += 1;

    // payload
    b.payload = try allocator.dupe(u8, data[header_size..]);

    return b;
}
