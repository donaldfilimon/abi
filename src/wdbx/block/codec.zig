//! Binary encode and decode logic.

const std = @import("std");
const block = @import("block.zig");

pub fn encodeBlock(allocator: std.mem.Allocator, b: block.StoredBlock) ![]u8 {
    const header_size = 32 + 4 + 4 + 32 + 8 + 4 + 2 + 1; // [32]u8, u32, u32, [32]u8, u64, u32, u16, u8
    const total_size = header_size + b.payload.len;
    const data = try allocator.alloc(u8, total_size);
    errdefer allocator.free(data);

    var fbs = std.io.fixedBufferStream(data);
    const writer = fbs.writer();

    try writer.writeAll(&b.header.id.id);
    try writer.writeInt(u32, @intFromEnum(b.header.kind), .little);
    try writer.writeInt(u32, b.header.version, .little);
    try writer.writeAll(&b.header.content_hash);
    try writer.writeInt(u64, b.header.timestamp.counter, .little);
    try writer.writeInt(u32, b.header.size, .little);
    try writer.writeInt(u16, b.header.flags, .little);
    try writer.writeByte(b.header.compression_marker);
    try writer.writeAll(b.payload);

    return data;
}

pub fn decodeBlock(allocator: std.mem.Allocator, data: []const u8) !block.StoredBlock {
    const header_size = 32 + 4 + 4 + 32 + 8 + 4 + 2 + 1;
    if (data.len < header_size) return error.MalformedBlock;

    var fbs = std.io.fixedBufferStream(data);
    const reader = fbs.reader();

    var b: block.StoredBlock = undefined;
    
    try reader.readNoEof(&b.header.id.id);
    b.header.kind = @enumFromInt(try reader.readInt(u32, .little));
    b.header.version = try reader.readInt(u32, .little);
    try reader.readNoEof(&b.header.content_hash);
    b.header.timestamp.counter = try reader.readInt(u64, .little);
    b.header.size = try reader.readInt(u32, .little);
    b.header.flags = try reader.readInt(u16, .little);
    b.header.compression_marker = try reader.readByte();
    
    b.payload = try allocator.dupe(u8, data[header_size..]);

    return b;
}
