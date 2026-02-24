//! Binary serialization utilities.
//!
//! Provides helpers for reading and writing binary data in a structured way.

const std = @import("std");

/// Writer for building binary data incrementally.
pub const SerializationWriter = struct {
    allocator: std.mem.Allocator,
    buffer: std.ArrayListUnmanaged(u8),

    pub fn init(allocator: std.mem.Allocator) SerializationWriter {
        return .{
            .allocator = allocator,
            .buffer = .{},
        };
    }

    pub fn deinit(self: *SerializationWriter) void {
        self.buffer.deinit(self.allocator);
    }

    /// Append raw bytes to the buffer.
    pub fn appendBytes(self: *SerializationWriter, bytes: []const u8) !void {
        try self.buffer.appendSlice(self.allocator, bytes);
    }

    /// Append an integer value in little-endian format.
    pub fn appendInt(self: *SerializationWriter, comptime T: type, value: T) !void {
        const bytes = std.mem.asBytes(&std.mem.nativeToLittle(T, value));
        try self.buffer.appendSlice(self.allocator, bytes);
    }

    /// Append a float value.
    pub fn appendFloat(self: *SerializationWriter, comptime T: type, value: T) !void {
        const IntType = std.meta.Int(.unsigned, @bitSizeOf(T));
        try self.appendInt(IntType, @bitCast(value));
    }

    /// Get the accumulated data as an owned slice.
    pub fn toOwnedSlice(self: *SerializationWriter) ![]u8 {
        return try self.buffer.toOwnedSlice(self.allocator);
    }

    /// Get the current length of accumulated data.
    pub fn len(self: *const SerializationWriter) usize {
        return self.buffer.items.len;
    }
};

/// Cursor for reading binary data.
pub const SerializationCursor = struct {
    data: []const u8,
    pos: usize,

    pub const Error = error{
        EndOfData,
    };

    pub fn init(data: []const u8) SerializationCursor {
        return .{
            .data = data,
            .pos = 0,
        };
    }

    /// Read a fixed number of bytes.
    pub fn readBytes(self: *SerializationCursor, n: usize) Error![]const u8 {
        if (self.pos + n > self.data.len) return Error.EndOfData;
        const result = self.data[self.pos .. self.pos + n];
        self.pos += n;
        return result;
    }

    /// Read an integer value in little-endian format.
    pub fn readInt(self: *SerializationCursor, comptime T: type) Error!T {
        const size = @sizeOf(T);
        if (self.pos + size > self.data.len) return Error.EndOfData;
        const bytes = self.data[self.pos..][0..size];
        self.pos += size;
        return std.mem.littleToNative(T, @as(*align(1) const T, @ptrCast(bytes)).*);
    }

    /// Read a float value.
    pub fn readFloat(self: *SerializationCursor, comptime T: type) Error!T {
        const IntType = std.meta.Int(.unsigned, @bitSizeOf(T));
        const bits = try self.readInt(IntType);
        return @bitCast(bits);
    }

    /// Check if there's more data to read.
    pub fn hasRemaining(self: *const SerializationCursor) bool {
        return self.pos < self.data.len;
    }

    /// Get remaining bytes count.
    pub fn remaining(self: *const SerializationCursor) usize {
        return self.data.len - self.pos;
    }

    /// Skip n bytes.
    pub fn skip(self: *SerializationCursor, n: usize) Error!void {
        if (self.pos + n > self.data.len) return Error.EndOfData;
        self.pos += n;
    }
};

test "serialization writer basic" {
    const allocator = std.testing.allocator;
    var writer = SerializationWriter.init(allocator);
    defer writer.deinit();

    try writer.appendBytes("MAGIC");
    try writer.appendInt(u16, 42);
    try writer.appendInt(u32, 0xDEADBEEF);

    const data = try writer.toOwnedSlice();
    defer allocator.free(data);

    try std.testing.expectEqual(@as(usize, 11), data.len);
    try std.testing.expectEqualSlices(u8, "MAGIC", data[0..5]);
}

test "serialization cursor basic" {
    var data: [11]u8 = undefined;
    @memcpy(data[0..5], "MAGIC");
    std.mem.writeInt(u16, data[5..7], 42, .little);
    std.mem.writeInt(u32, data[7..11], 0xDEADBEEF, .little);

    var cursor = SerializationCursor.init(&data);

    const magic = try cursor.readBytes(5);
    try std.testing.expectEqualSlices(u8, "MAGIC", magic);

    const short = try cursor.readInt(u16);
    try std.testing.expectEqual(@as(u16, 42), short);

    const int = try cursor.readInt(u32);
    try std.testing.expectEqual(@as(u32, 0xDEADBEEF), int);

    try std.testing.expect(!cursor.hasRemaining());
}

test {
    std.testing.refAllDecls(@This());
}
