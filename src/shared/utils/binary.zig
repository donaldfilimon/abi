//! Binary serialization utilities.
//!
//! Provides common binary serialization primitives used across database,
//! network, and other modules. Uses packed structs for optimal memory layout.

const std = @import("std");

/// Binary cursor for reading serialized data.
pub const SerializationCursor = struct {
    data: []const u8,
    index: usize = 0,

    pub fn init(data: []const u8) SerializationCursor {
        return .{ .data = data, .index = 0 };
    }

    /// Read bytes from cursor.
    pub fn readBytes(self: *SerializationCursor, len: usize) ![]const u8 {
        if (self.index + len > self.data.len) {
            return error.OutOfBounds;
        }
        const result = self.data[self.index..][0..len];
        self.index += len;
        return result;
    }

    /// Read an integer value from cursor.
    pub fn readInt(self: *SerializationCursor, comptime T: type) !T {
        comptime {
            if (!std.meta.trait.isUnsignedInt(T)) {
                @compileError("readInt only supports unsigned integer types");
            }
            std.debug.assert(@sizeOf(T) <= 8); // Reasonable size limit
        }
        const size = @sizeOf(T);
        if (self.index + size > self.data.len) {
            return error.OutOfBounds;
        }
        const bytes = self.data[self.index..][0..size];
        self.index += size;
        return std.mem.readInt(T, bytes, .little);
    }

    /// Read a slice length (as u32) followed by slice.
    pub fn readSlice(self: *SerializationCursor) ![]const u8 {
        const len = try self.readInt(u32);
        return try self.readBytes(len);
    }

    /// Read a packed struct directly.
    pub fn readStruct(self: *SerializationCursor, comptime T: type) !T {
        const size = @sizeOf(T);
        if (self.index + size > self.data.len) {
            return error.OutOfBounds;
        }
        const bytes = self.data[self.index..][0..size];
        self.index += size;
        return std.mem.bytesAsValue(T, bytes[0..size]);
    }

    /// Check if cursor is at end of data.
    pub fn isAtEnd(self: *const SerializationCursor) bool {
        return self.index >= self.data.len;
    }

    /// Get remaining bytes.
    pub fn remaining(self: *const SerializationCursor) []const u8 {
        return self.data[self.index..];
    }
};

/// Binary writer for serializing data.
pub const SerializationWriter = struct {
    buffer: std.ArrayList(u8),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) SerializationWriter {
        return .{
            .buffer = std.ArrayList(u8).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SerializationWriter) void {
        self.buffer.deinit();
        self.* = undefined;
    }

    /// Append bytes to buffer.
    pub fn appendBytes(self: *SerializationWriter, data: []const u8) !void {
        try self.buffer.appendSlice(data);
    }

    /// Append an integer value.
    pub fn appendInt(self: *SerializationWriter, comptime T: type, value: T) !void {
        comptime {
            if (!std.meta.trait.isUnsignedInt(T)) {
                @compileError("appendInt only supports unsigned integer types");
            }
        }
        const size = @sizeOf(T);
        const start = self.buffer.items.len;
        try self.buffer.resize(start + size);
        std.mem.writeInt(T, self.buffer.items[start..][0..size], value, .little);
    }

    /// Append a slice with length prefix (u32).
    pub fn appendSlice(self: *SerializationWriter, data: []const u8) !void {
        try self.writeInt(u32, @intCast(data.len));
        try self.appendBytes(data);
    }

    /// Append a string with length prefix (u32).
    pub fn appendString(self: *SerializationWriter, data: []const u8) !void {
        try self.appendSlice(data);
    }

    /// Append a packed struct directly.
    pub fn appendStruct(self: *SerializationWriter, comptime T: type, value: T) !void {
        comptime std.mem.validateStruct(T);
        const size = @sizeOf(T);
        const start = self.buffer.items.len;
        try self.buffer.resize(start + size);
        std.mem.bytesAsValue([size]u8, self.buffer.items[start..][0..size], value);
    }

    /// Get serialized bytes.
    pub fn toOwnedSlice(self: *SerializationWriter) ![]const u8 {
        return self.buffer.toOwnedSlice();
    }

    /// Get current buffer length without copying.
    pub fn len(self: *const SerializationWriter) usize {
        return self.buffer.items.len;
    }

    /// Clear the buffer while keeping capacity.
    pub fn clear(self: *SerializationWriter) void {
        self.buffer.clearRetainingCapacity();
    }
};

test "SerializationCursor reads data correctly" {
    const data = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var cursor = SerializationCursor.init(&data);

    try std.testing.expectEqual(@as(u8, 1), try cursor.readInt(u8));
    try std.testing.expectEqual(@as(u16, 0x0504), try cursor.readInt(u16));
    try std.testing.expectEqual(@as(u32, 0x08070605), try cursor.readInt(u32));
    try std.testing.expect(cursor.isAtEnd());
}

test "SerializationCursor reads packed structs" {
    const TestStruct = packed struct { a: u8, b: u16, c: u32 };
    const expected = TestStruct{ .a = 1, .b = 0x0201, .c = 0x05040302 };
    const bytes = std.mem.asBytes(&expected);
    var cursor = SerializationCursor.init(bytes);

    const result = try cursor.readStruct(TestStruct);
    try std.testing.expectEqual(expected.a, result.a);
    try std.testing.expectEqual(expected.b, result.b);
    try std.testing.expectEqual(expected.c, result.c);
    try std.testing.expect(cursor.isAtEnd());
}

test "SerializationCursor reads slices correctly" {
    const len_bytes = [_]u8{ 0x03, 0x00, 0x00, 0x00 };
    const payload = "ABC";
    const data = &([_]u8{ len_bytes[0], len_bytes[1], len_bytes[2], len_bytes[3] }) ++ payload;
    var cursor = SerializationCursor.init(data);

    const len = try cursor.readInt(u32);
    try std.testing.expectEqual(@as(u32, 3), len);
    const slice = try cursor.readBytes(3);
    try std.testing.expectEqualSlices(u8, payload, slice);
    try std.testing.expect(cursor.isAtEnd());
}

test "SerializationWriter writes data correctly" {
    var writer = try SerializationWriter.init(std.testing.allocator);
    defer writer.deinit();

    try writer.appendInt(u8, 1);
    try writer.appendInt(u16, 0x0201);
    try writer.appendInt(u32, 0x05040302);

    const result = try writer.toOwnedSlice();
    defer std.testing.allocator.free(result);
    try std.testing.expectEqualSlices(u8, result, &[_]u8{ 1, 2, 1, 2, 1, 2 });
}

test "SerializationWriter writes packed structs" {
    const TestStruct = packed struct { a: u8, b: u16, c: u32 };
    const value = TestStruct{ .a = 1, .b = 0x0201, .c = 0x05040302 };
    var writer = try SerializationWriter.init(std.testing.allocator);
    defer writer.deinit();

    try writer.appendStruct(TestStruct, value);

    const result = try writer.toOwnedSlice();
    defer std.testing.allocator.free(result);
    const expected_bytes = std.mem.asBytes(&value);
    try std.testing.expectEqualSlices(u8, expected_bytes, result);
}

test "SerializationWriter writes slices correctly" {
    var writer = try SerializationWriter.init(std.testing.allocator);
    defer writer.deinit();

    try writer.appendSlice("ABC");

    const result = try writer.toOwnedSlice();
    defer std.testing.allocator.free(result);
    try std.testing.expectEqualSlices(u8, result, &[_]u8{ 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 'A', 'B', 'C' });
}

test "round trip serialization" {
    var writer = try SerializationWriter.init(std.testing.allocator);
    defer writer.deinit();

    const original_string = "Hello, world!";
    const original_int: u64 = 42;

    try writer.appendString(original_string);
    try writer.appendInt(u64, original_int);

    const data = try writer.toOwnedSlice();
    defer std.testing.allocator.free(data);

    var cursor = SerializationCursor.init(data);
    const string_len = try cursor.readInt(u32);
    try std.testing.expectEqual(@as(u32, 13), string_len);
    const read_string = try cursor.readBytes(13);
    const read_int = try cursor.readInt(u64);

    try std.testing.expectEqualStrings(original_string, read_string);
    try std.testing.expectEqual(@as(u64, 42), read_int);
}

test "SerializationWriter clear and len" {
    var writer = try SerializationWriter.init(std.testing.allocator);
    defer writer.deinit();

    try writer.appendSlice("test");
    const len1 = writer.len();
    try std.testing.expect(len1 > 0);

    writer.clear();
    const len2 = writer.len();
    try std.testing.expectEqual(@as(usize, 0), len2);
}
