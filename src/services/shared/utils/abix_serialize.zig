//! ABIX Binary Serializer / Deserializer
//!
//! Adapted from the v2.0 serialize.zig module for the ABI framework.
//! Uses `std.ArrayListUnmanaged(u8)` per Zig 0.16 convention — the allocator
//! is threaded explicitly rather than stored inside the list.
//!
//! Schema-driven binary encoding for the WDBX distributed block exchange.
//! Produces compact, zero-copy-friendly wire format with the following layout:
//!
//!   +----------+----------+---------+----------------------+
//!   | magic(4) | ver(2)   | len(4)  | payload ...          |
//!   +----------+----------+---------+----------------------+
//!
//! Payload encoding:
//!   - Integers: fixed-width little-endian
//!   - Floats: IEEE 754 little-endian
//!   - Slices: u32 length prefix + raw bytes
//!   - Structs: sequential field encoding (reflection-driven)
//!   - Arrays: element count + elements
//!
//! Zero-copy deserialization: `readSlice` returns a borrowed pointer
//! into the input buffer — no allocation for large embedding vectors.

const std = @import("std");

// ---- Wire Format Constants --------------------------------------------------

pub const MAGIC: [4]u8 = .{ 'A', 'B', 'I', 'X' };
pub const VERSION: u16 = 1;
pub const HEADER_SIZE = 10; // magic(4) + version(2) + payload_len(4)

// ---- Writer -----------------------------------------------------------------

/// Appends binary data to a growable buffer. The allocator is stored
/// alongside the unmanaged list so callers do not need to pass it on
/// every write call.
pub const Writer = struct {
    buffer: std.ArrayListUnmanaged(u8) = .{},
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Writer {
        return .{ .allocator = allocator };
    }

    pub fn initCapacity(allocator: std.mem.Allocator, capacity: usize) !Writer {
        var w = Writer{ .allocator = allocator };
        try w.buffer.ensureTotalCapacity(allocator, capacity);
        return w;
    }

    pub fn deinit(self: *Writer) void {
        self.buffer.deinit(self.allocator);
    }

    /// Write the ABIX header. Call this first, then finalize() after payload.
    pub fn writeHeader(self: *Writer) !void {
        try self.writeBytes(&MAGIC);
        try self.writeU16(VERSION);
        try self.writeU32(0); // placeholder for payload length
    }

    /// Patch the payload length in the header. Call after all payload writes.
    pub fn finalize(self: *Writer) void {
        if (self.buffer.items.len >= HEADER_SIZE) {
            const payload_len: u32 = @intCast(self.buffer.items.len - HEADER_SIZE);
            const len_bytes = std.mem.toBytes(std.mem.nativeToLittle(u32, payload_len));
            @memcpy(self.buffer.items[6..10], &len_bytes);
        }
    }

    /// Get the serialized output.
    pub fn getOutput(self: *const Writer) []const u8 {
        return self.buffer.items;
    }

    pub fn reset(self: *Writer) void {
        self.buffer.clearRetainingCapacity();
    }

    // ---- Primitive Writes ---------------------------------------------------

    pub fn writeU8(self: *Writer, val: u8) !void {
        try self.buffer.append(self.allocator, val);
    }

    pub fn writeU16(self: *Writer, val: u16) !void {
        try self.writeBytes(&std.mem.toBytes(std.mem.nativeToLittle(u16, val)));
    }

    pub fn writeU32(self: *Writer, val: u32) !void {
        try self.writeBytes(&std.mem.toBytes(std.mem.nativeToLittle(u32, val)));
    }

    pub fn writeU64(self: *Writer, val: u64) !void {
        try self.writeBytes(&std.mem.toBytes(std.mem.nativeToLittle(u64, val)));
    }

    pub fn writeI32(self: *Writer, val: i32) !void {
        try self.writeBytes(&std.mem.toBytes(std.mem.nativeToLittle(i32, val)));
    }

    pub fn writeI64(self: *Writer, val: i64) !void {
        try self.writeBytes(&std.mem.toBytes(std.mem.nativeToLittle(i64, val)));
    }

    pub fn writeF32(self: *Writer, val: f32) !void {
        try self.writeBytes(&std.mem.toBytes(val));
    }

    pub fn writeF64(self: *Writer, val: f64) !void {
        try self.writeBytes(&std.mem.toBytes(val));
    }

    pub fn writeBool(self: *Writer, val: bool) !void {
        try self.writeU8(if (val) 1 else 0);
    }

    // ---- Composite Writes ---------------------------------------------------

    /// Write a length-prefixed byte slice.
    pub fn writeSlice(self: *Writer, data: []const u8) !void {
        try self.writeU32(@intCast(data.len));
        try self.writeBytes(data);
    }

    /// Write a length-prefixed string (alias for writeSlice).
    pub fn writeString(self: *Writer, str: []const u8) !void {
        try self.writeSlice(str);
    }

    /// Write a typed array: u32 count + raw elements.
    pub fn writeArray(self: *Writer, comptime T: type, items: []const T) !void {
        try self.writeU32(@intCast(items.len));
        const bytes = std.mem.sliceAsBytes(items);
        try self.writeBytes(bytes);
    }

    /// Write raw bytes (no length prefix).
    pub fn writeBytes(self: *Writer, data: []const u8) !void {
        try self.buffer.appendSlice(self.allocator, data);
    }

    /// Serialize a struct via comptime reflection.
    pub fn writeStruct(self: *Writer, comptime T: type, value: T) !void {
        const fields = @typeInfo(T).@"struct".fields;
        inline for (fields) |field| {
            const fval = @field(value, field.name);
            try self.writeField(field.type, fval);
        }
    }

    fn writeField(self: *Writer, comptime T: type, value: T) !void {
        switch (@typeInfo(T)) {
            .bool => try self.writeBool(value),
            .int => |info| {
                switch (info.bits) {
                    8 => try self.writeU8(@bitCast(value)),
                    16 => try self.writeU16(@bitCast(value)),
                    32 => try self.writeU32(@bitCast(value)),
                    64 => try self.writeU64(@bitCast(value)),
                    else => @compileError("Unsupported integer width"),
                }
            },
            .float => |info| {
                switch (info.bits) {
                    32 => try self.writeF32(value),
                    64 => try self.writeF64(value),
                    else => @compileError("Unsupported float width"),
                }
            },
            .pointer => |ptr| {
                if (ptr.size == .slice and ptr.child == u8) {
                    try self.writeSlice(value);
                } else if (ptr.size == .slice) {
                    try self.writeArray(ptr.child, value);
                } else {
                    @compileError("Unsupported pointer type");
                }
            },
            else => @compileError("Unsupported field type: " ++ @typeName(T)),
        }
    }

    pub fn bytesWritten(self: *const Writer) usize {
        return self.buffer.items.len;
    }
};

// ---- Reader -----------------------------------------------------------------

/// Zero-copy reader over a byte buffer. Slice reads return borrowed
/// pointers into the original buffer — no allocation needed.
pub const Reader = struct {
    data: []const u8,
    pos: usize = 0,

    pub fn init(data: []const u8) Reader {
        return .{ .data = data };
    }

    /// Validate and skip past the ABIX header. Returns payload length.
    pub fn readHeader(self: *Reader) !u32 {
        if (self.remaining() < HEADER_SIZE) return error.BufferTooSmall;

        const magic = self.data[self.pos..][0..4];
        if (!std.mem.eql(u8, magic, &MAGIC)) return error.InvalidMagic;
        self.pos += 4;

        const version = self.readU16() catch return error.BufferTooSmall;
        if (version > VERSION) return error.UnsupportedVersion;

        const payload_len = self.readU32() catch return error.BufferTooSmall;
        if (payload_len > self.remaining()) return error.BufferTooSmall;
        return payload_len;
    }

    // ---- Primitive Reads ----------------------------------------------------

    pub fn readU8(self: *Reader) !u8 {
        if (self.pos >= self.data.len) return error.BufferTooSmall;
        const val = self.data[self.pos];
        self.pos += 1;
        return val;
    }

    pub fn readU16(self: *Reader) !u16 {
        if (self.pos + 2 > self.data.len) return error.BufferTooSmall;
        const val = std.mem.readInt(u16, self.data[self.pos..][0..2], .little);
        self.pos += 2;
        return val;
    }

    pub fn readU32(self: *Reader) !u32 {
        if (self.pos + 4 > self.data.len) return error.BufferTooSmall;
        const val = std.mem.readInt(u32, self.data[self.pos..][0..4], .little);
        self.pos += 4;
        return val;
    }

    pub fn readU64(self: *Reader) !u64 {
        if (self.pos + 8 > self.data.len) return error.BufferTooSmall;
        const val = std.mem.readInt(u64, self.data[self.pos..][0..8], .little);
        self.pos += 8;
        return val;
    }

    pub fn readI32(self: *Reader) !i32 {
        return @bitCast(try self.readU32());
    }

    pub fn readI64(self: *Reader) !i64 {
        return @bitCast(try self.readU64());
    }

    pub fn readF32(self: *Reader) !f32 {
        if (self.pos + 4 > self.data.len) return error.BufferTooSmall;
        const bytes = self.data[self.pos..][0..4];
        self.pos += 4;
        return @bitCast(bytes.*);
    }

    pub fn readF64(self: *Reader) !f64 {
        if (self.pos + 8 > self.data.len) return error.BufferTooSmall;
        const bytes = self.data[self.pos..][0..8];
        self.pos += 8;
        return @bitCast(bytes.*);
    }

    pub fn readBool(self: *Reader) !bool {
        return (try self.readU8()) != 0;
    }

    // ---- Composite Reads ----------------------------------------------------

    /// Read a length-prefixed byte slice (zero-copy -- borrows from input buffer).
    pub fn readSlice(self: *Reader) ![]const u8 {
        const len: usize = try self.readU32();
        if (len > self.remaining()) return error.BufferTooSmall;
        const result = self.data[self.pos .. self.pos + len];
        self.pos += len;
        return result;
    }

    /// Read a length-prefixed string (alias for readSlice).
    pub fn readString(self: *Reader) ![]const u8 {
        return self.readSlice();
    }

    /// Read a typed f32 array (zero-copy for the element data).
    pub fn readArrayF32(self: *Reader) ![]const f32 {
        const count: usize = try self.readU32();
        // Validate count against remaining bytes to prevent overflow in multiplication
        if (count > self.remaining() / @sizeOf(f32)) return error.BufferTooSmall;
        const byte_len = count * @sizeOf(f32);
        if (self.pos % @alignOf(f32) != 0) return error.InvalidAlignment;
        const bytes = self.data[self.pos .. self.pos + byte_len];
        self.pos += byte_len;
        return std.mem.bytesAsSlice(f32, @alignCast(bytes));
    }

    /// Deserialize a struct via comptime reflection.
    pub fn readStruct(self: *Reader, comptime T: type) !T {
        var result: T = undefined;
        const fields = @typeInfo(T).@"struct".fields;
        inline for (fields) |field| {
            @field(result, field.name) = try self.readField(field.type);
        }
        return result;
    }

    fn readField(self: *Reader, comptime T: type) !T {
        return switch (@typeInfo(T)) {
            .bool => self.readBool(),
            .int => |info| switch (info.bits) {
                8 => @bitCast(try self.readU8()),
                16 => @bitCast(try self.readU16()),
                32 => @bitCast(try self.readU32()),
                64 => @bitCast(try self.readU64()),
                else => @compileError("Unsupported integer width"),
            },
            .float => |info| switch (info.bits) {
                32 => self.readF32(),
                64 => self.readF64(),
                else => @compileError("Unsupported float width"),
            },
            else => @compileError("Unsupported field type for auto-deserialization: " ++ @typeName(T)),
        };
    }

    pub fn remaining(self: *const Reader) usize {
        return if (self.pos < self.data.len) self.data.len - self.pos else 0;
    }

    pub fn isComplete(self: *const Reader) bool {
        return self.pos >= self.data.len;
    }
};

// ---- Message Builder (High-Level API) ---------------------------------------

/// Convenience wrapper that handles header/finalize automatically.
pub const MessageBuilder = struct {
    writer: Writer,

    pub fn init(allocator: std.mem.Allocator) !MessageBuilder {
        var mb = MessageBuilder{ .writer = try Writer.initCapacity(allocator, 256) };
        try mb.writer.writeHeader();
        return mb;
    }

    pub fn deinit(self: *MessageBuilder) void {
        self.writer.deinit();
    }

    /// Finalize and return the serialized message bytes.
    pub fn finish(self: *MessageBuilder) []const u8 {
        self.writer.finalize();
        return self.writer.getOutput();
    }

    // Forward all write methods to the underlying Writer.

    pub fn writeU8(self: *MessageBuilder, v: u8) !void {
        try self.writer.writeU8(v);
    }

    pub fn writeU16(self: *MessageBuilder, v: u16) !void {
        try self.writer.writeU16(v);
    }

    pub fn writeU32(self: *MessageBuilder, v: u32) !void {
        try self.writer.writeU32(v);
    }

    pub fn writeU64(self: *MessageBuilder, v: u64) !void {
        try self.writer.writeU64(v);
    }

    pub fn writeF32(self: *MessageBuilder, v: f32) !void {
        try self.writer.writeF32(v);
    }

    pub fn writeF64(self: *MessageBuilder, v: f64) !void {
        try self.writer.writeF64(v);
    }

    pub fn writeString(self: *MessageBuilder, s: []const u8) !void {
        try self.writer.writeString(s);
    }

    pub fn writeArray(self: *MessageBuilder, comptime T: type, items: []const T) !void {
        try self.writer.writeArray(T, items);
    }

    pub fn writeStruct(self: *MessageBuilder, comptime T: type, value: T) !void {
        try self.writer.writeStruct(T, value);
    }
};

// ---- Tests ------------------------------------------------------------------

test "Writer/Reader round-trip primitives" {
    const alloc = std.testing.allocator;
    var w = Writer.init(alloc);
    defer w.deinit();

    try w.writeU8(0xAB);
    try w.writeU16(0x1234);
    try w.writeU32(0xDEADBEEF);
    try w.writeU64(0x0102030405060708);
    try w.writeI32(-42);
    try w.writeF32(3.14);
    try w.writeBool(true);
    try w.writeBool(false);

    var r = Reader.init(w.getOutput());
    try std.testing.expectEqual(@as(u8, 0xAB), try r.readU8());
    try std.testing.expectEqual(@as(u16, 0x1234), try r.readU16());
    try std.testing.expectEqual(@as(u32, 0xDEADBEEF), try r.readU32());
    try std.testing.expectEqual(@as(u64, 0x0102030405060708), try r.readU64());
    try std.testing.expectEqual(@as(i32, -42), try r.readI32());
    try std.testing.expectApproxEqAbs(@as(f32, 3.14), try r.readF32(), 0.001);
    try std.testing.expect(try r.readBool());
    try std.testing.expect(!(try r.readBool()));
    try std.testing.expect(r.isComplete());
}

test "Writer/Reader round-trip slices and strings" {
    const alloc = std.testing.allocator;
    var w = Writer.init(alloc);
    defer w.deinit();

    try w.writeString("hello world");
    try w.writeSlice(&[_]u8{ 0xDE, 0xAD });

    var r = Reader.init(w.getOutput());
    const s = try r.readString();
    try std.testing.expectEqualStrings("hello world", s);
    const sl = try r.readSlice();
    try std.testing.expectEqual(@as(usize, 2), sl.len);
    try std.testing.expectEqual(@as(u8, 0xDE), sl[0]);
    try std.testing.expect(r.isComplete());
}

test "header write/read and finalize" {
    const alloc = std.testing.allocator;
    var w = Writer.init(alloc);
    defer w.deinit();

    try w.writeHeader();
    try w.writeU32(42);
    try w.writeString("test");
    w.finalize();

    var r = Reader.init(w.getOutput());
    const payload_len = try r.readHeader();
    // payload = u32(42) + u32(len=4) + "test" = 4 + 4 + 4 = 12
    try std.testing.expectEqual(@as(u32, 12), payload_len);
    try std.testing.expectEqual(@as(u32, 42), try r.readU32());
    try std.testing.expectEqualStrings("test", try r.readString());
}

test "Reader errors on truncated data" {
    var r = Reader.init(&[_]u8{0x01});
    try std.testing.expectEqual(@as(u8, 0x01), try r.readU8());
    try std.testing.expectError(error.BufferTooSmall, r.readU8());

    var r2 = Reader.init(&[_]u8{ 0x01, 0x02 });
    try std.testing.expectError(error.BufferTooSmall, r2.readU32());
}

test "Reader rejects invalid magic" {
    var bad_header = [_]u8{ 'X', 'Y', 'Z', 'W', 0x01, 0x00, 0x00, 0x00, 0x00, 0x00 };
    var r = Reader.init(&bad_header);
    try std.testing.expectError(error.InvalidMagic, r.readHeader());
}

test "MessageBuilder convenience API" {
    const alloc = std.testing.allocator;
    var mb = try MessageBuilder.init(alloc);
    defer mb.deinit();

    try mb.writeU32(100);
    try mb.writeString("payload");
    const data = mb.finish();

    var r = Reader.init(data);
    _ = try r.readHeader();
    try std.testing.expectEqual(@as(u32, 100), try r.readU32());
    try std.testing.expectEqualStrings("payload", try r.readString());
}

test "struct round-trip via reflection" {
    const TestStruct = struct {
        x: u32,
        y: i32,
        z: f32,
        flag: bool,
    };

    const alloc = std.testing.allocator;
    var w = Writer.init(alloc);
    defer w.deinit();

    const original = TestStruct{ .x = 42, .y = -7, .z = 2.5, .flag = true };
    try w.writeStruct(TestStruct, original);

    var r = Reader.init(w.getOutput());
    const decoded = try r.readStruct(TestStruct);
    try std.testing.expectEqual(original.x, decoded.x);
    try std.testing.expectEqual(original.y, decoded.y);
    try std.testing.expectApproxEqAbs(original.z, decoded.z, 0.001);
    try std.testing.expectEqual(original.flag, decoded.flag);
}

test "Writer reset and reuse" {
    const alloc = std.testing.allocator;
    var w = Writer.init(alloc);
    defer w.deinit();

    try w.writeU32(1);
    try std.testing.expectEqual(@as(usize, 4), w.bytesWritten());
    w.reset();
    try std.testing.expectEqual(@as(usize, 0), w.bytesWritten());
    try w.writeU8(2);
    try std.testing.expectEqual(@as(usize, 1), w.bytesWritten());
}

test {
    std.testing.refAllDecls(@This());
}
