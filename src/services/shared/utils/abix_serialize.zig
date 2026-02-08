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
        if (self.pos + len > self.data.len) return error.BufferTooSmall;
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
        const byte_len = count * @sizeOf(f32);
        if (self.pos + byte_len > self.data.len) return error.BufferTooSmall;
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
