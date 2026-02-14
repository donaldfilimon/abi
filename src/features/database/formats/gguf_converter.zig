//! GGUF Format Converter
//!
//! Bidirectional conversion between ABI Unified format and GGUF.
//! Supports all GGUF v2/v3 features including quantization types.

const std = @import("std");
const unified = @import("unified.zig");
const mod = @import("mod.zig");

pub const GgufConversionError = error{
    InvalidMagic,
    UnsupportedVersion,
    InvalidMetadataType,
    InvalidTensorType,
    ParseError,
    OutOfMemory,
    IoError,
};

/// GGUF magic number
pub const GGUF_MAGIC: u32 = 0x46554747; // "GGUF"

/// GGUF file header
pub const GgufHeader = extern struct {
    magic: u32,
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
};

/// GGUF metadata value types
pub const GgufMetadataType = enum(u32) {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    bool_ = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,
};

/// GGUF tensor types
pub const GgufTensorType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    iq1_m = 29,
    bf16 = 30,
    tq1_0 = 34,
    tq2_0 = 35,
    mxfp4 = 39,

    pub fn fromInt(value: u32) ?GgufTensorType {
        return switch (value) {
            0 => .f32,
            1 => .f16,
            2 => .q4_0,
            3 => .q4_1,
            6 => .q5_0,
            7 => .q5_1,
            8 => .q8_0,
            9 => .q8_1,
            10 => .q2_k,
            11 => .q3_k,
            12 => .q4_k,
            13 => .q5_k,
            14 => .q6_k,
            15 => .q8_k,
            24 => .i8,
            25 => .i16,
            26 => .i32,
            27 => .i64,
            28 => .f64,
            29 => .iq1_m,
            30 => .bf16,
            34 => .tq1_0,
            35 => .tq2_0,
            39 => .mxfp4,
            else => null,
        };
    }

    pub fn toUnified(self: GgufTensorType) unified.DataType {
        return switch (self) {
            .f32 => .f32,
            .f16 => .f16,
            .bf16 => .bf16,
            .f64 => .f64,
            .i8 => .i8,
            .i16 => .i16,
            .i32 => .i32,
            .i64 => .i64,
            .q4_0 => .q4_0,
            .q4_1 => .q4_1,
            .q5_0 => .q5_0,
            .q5_1 => .q5_1,
            .q8_0 => .q8_0,
            .q8_1 => .q8_1,
            .q2_k => .q2_k,
            .q3_k => .q3_k,
            .q4_k => .q4_k,
            .q5_k => .q5_k,
            .q6_k => .q6_k,
            .q8_k => .q8_k,
            .iq1_m, .tq1_0, .tq2_0, .mxfp4 => .custom,
        };
    }

    pub fn fromUnified(dtype: unified.DataType) ?GgufTensorType {
        return switch (dtype) {
            .f32 => .f32,
            .f16 => .f16,
            .bf16 => .bf16,
            .f64 => .f64,
            .i8 => .i8,
            .i16 => .i16,
            .i32 => .i32,
            .i64 => .i64,
            .q4_0 => .q4_0,
            .q4_1 => .q4_1,
            .q5_0 => .q5_0,
            .q5_1 => .q5_1,
            .q8_0 => .q8_0,
            .q8_1 => .q8_1,
            .q2_k => .q2_k,
            .q3_k => .q3_k,
            .q4_k => .q4_k,
            .q5_k => .q5_k,
            .q6_k => .q6_k,
            .q8_k => .q8_k,
            else => null,
        };
    }

    pub fn bytesPerBlock(self: GgufTensorType) usize {
        return switch (self) {
            .f32 => 4,
            .f16, .bf16 => 2,
            .f64 => 8,
            .i8 => 1,
            .i16 => 2,
            .i32 => 4,
            .i64 => 8,
            .q4_0 => 18,
            .q4_1 => 20,
            .q5_0 => 22,
            .q5_1 => 24,
            .q8_0 => 34,
            .q8_1 => 36,
            .q2_k => 84,
            .q3_k => 110,
            .q4_k => 144,
            .q5_k => 176,
            .q6_k => 210,
            .q8_k => 292,
            .iq1_m => 56,
            .tq1_0 => 54,
            .tq2_0 => 66,
            .mxfp4 => 17,
        };
    }

    pub fn blockSize(self: GgufTensorType) usize {
        return switch (self) {
            .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1 => 32,
            .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q8_k => 256,
            .mxfp4 => 32,
            .iq1_m, .tq1_0, .tq2_0 => 256,
            else => 1,
        };
    }
};

/// GGUF default alignment
pub const GGUF_ALIGNMENT: usize = 32;

/// Convert GGUF data to Unified format
pub fn fromGguf(allocator: std.mem.Allocator, data: []const u8) GgufConversionError!unified.UnifiedFormat {
    if (data.len < @sizeOf(GgufHeader)) return error.ParseError;

    // Read header
    const header: *const GgufHeader = @ptrCast(@alignCast(data.ptr));
    if (header.magic != GGUF_MAGIC) return error.InvalidMagic;
    if (header.version < 2 or header.version > 3) return error.UnsupportedVersion;

    var result = unified.UnifiedFormat.init(allocator);
    errdefer result.deinit();

    var cursor: usize = @sizeOf(GgufHeader);

    // Parse metadata
    for (0..header.metadata_kv_count) |_| {
        const key_len = readU64(data, &cursor) orelse return error.ParseError;
        if (cursor + key_len > data.len) return error.ParseError;
        const key = data[cursor..][0..key_len];
        cursor += key_len;

        const value_type = readU32(data, &cursor) orelse return error.ParseError;
        // Skip value based on type
        cursor = skipMetadataValue(data, cursor, value_type) orelse return error.ParseError;

        // Store key for reference
        _ = key;
    }

    // Parse tensor info
    var tensor_infos = std.ArrayListUnmanaged(TensorInfo).empty;
    defer tensor_infos.deinit(allocator);

    for (0..header.tensor_count) |_| {
        const name_len = readU64(data, &cursor) orelse return error.ParseError;
        if (cursor + name_len > data.len) return error.ParseError;
        const name = data[cursor..][0..name_len];
        cursor += name_len;

        const n_dims = readU32(data, &cursor) orelse return error.ParseError;
        if (n_dims > 4) return error.ParseError;

        var dims: [4]u64 = .{ 1, 1, 1, 1 };
        for (0..n_dims) |i| {
            dims[i] = readU64(data, &cursor) orelse return error.ParseError;
        }

        const tensor_type_int = readU32(data, &cursor) orelse return error.ParseError;
        const tensor_type = GgufTensorType.fromInt(tensor_type_int) orelse return error.InvalidTensorType;

        const offset = readU64(data, &cursor) orelse return error.ParseError;

        tensor_infos.append(allocator, .{
            .name = name,
            .n_dims = @intCast(n_dims),
            .dims = dims,
            .tensor_type = tensor_type,
            .offset = offset,
        }) catch return error.OutOfMemory;
    }

    // Align to data section
    cursor = alignUp(cursor, GGUF_ALIGNMENT);
    const data_start = cursor;

    // Add tensors to result
    for (tensor_infos.items) |info| {
        var elem_count: u64 = 1;
        for (0..info.n_dims) |i| {
            elem_count *= info.dims[i];
        }

        const bs = info.tensor_type.blockSize();
        const num_blocks = (elem_count + bs - 1) / bs;
        const byte_size = num_blocks * info.tensor_type.bytesPerBlock();

        const desc = unified.TensorDescriptor{
            .name_hash = std.hash.Wyhash.hash(0, info.name),
            .name_offset = 0,
            .name_length = @intCast(info.name.len),
            .data_type = info.tensor_type.toUnified(),
            .n_dims = info.n_dims,
            .dims = info.dims,
            .data_offset = @intCast(data_start + info.offset),
            .data_size = byte_size,
        };

        result.tensors.put(allocator, info.name, desc) catch return error.OutOfMemory;
    }

    result.data = data;
    result.owns_data = false;

    return result;
}

/// Convert Unified format to GGUF
pub fn toGguf(allocator: std.mem.Allocator, source: *const unified.UnifiedFormat) GgufConversionError![]u8 {
    var output = std.ArrayListUnmanaged(u8).empty;
    errdefer output.deinit(allocator);

    // Count valid tensors (only GGUF-compatible types)
    var tensor_count: u64 = 0;
    var tensor_iter = source.tensors.iterator();
    while (tensor_iter.next()) |entry| {
        if (GgufTensorType.fromUnified(entry.value_ptr.data_type) != null) {
            tensor_count += 1;
        }
    }

    // Write header
    var header = GgufHeader{
        .magic = GGUF_MAGIC,
        .version = 3,
        .tensor_count = tensor_count,
        .metadata_kv_count = 0,
    };
    output.appendSlice(allocator, std.mem.asBytes(&header)) catch return error.OutOfMemory;

    // Write tensor info
    var data_offset: u64 = 0;
    tensor_iter = source.tensors.iterator();
    while (tensor_iter.next()) |entry| {
        const name = entry.key_ptr.*;
        const desc = entry.value_ptr.*;

        const gguf_type = GgufTensorType.fromUnified(desc.data_type) orelse continue;

        // Write name length and name
        writeU64(&output, allocator, name.len) catch return error.OutOfMemory;
        output.appendSlice(allocator, name) catch return error.OutOfMemory;

        // Write n_dims
        writeU32(&output, allocator, desc.n_dims) catch return error.OutOfMemory;

        // Write dims
        for (0..desc.n_dims) |i| {
            writeU64(&output, allocator, desc.dims[i]) catch return error.OutOfMemory;
        }

        // Write type
        writeU32(&output, allocator, @intFromEnum(gguf_type)) catch return error.OutOfMemory;

        // Write offset
        writeU64(&output, allocator, data_offset) catch return error.OutOfMemory;

        data_offset += desc.data_size;
    }

    // Align to data section
    const current_len = output.items.len;
    const aligned_len = alignUp(current_len, GGUF_ALIGNMENT);
    for (current_len..aligned_len) |_| {
        output.append(allocator, 0) catch return error.OutOfMemory;
    }

    // Write tensor data
    tensor_iter = source.tensors.iterator();
    while (tensor_iter.next()) |entry| {
        const name = entry.key_ptr.*;
        const desc = entry.value_ptr.*;

        if (GgufTensorType.fromUnified(desc.data_type) == null) continue;

        const tensor_data = source.getTensorData(allocator, name) catch continue;
        defer if (desc.compressed_size > 0) allocator.free(tensor_data);

        output.appendSlice(allocator, tensor_data) catch return error.OutOfMemory;
    }

    return output.toOwnedSlice(allocator) catch return error.OutOfMemory;
}

const TensorInfo = struct {
    name: []const u8,
    n_dims: u8,
    dims: [4]u64,
    tensor_type: GgufTensorType,
    offset: u64,
};

fn readU32(data: []const u8, cursor: *usize) ?u32 {
    if (cursor.* + 4 > data.len) return null;
    const result = std.mem.readInt(u32, data[cursor.*..][0..4], .little);
    cursor.* += 4;
    return result;
}

fn readU64(data: []const u8, cursor: *usize) ?u64 {
    if (cursor.* + 8 > data.len) return null;
    const result = std.mem.readInt(u64, data[cursor.*..][0..8], .little);
    cursor.* += 8;
    return result;
}

fn writeU32(output: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, value: u32) !void {
    var bytes: [4]u8 = undefined;
    std.mem.writeInt(u32, &bytes, value, .little);
    try output.appendSlice(allocator, &bytes);
}

fn writeU64(output: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, value: u64) !void {
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, value, .little);
    try output.appendSlice(allocator, &bytes);
}

fn skipMetadataValue(data: []const u8, start: usize, value_type: u32) ?usize {
    var cursor = start;
    switch (value_type) {
        0, 1, 7 => cursor += 1, // u8, i8, bool
        2, 3 => cursor += 2, // u16, i16
        4, 5, 6 => cursor += 4, // u32, i32, f32
        10, 11, 12 => cursor += 8, // u64, i64, f64
        8 => { // string
            const len = readU64(data, &cursor) orelse return null;
            cursor += len;
        },
        9 => { // array
            const elem_type = readU32(data, &cursor) orelse return null;
            const count = readU64(data, &cursor) orelse return null;
            for (0..count) |_| {
                cursor = skipMetadataValue(data, cursor, elem_type) orelse return null;
            }
        },
        else => return null,
    }
    return cursor;
}

fn alignUp(value: usize, alignment: usize) usize {
    return (value + alignment - 1) & ~(alignment - 1);
}

test "gguf type conversion" {
    try std.testing.expectEqual(unified.DataType.f32, GgufTensorType.f32.toUnified());
    try std.testing.expectEqual(unified.DataType.q4_0, GgufTensorType.q4_0.toUnified());
    try std.testing.expectEqual(GgufTensorType.f32, GgufTensorType.fromUnified(.f32).?);
    try std.testing.expectEqual(GgufTensorType.q8_0, GgufTensorType.fromUnified(.q8_0).?);
}
