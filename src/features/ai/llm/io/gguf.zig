//! GGUF file format parser for loading llama.cpp-compatible models.
//!
//! GGUF (GGML Universal Format) is the standard format for quantized LLMs.
//! This parser supports GGUF v2 and v3 formats.
//!
//! File structure:
//! 1. Header (magic, version, tensor count, metadata count)
//! 2. Metadata key-value pairs (model config, tokenizer vocab, etc.)
//! 3. Tensor info table (names, shapes, types, offsets)
//! 4. Tensor data (aligned to GGUF_DEFAULT_ALIGNMENT)

const std = @import("std");
const mmap = @import("mmap.zig");

pub const GgufError = error{
    InvalidMagic,
    UnsupportedVersion,
    InvalidMetadataType,
    InvalidTensorType,
    MissingMetadata,
    TensorNotFound,
    ParseError,
    OutOfMemory,
    AlignmentError,
};

/// GGUF magic number: "GGUF" in little-endian
pub const GGUF_MAGIC: u32 = 0x46554747; // "GGUF"

/// Default alignment for tensor data
pub const GGUF_DEFAULT_ALIGNMENT: usize = 32;

/// Supported GGUF versions
pub const GGUF_VERSION_2: u32 = 2;
pub const GGUF_VERSION_3: u32 = 3;

/// GGUF file header (v2/v3)
pub const GgufHeader = extern struct {
    magic: u32,
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,

    pub fn isValid(self: GgufHeader) bool {
        return self.magic == GGUF_MAGIC and
            (self.version == GGUF_VERSION_2 or self.version == GGUF_VERSION_3);
    }
};

/// Metadata value types
pub const GgufMetadataValueType = enum(u32) {
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

/// Tensor data types
pub const GgufTensorType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    // q4_2 = 4, // deprecated
    // q4_3 = 5, // deprecated
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
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    bf16 = 29,

    /// Get bytes per element for unquantized types
    pub fn elementSize(self: GgufTensorType) ?usize {
        return switch (self) {
            .f32 => 4,
            .f16 => 2,
            .bf16 => 2,
            .i8 => 1,
            .i16 => 2,
            .i32 => 4,
            .i64 => 8,
            .f64 => 8,
            else => null, // Quantized types have block sizes
        };
    }

    /// Get block size for quantized types
    pub fn blockSize(self: GgufTensorType) usize {
        return switch (self) {
            .q4_0, .q4_1 => 32,
            .q5_0, .q5_1 => 32,
            .q8_0, .q8_1 => 32,
            .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q8_k => 256,
            else => 1,
        };
    }

    /// Get bytes per block for quantized types
    pub fn bytesPerBlock(self: GgufTensorType) usize {
        return switch (self) {
            .f32 => 4,
            .f16, .bf16 => 2,
            .q4_0 => 18, // 32 values: 16 bytes + 2 byte scale
            .q4_1 => 20, // 32 values: 16 bytes + 2 scale + 2 min
            .q5_0 => 22, // 32 values: 16 bytes + 4 bits + 2 scale
            .q5_1 => 24,
            .q8_0 => 34, // 32 values: 32 bytes + 2 byte scale
            .q8_1 => 36,
            .q2_k => 84,
            .q3_k => 110,
            .q4_k => 144,
            .q5_k => 176,
            .q6_k => 210,
            .q8_k => 292,
            .i8 => 1,
            .i16 => 2,
            .i32 => 4,
            .i64 => 8,
            .f64 => 8,
            else => 1,
        };
    }

    /// Calculate total bytes for a tensor of this type
    pub fn tensorBytes(self: GgufTensorType, element_count: u64) u64 {
        const bs = self.blockSize();
        const num_blocks = (element_count + bs - 1) / bs;
        return num_blocks * self.bytesPerBlock();
    }
};

/// Metadata value union
pub const GgufMetadataValue = union(GgufMetadataValueType) {
    uint8: u8,
    int8: i8,
    uint16: u16,
    int16: i16,
    uint32: u32,
    int32: i32,
    float32: f32,
    bool_: bool,
    string: []const u8,
    array: ArrayValue,
    uint64: u64,
    int64: i64,
    float64: f64,

    pub const ArrayValue = struct {
        element_type: GgufMetadataValueType,
        data: []const u8,
        count: u64,
    };

    pub fn asU32(self: GgufMetadataValue) ?u32 {
        return switch (self) {
            .uint8 => |v| v,
            .uint16 => |v| v,
            .uint32 => |v| v,
            .int8 => |v| if (v >= 0) @intCast(v) else null,
            .int16 => |v| if (v >= 0) @intCast(v) else null,
            .int32 => |v| if (v >= 0) @intCast(v) else null,
            else => null,
        };
    }

    pub fn asU64(self: GgufMetadataValue) ?u64 {
        return switch (self) {
            .uint8 => |v| v,
            .uint16 => |v| v,
            .uint32 => |v| v,
            .uint64 => |v| v,
            .int8 => |v| if (v >= 0) @intCast(v) else null,
            .int16 => |v| if (v >= 0) @intCast(v) else null,
            .int32 => |v| if (v >= 0) @intCast(v) else null,
            .int64 => |v| if (v >= 0) @intCast(v) else null,
            else => null,
        };
    }

    pub fn asF32(self: GgufMetadataValue) ?f32 {
        return switch (self) {
            .float32 => |v| v,
            .float64 => |v| @floatCast(v),
            .uint8 => |v| @floatFromInt(v),
            .uint16 => |v| @floatFromInt(v),
            .uint32 => |v| @floatFromInt(v),
            .int8 => |v| @floatFromInt(v),
            .int16 => |v| @floatFromInt(v),
            .int32 => |v| @floatFromInt(v),
            else => null,
        };
    }

    pub fn asString(self: GgufMetadataValue) ?[]const u8 {
        return switch (self) {
            .string => |v| v,
            else => null,
        };
    }

    pub fn asBool(self: GgufMetadataValue) ?bool {
        return switch (self) {
            .bool_ => |v| v,
            .uint8 => |v| v != 0,
            .int8 => |v| v != 0,
            else => null,
        };
    }
};

/// Parsed metadata entry
pub const GgufMetadata = struct {
    key: []const u8,
    value: GgufMetadataValue,
};

/// Tensor information from the info table
pub const TensorInfo = struct {
    name: []const u8,
    n_dims: u32,
    dims: [4]u64,
    tensor_type: GgufTensorType,
    offset: u64,

    /// Calculate total number of elements
    pub fn elementCount(self: TensorInfo) u64 {
        var count: u64 = 1;
        for (0..self.n_dims) |i| {
            count *= self.dims[i];
        }
        return count;
    }

    /// Calculate total bytes
    pub fn byteSize(self: TensorInfo) u64 {
        return self.tensor_type.tensorBytes(self.elementCount());
    }

    /// Get shape as slice
    pub fn shape(self: *const TensorInfo) []const u64 {
        return self.dims[0..self.n_dims];
    }
};

/// Parsed GGUF file
pub const GgufFile = struct {
    allocator: std.mem.Allocator,
    mapped: ?mmap.MappedFile,
    header: GgufHeader,
    metadata: std.StringHashMapUnmanaged(GgufMetadataValue),
    tensors: std.StringHashMapUnmanaged(TensorInfo),
    tensor_data_offset: u64,

    pub fn open(allocator: std.mem.Allocator, path: []const u8) !GgufFile {
        var mapped = try mmap.MappedFile.open(allocator, path);
        errdefer mapped.close();

        return parse(allocator, &mapped);
    }

    pub fn parse(allocator: std.mem.Allocator, mapped: *mmap.MappedFile) !GgufFile {
        var cursor = mmap.MemoryCursor.fromMapped(mapped);

        // Read and validate header
        const header = cursor.read(GgufHeader) orelse return GgufError.ParseError;
        if (!header.isValid()) {
            if (header.magic != GGUF_MAGIC) return GgufError.InvalidMagic;
            return GgufError.UnsupportedVersion;
        }

        var self = GgufFile{
            .allocator = allocator,
            .mapped = mapped.*,
            .header = header,
            .metadata = std.StringHashMapUnmanaged(GgufMetadataValue).empty,
            .tensors = std.StringHashMapUnmanaged(TensorInfo).empty,
            .tensor_data_offset = 0,
        };
        errdefer self.deinit();

        // Parse metadata
        for (0..header.metadata_kv_count) |_| {
            const key = cursor.readString() orelse return GgufError.ParseError;
            const value_type_int = cursor.read(u32) orelse return GgufError.ParseError;

            if (value_type_int > @intFromEnum(GgufMetadataValueType.float64)) {
                return GgufError.InvalidMetadataType;
            }
            const value_type: GgufMetadataValueType = @enumFromInt(value_type_int);
            const value = try readMetadataValue(&cursor, value_type);

            try self.metadata.put(allocator, key, value);
        }

        // Parse tensor info
        for (0..header.tensor_count) |_| {
            const name = cursor.readString() orelse return GgufError.ParseError;
            const n_dims = cursor.read(u32) orelse return GgufError.ParseError;

            if (n_dims > 4) return GgufError.ParseError;

            var dims: [4]u64 = .{ 1, 1, 1, 1 };
            for (0..n_dims) |i| {
                dims[i] = cursor.read(u64) orelse return GgufError.ParseError;
            }

            const tensor_type_int = cursor.read(u32) orelse return GgufError.ParseError;
            if (tensor_type_int > @intFromEnum(GgufTensorType.bf16)) {
                return GgufError.InvalidTensorType;
            }
            const tensor_type: GgufTensorType = @enumFromInt(tensor_type_int);
            const offset = cursor.read(u64) orelse return GgufError.ParseError;

            try self.tensors.put(allocator, name, .{
                .name = name,
                .n_dims = n_dims,
                .dims = dims,
                .tensor_type = tensor_type,
                .offset = offset,
            });
        }

        // Calculate tensor data start (aligned)
        cursor.alignTo(GGUF_DEFAULT_ALIGNMENT);
        self.tensor_data_offset = cursor.position;

        return self;
    }

    pub fn deinit(self: *GgufFile) void {
        self.metadata.deinit(self.allocator);
        self.tensors.deinit(self.allocator);
        if (self.mapped) |*m| {
            m.close();
        }
        self.* = undefined;
    }

    /// Get metadata value by key
    pub fn getMetadata(self: *const GgufFile, key: []const u8) ?GgufMetadataValue {
        return self.metadata.get(key);
    }

    /// Get tensor info by name
    pub fn getTensor(self: *const GgufFile, name: []const u8) ?TensorInfo {
        return self.tensors.get(name);
    }

    /// Get pointer to tensor data
    pub fn getTensorData(self: *const GgufFile, name: []const u8) ?[]const u8 {
        const info = self.tensors.get(name) orelse return null;
        const mapped = self.mapped orelse return null;
        const start = self.tensor_data_offset + info.offset;
        const end = start + info.byteSize();
        if (end > mapped.size) return null;
        return mapped.data[start..end];
    }

    /// Common metadata accessors
    pub fn getArchitecture(self: *const GgufFile) ?[]const u8 {
        const val = self.getMetadata("general.architecture") orelse return null;
        return val.asString();
    }

    pub fn getName(self: *const GgufFile) ?[]const u8 {
        const val = self.getMetadata("general.name") orelse return null;
        return val.asString();
    }

    pub fn getContextLength(self: *const GgufFile) ?u32 {
        // Try various keys used by different models
        const keys = [_][]const u8{
            "llama.context_length",
            "mistral.context_length",
            "phi.context_length",
            "general.context_length",
        };
        for (keys) |key| {
            const val = self.getMetadata(key) orelse continue;
            return val.asU32();
        }
        return null;
    }

    pub fn getEmbeddingLength(self: *const GgufFile) ?u32 {
        const arch = self.getArchitecture() orelse "llama";
        var buf: [64]u8 = undefined;
        const key = std.fmt.bufPrint(&buf, "{s}.embedding_length", .{arch}) catch return null;
        const val = self.getMetadata(key) orelse return null;
        return val.asU32();
    }

    pub fn getBlockCount(self: *const GgufFile) ?u32 {
        const arch = self.getArchitecture() orelse "llama";
        var buf: [64]u8 = undefined;
        const key = std.fmt.bufPrint(&buf, "{s}.block_count", .{arch}) catch return null;
        const val = self.getMetadata(key) orelse return null;
        return val.asU32();
    }

    pub fn getHeadCount(self: *const GgufFile) ?u32 {
        const arch = self.getArchitecture() orelse "llama";
        var buf: [64]u8 = undefined;
        const key = std.fmt.bufPrint(&buf, "{s}.attention.head_count", .{arch}) catch return null;
        const val = self.getMetadata(key) orelse return null;
        return val.asU32();
    }

    pub fn getHeadCountKV(self: *const GgufFile) ?u32 {
        const arch = self.getArchitecture() orelse "llama";
        var buf: [64]u8 = undefined;
        const key = std.fmt.bufPrint(&buf, "{s}.attention.head_count_kv", .{arch}) catch return null;
        const val = self.getMetadata(key) orelse self.getMetadata("llama.attention.head_count_kv") orelse return null;
        return val.asU32();
    }

    pub fn getVocabSize(self: *const GgufFile) ?u32 {
        const val = self.getMetadata("tokenizer.ggml.tokens") orelse return null;
        return switch (val) {
            .array => |arr| @intCast(arr.count),
            else => null,
        };
    }

    /// Print summary of the model
    pub fn printSummary(self: *const GgufFile, writer: anytype) !void {
        try writer.print("GGUF Model Summary\n", .{});
        try writer.print("==================\n", .{});
        try writer.print("Version: {d}\n", .{self.header.version});
        try writer.print("Tensor count: {d}\n", .{self.header.tensor_count});
        try writer.print("Metadata count: {d}\n", .{self.header.metadata_kv_count});

        if (self.getArchitecture()) |arch| {
            try writer.print("Architecture: {s}\n", .{arch});
        }
        if (self.getName()) |name| {
            try writer.print("Name: {s}\n", .{name});
        }
        if (self.getContextLength()) |ctx| {
            try writer.print("Context length: {d}\n", .{ctx});
        }
        if (self.getEmbeddingLength()) |emb| {
            try writer.print("Embedding dim: {d}\n", .{emb});
        }
        if (self.getBlockCount()) |blocks| {
            try writer.print("Layers: {d}\n", .{blocks});
        }
        if (self.getHeadCount()) |heads| {
            try writer.print("Attention heads: {d}\n", .{heads});
        }
        if (self.getVocabSize()) |vocab| {
            try writer.print("Vocab size: {d}\n", .{vocab});
        }
    }

    /// Print summary using std.debug.print (for CLI usage)
    pub fn printSummaryDebug(self: *const GgufFile) void {
        std.debug.print("GGUF Model Summary\n", .{});
        std.debug.print("==================\n", .{});
        std.debug.print("Version: {d}\n", .{self.header.version});
        std.debug.print("Tensor count: {d}\n", .{self.header.tensor_count});
        std.debug.print("Metadata count: {d}\n", .{self.header.metadata_kv_count});

        if (self.getArchitecture()) |arch| {
            std.debug.print("Architecture: {s}\n", .{arch});
        }
        if (self.getName()) |name| {
            std.debug.print("Name: {s}\n", .{name});
        }
        if (self.getContextLength()) |ctx| {
            std.debug.print("Context length: {d}\n", .{ctx});
        }
        if (self.getEmbeddingLength()) |emb| {
            std.debug.print("Embedding dim: {d}\n", .{emb});
        }
        if (self.getBlockCount()) |blocks| {
            std.debug.print("Layers: {d}\n", .{blocks});
        }
        if (self.getHeadCount()) |heads| {
            std.debug.print("Attention heads: {d}\n", .{heads});
        }
        if (self.getVocabSize()) |vocab| {
            std.debug.print("Vocab size: {d}\n", .{vocab});
        }
    }
};

fn readMetadataValue(cursor: *mmap.MemoryCursor, value_type: GgufMetadataValueType) !GgufMetadataValue {
    return switch (value_type) {
        .uint8 => .{ .uint8 = cursor.read(u8) orelse return GgufError.ParseError },
        .int8 => .{ .int8 = cursor.read(i8) orelse return GgufError.ParseError },
        .uint16 => .{ .uint16 = cursor.read(u16) orelse return GgufError.ParseError },
        .int16 => .{ .int16 = cursor.read(i16) orelse return GgufError.ParseError },
        .uint32 => .{ .uint32 = cursor.read(u32) orelse return GgufError.ParseError },
        .int32 => .{ .int32 = cursor.read(i32) orelse return GgufError.ParseError },
        .float32 => .{ .float32 = cursor.read(f32) orelse return GgufError.ParseError },
        .bool_ => .{ .bool_ = (cursor.read(u8) orelse return GgufError.ParseError) != 0 },
        .string => .{ .string = cursor.readString() orelse return GgufError.ParseError },
        .array => blk: {
            const elem_type_int = cursor.read(u32) orelse return GgufError.ParseError;
            if (elem_type_int > @intFromEnum(GgufMetadataValueType.float64)) {
                return GgufError.InvalidMetadataType;
            }
            const elem_type: GgufMetadataValueType = @enumFromInt(elem_type_int);
            const count = cursor.read(u64) orelse return GgufError.ParseError;

            // Calculate data size and read raw bytes
            const elem_size: u64 = switch (elem_type) {
                .uint8, .int8, .bool_ => 1,
                .uint16, .int16 => 2,
                .uint32, .int32, .float32 => 4,
                .uint64, .int64, .float64 => 8,
                .string, .array => return GgufError.ParseError, // Nested not supported inline
            };

            const data_size: usize = @intCast(count * elem_size);
            const data = cursor.readBytes(data_size) orelse return GgufError.ParseError;

            break :blk .{
                .array = .{
                    .element_type = elem_type,
                    .data = data,
                    .count = count,
                },
            };
        },
        .uint64 => .{ .uint64 = cursor.read(u64) orelse return GgufError.ParseError },
        .int64 => .{ .int64 = cursor.read(i64) orelse return GgufError.ParseError },
        .float64 => .{ .float64 = cursor.read(f64) orelse return GgufError.ParseError },
    };
}

test "gguf header validation" {
    const valid_header = GgufHeader{
        .magic = GGUF_MAGIC,
        .version = GGUF_VERSION_3,
        .tensor_count = 100,
        .metadata_kv_count = 50,
    };
    try std.testing.expect(valid_header.isValid());

    const invalid_magic = GgufHeader{
        .magic = 0x12345678,
        .version = GGUF_VERSION_3,
        .tensor_count = 0,
        .metadata_kv_count = 0,
    };
    try std.testing.expect(!invalid_magic.isValid());

    const invalid_version = GgufHeader{
        .magic = GGUF_MAGIC,
        .version = 99,
        .tensor_count = 0,
        .metadata_kv_count = 0,
    };
    try std.testing.expect(!invalid_version.isValid());
}

test "tensor type bytes calculation" {
    // Q4_0: 32 values packed into 18 bytes per block
    try std.testing.expectEqual(@as(usize, 18), GgufTensorType.q4_0.bytesPerBlock());
    try std.testing.expectEqual(@as(usize, 32), GgufTensorType.q4_0.blockSize());

    // 64 elements = 2 blocks of Q4_0 = 36 bytes
    try std.testing.expectEqual(@as(u64, 36), GgufTensorType.q4_0.tensorBytes(64));

    // F32: 4 bytes per element
    try std.testing.expectEqual(@as(u64, 256), GgufTensorType.f32.tensorBytes(64));
}

test "tensor info element count" {
    const info = TensorInfo{
        .name = "test",
        .n_dims = 3,
        .dims = .{ 4, 8, 16, 1 },
        .tensor_type = .f32,
        .offset = 0,
    };

    try std.testing.expectEqual(@as(u64, 4 * 8 * 16), info.elementCount());
    try std.testing.expectEqual(@as(u64, 4 * 8 * 16 * 4), info.byteSize());
}
