//! Unified Storage Format Implementation
//!
//! Zero-copy, memory-mapped storage format optimized for maximum throughput.
//! Designed to be faster than GGUF while supporting more data types.

const std = @import("std");
const mod = @import("mod.zig");
const compression = @import("compression.zig");

pub const UnifiedError = error{
    InvalidMagic,
    UnsupportedVersion,
    CorruptedData,
    ChecksumMismatch,
    CompressionError,
    DecompressionError,
    TensorNotFound,
    InvalidAlignment,
    OutOfBounds,
    OutOfMemory,
};

/// Format flags for optional features
pub const FormatFlags = packed struct(u32) {
    compressed: bool = false,
    checksummed: bool = true,
    memory_mapped: bool = true,
    streaming: bool = false,
    encrypted: bool = false,
    sparse: bool = false,
    _reserved: u26 = 0,
};

/// Data types supported by the format
pub const DataType = enum(u8) {
    // Standard types
    f32 = 0,
    f16 = 1,
    bf16 = 2,
    f64 = 3,
    i8 = 4,
    i16 = 5,
    i32 = 6,
    i64 = 7,
    u8 = 8,
    u16 = 9,
    u32 = 10,
    u64 = 11,

    // Quantized types (compatible with GGUF)
    q4_0 = 32,
    q4_1 = 33,
    q5_0 = 34,
    q5_1 = 35,
    q8_0 = 36,
    q8_1 = 37,
    q2_k = 40,
    q3_k = 41,
    q4_k = 42,
    q5_k = 43,
    q6_k = 44,
    q8_k = 45,

    // Vector types
    vec_f32 = 64,
    vec_f16 = 65,

    // Custom/mixed
    custom = 127,

    /// Get bytes per element for unquantized types
    pub fn elementSize(self: DataType) ?usize {
        return switch (self) {
            .f32 => 4,
            .f16, .bf16 => 2,
            .f64 => 8,
            .i8, .u8 => 1,
            .i16, .u16 => 2,
            .i32, .u32 => 4,
            .i64, .u64 => 8,
            .vec_f32 => 4,
            .vec_f16 => 2,
            else => null,
        };
    }

    /// Get block size for quantized types
    pub fn blockSize(self: DataType) usize {
        return switch (self) {
            .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1 => 32,
            .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q8_k => 256,
            else => 1,
        };
    }

    /// Get bytes per block
    pub fn bytesPerBlock(self: DataType) usize {
        return switch (self) {
            .f32, .vec_f32 => 4,
            .f16, .bf16, .vec_f16 => 2,
            .f64 => 8,
            .i8, .u8 => 1,
            .i16, .u16 => 2,
            .i32, .u32 => 4,
            .i64, .u64 => 8,
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
            .custom => 1,
        };
    }
};

/// File header (64 bytes, aligned)
pub const FormatHeader = extern struct {
    magic: u32 = mod.FORMAT_MAGIC,
    version: u16 = mod.FORMAT_VERSION,
    header_size: u16 = 64,
    flags: FormatFlags = .{},
    compression_type: u8 = 0,
    _reserved1: [3]u8 = .{ 0, 0, 0 },
    tensor_count: u64 = 0,
    metadata_count: u64 = 0,
    metadata_offset: u64 = 0,
    index_offset: u64 = 0,
    data_offset: u64 = 0,
    total_size: u64 = 0,
    checksum: u32 = 0,
    _reserved2: [4]u8 = .{ 0, 0, 0, 0 },

    pub fn isValid(self: FormatHeader) bool {
        return self.magic == mod.FORMAT_MAGIC and
            self.version <= mod.FORMAT_VERSION and
            self.header_size >= 64;
    }

    /// Compute header checksum (CRC32)
    pub fn computeChecksum(self: *FormatHeader) u32 {
        const bytes = std.mem.asBytes(self);
        // Exclude checksum field from calculation
        return std.hash.Crc32.hash(bytes[0..56]);
    }

    pub fn updateChecksum(self: *FormatHeader) void {
        self.checksum = self.computeChecksum();
    }
};

/// Tensor descriptor in index table
pub const TensorDescriptor = extern struct {
    name_hash: u64 = 0,
    name_offset: u32 = 0,
    name_length: u16 = 0,
    data_type: DataType = .f32,
    n_dims: u8 = 0,
    dims: [4]u64 = .{ 0, 0, 0, 0 },
    data_offset: u64 = 0,
    data_size: u64 = 0,
    compressed_size: u64 = 0, // 0 if uncompressed
    flags: u32 = 0,
    _reserved: [4]u8 = .{ 0, 0, 0, 0 },

    /// Calculate total elements
    pub fn elementCount(self: TensorDescriptor) u64 {
        var count: u64 = 1;
        for (0..self.n_dims) |i| {
            if (self.dims[i] == 0) break;
            count *= self.dims[i];
        }
        return count;
    }

    /// Get shape as slice
    pub fn shape(self: *const TensorDescriptor) []const u64 {
        return self.dims[0..self.n_dims];
    }

    /// Calculate expected byte size
    pub fn byteSize(self: TensorDescriptor) u64 {
        const elem_count = self.elementCount();
        const bs = self.data_type.blockSize();
        const num_blocks = (elem_count + bs - 1) / bs;
        return num_blocks * self.data_type.bytesPerBlock();
    }
};

/// Unified format file handler
pub const UnifiedFormat = struct {
    allocator: std.mem.Allocator,
    header: FormatHeader,
    tensors: std.StringHashMapUnmanaged(TensorDescriptor),
    metadata: std.StringHashMapUnmanaged([]const u8),
    data: []const u8,
    owns_data: bool,

    /// Create new empty format
    pub fn init(allocator: std.mem.Allocator) UnifiedFormat {
        return .{
            .allocator = allocator,
            .header = .{},
            .tensors = std.StringHashMapUnmanaged(TensorDescriptor).empty,
            .metadata = std.StringHashMapUnmanaged([]const u8).empty,
            .data = &.{},
            .owns_data = false,
        };
    }

    /// Load from memory (zero-copy if possible)
    pub fn fromMemory(allocator: std.mem.Allocator, data: []const u8) UnifiedError!UnifiedFormat {
        if (data.len < @sizeOf(FormatHeader)) return UnifiedError.CorruptedData;

        const header: *const FormatHeader = @ptrCast(@alignCast(data.ptr));
        if (!header.isValid()) {
            if (header.magic != mod.FORMAT_MAGIC) return UnifiedError.InvalidMagic;
            return UnifiedError.UnsupportedVersion;
        }

        // Verify checksum if enabled
        if (header.flags.checksummed) {
            var header_copy = header.*;
            if (header_copy.computeChecksum() != header.checksum) {
                return UnifiedError.ChecksumMismatch;
            }
        }

        var self = UnifiedFormat{
            .allocator = allocator,
            .header = header.*,
            .tensors = std.StringHashMapUnmanaged(TensorDescriptor).empty,
            .metadata = std.StringHashMapUnmanaged([]const u8).empty,
            .data = data,
            .owns_data = false,
        };
        errdefer self.deinit();

        // Parse index table
        try self.parseIndex();

        return self;
    }

    fn parseIndex(self: *UnifiedFormat) UnifiedError!void {
        // Parse metadata section first
        const metadata_count = self.header.metadata_count;
        var meta_offset = self.header.metadata_offset;
        for (0..metadata_count) |_| {
            if (meta_offset + 8 > self.data.len) break;

            const key_len = std.mem.readInt(u32, self.data[meta_offset..][0..4], .little);
            meta_offset += 4;
            const value_len = std.mem.readInt(u32, self.data[meta_offset..][0..4], .little);
            meta_offset += 4;

            if (meta_offset + key_len + value_len > self.data.len) break;

            const key = self.data[meta_offset..][0..key_len];
            meta_offset += key_len;
            const value = self.data[meta_offset..][0..value_len];
            meta_offset += value_len;

            self.metadata.put(self.allocator, key, value) catch return UnifiedError.OutOfMemory;
        }

        // Parse tensor index
        const index_start = self.header.index_offset;
        const tensor_count = self.header.tensor_count;

        if (index_start + tensor_count * @sizeOf(TensorDescriptor) > self.data.len) {
            return UnifiedError.CorruptedData;
        }

        // Read tensor descriptors
        var offset = index_start;
        for (0..tensor_count) |_| {
            if (offset + @sizeOf(TensorDescriptor) > self.data.len) {
                return UnifiedError.CorruptedData;
            }

            const desc: *const TensorDescriptor = @ptrCast(@alignCast(self.data.ptr + offset));
            offset += @sizeOf(TensorDescriptor);

            // Read tensor name from name table
            const name_start = desc.name_offset;
            const name_end = name_start + desc.name_length;
            if (name_end > self.data.len) {
                return UnifiedError.CorruptedData;
            }

            const name = self.data[name_start..name_end];
            self.tensors.put(self.allocator, name, desc.*) catch return UnifiedError.OutOfMemory;
        }
    }

    pub fn deinit(self: *UnifiedFormat) void {
        self.tensors.deinit(self.allocator);
        self.metadata.deinit(self.allocator);
        if (self.owns_data) {
            self.allocator.free(self.data);
        }
        self.* = undefined;
    }

    /// Get tensor descriptor by name
    pub fn getTensor(self: *const UnifiedFormat, name: []const u8) ?TensorDescriptor {
        return self.tensors.get(name);
    }

    /// Get tensor data (zero-copy, decompresses if needed)
    pub fn getTensorData(self: *const UnifiedFormat, allocator: std.mem.Allocator, name: []const u8) UnifiedError![]const u8 {
        const desc = self.tensors.get(name) orelse return UnifiedError.TensorNotFound;

        const start = desc.data_offset;
        const size = if (desc.compressed_size > 0) desc.compressed_size else desc.data_size;
        const end = start + size;

        if (end > self.data.len) return UnifiedError.OutOfBounds;

        const raw_data = self.data[start..end];

        // Decompress if needed
        if (desc.compressed_size > 0) {
            const comp_type: compression.CompressionType = @enumFromInt(self.header.compression_type);
            return compression.decompress(allocator, raw_data, @intCast(desc.data_size), comp_type) catch {
                return UnifiedError.DecompressionError;
            };
        }

        return raw_data;
    }

    /// Get tensor data as typed slice (zero-copy for uncompressed)
    pub fn getTensorSlice(self: *const UnifiedFormat, comptime T: type, name: []const u8) UnifiedError![]const T {
        const desc = self.tensors.get(name) orelse return UnifiedError.TensorNotFound;

        // Only zero-copy for uncompressed data
        if (desc.compressed_size > 0) return UnifiedError.CompressionError;

        const start = desc.data_offset;
        const end = start + desc.data_size;
        if (end > self.data.len) return UnifiedError.OutOfBounds;

        const raw = self.data[start..end];
        const elem_count = raw.len / @sizeOf(T);

        return @as([*]const T, @ptrCast(@alignCast(raw.ptr)))[0..elem_count];
    }

    /// Get metadata value
    pub fn getMetadata(self: *const UnifiedFormat, key: []const u8) ?[]const u8 {
        return self.metadata.get(key);
    }
};

/// Builder for creating unified format files
pub const UnifiedFormatBuilder = struct {
    allocator: std.mem.Allocator,
    tensors: std.ArrayListUnmanaged(TensorEntry),
    metadata: std.ArrayListUnmanaged(MetadataEntry),
    compression_type: compression.CompressionType,
    flags: FormatFlags,

    const TensorEntry = struct {
        name: []const u8,
        owned_name: ?[]u8, // Separately track owned name to avoid @constCast
        data: []const u8,
        data_type: DataType,
        dims: [4]u64,
        n_dims: u8,
        owns_data: bool,
    };

    const MetadataEntry = struct {
        key: []const u8,
        value: []const u8,
    };

    pub fn init(allocator: std.mem.Allocator) UnifiedFormatBuilder {
        return .{
            .allocator = allocator,
            .tensors = std.ArrayListUnmanaged(TensorEntry).empty,
            .metadata = std.ArrayListUnmanaged(MetadataEntry).empty,
            .compression_type = .none,
            .flags = .{},
        };
    }

    pub fn deinit(self: *UnifiedFormatBuilder) void {
        for (self.tensors.items) |entry| {
            if (entry.owns_data) {
                self.allocator.free(entry.data);
            }
            if (entry.owned_name) |owned| {
                self.allocator.free(owned);
            }
        }
        self.tensors.deinit(self.allocator);
        self.metadata.deinit(self.allocator);
    }

    /// Set compression type
    pub fn setCompression(self: *UnifiedFormatBuilder, comp_type: compression.CompressionType) *UnifiedFormatBuilder {
        self.compression_type = comp_type;
        self.flags.compressed = (comp_type != .none);
        return self;
    }

    /// Add tensor
    pub fn addTensor(
        self: *UnifiedFormatBuilder,
        name: []const u8,
        data: []const u8,
        data_type: DataType,
        dims: []const u64,
    ) !*UnifiedFormatBuilder {
        var dims_array: [4]u64 = .{ 0, 0, 0, 0 };
        const n_dims: u8 = @intCast(@min(dims.len, 4));
        for (0..n_dims) |i| {
            dims_array[i] = dims[i];
        }

        // Copy the name to ensure it's not invalidated when the caller reuses a buffer
        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);

        try self.tensors.append(self.allocator, .{
            .name = name_copy,
            .owned_name = name_copy, // Track owned allocation separately
            .data = data,
            .data_type = data_type,
            .dims = dims_array,
            .n_dims = n_dims,
            .owns_data = false,
        });
        return self;
    }

    /// Add typed tensor
    pub fn addTypedTensor(
        self: *UnifiedFormatBuilder,
        comptime T: type,
        name: []const u8,
        data: []const T,
        dims: []const u64,
    ) !*UnifiedFormatBuilder {
        const data_type: DataType = switch (T) {
            f32 => .f32,
            f16 => .f16,
            f64 => .f64,
            i8 => .i8,
            i16 => .i16,
            i32 => .i32,
            i64 => .i64,
            u8 => .u8,
            u16 => .u16,
            u32 => .u32,
            u64 => .u64,
            else => .custom,
        };

        const bytes = std.mem.sliceAsBytes(data);
        return self.addTensor(name, bytes, data_type, dims);
    }

    /// Add metadata
    pub fn addMetadata(self: *UnifiedFormatBuilder, key: []const u8, value: []const u8) !*UnifiedFormatBuilder {
        try self.metadata.append(self.allocator, .{
            .key = key,
            .value = value,
        });
        return self;
    }

    /// Build and return serialized data
    pub fn build(self: *UnifiedFormatBuilder) ![]u8 {
        // Calculate offsets
        const header_size: u64 = @sizeOf(FormatHeader);

        // Name table size
        var name_table_size: u64 = 0;
        for (self.tensors.items) |entry| {
            name_table_size += entry.name.len;
        }
        name_table_size = alignUp(name_table_size, mod.DEFAULT_ALIGNMENT);

        // Metadata section size
        var metadata_size: u64 = 0;
        for (self.metadata.items) |entry| {
            metadata_size += 8 + entry.key.len + entry.value.len; // lengths + data
        }
        metadata_size = alignUp(metadata_size, mod.DEFAULT_ALIGNMENT);

        // Index table size
        const index_size = alignUp(
            self.tensors.items.len * @sizeOf(TensorDescriptor),
            mod.DEFAULT_ALIGNMENT,
        );

        // Calculate data section with potential compression
        var data_entries = std.ArrayListUnmanaged(struct { data: []const u8, compressed: bool }).empty;
        defer {
            for (data_entries.items) |entry| {
                if (entry.compressed) {
                    self.allocator.free(entry.data);
                }
            }
            data_entries.deinit(self.allocator);
        }

        var data_size: u64 = 0;
        for (self.tensors.items) |entry| {
            if (self.compression_type != .none) {
                const compressed = compression.compress(
                    self.allocator,
                    entry.data,
                    self.compression_type,
                ) catch {
                    // Fall back to uncompressed
                    try data_entries.append(self.allocator, .{ .data = entry.data, .compressed = false });
                    data_size += entry.data.len;
                    continue;
                };
                try data_entries.append(self.allocator, .{ .data = compressed, .compressed = true });
                data_size += compressed.len;
            } else {
                try data_entries.append(self.allocator, .{ .data = entry.data, .compressed = false });
                data_size += entry.data.len;
            }
        }

        // Calculate total size
        const metadata_offset = header_size;
        const name_table_offset = metadata_offset + metadata_size;
        const index_offset = name_table_offset + name_table_size;
        const data_offset = index_offset + index_size;
        const total_size = data_offset + data_size + 32; // +32 for final checksum

        // Allocate output buffer
        const output = try self.allocator.alloc(u8, @intCast(total_size));
        errdefer self.allocator.free(output);

        // Write header
        var header = FormatHeader{
            .flags = self.flags,
            .compression_type = @intFromEnum(self.compression_type),
            .tensor_count = self.tensors.items.len,
            .metadata_count = self.metadata.items.len,
            .metadata_offset = metadata_offset,
            .index_offset = index_offset,
            .data_offset = data_offset,
            .total_size = total_size,
        };
        header.updateChecksum();
        @memcpy(output[0..@sizeOf(FormatHeader)], std.mem.asBytes(&header));

        // Write metadata section
        var meta_pos: usize = @intCast(metadata_offset);
        for (self.metadata.items) |entry| {
            std.mem.writeInt(u32, output[meta_pos..][0..4], @intCast(entry.key.len), .little);
            meta_pos += 4;
            std.mem.writeInt(u32, output[meta_pos..][0..4], @intCast(entry.value.len), .little);
            meta_pos += 4;
            @memcpy(output[meta_pos..][0..entry.key.len], entry.key);
            meta_pos += entry.key.len;
            @memcpy(output[meta_pos..][0..entry.value.len], entry.value);
            meta_pos += entry.value.len;
        }

        // Write name table and build index
        var name_pos: usize = @intCast(name_table_offset);
        var data_pos: usize = @intCast(data_offset);
        var idx_pos: usize = @intCast(index_offset);

        for (self.tensors.items, 0..) |entry, i| {
            // Write name
            @memcpy(output[name_pos..][0..entry.name.len], entry.name);

            // Get data entry
            const data_entry = data_entries.items[i];

            // Build descriptor
            var desc = TensorDescriptor{
                .name_hash = std.hash.Wyhash.hash(0, entry.name),
                .name_offset = @intCast(name_pos),
                .name_length = @intCast(entry.name.len),
                .data_type = entry.data_type,
                .n_dims = entry.n_dims,
                .dims = entry.dims,
                .data_offset = @intCast(data_pos),
                .data_size = @intCast(entry.data.len),
                .compressed_size = if (data_entry.compressed) @intCast(data_entry.data.len) else 0,
            };

            // Write descriptor
            @memcpy(output[idx_pos..][0..@sizeOf(TensorDescriptor)], std.mem.asBytes(&desc));
            idx_pos += @sizeOf(TensorDescriptor);

            // Write data
            @memcpy(output[data_pos..][0..data_entry.data.len], data_entry.data);
            data_pos += data_entry.data.len;

            name_pos += entry.name.len;
        }

        // Write final checksum
        const final_checksum = std.hash.Crc32.hash(output[0 .. output.len - 32]);
        std.mem.writeInt(u32, output[output.len - 32 ..][0..4], final_checksum, .little);

        return output;
    }
};

fn alignUp(value: u64, alignment: u64) u64 {
    return (value + alignment - 1) & ~(alignment - 1);
}

test "format header validation" {
    var header = FormatHeader{};
    try std.testing.expect(header.isValid());
    header.updateChecksum();

    const computed = header.computeChecksum();
    try std.testing.expectEqual(header.checksum, computed);
}

test "data type sizes" {
    try std.testing.expectEqual(@as(?usize, 4), DataType.f32.elementSize());
    try std.testing.expectEqual(@as(?usize, 2), DataType.f16.elementSize());
    try std.testing.expectEqual(@as(usize, 32), DataType.q4_0.blockSize());
    try std.testing.expectEqual(@as(usize, 18), DataType.q4_0.bytesPerBlock());
}

test "builder basic" {
    const allocator = std.testing.allocator;

    var builder = UnifiedFormatBuilder.init(allocator);
    defer builder.deinit();

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    _ = try builder.addTypedTensor(f32, "test_tensor", &data, &.{4});
    _ = try builder.addMetadata("name", "test");

    const output = try builder.build();
    defer allocator.free(output);

    // Verify we can parse it back
    var parsed = try UnifiedFormat.fromMemory(allocator, output);
    defer parsed.deinit();

    try std.testing.expectEqual(@as(u64, 1), parsed.header.tensor_count);
    const desc = parsed.getTensor("test_tensor");
    try std.testing.expect(desc != null);
    try std.testing.expectEqual(DataType.f32, desc.?.data_type);
}

test {
    std.testing.refAllDecls(@This());
}
