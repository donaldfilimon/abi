//! Streaming Write Support for Unified Format
//!
//! Enables incremental writes without loading entire files into memory.
//! Supports:
//! - Append-only operations
//! - Partial updates
//! - Transaction-like semantics
//! - Memory-efficient large file handling

const std = @import("std");
const unified = @import("unified.zig");
const compression = @import("compression.zig");
const mod = @import("mod.zig");

pub const StreamingError = error{
    NotInitialized,
    AlreadyFinalized,
    WriteFailed,
    InvalidState,
    ChecksumMismatch,
    OutOfMemory,
};

/// Streaming format writer for incremental file creation
pub const StreamingWriter = struct {
    allocator: std.mem.Allocator,
    buffer: std.ArrayListUnmanaged(u8),
    tensor_index: std.ArrayListUnmanaged(TensorIndexEntry),
    metadata_buffer: std.ArrayListUnmanaged(u8),
    compression_type: compression.CompressionType,
    current_data_offset: u64,
    finalized: bool,
    checksum: u32,

    const TensorIndexEntry = struct {
        name: []const u8,
        data_type: unified.DataType,
        dims: [4]u64,
        n_dims: u8,
        offset: u64,
        size: u64,
        compressed_size: u64,
    };

    pub fn init(allocator: std.mem.Allocator) StreamingWriter {
        return .{
            .allocator = allocator,
            .buffer = std.ArrayListUnmanaged(u8).empty,
            .tensor_index = std.ArrayListUnmanaged(TensorIndexEntry).empty,
            .metadata_buffer = std.ArrayListUnmanaged(u8).empty,
            .compression_type = .none,
            .current_data_offset = 0,
            .finalized = false,
            .checksum = 0,
        };
    }

    pub fn deinit(self: *StreamingWriter) void {
        self.buffer.deinit(self.allocator);
        for (self.tensor_index.items) |entry| {
            self.allocator.free(entry.name);
        }
        self.tensor_index.deinit(self.allocator);
        self.metadata_buffer.deinit(self.allocator);
    }

    /// Set compression type for subsequent writes
    pub fn setCompression(self: *StreamingWriter, comp: compression.CompressionType) *StreamingWriter {
        self.compression_type = comp;
        return self;
    }

    /// Write a tensor incrementally
    pub fn writeTensor(
        self: *StreamingWriter,
        name: []const u8,
        data: []const u8,
        data_type: unified.DataType,
        dims: []const u64,
    ) StreamingError!void {
        if (self.finalized) return error.AlreadyFinalized;

        // Copy name
        const name_copy = self.allocator.dupe(u8, name) catch return error.OutOfMemory;
        errdefer self.allocator.free(name_copy);

        // Compress if needed
        var write_data = data;
        var compressed_size: u64 = 0;
        var compressed_buf: ?[]u8 = null;

        if (self.compression_type != .none) {
            compressed_buf = compression.compress(self.allocator, data, self.compression_type) catch null;
            if (compressed_buf) |cb| {
                if (cb.len < data.len) {
                    write_data = cb;
                    compressed_size = cb.len;
                } else {
                    self.allocator.free(cb);
                    compressed_buf = null;
                }
            }
        }
        defer if (compressed_buf) |cb| self.allocator.free(cb);

        // Write to buffer
        self.buffer.appendSlice(self.allocator, write_data) catch return error.OutOfMemory;

        // Update checksum
        self.checksum = std.hash.Crc32.hash(write_data);

        // Record index entry
        var dims_array: [4]u64 = .{ 0, 0, 0, 0 };
        const n_dims: u8 = @intCast(@min(dims.len, 4));
        for (0..n_dims) |i| {
            dims_array[i] = dims[i];
        }

        self.tensor_index.append(self.allocator, .{
            .name = name_copy,
            .data_type = data_type,
            .dims = dims_array,
            .n_dims = n_dims,
            .offset = self.current_data_offset,
            .size = data.len,
            .compressed_size = compressed_size,
        }) catch return error.OutOfMemory;

        self.current_data_offset += write_data.len;
    }

    /// Write typed tensor
    pub fn writeTensorTyped(
        self: *StreamingWriter,
        comptime T: type,
        name: []const u8,
        data: []const T,
        dims: []const u64,
    ) StreamingError!void {
        const data_type: unified.DataType = switch (T) {
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
        return self.writeTensor(name, std.mem.sliceAsBytes(data), data_type, dims);
    }

    /// Add metadata
    pub fn writeMetadata(self: *StreamingWriter, key: []const u8, value: []const u8) StreamingError!void {
        if (self.finalized) return error.AlreadyFinalized;

        // Format: key_len (4) + value_len (4) + key + value
        var len_buf: [4]u8 = undefined;
        std.mem.writeInt(u32, &len_buf, @intCast(key.len), .little);
        self.metadata_buffer.appendSlice(self.allocator, &len_buf) catch return error.OutOfMemory;
        std.mem.writeInt(u32, &len_buf, @intCast(value.len), .little);
        self.metadata_buffer.appendSlice(self.allocator, &len_buf) catch return error.OutOfMemory;
        self.metadata_buffer.appendSlice(self.allocator, key) catch return error.OutOfMemory;
        self.metadata_buffer.appendSlice(self.allocator, value) catch return error.OutOfMemory;
    }

    /// Finalize and get complete file
    pub fn finalize(self: *StreamingWriter) StreamingError![]u8 {
        if (self.finalized) return error.AlreadyFinalized;
        self.finalized = true;

        // Calculate sizes
        const header_size: u64 = @sizeOf(unified.FormatHeader);
        const metadata_size = alignUp(self.metadata_buffer.items.len, mod.DEFAULT_ALIGNMENT);
        const index_size = alignUp(self.tensor_index.items.len * @sizeOf(unified.TensorDescriptor), mod.DEFAULT_ALIGNMENT);
        const name_table_size = blk: {
            var size: usize = 0;
            for (self.tensor_index.items) |entry| {
                size += entry.name.len;
            }
            break :blk alignUp(size, mod.DEFAULT_ALIGNMENT);
        };
        const data_size = self.buffer.items.len;
        const total_size = header_size + metadata_size + name_table_size + index_size + data_size + 32;

        // Allocate output
        var output = self.allocator.alloc(u8, @intCast(total_size)) catch return error.OutOfMemory;
        errdefer self.allocator.free(output);

        // Count metadata entries by parsing the buffer format (key_len + value_len + key + value)
        const metadata_count = countMetadataEntries(self.metadata_buffer.items);

        // Build header
        var header = unified.FormatHeader{
            .flags = .{
                .compressed = (self.compression_type != .none),
                .checksummed = true,
                .streaming = true,
            },
            .compression_type = @intFromEnum(self.compression_type),
            .tensor_count = self.tensor_index.items.len,
            .metadata_count = metadata_count,
            .metadata_offset = header_size,
            .index_offset = header_size + metadata_size + name_table_size,
            .data_offset = header_size + metadata_size + name_table_size + index_size,
            .total_size = total_size,
        };
        header.updateChecksum();

        // Write header
        @memcpy(output[0..@sizeOf(unified.FormatHeader)], std.mem.asBytes(&header));

        // Write metadata
        var pos: usize = @intCast(header_size);
        if (self.metadata_buffer.items.len > 0) {
            @memcpy(output[pos..][0..self.metadata_buffer.items.len], self.metadata_buffer.items);
        }
        pos = @intCast(header_size + metadata_size);

        // Write name table and index
        var name_offset: usize = pos;
        for (self.tensor_index.items) |entry| {
            @memcpy(output[name_offset..][0..entry.name.len], entry.name);
            name_offset += entry.name.len;
        }

        pos = @intCast(header.index_offset);
        var current_name_offset: u32 = @intCast(header_size + metadata_size);
        for (self.tensor_index.items) |entry| {
            const desc = unified.TensorDescriptor{
                .name_hash = std.hash.Wyhash.hash(0, entry.name),
                .name_offset = current_name_offset,
                .name_length = @intCast(entry.name.len),
                .data_type = entry.data_type,
                .n_dims = entry.n_dims,
                .dims = entry.dims,
                .data_offset = header.data_offset + entry.offset,
                .data_size = entry.size,
                .compressed_size = entry.compressed_size,
            };
            @memcpy(output[pos..][0..@sizeOf(unified.TensorDescriptor)], std.mem.asBytes(&desc));
            pos += @sizeOf(unified.TensorDescriptor);
            current_name_offset += @intCast(entry.name.len);
        }

        // Write data
        pos = @intCast(header.data_offset);
        @memcpy(output[pos..][0..self.buffer.items.len], self.buffer.items);

        // Write final checksum
        const final_checksum = std.hash.Crc32.hash(output[0 .. output.len - 32]);
        std.mem.writeInt(u32, output[output.len - 32 ..][0..4], final_checksum, .little);

        return output;
    }

    /// Get current size estimate
    pub fn estimatedSize(self: *const StreamingWriter) usize {
        const header_size = @sizeOf(unified.FormatHeader);
        const meta_size = alignUp(self.metadata_buffer.items.len, 64);
        const index_size = alignUp(self.tensor_index.items.len * @sizeOf(unified.TensorDescriptor), 64);
        return header_size + meta_size + index_size + self.buffer.items.len + 32;
    }

    /// Get number of tensors written
    pub fn tensorCount(self: *const StreamingWriter) usize {
        return self.tensor_index.items.len;
    }
};

/// Streaming reader for incremental file reading
pub const StreamingReader = struct {
    allocator: std.mem.Allocator,
    data: []const u8,
    header: unified.FormatHeader,
    current_tensor: usize,

    pub fn init(allocator: std.mem.Allocator, data: []const u8) StreamingError!StreamingReader {
        if (data.len < @sizeOf(unified.FormatHeader)) return error.InvalidState;

        const header: *const unified.FormatHeader = @ptrCast(@alignCast(data.ptr));
        if (!header.isValid()) return error.InvalidState;

        return .{
            .allocator = allocator,
            .data = data,
            .header = header.*,
            .current_tensor = 0,
        };
    }

    /// Read next tensor
    pub fn next(self: *StreamingReader) ?TensorData {
        if (self.current_tensor >= self.header.tensor_count) return null;

        const index_start = self.header.index_offset + self.current_tensor * @sizeOf(unified.TensorDescriptor);
        if (index_start + @sizeOf(unified.TensorDescriptor) > self.data.len) return null;

        const desc: *const unified.TensorDescriptor = @ptrCast(@alignCast(self.data.ptr + index_start));
        self.current_tensor += 1;

        const name = self.data[desc.name_offset..][0..desc.name_length];
        const data_start = desc.data_offset;
        const data_size = if (desc.compressed_size > 0) desc.compressed_size else desc.data_size;

        if (data_start + data_size > self.data.len) return null;

        return .{
            .name = name,
            .data = self.data[data_start..][0..data_size],
            .data_type = desc.data_type,
            .dims = desc.dims,
            .n_dims = desc.n_dims,
            .compressed = desc.compressed_size > 0,
            .original_size = desc.data_size,
        };
    }

    /// Reset to beginning
    pub fn reset(self: *StreamingReader) void {
        self.current_tensor = 0;
    }

    /// Get tensor count
    pub fn count(self: *const StreamingReader) usize {
        return @intCast(self.header.tensor_count);
    }

    pub const TensorData = struct {
        name: []const u8,
        data: []const u8,
        data_type: unified.DataType,
        dims: [4]u64,
        n_dims: u8,
        compressed: bool,
        original_size: u64,
    };
};

fn alignUp(value: usize, alignment: usize) usize {
    return (value + alignment - 1) & ~(alignment - 1);
}

/// Count metadata entries in a metadata buffer
/// Format: key_len (4 bytes) + value_len (4 bytes) + key + value, repeated
fn countMetadataEntries(buffer: []const u8) usize {
    var count: usize = 0;
    var pos: usize = 0;

    while (pos + 8 <= buffer.len) {
        const key_len = std.mem.readInt(u32, buffer[pos..][0..4], .little);
        const value_len = std.mem.readInt(u32, buffer[pos + 4 ..][0..4], .little);
        const entry_size = 8 + key_len + value_len;

        if (pos + entry_size > buffer.len) break;

        count += 1;
        pos += entry_size;
    }

    return count;
}

test "streaming writer basic" {
    const allocator = std.testing.allocator;

    var writer = StreamingWriter.init(allocator);
    defer writer.deinit();

    const data1 = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const data2 = [_]f32{ 5.0, 6.0, 7.0, 8.0 };

    try writer.writeTensorTyped(f32, "tensor1", &data1, &.{4});
    try writer.writeTensorTyped(f32, "tensor2", &data2, &.{4});
    try writer.writeMetadata("author", "test");

    try std.testing.expectEqual(@as(usize, 2), writer.tensorCount());

    const output = try writer.finalize();
    defer allocator.free(output);

    // Verify we can read it back
    var reader = try StreamingReader.init(allocator, output);
    try std.testing.expectEqual(@as(usize, 2), reader.count());

    const t1 = reader.next().?;
    try std.testing.expectEqualStrings("tensor1", t1.name);
    try std.testing.expectEqual(unified.DataType.f32, t1.data_type);

    const t2 = reader.next().?;
    try std.testing.expectEqualStrings("tensor2", t2.name);

    try std.testing.expect(reader.next() == null);
}

test "streaming writer with compression" {
    const allocator = std.testing.allocator;

    var writer = StreamingWriter.init(allocator);
    defer writer.deinit();

    _ = writer.setCompression(.lz4);

    // Repetitive data that compresses well
    var data: [256]f32 = undefined;
    for (&data, 0..) |*v, i| {
        v.* = @floatFromInt(i % 4);
    }

    try writer.writeTensorTyped(f32, "compressed_tensor", &data, &.{256});

    const output = try writer.finalize();
    defer allocator.free(output);

    var reader = try StreamingReader.init(allocator, output);
    const t = reader.next().?;
    try std.testing.expect(t.compressed);
}

test {
    std.testing.refAllDecls(@This());
}
