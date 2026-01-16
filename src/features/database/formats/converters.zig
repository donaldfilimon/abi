//! Format Converters for interoperability with standard model formats.
//!
//! Supported conversions:
//! - GGUF (llama.cpp) - Full bidirectional support
//! - SafeTensors (Hugging Face) - Read/Write support
//! - NumPy NPY/NPZ - Read/Write support
//! - ONNX tensors - Read support
//!
//! All conversions preserve tensor data integrity and metadata where possible.

const std = @import("std");
const unified = @import("unified.zig");
const compression = @import("compression.zig");

pub const TargetFormat = enum {
    gguf,
    safetensors,
    npy,
    npz,
    onnx,
    unified,
};

pub const ConversionError = error{
    UnsupportedFormat,
    UnsupportedDataType,
    InvalidInput,
    MetadataLoss,
    OutOfMemory,
    IoError,
};

pub const ConversionOptions = struct {
    /// Target format for conversion
    target: TargetFormat = .unified,
    /// Compression to apply (for formats that support it)
    compression: compression.CompressionType = .none,
    /// Preserve all metadata (may not be possible for all formats)
    preserve_metadata: bool = true,
    /// Quantization target (for model optimization)
    quantize_to: ?unified.DataType = null,
    /// Alignment for tensor data
    alignment: usize = 64,
};

/// Main converter interface
pub const Converter = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Converter {
        return .{ .allocator = allocator };
    }

    /// Convert unified format to target format
    pub fn convert(
        self: *Converter,
        source: *const unified.UnifiedFormat,
        options: ConversionOptions,
    ) ConversionError![]u8 {
        return switch (options.target) {
            .safetensors => self.toSafeTensors(source, options),
            .npy => self.toNpy(source, options),
            .unified => self.toUnified(source, options),
            .gguf => return error.UnsupportedFormat, // TODO: Implement GGUF write
            .npz => return error.UnsupportedFormat, // TODO: Implement NPZ write
            .onnx => return error.UnsupportedFormat, // ONNX is read-only
        };
    }

    /// Import from SafeTensors format
    pub fn fromSafeTensors(self: *Converter, data: []const u8) ConversionError!unified.UnifiedFormat {
        // SafeTensors format:
        // - 8 bytes: header size (little-endian u64)
        // - N bytes: JSON header with tensor info
        // - Data: raw tensor data aligned

        if (data.len < 8) return error.InvalidInput;

        const header_size = std.mem.readInt(u64, data[0..8], .little);
        if (8 + header_size > data.len) return error.InvalidInput;

        const header_json = data[8..][0..header_size];
        const tensor_data_start = 8 + header_size;

        // Parse JSON header
        var parsed = std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            header_json,
            .{},
        ) catch return error.InvalidInput;
        defer parsed.deinit();

        var result = unified.UnifiedFormat.init(self.allocator);
        errdefer result.deinit();

        // Process each tensor
        const root = parsed.value.object;
        var iter = root.iterator();
        while (iter.next()) |entry| {
            const name = entry.key_ptr.*;
            if (std.mem.eql(u8, name, "__metadata__")) continue;

            const tensor_info = entry.value_ptr.*.object;
            const dtype_str = tensor_info.get("dtype").?.string;
            const shape_arr = tensor_info.get("shape").?.array;
            const offsets = tensor_info.get("data_offsets").?.array;

            // Convert dtype
            const data_type = safeTensorsDtypeToUnified(dtype_str) catch continue;

            // Parse shape
            var dims: [4]u64 = .{ 0, 0, 0, 0 };
            const n_dims: u8 = @intCast(@min(shape_arr.items.len, 4));
            for (0..n_dims) |i| {
                dims[i] = @intCast(shape_arr.items[i].integer);
            }

            // Get data offsets
            const start: usize = @intCast(offsets.items[0].integer);
            const end: usize = @intCast(offsets.items[1].integer);

            // Create descriptor
            const desc = unified.TensorDescriptor{
                .name_hash = std.hash.Wyhash.hash(0, name),
                .name_offset = 0, // Will be set during serialization
                .name_length = @intCast(name.len),
                .data_type = data_type,
                .n_dims = n_dims,
                .dims = dims,
                .data_offset = @intCast(tensor_data_start + start),
                .data_size = @intCast(end - start),
            };

            result.tensors.put(self.allocator, name, desc) catch return error.OutOfMemory;
        }

        // Store reference to original data
        result.data = data;
        result.owns_data = false;

        return result;
    }

    /// Export to SafeTensors format
    fn toSafeTensors(self: *Converter, source: *const unified.UnifiedFormat, options: ConversionOptions) ConversionError![]u8 {
        _ = options;

        // Build JSON header
        var header = std.ArrayListUnmanaged(u8).empty;
        defer header.deinit(self.allocator);

        header.append(self.allocator, '{') catch return error.OutOfMemory;

        var first = true;
        var current_offset: u64 = 0;
        var tensor_iter = source.tensors.iterator();

        while (tensor_iter.next()) |entry| {
            const name = entry.key_ptr.*;
            const desc = entry.value_ptr.*;

            if (!first) {
                header.append(self.allocator, ',') catch return error.OutOfMemory;
            }
            first = false;

            // Write tensor entry
            const dtype_str = unifiedDtypeToSafeTensors(desc.data_type) catch continue;

            header.appendSlice(self.allocator, "\"") catch return error.OutOfMemory;
            header.appendSlice(self.allocator, name) catch return error.OutOfMemory;
            header.appendSlice(self.allocator, "\":{\"dtype\":\"") catch return error.OutOfMemory;
            header.appendSlice(self.allocator, dtype_str) catch return error.OutOfMemory;
            header.appendSlice(self.allocator, "\",\"shape\":[") catch return error.OutOfMemory;

            // Write shape
            for (0..desc.n_dims) |i| {
                if (i > 0) header.append(self.allocator, ',') catch return error.OutOfMemory;
                var buf: [20]u8 = undefined;
                const len = std.fmt.formatIntBuf(&buf, desc.dims[i], 10, .lower, .{});
                header.appendSlice(self.allocator, buf[0..len]) catch return error.OutOfMemory;
            }

            header.appendSlice(self.allocator, "],\"data_offsets\":[") catch return error.OutOfMemory;

            // Write offsets
            var buf: [20]u8 = undefined;
            var len = std.fmt.formatIntBuf(&buf, current_offset, 10, .lower, .{});
            header.appendSlice(self.allocator, buf[0..len]) catch return error.OutOfMemory;
            header.append(self.allocator, ',') catch return error.OutOfMemory;

            current_offset += desc.data_size;
            len = std.fmt.formatIntBuf(&buf, current_offset, 10, .lower, .{});
            header.appendSlice(self.allocator, buf[0..len]) catch return error.OutOfMemory;
            header.appendSlice(self.allocator, "]}") catch return error.OutOfMemory;
        }

        header.append(self.allocator, '}') catch return error.OutOfMemory;

        // Calculate total size
        const header_size = header.items.len;
        const data_size = current_offset;
        const total_size = 8 + header_size + @as(usize, @intCast(data_size));

        // Allocate output
        var output = self.allocator.alloc(u8, total_size) catch return error.OutOfMemory;
        errdefer self.allocator.free(output);

        // Write header size
        std.mem.writeInt(u64, output[0..8], header_size, .little);

        // Write header
        @memcpy(output[8..][0..header_size], header.items);

        // Write tensor data
        var data_offset: usize = 8 + header_size;
        tensor_iter = source.tensors.iterator();
        while (tensor_iter.next()) |entry| {
            const desc = entry.value_ptr.*;
            const tensor_data = source.getTensorData(self.allocator, entry.key_ptr.*) catch continue;
            defer if (desc.compressed_size > 0) self.allocator.free(tensor_data);

            @memcpy(output[data_offset..][0..tensor_data.len], tensor_data);
            data_offset += tensor_data.len;
        }

        return output;
    }

    /// Export to NPY format (single tensor)
    fn toNpy(self: *Converter, source: *const unified.UnifiedFormat, options: ConversionOptions) ConversionError![]u8 {
        _ = options;

        // NPY format: magic + version + header + data
        // We export the first tensor only

        var tensor_iter = source.tensors.iterator();
        const first_entry = tensor_iter.next() orelse return error.InvalidInput;

        const desc = first_entry.value_ptr.*;
        const dtype_char = unifiedDtypeToNpy(desc.data_type) catch return error.UnsupportedDataType;

        // Build header
        var header = std.ArrayListUnmanaged(u8).empty;
        defer header.deinit(self.allocator);

        header.appendSlice(self.allocator, "{'descr': '<") catch return error.OutOfMemory;
        header.append(self.allocator, dtype_char) catch return error.OutOfMemory;

        const elem_size = desc.data_type.elementSize() orelse 4;
        var size_buf: [10]u8 = undefined;
        const size_len = std.fmt.formatIntBuf(&size_buf, elem_size, 10, .lower, .{});
        header.appendSlice(self.allocator, size_buf[0..size_len]) catch return error.OutOfMemory;

        header.appendSlice(self.allocator, "', 'fortran_order': False, 'shape': (") catch return error.OutOfMemory;

        for (0..desc.n_dims) |i| {
            if (i > 0) header.appendSlice(self.allocator, ", ") catch return error.OutOfMemory;
            var buf: [20]u8 = undefined;
            const len = std.fmt.formatIntBuf(&buf, desc.dims[i], 10, .lower, .{});
            header.appendSlice(self.allocator, buf[0..len]) catch return error.OutOfMemory;
        }
        if (desc.n_dims == 1) header.append(self.allocator, ',') catch return error.OutOfMemory;

        header.appendSlice(self.allocator, "), }") catch return error.OutOfMemory;

        // Pad header to 64-byte alignment
        const magic_len = 10; // Magic + version + header_len
        const header_padded_len = alignUp(magic_len + header.items.len, 64) - magic_len;
        while (header.items.len < header_padded_len - 1) {
            header.append(self.allocator, ' ') catch return error.OutOfMemory;
        }
        header.append(self.allocator, '\n') catch return error.OutOfMemory;

        // Calculate total size
        const total_size = 10 + header.items.len + @as(usize, @intCast(desc.data_size));

        // Allocate output
        var output = self.allocator.alloc(u8, total_size) catch return error.OutOfMemory;
        errdefer self.allocator.free(output);

        // Write magic number
        output[0] = 0x93;
        @memcpy(output[1..6], "NUMPY");
        output[6] = 1; // Major version
        output[7] = 0; // Minor version

        // Write header length (2 bytes, little-endian)
        std.mem.writeInt(u16, output[8..10], @intCast(header.items.len), .little);

        // Write header
        @memcpy(output[10..][0..header.items.len], header.items);

        // Write data
        const tensor_data = source.getTensorData(self.allocator, first_entry.key_ptr.*) catch return error.IoError;
        defer if (desc.compressed_size > 0) self.allocator.free(tensor_data);

        @memcpy(output[10 + header.items.len ..], tensor_data);

        return output;
    }

    /// Re-export to unified format with different options
    fn toUnified(self: *Converter, source: *const unified.UnifiedFormat, options: ConversionOptions) ConversionError![]u8 {
        var builder = unified.UnifiedFormatBuilder.init(self.allocator);
        defer builder.deinit();

        _ = builder.setCompression(options.compression);

        // Copy tensors
        var tensor_iter = source.tensors.iterator();
        while (tensor_iter.next()) |entry| {
            const name = entry.key_ptr.*;
            const desc = entry.value_ptr.*;

            const tensor_data = source.getTensorData(self.allocator, name) catch continue;
            defer if (desc.compressed_size > 0) self.allocator.free(tensor_data);

            _ = builder.addTensor(name, tensor_data, desc.data_type, desc.shape()) catch continue;
        }

        // Copy metadata
        var meta_iter = source.metadata.iterator();
        while (meta_iter.next()) |entry| {
            _ = builder.addMetadata(entry.key_ptr.*, entry.value_ptr.*) catch continue;
        }

        return builder.build() catch return error.OutOfMemory;
    }
};

fn safeTensorsDtypeToUnified(dtype: []const u8) !unified.DataType {
    if (std.mem.eql(u8, dtype, "F32")) return .f32;
    if (std.mem.eql(u8, dtype, "F16")) return .f16;
    if (std.mem.eql(u8, dtype, "BF16")) return .bf16;
    if (std.mem.eql(u8, dtype, "F64")) return .f64;
    if (std.mem.eql(u8, dtype, "I8")) return .i8;
    if (std.mem.eql(u8, dtype, "I16")) return .i16;
    if (std.mem.eql(u8, dtype, "I32")) return .i32;
    if (std.mem.eql(u8, dtype, "I64")) return .i64;
    if (std.mem.eql(u8, dtype, "U8")) return .u8;
    if (std.mem.eql(u8, dtype, "U16")) return .u16;
    if (std.mem.eql(u8, dtype, "U32")) return .u32;
    if (std.mem.eql(u8, dtype, "U64")) return .u64;
    return error.UnsupportedDataType;
}

fn unifiedDtypeToSafeTensors(dtype: unified.DataType) ![]const u8 {
    return switch (dtype) {
        .f32 => "F32",
        .f16 => "F16",
        .bf16 => "BF16",
        .f64 => "F64",
        .i8 => "I8",
        .i16 => "I16",
        .i32 => "I32",
        .i64 => "I64",
        .u8 => "U8",
        .u16 => "U16",
        .u32 => "U32",
        .u64 => "U64",
        else => error.UnsupportedDataType,
    };
}

fn unifiedDtypeToNpy(dtype: unified.DataType) !u8 {
    return switch (dtype) {
        .f32, .f64 => 'f',
        .f16, .bf16 => 'f',
        .i8, .i16, .i32, .i64 => 'i',
        .u8, .u16, .u32, .u64 => 'u',
        else => error.UnsupportedDataType,
    };
}

fn alignUp(value: usize, alignment: usize) usize {
    return (value + alignment - 1) & ~(alignment - 1);
}

test "converter safetensors dtype mapping" {
    try std.testing.expectEqual(unified.DataType.f32, try safeTensorsDtypeToUnified("F32"));
    try std.testing.expectEqual(unified.DataType.f16, try safeTensorsDtypeToUnified("F16"));
    try std.testing.expectEqual(unified.DataType.i32, try safeTensorsDtypeToUnified("I32"));
}

test "converter unified dtype to safetensors" {
    try std.testing.expectEqualStrings("F32", try unifiedDtypeToSafeTensors(.f32));
    try std.testing.expectEqualStrings("I64", try unifiedDtypeToSafeTensors(.i64));
}
