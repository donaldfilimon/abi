//! Tensor loading utilities for deserializing weights from GGUF files.
//!
//! Handles loading tensors from memory-mapped files, including on-the-fly
//! dequantization from Q4_0, Q8_0, and other quantized formats.

const std = @import("std");
const gguf = @import("gguf.zig");
const mmap = @import("mmap.zig");

pub const TensorLoader = struct {
    allocator: std.mem.Allocator,
    gguf_file: *const gguf.GgufFile,

    pub fn init(allocator: std.mem.Allocator, gguf_file: *const gguf.GgufFile) TensorLoader {
        return .{
            .allocator = allocator,
            .gguf_file = gguf_file,
        };
    }

    /// Load a tensor as f32, dequantizing if necessary.
    pub fn loadAsF32(self: *TensorLoader, name: []const u8) ![]f32 {
        const info = self.gguf_file.getTensor(name) orelse return error.TensorNotFound;
        const data = self.gguf_file.getTensorData(name) orelse return error.TensorNotFound;

        const element_count = info.elementCount();
        const result = try self.allocator.alloc(f32, @intCast(element_count));
        errdefer self.allocator.free(result);

        switch (info.tensor_type) {
            .f32 => {
                const src: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, data));
                @memcpy(result, src);
            },
            .f16 => {
                const src: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, data));
                for (src, 0..) |v, i| {
                    result[i] = @floatCast(v);
                }
            },
            .q4_0 => try dequantizeQ4_0(data, result),
            .q8_0 => try dequantizeQ8_0(data, result),
            else => return error.UnsupportedQuantization,
        }

        return result;
    }

    /// Load a tensor as raw bytes (no conversion).
    pub fn loadRaw(self: *TensorLoader, name: []const u8) ![]const u8 {
        return self.gguf_file.getTensorData(name) orelse error.TensorNotFound;
    }

    /// Get tensor info without loading data.
    pub fn getInfo(self: *TensorLoader, name: []const u8) ?gguf.TensorInfo {
        return self.gguf_file.getTensor(name);
    }

    /// Check if a tensor exists.
    pub fn hasTensor(self: *TensorLoader, name: []const u8) bool {
        return self.gguf_file.getTensor(name) != null;
    }

    /// Get list of all tensor names.
    pub fn getTensorNames(self: *TensorLoader) []const []const u8 {
        var names = std.ArrayListUnmanaged([]const u8){};
        errdefer names.deinit(self.allocator);

        var iter = self.gguf_file.tensors.keyIterator();
        while (iter.next()) |key| {
            names.append(self.allocator, key.*) catch continue;
        }

        return names.toOwnedSlice(self.allocator) catch &[_][]const u8{};
    }
};

/// Q4_0 block: 32 x 4-bit values + f16 scale
pub const Q4_0Block = extern struct {
    scale: f16,
    quants: [16]u8, // 32 x 4-bit values packed
};

/// Q8_0 block: 32 x 8-bit values + f16 scale
pub const Q8_0Block = extern struct {
    scale: f16,
    quants: [32]i8,
};

/// Dequantize Q4_0 data to f32
pub fn dequantizeQ4_0(data: []const u8, output: []f32) !void {
    const block_size = 32;
    const bytes_per_block = @sizeOf(Q4_0Block);
    const num_blocks = data.len / bytes_per_block;

    if (output.len < num_blocks * block_size) {
        return error.ShapeMismatch;
    }

    const blocks: []const Q4_0Block = @alignCast(std.mem.bytesAsSlice(Q4_0Block, data[0 .. num_blocks * bytes_per_block]));

    for (blocks, 0..) |block, block_idx| {
        const scale: f32 = @floatCast(block.scale);
        const base = block_idx * block_size;

        // Each byte contains two 4-bit values
        for (0..16) |i| {
            const byte = block.quants[i];
            const lo: i8 = @as(i8, @intCast(byte & 0xF)) - 8;
            const hi: i8 = @as(i8, @intCast(byte >> 4)) - 8;

            output[base + i] = @as(f32, @floatFromInt(lo)) * scale;
            output[base + i + 16] = @as(f32, @floatFromInt(hi)) * scale;
        }
    }
}

/// Dequantize Q8_0 data to f32
pub fn dequantizeQ8_0(data: []const u8, output: []f32) !void {
    const block_size = 32;
    const bytes_per_block = @sizeOf(Q8_0Block);
    const num_blocks = data.len / bytes_per_block;

    if (output.len < num_blocks * block_size) {
        return error.ShapeMismatch;
    }

    const blocks: []const Q8_0Block = @alignCast(std.mem.bytesAsSlice(Q8_0Block, data[0 .. num_blocks * bytes_per_block]));

    for (blocks, 0..) |block, block_idx| {
        const scale: f32 = @floatCast(block.scale);
        const base = block_idx * block_size;

        for (0..32) |i| {
            output[base + i] = @as(f32, @floatFromInt(block.quants[i])) * scale;
        }
    }
}

/// Calculate total memory needed for dequantized tensors
pub fn calculateDequantizedSize(tensor_type: gguf.GgufTensorType, element_count: u64) u64 {
    _ = tensor_type; // All dequantize to f32
    return element_count * @sizeOf(f32);
}

test "q4_0 dequantization" {
    // Create a simple Q4_0 block
    var block_data: [@sizeOf(Q4_0Block)]u8 = undefined;
    const block: *Q4_0Block = @ptrCast(@alignCast(&block_data));

    block.scale = @as(f16, 0.5);
    // Pack two values per byte: (val + 8) for 4-bit unsigned
    // Value 0 -> 8, Value 1 -> 9, etc.
    for (0..16) |i| {
        block.quants[i] = @intCast((8 + i) | ((8 + i) << 4)); // lo = i, hi = i
    }

    var output: [32]f32 = undefined;
    try dequantizeQ4_0(&block_data, &output);

    // First value: (8 - 8) * 0.5 = 0.0, then (9 - 8) * 0.5 = 0.5, etc.
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), output[1], 0.001);
}

test "q8_0 dequantization" {
    var block_data: [@sizeOf(Q8_0Block)]u8 = undefined;
    const block: *Q8_0Block = @ptrCast(@alignCast(&block_data));

    block.scale = @as(f16, 0.1);
    for (0..32) |i| {
        block.quants[i] = @intCast(i);
    }

    var output: [32]f32 = undefined;
    try dequantizeQ8_0(&block_data, &output);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), output[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.1), output[31], 0.001);
}
