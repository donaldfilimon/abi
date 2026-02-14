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
            .bf16 => {
                const src: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, data));
                for (src, 0..) |v, i| {
                    result[i] = bf16ToF32(v);
                }
            },
            .q4_0 => try dequantizeQ4_0(data, result),
            .q8_0 => try dequantizeQ8_0(data, result),
            .q6_k => try dequantizeQ6_K(data, result),
            .mxfp4 => try dequantizeMXFP4(data, result),
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

/// MXFP4 block: 32 values with shared E8M0 scale.
/// Layout matches ggml `block_mxfp4`.
pub const MXFP4Block = extern struct {
    e: u8,
    qs: [16]u8,
};

/// Q6_K block: 256 values with per-16 scale and a global scale
pub const Q6_KBlock = extern struct {
    ql: [128]u8, // low 4 bits
    qh: [64]u8, // high 2 bits
    scales: [16]i8, // per-16 scale
    d: f16, // global scale
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

/// Dequantize Q6_K data to f32
pub fn dequantizeQ6_K(data: []const u8, output: []f32) !void {
    const block_size = 256;
    const bytes_per_block = @sizeOf(Q6_KBlock);
    const num_blocks = data.len / bytes_per_block;

    if (output.len < num_blocks * block_size) {
        return error.ShapeMismatch;
    }

    const blocks: []const Q6_KBlock = @alignCast(std.mem.bytesAsSlice(Q6_KBlock, data[0 .. num_blocks * bytes_per_block]));

    for (blocks, 0..) |block, block_idx| {
        const d: f32 = @floatCast(block.d);
        var ql_index: usize = 0;
        var qh_index: usize = 0;
        var sc_index: usize = 0;
        var out_index: usize = block_idx * block_size;

        var n: usize = 0;
        while (n < block_size) : (n += 128) {
            for (0..32) |l| {
                const is: usize = l / 16;
                const ql0 = block.ql[ql_index + l];
                const ql32 = block.ql[ql_index + l + 32];
                const qh = block.qh[qh_index + l];

                const q1 = @as(i32, @intCast(ql0 & 0xF)) | (@as(i32, @intCast((qh >> 0) & 3)) << 4);
                const q2 = @as(i32, @intCast(ql32 & 0xF)) | (@as(i32, @intCast((qh >> 2) & 3)) << 4);
                const q3 = @as(i32, @intCast(ql0 >> 4)) | (@as(i32, @intCast((qh >> 4) & 3)) << 4);
                const q4 = @as(i32, @intCast(ql32 >> 4)) | (@as(i32, @intCast((qh >> 6) & 3)) << 4);

                const sc0 = block.scales[sc_index + is + 0];
                const sc2 = block.scales[sc_index + is + 2];
                const sc4 = block.scales[sc_index + is + 4];
                const sc6 = block.scales[sc_index + is + 6];

                output[out_index + l + 0] = d * @as(f32, @floatFromInt(sc0)) * @as(f32, @floatFromInt(q1 - 32));
                output[out_index + l + 32] = d * @as(f32, @floatFromInt(sc2)) * @as(f32, @floatFromInt(q2 - 32));
                output[out_index + l + 64] = d * @as(f32, @floatFromInt(sc4)) * @as(f32, @floatFromInt(q3 - 32));
                output[out_index + l + 96] = d * @as(f32, @floatFromInt(sc6)) * @as(f32, @floatFromInt(q4 - 32));
            }

            out_index += 128;
            ql_index += 64;
            qh_index += 32;
            sc_index += 8;
        }
    }
}

const mxfp4_values = [_]i8{
    0, 1, 2, 3, 4, 6, 8, 12,
    0, -1, -2, -3, -4, -6, -8, -12,
};

fn e8m0ToF32Half(x: u8) f32 {
    const bits: u32 = if (x < 2)
        @as(u32, 0x0020_0000) << @intCast(x)
    else
        @as(u32, x - 1) << 23;
    return @bitCast(bits);
}

pub fn bf16ToF32(value: u16) f32 {
    const bits: u32 = @as(u32, value) << 16;
    return @bitCast(bits);
}

/// Dequantize MXFP4 data to f32.
pub fn dequantizeMXFP4(data: []const u8, output: []f32) !void {
    const block_size = 32;
    const bytes_per_block = @sizeOf(MXFP4Block);
    const num_blocks = data.len / bytes_per_block;

    if (output.len < num_blocks * block_size) {
        return error.ShapeMismatch;
    }

    const blocks: []const MXFP4Block = @alignCast(std.mem.bytesAsSlice(MXFP4Block, data[0 .. num_blocks * bytes_per_block]));

    for (blocks, 0..) |block, block_idx| {
        const d = e8m0ToF32Half(block.e);
        const base = block_idx * block_size;

        for (0..16) |j| {
            const packed_byte = block.qs[j];
            const lo = mxfp4_values[@as(usize, packed_byte & 0x0F)];
            const hi = mxfp4_values[@as(usize, packed_byte >> 4)];

            output[base + j] = @as(f32, @floatFromInt(lo)) * d;
            output[base + j + 16] = @as(f32, @floatFromInt(hi)) * d;
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
    // Pack two 4-bit values per byte: lo nibble = i, hi nibble = i
    // Values 0..15 fit in 4 bits; dequantization subtracts 8 to center around zero.
    for (0..16) |i| {
        const v: u8 = @intCast(i);
        block.quants[i] = v | (v << 4);
    }

    var output: [32]f32 = undefined;
    try dequantizeQ4_0(&block_data, &output);

    // Q4_0: output[i] = lo nibble of byte i (i=0..15), output[16+i] = hi nibble of byte i
    // Byte 0: lo=0 -> (0-8)*0.5 = -4.0, Byte 2: lo=2 -> (2-8)*0.5 = -3.0
    try std.testing.expectApproxEqAbs(@as(f32, -4.0), output[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, -3.0), output[2], 0.01);
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

test "q6_k dequantization" {
    var block_data: [@sizeOf(Q6_KBlock)]u8 = undefined;
    const block: *Q6_KBlock = @ptrCast(@alignCast(&block_data));

    @memset(block.ql[0..], 0);
    @memset(block.qh[0..], 0);
    for (block.scales[0..]) |*s| {
        s.* = 1;
    }
    block.d = @as(f16, 1.0);

    var output: [256]f32 = undefined;
    try dequantizeQ6_K(block_data[0..], output[0..]);

    for (output) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, -32.0), v, 0.001);
    }
}

test "mxfp4 dequantization" {
    var block_data: [@sizeOf(MXFP4Block)]u8 = undefined;
    const block: *MXFP4Block = @ptrCast(@alignCast(&block_data));

    // Produces d = 1.0 in e8m0ToF32Half.
    block.e = 128;
    for (0..16) |i| {
        block.qs[i] = 0x21; // low nibble = +1, high nibble = +2
    }

    var output: [32]f32 = undefined;
    try dequantizeMXFP4(block_data[0..], output[0..]);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), output[16], 0.001);
}
