//! Quantized tensor types and dequantization routines.
//!
//! Supports Q4_0, Q4_1, Q5_0, Q5_1, and Q8_0 formats used by llama.cpp
//! for memory-efficient storage of model weights.

const std = @import("std");

/// Q4_0 quantization block: 32 values stored in 18 bytes.
/// Format: 2-byte f16 scale + 16 bytes of 4-bit values
pub const Q4_0Block = extern struct {
    scale: f16,
    quants: [16]u8, // 32 x 4-bit values packed (2 per byte)

    pub const BLOCK_SIZE: usize = 32;
    pub const BYTE_SIZE: usize = 18;

    /// Dequantize this block to f32 values.
    pub fn dequantize(self: *const Q4_0Block, output: *[32]f32) void {
        const scale: f32 = @floatCast(self.scale);

        for (0..16) |i| {
            const byte = self.quants[i];
            // Low 4 bits: value - 8 to center around 0
            const lo: i8 = @as(i8, @intCast(byte & 0xF)) - 8;
            // High 4 bits
            const hi: i8 = @as(i8, @intCast(byte >> 4)) - 8;

            output[i] = @as(f32, @floatFromInt(lo)) * scale;
            output[i + 16] = @as(f32, @floatFromInt(hi)) * scale;
        }
    }
};

/// Q4_1 quantization block: 32 values stored in 20 bytes.
/// Format: 2-byte f16 scale + 2-byte f16 min + 16 bytes of 4-bit values
/// More accurate than Q4_0 due to min offset.
pub const Q4_1Block = extern struct {
    scale: f16,
    min: f16,
    quants: [16]u8, // 32 x 4-bit values packed (2 per byte)

    pub const BLOCK_SIZE: usize = 32;
    pub const BYTE_SIZE: usize = 20;

    /// Dequantize this block to f32 values.
    pub fn dequantize(self: *const Q4_1Block, output: *[32]f32) void {
        const scale: f32 = @floatCast(self.scale);
        const min: f32 = @floatCast(self.min);

        for (0..16) |i| {
            const byte = self.quants[i];
            // Low 4 bits: value * scale + min
            const lo: u8 = byte & 0xF;
            // High 4 bits
            const hi: u8 = byte >> 4;

            output[i] = @as(f32, @floatFromInt(lo)) * scale + min;
            output[i + 16] = @as(f32, @floatFromInt(hi)) * scale + min;
        }
    }

    /// Quantize f32 values to this block.
    pub fn quantize(self: *Q4_1Block, input: *const [32]f32) void {
        // Find min and max values
        var min_val: f32 = input[0];
        var max_val: f32 = input[0];
        for (input) |v| {
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }

        // Calculate scale and min
        const range = max_val - min_val;
        const scale: f32 = if (range > 0) range / 15.0 else 1.0;
        self.scale = @floatCast(scale);
        self.min = @floatCast(min_val);

        // Quantize values
        const inv_scale = if (scale > 0) 1.0 / scale else 0;
        for (0..16) |i| {
            const lo_scaled = (input[i] - min_val) * inv_scale;
            const hi_scaled = (input[i + 16] - min_val) * inv_scale;
            const lo_clamped = @max(0.0, @min(15.0, lo_scaled));
            const hi_clamped = @max(0.0, @min(15.0, hi_scaled));
            const lo: u8 = @intFromFloat(@round(lo_clamped));
            const hi: u8 = @intFromFloat(@round(hi_clamped));
            self.quants[i] = lo | (hi << 4);
        }
    }
};

/// Q5_0 quantization block: 32 values stored in 22 bytes.
/// Format: 2-byte f16 scale + 4-byte high bits + 16 bytes of low 4-bit values
/// Each value is stored as: 4 low bits in quants[], 1 high bit in qh
/// Total 5 bits per value, centered around 0 (-16 to 15 range)
pub const Q5_0Block = extern struct {
    scale: f16,
    qh: u32, // High bits (bit 4) for 32 values
    quants: [16]u8, // Low 4 bits for 32 values packed (2 per byte)

    pub const BLOCK_SIZE: usize = 32;
    pub const BYTE_SIZE: usize = 22;

    /// Dequantize this block to f32 values.
    pub fn dequantize(self: *const Q5_0Block, output: *[32]f32) void {
        const scale: f32 = @floatCast(self.scale);
        const qh = self.qh;

        for (0..16) |i| {
            const byte = self.quants[i];
            // Low 4 bits of first value
            const lo4: u8 = byte & 0xF;
            // Low 4 bits of second value
            const hi4: u8 = byte >> 4;
            // High bit for first value (bit i)
            const hb_lo: u8 = @truncate((qh >> @intCast(i)) & 1);
            // High bit for second value (bit i+16)
            const hb_hi: u8 = @truncate((qh >> @intCast(i + 16)) & 1);

            // Combine to 5-bit value and center around 0
            const lo5: i8 = @as(i8, @intCast(lo4 | (hb_lo << 4))) - 16;
            const hi5: i8 = @as(i8, @intCast(hi4 | (hb_hi << 4))) - 16;

            output[i] = @as(f32, @floatFromInt(lo5)) * scale;
            output[i + 16] = @as(f32, @floatFromInt(hi5)) * scale;
        }
    }

    /// Quantize f32 values to this block.
    pub fn quantize(self: *Q5_0Block, input: *const [32]f32) void {
        // Find max absolute value for scale
        var amax: f32 = 0;
        for (input) |v| {
            const abs_v = @abs(v);
            if (abs_v > amax) amax = abs_v;
        }

        // Calculate scale: 5-bit centered around 0 means range [-16, 15]
        const scale: f32 = if (amax > 0) amax / 15.0 else 1.0;
        self.scale = @floatCast(scale);

        const inv_scale = if (scale > 0) 1.0 / scale else 0;
        var qh: u32 = 0;

        for (0..16) |i| {
            // Quantize and center: add 16 to shift range from [-16,15] to [0,31]
            const lo_scaled = input[i] * inv_scale + 16.0;
            const hi_scaled = input[i + 16] * inv_scale + 16.0;

            const lo_clamped = @max(0.0, @min(31.0, lo_scaled));
            const hi_clamped = @max(0.0, @min(31.0, hi_scaled));

            const lo5: u8 = @intFromFloat(@round(lo_clamped));
            const hi5: u8 = @intFromFloat(@round(hi_clamped));

            // Extract low 4 bits for quants
            self.quants[i] = (lo5 & 0xF) | ((hi5 & 0xF) << 4);

            // Extract high bits for qh
            if (lo5 & 0x10 != 0) qh |= @as(u32, 1) << @intCast(i);
            if (hi5 & 0x10 != 0) qh |= @as(u32, 1) << @intCast(i + 16);
        }

        self.qh = qh;
    }
};

/// Q5_1 quantization block: 32 values stored in 24 bytes.
/// Format: 2-byte f16 scale + 2-byte f16 min + 4-byte high bits + 16 bytes of low 4-bit values
/// More accurate than Q5_0 due to min offset (asymmetric quantization).
pub const Q5_1Block = extern struct {
    scale: f16,
    min: f16,
    qh: u32, // High bits (bit 4) for 32 values
    quants: [16]u8, // Low 4 bits for 32 values packed (2 per byte)

    pub const BLOCK_SIZE: usize = 32;
    pub const BYTE_SIZE: usize = 24;

    /// Dequantize this block to f32 values.
    pub fn dequantize(self: *const Q5_1Block, output: *[32]f32) void {
        const scale: f32 = @floatCast(self.scale);
        const min: f32 = @floatCast(self.min);
        const qh = self.qh;

        for (0..16) |i| {
            const byte = self.quants[i];
            // Low 4 bits of values
            const lo4: u8 = byte & 0xF;
            const hi4: u8 = byte >> 4;
            // High bits
            const hb_lo: u8 = @truncate((qh >> @intCast(i)) & 1);
            const hb_hi: u8 = @truncate((qh >> @intCast(i + 16)) & 1);

            // Combine to 5-bit value (no centering, asymmetric)
            const lo5: u8 = lo4 | (hb_lo << 4);
            const hi5: u8 = hi4 | (hb_hi << 4);

            output[i] = @as(f32, @floatFromInt(lo5)) * scale + min;
            output[i + 16] = @as(f32, @floatFromInt(hi5)) * scale + min;
        }
    }

    /// Quantize f32 values to this block.
    pub fn quantize(self: *Q5_1Block, input: *const [32]f32) void {
        // Find min and max values
        var min_val: f32 = input[0];
        var max_val: f32 = input[0];
        for (input) |v| {
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }

        // Calculate scale and min
        const range = max_val - min_val;
        const scale: f32 = if (range > 0) range / 31.0 else 1.0;
        self.scale = @floatCast(scale);
        self.min = @floatCast(min_val);

        const inv_scale = if (scale > 0) 1.0 / scale else 0;
        var qh: u32 = 0;

        for (0..16) |i| {
            const lo_scaled = (input[i] - min_val) * inv_scale;
            const hi_scaled = (input[i + 16] - min_val) * inv_scale;

            const lo_clamped = @max(0.0, @min(31.0, lo_scaled));
            const hi_clamped = @max(0.0, @min(31.0, hi_scaled));

            const lo5: u8 = @intFromFloat(@round(lo_clamped));
            const hi5: u8 = @intFromFloat(@round(hi_clamped));

            // Extract low 4 bits for quants
            self.quants[i] = (lo5 & 0xF) | ((hi5 & 0xF) << 4);

            // Extract high bits for qh
            if (lo5 & 0x10 != 0) qh |= @as(u32, 1) << @intCast(i);
            if (hi5 & 0x10 != 0) qh |= @as(u32, 1) << @intCast(i + 16);
        }

        self.qh = qh;
    }
};

/// Q8_0 quantization block: 32 values stored in 34 bytes.
/// Format: 2-byte f16 scale + 32 bytes of 8-bit signed values
pub const Q8_0Block = extern struct {
    scale: f16,
    quants: [32]i8,

    pub const BLOCK_SIZE: usize = 32;
    pub const BYTE_SIZE: usize = 34;

    /// Dequantize this block to f32 values.
    pub fn dequantize(self: *const Q8_0Block, output: *[32]f32) void {
        const scale: f32 = @floatCast(self.scale);

        for (0..32) |i| {
            output[i] = @as(f32, @floatFromInt(self.quants[i])) * scale;
        }
    }

    /// Quantize f32 values to this block.
    pub fn quantize(self: *Q8_0Block, input: *const [32]f32) void {
        // Find max absolute value for scale
        var amax: f32 = 0;
        for (input) |v| {
            const abs_v = @abs(v);
            if (abs_v > amax) amax = abs_v;
        }

        // Calculate scale
        const scale: f32 = if (amax > 0) amax / 127.0 else 1.0;
        self.scale = @floatCast(scale);

        // Quantize values
        const inv_scale = if (scale > 0) 1.0 / scale else 0;
        for (input, 0..) |v, i| {
            const scaled = v * inv_scale;
            const clamped = @max(-128.0, @min(127.0, scaled));
            self.quants[i] = @intFromFloat(@round(clamped));
        }
    }
};

/// Dequantize Q4_0 data to f32.
pub fn dequantizeQ4_0(data: []const u8, output: []f32) !void {
    const num_blocks = data.len / Q4_0Block.BYTE_SIZE;
    const expected_output = num_blocks * Q4_0Block.BLOCK_SIZE;

    if (output.len < expected_output) {
        return error.ShapeMismatch;
    }

    const blocks: []const Q4_0Block = @alignCast(std.mem.bytesAsSlice(
        Q4_0Block,
        data[0 .. num_blocks * Q4_0Block.BYTE_SIZE],
    ));

    for (blocks, 0..) |*block, i| {
        var block_output: [32]f32 = undefined;
        block.dequantize(&block_output);
        @memcpy(output[i * 32 .. (i + 1) * 32], &block_output);
    }
}

/// Dequantize Q4_1 data to f32.
pub fn dequantizeQ4_1(data: []const u8, output: []f32) !void {
    const num_blocks = data.len / Q4_1Block.BYTE_SIZE;
    const expected_output = num_blocks * Q4_1Block.BLOCK_SIZE;

    if (output.len < expected_output) {
        return error.ShapeMismatch;
    }

    const blocks: []const Q4_1Block = @alignCast(std.mem.bytesAsSlice(
        Q4_1Block,
        data[0 .. num_blocks * Q4_1Block.BYTE_SIZE],
    ));

    for (blocks, 0..) |*block, i| {
        var block_output: [32]f32 = undefined;
        block.dequantize(&block_output);
        @memcpy(output[i * 32 .. (i + 1) * 32], &block_output);
    }
}

/// Dequantize Q5_0 data to f32.
pub fn dequantizeQ5_0(data: []const u8, output: []f32) !void {
    const num_blocks = data.len / Q5_0Block.BYTE_SIZE;
    const expected_output = num_blocks * Q5_0Block.BLOCK_SIZE;

    if (output.len < expected_output) {
        return error.ShapeMismatch;
    }

    const blocks: []const Q5_0Block = @alignCast(std.mem.bytesAsSlice(
        Q5_0Block,
        data[0 .. num_blocks * Q5_0Block.BYTE_SIZE],
    ));

    for (blocks, 0..) |*block, i| {
        var block_output: [32]f32 = undefined;
        block.dequantize(&block_output);
        @memcpy(output[i * 32 .. (i + 1) * 32], &block_output);
    }
}

/// Dequantize Q5_1 data to f32.
pub fn dequantizeQ5_1(data: []const u8, output: []f32) !void {
    const num_blocks = data.len / Q5_1Block.BYTE_SIZE;
    const expected_output = num_blocks * Q5_1Block.BLOCK_SIZE;

    if (output.len < expected_output) {
        return error.ShapeMismatch;
    }

    const blocks: []const Q5_1Block = @alignCast(std.mem.bytesAsSlice(
        Q5_1Block,
        data[0 .. num_blocks * Q5_1Block.BYTE_SIZE],
    ));

    for (blocks, 0..) |*block, i| {
        var block_output: [32]f32 = undefined;
        block.dequantize(&block_output);
        @memcpy(output[i * 32 .. (i + 1) * 32], &block_output);
    }
}

/// Dequantize Q8_0 data to f32.
pub fn dequantizeQ8_0(data: []const u8, output: []f32) !void {
    const num_blocks = data.len / Q8_0Block.BYTE_SIZE;
    const expected_output = num_blocks * Q8_0Block.BLOCK_SIZE;

    if (output.len < expected_output) {
        return error.ShapeMismatch;
    }

    const blocks: []const Q8_0Block = @alignCast(std.mem.bytesAsSlice(
        Q8_0Block,
        data[0 .. num_blocks * Q8_0Block.BYTE_SIZE],
    ));

    for (blocks, 0..) |*block, i| {
        var block_output: [32]f32 = undefined;
        block.dequantize(&block_output);
        @memcpy(output[i * 32 .. (i + 1) * 32], &block_output);
    }
}

/// Quantize f32 data to Q8_0.
pub fn quantizeToQ8_0(input: []const f32, output: []u8) !void {
    const num_blocks = input.len / Q8_0Block.BLOCK_SIZE;
    const expected_output = num_blocks * Q8_0Block.BYTE_SIZE;

    if (output.len < expected_output) {
        return error.ShapeMismatch;
    }
    if (input.len % Q8_0Block.BLOCK_SIZE != 0) {
        return error.ShapeMismatch;
    }

    var blocks: []Q8_0Block = @alignCast(std.mem.bytesAsSlice(
        Q8_0Block,
        output[0..expected_output],
    ));

    for (0..num_blocks) |i| {
        const start = i * 32;
        const block_input: *const [32]f32 = @ptrCast(input[start .. start + 32]);
        blocks[i].quantize(block_input);
    }
}

/// Quantize f32 data to Q4_1.
pub fn quantizeToQ4_1(input: []const f32, output: []u8) !void {
    const num_blocks = input.len / Q4_1Block.BLOCK_SIZE;
    const expected_output = num_blocks * Q4_1Block.BYTE_SIZE;

    if (output.len < expected_output) {
        return error.ShapeMismatch;
    }
    if (input.len % Q4_1Block.BLOCK_SIZE != 0) {
        return error.ShapeMismatch;
    }

    var blocks: []Q4_1Block = @alignCast(std.mem.bytesAsSlice(
        Q4_1Block,
        output[0..expected_output],
    ));

    for (0..num_blocks) |i| {
        const start = i * 32;
        const block_input: *const [32]f32 = @ptrCast(input[start .. start + 32]);
        blocks[i].quantize(block_input);
    }
}

/// Quantize f32 data to Q5_0.
pub fn quantizeToQ5_0(input: []const f32, output: []u8) !void {
    const num_blocks = input.len / Q5_0Block.BLOCK_SIZE;
    const expected_output = num_blocks * Q5_0Block.BYTE_SIZE;

    if (output.len < expected_output) {
        return error.ShapeMismatch;
    }
    if (input.len % Q5_0Block.BLOCK_SIZE != 0) {
        return error.ShapeMismatch;
    }

    var blocks: []Q5_0Block = @alignCast(std.mem.bytesAsSlice(
        Q5_0Block,
        output[0..expected_output],
    ));

    for (0..num_blocks) |i| {
        const start = i * 32;
        const block_input: *const [32]f32 = @ptrCast(input[start .. start + 32]);
        blocks[i].quantize(block_input);
    }
}

/// Quantize f32 data to Q5_1.
pub fn quantizeToQ5_1(input: []const f32, output: []u8) !void {
    const num_blocks = input.len / Q5_1Block.BLOCK_SIZE;
    const expected_output = num_blocks * Q5_1Block.BYTE_SIZE;

    if (output.len < expected_output) {
        return error.ShapeMismatch;
    }
    if (input.len % Q5_1Block.BLOCK_SIZE != 0) {
        return error.ShapeMismatch;
    }

    var blocks: []Q5_1Block = @alignCast(std.mem.bytesAsSlice(
        Q5_1Block,
        output[0..expected_output],
    ));

    for (0..num_blocks) |i| {
        const start = i * 32;
        const block_input: *const [32]f32 = @ptrCast(input[start .. start + 32]);
        blocks[i].quantize(block_input);
    }
}

/// Calculate Q4_0 dequantized dot product with f32 vector.
/// Optimized for vectorized operations.
pub fn dotQ4_0F32(q4_data: []const u8, f32_data: []const f32) f32 {
    const num_blocks = q4_data.len / Q4_0Block.BYTE_SIZE;
    const blocks: []const Q4_0Block = @alignCast(std.mem.bytesAsSlice(
        Q4_0Block,
        q4_data[0 .. num_blocks * Q4_0Block.BYTE_SIZE],
    ));

    var sum: f32 = 0;

    for (blocks, 0..) |*block, block_idx| {
        const scale: f32 = @floatCast(block.scale);
        const base = block_idx * 32;

        // Accumulate for this block
        var block_sum: f32 = 0;

        for (0..16) |i| {
            const byte = block.quants[i];
            const lo: i8 = @as(i8, @intCast(byte & 0xF)) - 8;
            const hi: i8 = @as(i8, @intCast(byte >> 4)) - 8;

            block_sum += @as(f32, @floatFromInt(lo)) * f32_data[base + i];
            block_sum += @as(f32, @floatFromInt(hi)) * f32_data[base + i + 16];
        }

        sum += block_sum * scale;
    }

    return sum;
}

/// Calculate Q4_1 dequantized dot product with f32 vector.
/// More accurate than Q4_0 due to min offset.
pub fn dotQ4_1F32(q4_data: []const u8, f32_data: []const f32) f32 {
    const num_blocks = q4_data.len / Q4_1Block.BYTE_SIZE;
    const blocks: []const Q4_1Block = @alignCast(std.mem.bytesAsSlice(
        Q4_1Block,
        q4_data[0 .. num_blocks * Q4_1Block.BYTE_SIZE],
    ));

    var sum: f32 = 0;

    for (blocks, 0..) |*block, block_idx| {
        const scale: f32 = @floatCast(block.scale);
        const min: f32 = @floatCast(block.min);
        const base = block_idx * 32;

        // Accumulate for this block
        var block_sum: f32 = 0;
        var min_sum: f32 = 0;

        for (0..16) |i| {
            const byte = block.quants[i];
            const lo: u8 = byte & 0xF;
            const hi: u8 = byte >> 4;

            block_sum += @as(f32, @floatFromInt(lo)) * f32_data[base + i];
            block_sum += @as(f32, @floatFromInt(hi)) * f32_data[base + i + 16];
            min_sum += f32_data[base + i] + f32_data[base + i + 16];
        }

        sum += block_sum * scale + min_sum * min;
    }

    return sum;
}

/// Calculate Q5_0 dequantized dot product with f32 vector.
pub fn dotQ5_0F32(q5_data: []const u8, f32_data: []const f32) f32 {
    const num_blocks = q5_data.len / Q5_0Block.BYTE_SIZE;
    const blocks: []const Q5_0Block = @alignCast(std.mem.bytesAsSlice(
        Q5_0Block,
        q5_data[0 .. num_blocks * Q5_0Block.BYTE_SIZE],
    ));

    var sum: f32 = 0;

    for (blocks, 0..) |*block, block_idx| {
        const scale: f32 = @floatCast(block.scale);
        const qh = block.qh;
        const base = block_idx * 32;

        var block_sum: f32 = 0;

        for (0..16) |i| {
            const byte = block.quants[i];
            const lo4: u8 = byte & 0xF;
            const hi4: u8 = byte >> 4;
            const hb_lo: u8 = @truncate((qh >> @intCast(i)) & 1);
            const hb_hi: u8 = @truncate((qh >> @intCast(i + 16)) & 1);

            const lo5: i8 = @as(i8, @intCast(lo4 | (hb_lo << 4))) - 16;
            const hi5: i8 = @as(i8, @intCast(hi4 | (hb_hi << 4))) - 16;

            block_sum += @as(f32, @floatFromInt(lo5)) * f32_data[base + i];
            block_sum += @as(f32, @floatFromInt(hi5)) * f32_data[base + i + 16];
        }

        sum += block_sum * scale;
    }

    return sum;
}

/// Calculate Q5_1 dequantized dot product with f32 vector.
pub fn dotQ5_1F32(q5_data: []const u8, f32_data: []const f32) f32 {
    const num_blocks = q5_data.len / Q5_1Block.BYTE_SIZE;
    const blocks: []const Q5_1Block = @alignCast(std.mem.bytesAsSlice(
        Q5_1Block,
        q5_data[0 .. num_blocks * Q5_1Block.BYTE_SIZE],
    ));

    var sum: f32 = 0;

    for (blocks, 0..) |*block, block_idx| {
        const scale: f32 = @floatCast(block.scale);
        const min: f32 = @floatCast(block.min);
        const qh = block.qh;
        const base = block_idx * 32;

        var block_sum: f32 = 0;
        var min_sum: f32 = 0;

        for (0..16) |i| {
            const byte = block.quants[i];
            const lo4: u8 = byte & 0xF;
            const hi4: u8 = byte >> 4;
            const hb_lo: u8 = @truncate((qh >> @intCast(i)) & 1);
            const hb_hi: u8 = @truncate((qh >> @intCast(i + 16)) & 1);

            const lo5: u8 = lo4 | (hb_lo << 4);
            const hi5: u8 = hi4 | (hb_hi << 4);

            block_sum += @as(f32, @floatFromInt(lo5)) * f32_data[base + i];
            block_sum += @as(f32, @floatFromInt(hi5)) * f32_data[base + i + 16];
            min_sum += f32_data[base + i] + f32_data[base + i + 16];
        }

        sum += block_sum * scale + min_sum * min;
    }

    return sum;
}

/// Calculate Q8_0 dequantized dot product with f32 vector.
pub fn dotQ8_0F32(q8_data: []const u8, f32_data: []const f32) f32 {
    const num_blocks = q8_data.len / Q8_0Block.BYTE_SIZE;
    const blocks: []const Q8_0Block = @alignCast(std.mem.bytesAsSlice(
        Q8_0Block,
        q8_data[0 .. num_blocks * Q8_0Block.BYTE_SIZE],
    ));

    var sum: f32 = 0;

    for (blocks, 0..) |*block, block_idx| {
        const scale: f32 = @floatCast(block.scale);
        const base = block_idx * 32;

        var block_sum: f32 = 0;
        for (0..32) |i| {
            block_sum += @as(f32, @floatFromInt(block.quants[i])) * f32_data[base + i];
        }

        sum += block_sum * scale;
    }

    return sum;
}

/// Quantization type enumeration.
pub const QuantType = enum {
    q4_0,
    q4_1,
    q5_0,
    q5_1,
    q8_0,

    /// Get the block size for this quantization type.
    pub fn blockSize(_: QuantType) usize {
        return 32; // All types use 32-element blocks
    }

    /// Get the byte size per block for this quantization type.
    pub fn byteSize(self: QuantType) usize {
        return switch (self) {
            .q4_0 => Q4_0Block.BYTE_SIZE,
            .q4_1 => Q4_1Block.BYTE_SIZE,
            .q5_0 => Q5_0Block.BYTE_SIZE,
            .q5_1 => Q5_1Block.BYTE_SIZE,
            .q8_0 => Q8_0Block.BYTE_SIZE,
        };
    }

    /// Get bits per weight for this quantization type.
    pub fn bitsPerWeight(self: QuantType) f32 {
        return switch (self) {
            .q4_0 => 4.5, // 18 bytes / 32 values * 8
            .q4_1 => 5.0, // 20 bytes / 32 values * 8
            .q5_0 => 5.5, // 22 bytes / 32 values * 8
            .q5_1 => 6.0, // 24 bytes / 32 values * 8
            .q8_0 => 8.5, // 34 bytes / 32 values * 8
        };
    }
};

/// Memory required for dequantized f32 representation.
pub fn dequantizedSize(quant_type: QuantType, num_elements: usize) usize {
    _ = quant_type;
    return num_elements * @sizeOf(f32);
}

/// Memory required for quantized representation.
pub fn quantizedSize(quant_type: QuantType, num_elements: usize) usize {
    const num_blocks = (num_elements + 31) / 32;
    return switch (quant_type) {
        .q4_0 => num_blocks * Q4_0Block.BYTE_SIZE,
        .q4_1 => num_blocks * Q4_1Block.BYTE_SIZE,
        .q5_0 => num_blocks * Q5_0Block.BYTE_SIZE,
        .q5_1 => num_blocks * Q5_1Block.BYTE_SIZE,
        .q8_0 => num_blocks * Q8_0Block.BYTE_SIZE,
    };
}

test "Q4_0 block dequantization" {
    var block: Q4_0Block = undefined;
    block.scale = @as(f16, 0.5);
    // Set all quants to 8 (which becomes 0 after subtracting 8)
    @memset(&block.quants, 0x88);

    var output: [32]f32 = undefined;
    block.dequantize(&output);

    // 8 - 8 = 0, so all values should be 0
    for (output) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), v, 0.001);
    }
}

test "Q8_0 roundtrip" {
    var input: [32]f32 = undefined;
    for (0..32) |i| {
        input[i] = @as(f32, @floatFromInt(i)) - 16.0;
    }

    var block: Q8_0Block = undefined;
    block.quantize(&input);

    var output: [32]f32 = undefined;
    block.dequantize(&output);

    // Should be close to original (within quantization error)
    for (input, 0..) |v, i| {
        try std.testing.expectApproxEqAbs(v, output[i], 0.5);
    }
}

test "Q4_1 roundtrip" {
    var input: [32]f32 = undefined;
    for (0..32) |i| {
        input[i] = @as(f32, @floatFromInt(i)) * 0.1;
    }

    var block: Q4_1Block = undefined;
    block.quantize(&input);

    var output: [32]f32 = undefined;
    block.dequantize(&output);

    // Should be close to original (within quantization error)
    // Q4_1 has 4-bit precision (16 levels), so error tolerance is higher
    for (input, 0..) |v, i| {
        try std.testing.expectApproxEqAbs(v, output[i], 0.3);
    }
}

test "Q5_0 roundtrip" {
    var input: [32]f32 = undefined;
    for (0..32) |i| {
        input[i] = @as(f32, @floatFromInt(i)) - 16.0;
    }

    var block: Q5_0Block = undefined;
    block.quantize(&input);

    var output: [32]f32 = undefined;
    block.dequantize(&output);

    // Should be close to original (within 5-bit quantization error)
    for (input, 0..) |v, i| {
        try std.testing.expectApproxEqAbs(v, output[i], 1.5);
    }
}

test "Q5_1 roundtrip" {
    var input: [32]f32 = undefined;
    for (0..32) |i| {
        input[i] = @as(f32, @floatFromInt(i)) * 0.1;
    }

    var block: Q5_1Block = undefined;
    block.quantize(&input);

    var output: [32]f32 = undefined;
    block.dequantize(&output);

    // Should be close to original (within 5-bit quantization error)
    for (input, 0..) |v, i| {
        try std.testing.expectApproxEqAbs(v, output[i], 0.15);
    }
}

test "quantized size calculations" {
    // 1024 elements = 32 blocks for Q4_0
    try std.testing.expectEqual(@as(usize, 32 * 18), quantizedSize(.q4_0, 1024));
    // 1024 elements = 32 blocks for Q4_1 (20 bytes per block)
    try std.testing.expectEqual(@as(usize, 32 * 20), quantizedSize(.q4_1, 1024));
    // 1024 elements = 32 blocks for Q5_0 (22 bytes per block)
    try std.testing.expectEqual(@as(usize, 32 * 22), quantizedSize(.q5_0, 1024));
    // 1024 elements = 32 blocks for Q5_1 (24 bytes per block)
    try std.testing.expectEqual(@as(usize, 32 * 24), quantizedSize(.q5_1, 1024));
    // 1024 elements = 32 blocks for Q8_0
    try std.testing.expectEqual(@as(usize, 32 * 34), quantizedSize(.q8_0, 1024));
}

test "quantization bits per weight" {
    try std.testing.expectApproxEqAbs(@as(f32, 4.5), QuantType.q4_0.bitsPerWeight(), 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), QuantType.q4_1.bitsPerWeight(), 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 5.5), QuantType.q5_0.bitsPerWeight(), 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), QuantType.q5_1.bitsPerWeight(), 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 8.5), QuantType.q8_0.bitsPerWeight(), 0.01);
}

test {
    std.testing.refAllDecls(@This());
}
