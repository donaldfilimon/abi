//! Quantized matrix multiplication for memory-efficient inference.
//!
//! Implements fused dequantization and matrix multiplication for
//! Q4_0 and Q8_0 quantized weights.

const std = @import("std");
const tensor_quantized = @import("../tensor/quantized.zig");
const Q4_0Block = tensor_quantized.Q4_0Block;
const Q8_0Block = tensor_quantized.Q8_0Block;

/// Quantized matrix-vector multiply: y = A_q @ x
/// A_q: [M, K] in Q4_0 format, x: [K] in f32, y: [M] in f32
pub fn quantizedMatmulQ4(
    a_quant: []const u8,
    x: []const f32,
    y: []f32,
    m: u32,
    k: u32,
) void {
    const blocks_per_row = k / Q4_0Block.BLOCK_SIZE;
    const bytes_per_row = blocks_per_row * Q4_0Block.BYTE_SIZE;

    for (0..m) |row| {
        const row_start = row * bytes_per_row;
        const row_data = a_quant[row_start .. row_start + bytes_per_row];
        y[row] = dotQ4F32(row_data, x);
    }
}

/// Quantized matrix-vector multiply for Q8_0.
pub fn quantizedMatmulQ8(
    a_quant: []const u8,
    x: []const f32,
    y: []f32,
    m: u32,
    k: u32,
) void {
    const blocks_per_row = k / Q8_0Block.BLOCK_SIZE;
    const bytes_per_row = blocks_per_row * Q8_0Block.BYTE_SIZE;

    for (0..m) |row| {
        const row_start = row * bytes_per_row;
        const row_data = a_quant[row_start .. row_start + bytes_per_row];
        y[row] = dotQ8F32(row_data, x);
    }
}

/// Q4_0 dot product with f32 vector.
fn dotQ4F32(q4_data: []const u8, f32_data: []const f32) f32 {
    const num_blocks = q4_data.len / Q4_0Block.BYTE_SIZE;
    const blocks: []const Q4_0Block = @alignCast(std.mem.bytesAsSlice(
        Q4_0Block,
        q4_data[0 .. num_blocks * Q4_0Block.BYTE_SIZE],
    ));

    var sum: f32 = 0;

    for (blocks, 0..) |*block, block_idx| {
        const scale: f32 = @floatCast(block.scale);
        const base = block_idx * 32;

        var block_sum: f32 = 0;

        // Unroll inner loop for better performance
        inline for (0..16) |i| {
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

/// Q8_0 dot product with f32 vector.
fn dotQ8F32(q8_data: []const u8, f32_data: []const f32) f32 {
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

        // Process 4 elements at a time
        var i: usize = 0;
        while (i + 4 <= 32) : (i += 4) {
            block_sum += @as(f32, @floatFromInt(block.quants[i + 0])) * f32_data[base + i + 0];
            block_sum += @as(f32, @floatFromInt(block.quants[i + 1])) * f32_data[base + i + 1];
            block_sum += @as(f32, @floatFromInt(block.quants[i + 2])) * f32_data[base + i + 2];
            block_sum += @as(f32, @floatFromInt(block.quants[i + 3])) * f32_data[base + i + 3];
        }

        sum += block_sum * scale;
    }

    return sum;
}

/// Batched quantized matmul for multiple vectors.
pub fn quantizedBatchMatmulQ8(
    a_quant: []const u8,
    x_batch: []const f32,
    y_batch: []f32,
    m: u32,
    k: u32,
    batch_size: u32,
) void {
    const blocks_per_row = k / Q8_0Block.BLOCK_SIZE;
    const bytes_per_row = blocks_per_row * Q8_0Block.BYTE_SIZE;

    for (0..batch_size) |b| {
        const x_offset = b * k;
        const y_offset = b * m;
        const x = x_batch[x_offset .. x_offset + k];
        const y = y_batch[y_offset .. y_offset + m];

        for (0..m) |row| {
            const row_start = row * bytes_per_row;
            const row_data = a_quant[row_start .. row_start + bytes_per_row];
            y[row] = dotQ8F32(row_data, x);
        }
    }
}

/// Calculate memory savings from quantization.
pub fn quantizationSavings(elements: usize, quant_type: enum { q4_0, q8_0 }) struct {
    f32_bytes: usize,
    quant_bytes: usize,
    ratio: f32,
} {
    const f32_bytes = elements * 4;
    const num_blocks = (elements + 31) / 32;
    const quant_bytes = switch (quant_type) {
        .q4_0 => num_blocks * Q4_0Block.BYTE_SIZE,
        .q8_0 => num_blocks * Q8_0Block.BYTE_SIZE,
    };

    return .{
        .f32_bytes = f32_bytes,
        .quant_bytes = quant_bytes,
        .ratio = @as(f32, @floatFromInt(f32_bytes)) / @as(f32, @floatFromInt(quant_bytes)),
    };
}

test "quantization savings calculation" {
    // 1024 elements
    const savings_q4 = quantizationSavings(1024, .q4_0);
    const savings_q8 = quantizationSavings(1024, .q8_0);

    // f32: 4096 bytes
    try std.testing.expectEqual(@as(usize, 4096), savings_q4.f32_bytes);
    try std.testing.expectEqual(@as(usize, 4096), savings_q8.f32_bytes);

    // Q4_0: 32 blocks * 18 bytes = 576 bytes (~7x savings)
    try std.testing.expectEqual(@as(usize, 576), savings_q4.quant_bytes);

    // Q8_0: 32 blocks * 34 bytes = 1088 bytes (~3.75x savings)
    try std.testing.expectEqual(@as(usize, 1088), savings_q8.quant_bytes);
}
