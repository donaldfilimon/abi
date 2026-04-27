//! Vector quantization utilities for WDBX
//!
//! Provides FP16, INT8, and binary quantization for reducing memory usage
//! of high-dimensional vectors. Re-exports the Quantize struct from simd.zig
//! and adds additional batch/dequantization helpers.

const std = @import("std");
const simd = @import("simd.zig");

pub const Quantize = simd.Quantize;

/// Quantized vector storage with metadata for reconstruction.
pub const QuantizedVector = struct {
    format: Format,
    dimension: u32,
    data: []u8,
    scale: f32,
    zero_point: i8,

    pub const Format = enum {
        f32,
        f16,
        int8,
        binary,
    };

    pub fn deinit(self: *QuantizedVector, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
        self.* = undefined;
    }

    /// Estimate memory savings vs f32.
    pub fn compressionRatio(self: *const QuantizedVector) f32 {
        const f32_bytes: f32 = @floatFromInt(self.dimension * 4);
        const actual_bytes: f32 = @floatFromInt(self.data.len);
        if (actual_bytes < 1e-10) return 1.0;
        return f32_bytes / actual_bytes;
    }
};

/// Quantize a batch of vectors to INT8, returning scales and zero-points per vector.
pub fn batchQuantizeInt8(
    allocator: std.mem.Allocator,
    vectors: []const []const f32,
    dst_vectors: [][]i8,
    scales: []f32,
    zero_points: []i8,
) !void {
    std.debug.assert(vectors.len == dst_vectors.len);
    std.debug.assert(vectors.len == scales.len);
    std.debug.assert(vectors.len == zero_points.len);
    _ = allocator;

    for (vectors, 0..) |vec, idx| {
        Quantize.toInt8(vec, dst_vectors[idx], &scales[idx], &zero_points[idx]);
    }
}

/// Dequantize INT8 back to f32.
pub fn dequantizeInt8(src: []const i8, dst: []f32, scale: f32, zero_point: i8) void {
    std.debug.assert(src.len == dst.len);
    for (src, 0..) |v, i| {
        dst[i] = scale * (@as(f32, @floatFromInt(v)) - @as(f32, @floatFromInt(zero_point)));
    }
}

// ============================================================================
// Tests
// ============================================================================

test "dequantize int8 round-trip approximation" {
    const src = [_]f32{ -1.0, 0.0, 0.5, 1.0 };
    var quantized: [4]i8 = undefined;
    var scale: f32 = undefined;
    var zp: i8 = undefined;
    Quantize.toInt8(&src, &quantized, &scale, &zp);

    var restored: [4]f32 = undefined;
    dequantizeInt8(&quantized, &restored, scale, zp);

    for (src, restored) |orig, rest| {
        try std.testing.expectApproxEqAbs(orig, rest, 0.05);
    }
}

test "QuantizedVector compression ratio" {
    var qv = QuantizedVector{
        .format = .int8,
        .dimension = 768,
        .data = @constCast(&[_]u8{0} ** 768),
        .scale = 0.01,
        .zero_point = 0,
    };
    const ratio = qv.compressionRatio();
    // 768*4 / 768 = 4.0
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), ratio, 0.01);
    _ = &qv;
}
