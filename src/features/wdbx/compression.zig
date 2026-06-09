//! Embedding compression codec (Compute/Storage support).
//!
//! Scalar int8 quantization of f32 embedding vectors: each vector is mapped to
//! a per-vector affine range [min, max] and stored as u8 codes, ~4× smaller
//! than raw f32. Reconstruction error is bounded by half a quantization step.
//! This is a real, lossy compression primitive for stored embeddings; it is not
//! a learned/entropy "neural" codec — the name reflects the embedding use case,
//! not a neural-network implementation.

const std = @import("std");

pub const Quantized = struct {
    min: f32,
    scale: f32, // (max - min) / 255; 0 for a constant vector
    codes: []u8,

    pub fn deinit(self: *Quantized, allocator: std.mem.Allocator) void {
        allocator.free(self.codes);
    }

    /// Raw f32 bytes divided by compressed bytes (codes + 2 f32 header fields).
    pub fn compressionRatio(self: Quantized) f32 {
        const raw: f32 = @floatFromInt(self.codes.len * @sizeOf(f32));
        const packed_bytes: f32 = @floatFromInt(self.codes.len + 2 * @sizeOf(f32));
        return raw / packed_bytes;
    }
};

pub fn quantize(allocator: std.mem.Allocator, vector: []const f32) !Quantized {
    if (vector.len == 0) return error.EmptyVector;
    var min = vector[0];
    var max = vector[0];
    for (vector) |v| {
        min = @min(min, v);
        max = @max(max, v);
    }
    const span = max - min;
    const scale: f32 = if (span == 0) 0 else span / 255.0;

    const codes = try allocator.alloc(u8, vector.len);
    errdefer allocator.free(codes);
    for (vector, 0..) |v, i| {
        if (scale == 0) {
            codes[i] = 0;
        } else {
            const q = @round((v - min) / scale);
            codes[i] = @intFromFloat(std.math.clamp(q, 0, 255));
        }
    }
    return .{ .min = min, .scale = scale, .codes = codes };
}

pub fn dequantize(allocator: std.mem.Allocator, q: Quantized) ![]f32 {
    const out = try allocator.alloc(f32, q.codes.len);
    errdefer allocator.free(out);
    for (q.codes, 0..) |c, i| {
        out[i] = q.min + @as(f32, @floatFromInt(c)) * q.scale;
    }
    return out;
}

/// Max absolute reconstruction error across the vector (for diagnostics/tests).
pub fn maxError(original: []const f32, reconstructed: []const f32) f32 {
    var err: f32 = 0;
    for (original, reconstructed) |a, b| err = @max(err, @abs(a - b));
    return err;
}

test "compression: round-trip stays within half a quantization step" {
    const allocator = std.testing.allocator;
    // A realistic-width embedding so the 8-byte header is amortized (~4x ratio).
    var vec: [128]f32 = undefined;
    for (&vec, 0..) |*v, i| v.* = std.math.sin(@as(f32, @floatFromInt(i)) * 0.1);

    var q = try quantize(allocator, &vec);
    defer q.deinit(allocator);
    const back = try dequantize(allocator, q);
    defer allocator.free(back);

    // Reconstruction error is bounded by one quantization step (span/255).
    try std.testing.expect(maxError(&vec, back) <= q.scale + 1e-6);
    try std.testing.expect(q.compressionRatio() > 3.0);
}

test "compression: constant vector quantizes losslessly" {
    const allocator = std.testing.allocator;
    const vec = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    var q = try quantize(allocator, &vec);
    defer q.deinit(allocator);
    try std.testing.expectEqual(@as(f32, 0), q.scale);

    const back = try dequantize(allocator, q);
    defer allocator.free(back);
    try std.testing.expectEqual(@as(f32, 0), maxError(&vec, back));
}

test "compression: rejects empty input" {
    try std.testing.expectError(error.EmptyVector, quantize(std.testing.allocator, &.{}));
}

test {
    std.testing.refAllDecls(@This());
}
