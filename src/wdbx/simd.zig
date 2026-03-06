//! SIMD-accelerated vector operations for WDBX
//!
//! Provides hardware-accelerated distance computations (cosine, L2, inner product,
//! hamming) using Zig's @Vector types. Falls back to scalar on unsupported platforms.
//! Zero allocations in all hot paths.

const std = @import("std");
const builtin = @import("builtin");

/// SIMD lane width chosen per architecture.
pub const simd_width: comptime_int = std.simd.suggestVectorLength(f32) orelse 4;

/// Native SIMD vector of f32.
pub const F32x = @Vector(simd_width, f32);

// ============================================================================
// Distance Functions
// ============================================================================

pub const Distance = struct {
    /// Cosine similarity in [−1, 1].
    /// Returns 0.0 for zero-magnitude vectors.
    pub fn cosine(a: []const f32, b: []const f32) f32 {
        std.debug.assert(a.len == b.len);
        if (a.len == 0) return 0.0;

        var i: usize = 0;
        var dot_acc: f32 = 0.0;
        var norm_a_acc: f32 = 0.0;
        var norm_b_acc: f32 = 0.0;

        if (comptime simd_width > 1) {
            const Vec = F32x;
            var vdot: Vec = @splat(0.0);
            var vna: Vec = @splat(0.0);
            var vnb: Vec = @splat(0.0);

            while (i + simd_width <= a.len) : (i += simd_width) {
                const va: Vec = a[i..][0..simd_width].*;
                const vb: Vec = b[i..][0..simd_width].*;
                vdot += va * vb;
                vna += va * va;
                vnb += vb * vb;
            }
            dot_acc = @reduce(.Add, vdot);
            norm_a_acc = @reduce(.Add, vna);
            norm_b_acc = @reduce(.Add, vnb);
        }

        // Scalar tail
        while (i < a.len) : (i += 1) {
            dot_acc += a[i] * b[i];
            norm_a_acc += a[i] * a[i];
            norm_b_acc += b[i] * b[i];
        }

        const denom = @sqrt(norm_a_acc) * @sqrt(norm_b_acc);
        if (denom < 1e-30) return 0.0;
        return dot_acc / denom;
    }

    /// L2 squared distance: sum((a[i] - b[i])^2).
    /// Avoids sqrt in the hot path — take sqrt of result if needed.
    pub fn l2Squared(a: []const f32, b: []const f32) f32 {
        std.debug.assert(a.len == b.len);
        if (a.len == 0) return 0.0;

        var i: usize = 0;
        var total: f32 = 0.0;

        if (comptime simd_width > 1) {
            const Vec = F32x;
            var acc: Vec = @splat(0.0);

            while (i + simd_width <= a.len) : (i += simd_width) {
                const va: Vec = a[i..][0..simd_width].*;
                const vb: Vec = b[i..][0..simd_width].*;
                const diff = va - vb;
                acc += diff * diff;
            }
            total = @reduce(.Add, acc);
        }

        while (i < a.len) : (i += 1) {
            const diff = a[i] - b[i];
            total += diff * diff;
        }
        return total;
    }

    /// Inner (dot) product: sum(a[i] * b[i]).
    pub fn innerProduct(a: []const f32, b: []const f32) f32 {
        std.debug.assert(a.len == b.len);
        if (a.len == 0) return 0.0;

        var i: usize = 0;
        var total: f32 = 0.0;

        if (comptime simd_width > 1) {
            const Vec = F32x;
            var acc: Vec = @splat(0.0);

            while (i + simd_width <= a.len) : (i += simd_width) {
                const va: Vec = a[i..][0..simd_width].*;
                const vb: Vec = b[i..][0..simd_width].*;
                acc += va * vb;
            }
            total = @reduce(.Add, acc);
        }

        while (i < a.len) : (i += 1) {
            total += a[i] * b[i];
        }
        return total;
    }

    /// Hamming distance between binary (u64-packed) vectors.
    pub fn hamming(a: []const u64, b: []const u64) u32 {
        std.debug.assert(a.len == b.len);
        var count: u32 = 0;
        for (a, b) |va, vb| {
            count += @popCount(va ^ vb);
        }
        return count;
    }
};

// ============================================================================
// Quantization
// ============================================================================

pub const Quantize = struct {
    /// Convert f32 → f16.
    pub fn toF16(src: []const f32, dst: []f16) void {
        std.debug.assert(src.len == dst.len);
        for (src, 0..) |v, i| {
            dst[i] = @floatCast(v);
        }
    }

    /// Convert f16 → f32.
    pub fn fromF16(src: []const f16, dst: []f32) void {
        std.debug.assert(src.len == dst.len);
        for (src, 0..) |v, i| {
            dst[i] = @floatCast(v);
        }
    }

    /// Affine INT8 quantization: value = scale * (int8 - zero_point).
    pub fn toInt8(src: []const f32, dst: []i8, scale: *f32, zero_point: *i8) void {
        std.debug.assert(src.len == dst.len);
        if (src.len == 0) return;

        var min_val: f32 = src[0];
        var max_val: f32 = src[0];
        for (src[1..]) |v| {
            min_val = @min(min_val, v);
            max_val = @max(max_val, v);
        }

        const range = max_val - min_val;
        if (range < 1e-30) {
            scale.* = 1.0;
            zero_point.* = 0;
            @memset(dst, 0);
            return;
        }

        scale.* = range / 255.0;
        const zp_f = -min_val / scale.* - 128.0;
        zero_point.* = @intFromFloat(std.math.clamp(zp_f, -128.0, 127.0));

        for (src, 0..) |v, i| {
            const quantized = v / scale.* + @as(f32, @floatFromInt(zero_point.*));
            dst[i] = @intFromFloat(std.math.clamp(quantized, -128.0, 127.0));
        }
    }

    /// Binary quantization: sign(v) packed into u64 words.
    pub fn toBinary(src: []const f32, dst: []u64) void {
        const needed = (src.len + 63) / 64;
        std.debug.assert(dst.len >= needed);

        @memset(dst[0..needed], 0);
        for (src, 0..) |v, i| {
            if (v > 0.0) {
                dst[i / 64] |= @as(u64, 1) << @intCast(i % 64);
            }
        }
    }
};

// ============================================================================
// Tests
// ============================================================================

test "cosine similarity - identical vectors" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const sim = Distance.cosine(&a, &a);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sim, 1e-5);
}

test "cosine similarity - orthogonal" {
    const a = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const b = [_]f32{ 0.0, 1.0, 0.0, 0.0 };
    const sim = Distance.cosine(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sim, 1e-5);
}

test "l2Squared distance" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    const dist = Distance.l2Squared(&a, &b);
    // (4^2)*4 = 64
    try std.testing.expectApproxEqAbs(@as(f32, 64.0), dist, 1e-5);
}

test "inner product" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 2.0, 3.0, 4.0, 5.0 };
    const ip = Distance.innerProduct(&a, &b);
    // 2 + 6 + 12 + 20 = 40
    try std.testing.expectApproxEqAbs(@as(f32, 40.0), ip, 1e-5);
}

test "hamming distance" {
    const a = [_]u64{0b1111_0000};
    const b = [_]u64{0b1010_0101};
    const dist = Distance.hamming(&a, &b);
    try std.testing.expect(dist > 0);
}

test "quantize f16 round-trip" {
    const src = [_]f32{ 1.0, -2.5, 3.14, 0.0 };
    var f16_buf: [4]f16 = undefined;
    var dst: [4]f32 = undefined;
    Quantize.toF16(&src, &f16_buf);
    Quantize.fromF16(&f16_buf, &dst);
    for (src, dst) |s, d| {
        try std.testing.expectApproxEqAbs(s, d, 0.01);
    }
}

test "quantize int8" {
    const src = [_]f32{ -1.0, 0.0, 0.5, 1.0 };
    var dst: [4]i8 = undefined;
    var scale: f32 = undefined;
    var zp: i8 = undefined;
    Quantize.toInt8(&src, &dst, &scale, &zp);
    try std.testing.expect(scale > 0);
}

test "quantize binary" {
    const src = [_]f32{ 1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.1 };
    var dst: [1]u64 = undefined;
    Quantize.toBinary(&src, &dst);
    // bits 0, 2, 5, 7 should be set (positive values)
    try std.testing.expect(dst[0] & 1 == 1); // bit 0
    try std.testing.expect(dst[0] & (1 << 2) != 0); // bit 2
}
