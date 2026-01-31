//! SIMD implementation for C bindings.
//! Standalone implementation that doesn't depend on the main ABI module.

const std = @import("std");

const VectorSize = std.simd.suggestVectorLength(f32) orelse 4;

/// Vector addition using SIMD when available
pub fn vectorAdd(a: []const f32, b: []const f32, result: []f32) void {
    std.debug.assert(a.len == b.len and a.len == result.len);
    if (a.len == 0) return;

    const len = a.len;
    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);

        while (i + VectorSize <= len) : (i += VectorSize) {
            const va: Vec = a[i..][0..VectorSize].*;
            const vb: Vec = b[i..][0..VectorSize].*;
            result[i..][0..VectorSize].* = va + vb;
        }
    }

    while (i < len) : (i += 1) {
        result[i] = a[i] + b[i];
    }
}

/// Vector dot product using SIMD when available
pub fn vectorDot(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    if (a.len == 0) return 0.0;

    const len = a.len;
    var dot_sum: f32 = 0.0;
    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        var vec_sum: Vec = @splat(0.0);

        while (i + VectorSize <= len) : (i += VectorSize) {
            const va: Vec = a[i..][0..VectorSize].*;
            const vb: Vec = b[i..][0..VectorSize].*;
            vec_sum += va * vb;
        }

        dot_sum += @reduce(.Add, vec_sum);
    }

    while (i < len) : (i += 1) {
        dot_sum += a[i] * b[i];
    }

    return dot_sum;
}

/// Vector L2 norm using SIMD when available
pub fn vectorL2Norm(v: []const f32) f32 {
    if (v.len == 0) return 0.0;

    const len = v.len;
    var norm_sum: f32 = 0.0;
    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        var vec_sum: Vec = @splat(0.0);

        while (i + VectorSize <= len) : (i += VectorSize) {
            const vv: Vec = v[i..][0..VectorSize].*;
            vec_sum += vv * vv;
        }

        norm_sum += @reduce(.Add, vec_sum);
    }

    while (i < len) : (i += 1) {
        norm_sum += v[i] * v[i];
    }

    return @sqrt(norm_sum);
}

/// Cosine similarity using SIMD operations
pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    if (a.len == 0 or b.len == 0) return 0.0;
    if (a.len != b.len) return 0.0;

    const dot_product = vectorDot(a, b);
    const norm_a = vectorL2Norm(a);
    const norm_b = vectorL2Norm(b);

    if (norm_a == 0.0 or norm_b == 0.0) {
        return 0.0;
    }

    return dot_product / (norm_a * norm_b);
}

test "simd impl" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 4.0, 3.0, 2.0, 1.0 };
    var result: [4]f32 = undefined;

    vectorAdd(&a, &b, &result);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), result[0], 0.001);

    const dot = vectorDot(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 20.0), dot, 0.001);
}
