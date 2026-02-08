//! Integer SIMD operations, FMA, and scalar vector operations
//!
//! Includes i32 vector ops (add, sum, max, min), fused multiply-add,
//! scalar-vector FMA, and in-place scaling/offset operations.

const std = @import("std");

const VectorSize = std.simd.suggestVectorLength(f32) orelse 4;

// ============================================================================
// Integer Vector Operations (Zig 0.16 @Vector)
// ============================================================================

/// Vector size for i32 operations
const VectorSizeI32 = std.simd.suggestVectorLength(i32) orelse 4;

/// Vector size for u8 operations (useful for quantization)
const VectorSizeU8 = std.simd.suggestVectorLength(u8) orelse 16;

/// SIMD integer addition
pub fn vectorAddI32(a: []const i32, b: []const i32, result: []i32) void {
    std.debug.assert(a.len > 0);
    std.debug.assert(a.len == b.len and a.len == result.len);

    var i: usize = 0;
    if (comptime VectorSizeI32 > 1) {
        const Vec = @Vector(VectorSizeI32, i32);
        while (i + VectorSizeI32 <= a.len) : (i += VectorSizeI32) {
            const va: Vec = a[i..][0..VectorSizeI32].*;
            const vb: Vec = b[i..][0..VectorSizeI32].*;
            result[i..][0..VectorSizeI32].* = va + vb;
        }
    }
    while (i < a.len) : (i += 1) {
        result[i] = a[i] + b[i];
    }
}

/// SIMD integer sum reduction
pub fn sumI32(data: []const i32) i64 {
    if (data.len == 0) return 0;

    var i: usize = 0;
    var total: i64 = 0;

    if (comptime VectorSizeI32 > 1) {
        const Vec = @Vector(VectorSizeI32, i32);
        var sum_vec: Vec = @splat(0);

        while (i + VectorSizeI32 <= data.len) : (i += VectorSizeI32) {
            const v: Vec = data[i..][0..VectorSizeI32].*;
            sum_vec += v;
        }

        // Horizontal sum using @reduce
        total = @reduce(.Add, sum_vec);
    }

    while (i < data.len) : (i += 1) {
        total += data[i];
    }

    return total;
}

/// SIMD max for i32
pub fn maxI32(data: []const i32) i32 {
    if (data.len == 0) return std.math.minInt(i32);

    var i: usize = 0;
    var max_val: i32 = data[0];

    if (comptime VectorSizeI32 > 1) {
        const Vec = @Vector(VectorSizeI32, i32);
        var max_vec: Vec = @splat(data[0]);

        while (i + VectorSizeI32 <= data.len) : (i += VectorSizeI32) {
            const v: Vec = data[i..][0..VectorSizeI32].*;
            max_vec = @max(max_vec, v);
        }

        max_val = @reduce(.Max, max_vec);
    }

    while (i < data.len) : (i += 1) {
        max_val = @max(max_val, data[i]);
    }

    return max_val;
}

/// SIMD min for i32
pub fn minI32(data: []const i32) i32 {
    if (data.len == 0) return std.math.maxInt(i32);

    var i: usize = 0;
    var min_val: i32 = data[0];

    if (comptime VectorSizeI32 > 1) {
        const Vec = @Vector(VectorSizeI32, i32);
        var min_vec: Vec = @splat(data[0]);

        while (i + VectorSizeI32 <= data.len) : (i += VectorSizeI32) {
            const v: Vec = data[i..][0..VectorSizeI32].*;
            min_vec = @min(min_vec, v);
        }

        min_val = @reduce(.Min, min_vec);
    }

    while (i < data.len) : (i += 1) {
        min_val = @min(min_val, data[i]);
    }

    return min_val;
}

// ============================================================================
// Fused Multiply-Add Operations (FMA)
// ============================================================================

/// Fused multiply-add: result = a * b + c
/// Uses SIMD FMA when available for better precision and performance
pub fn fma(a: []const f32, b: []const f32, c: []const f32, result: []f32) void {
    std.debug.assert(a.len == b.len and b.len == c.len and c.len == result.len);
    if (a.len == 0) return;

    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);

        while (i + VectorSize <= a.len) : (i += VectorSize) {
            const va: Vec = a[i..][0..VectorSize].*;
            const vb: Vec = b[i..][0..VectorSize].*;
            const vc: Vec = c[i..][0..VectorSize].*;
            result[i..][0..VectorSize].* = @mulAdd(Vec, va, vb, vc);
        }
    }

    while (i < a.len) : (i += 1) {
        result[i] = @mulAdd(f32, a[i], b[i], c[i]);
    }
}

/// Scalar-vector fused multiply-add: result = scalar * a + b
pub fn fmaScalar(scalar: f32, a: []const f32, b: []const f32, result: []f32) void {
    std.debug.assert(a.len == b.len and b.len == result.len);
    if (a.len == 0) return;

    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const s: Vec = @splat(scalar);

        while (i + VectorSize <= a.len) : (i += VectorSize) {
            const va: Vec = a[i..][0..VectorSize].*;
            const vb: Vec = b[i..][0..VectorSize].*;
            result[i..][0..VectorSize].* = @mulAdd(Vec, s, va, vb);
        }
    }

    while (i < a.len) : (i += 1) {
        result[i] = @mulAdd(f32, scalar, a[i], b[i]);
    }
}

// ============================================================================
// Vector Scaling Operations
// ============================================================================

/// Multiply vector by scalar in-place
pub fn scaleInPlace(data: []f32, scalar: f32) void {
    if (data.len == 0) return;

    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const s: Vec = @splat(scalar);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const v: Vec = data[i..][0..VectorSize].*;
            data[i..][0..VectorSize].* = v * s;
        }
    }

    while (i < data.len) : (i += 1) {
        data[i] *= scalar;
    }
}

/// Add scalar to vector in-place
pub fn addScalarInPlace(data: []f32, scalar: f32) void {
    if (data.len == 0) return;

    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const s: Vec = @splat(scalar);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const v: Vec = data[i..][0..VectorSize].*;
            data[i..][0..VectorSize].* = v + s;
        }
    }

    while (i < data.len) : (i += 1) {
        data[i] += scalar;
    }
}
