//! Element-wise operations, memory helpers, and v2 SIMD kernels
//!
//! Includes hadamard product, abs, clamp, countGreaterThan, SIMD copy/fill,
//! euclideanDistance, softmax (out-of-place), SAXPY, reduce{Sum,Min,Max},
//! and out-of-place scale.

const std = @import("std");

const VectorSize = std.simd.suggestVectorLength(f32) orelse 4;

const activations = @import("activations.zig");

// ============================================================================
// Element-wise Operations
// ============================================================================

/// Element-wise multiplication (Hadamard product)
pub fn hadamard(a: []const f32, b: []const f32, result: []f32) void {
    std.debug.assert(a.len == b.len and a.len == result.len);
    if (a.len == 0) return;

    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);

        while (i + VectorSize <= a.len) : (i += VectorSize) {
            const va: Vec = a[i..][0..VectorSize].*;
            const vb: Vec = b[i..][0..VectorSize].*;
            result[i..][0..VectorSize].* = va * vb;
        }
    }

    while (i < a.len) : (i += 1) {
        result[i] = a[i] * b[i];
    }
}

/// Element-wise absolute value
pub fn absInPlace(data: []f32) void {
    if (data.len == 0) return;

    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const v: Vec = data[i..][0..VectorSize].*;
            data[i..][0..VectorSize].* = @abs(v);
        }
    }

    while (i < data.len) : (i += 1) {
        data[i] = @abs(data[i]);
    }
}

/// Element-wise clamp
pub fn clampInPlace(data: []f32, min_val: f32, max_val: f32) void {
    if (data.len == 0) return;

    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const min_vec: Vec = @splat(min_val);
        const max_vec: Vec = @splat(max_val);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const v: Vec = data[i..][0..VectorSize].*;
            data[i..][0..VectorSize].* = @max(min_vec, @min(max_vec, v));
        }
    }

    while (i < data.len) : (i += 1) {
        data[i] = @max(min_val, @min(max_val, data[i]));
    }
}

// ============================================================================
// Comparison Operations (returning masks)
// ============================================================================

/// Count elements greater than threshold
pub fn countGreaterThan(data: []const f32, threshold: f32) usize {
    if (data.len == 0) return 0;

    var count: usize = 0;
    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const thresh: Vec = @splat(threshold);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const v: Vec = data[i..][0..VectorSize].*;
            const mask = v > thresh;
            // Count true values in mask
            const ones: @Vector(VectorSize, u1) = @bitCast(mask);
            count += @reduce(.Add, @as(@Vector(VectorSize, usize), ones));
        }
    }

    while (i < data.len) : (i += 1) {
        if (data[i] > threshold) count += 1;
    }

    return count;
}

// ============================================================================
// Memory Operations
// ============================================================================

/// Copy with SIMD acceleration
pub fn copyF32(src: []const f32, dst: []f32) void {
    std.debug.assert(src.len == dst.len);
    if (src.len == 0) return;

    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);

        while (i + VectorSize <= src.len) : (i += VectorSize) {
            const v: Vec = src[i..][0..VectorSize].*;
            dst[i..][0..VectorSize].* = v;
        }
    }

    while (i < src.len) : (i += 1) {
        dst[i] = src[i];
    }
}

/// Fill array with value using SIMD
pub fn fillF32(data: []f32, value: f32) void {
    if (data.len == 0) return;

    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const v: Vec = @splat(value);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            data[i..][0..VectorSize].* = v;
        }
    }

    while (i < data.len) : (i += 1) {
        data[i] = value;
    }
}

// ============================================================================
// v2.0 SIMD Kernels
// ============================================================================

/// Euclidean distance: sqrt(sum((a[i] - b[i])^2))
/// Single-pass SIMD computation of L2 distance between two vectors.
/// @param a First input vector (must not be empty)
/// @param b Second input vector (must have same length as a)
/// @return Euclidean distance between the two vectors
pub fn euclideanDistance(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len > 0);
    std.debug.assert(b.len > 0);
    std.debug.assert(a.len == b.len);

    const len = a.len;
    var dist_sum: f32 = 0.0;
    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        var vec_sum: Vec = @splat(0.0);

        while (i + VectorSize <= len) : (i += VectorSize) {
            const va: Vec = a[i..][0..VectorSize].*;
            const vb: Vec = b[i..][0..VectorSize].*;
            const diff = va - vb;
            vec_sum += diff * diff;
        }

        dist_sum += @reduce(.Add, vec_sum);
    }

    while (i < len) : (i += 1) {
        const diff = a[i] - b[i];
        dist_sum += diff * diff;
    }

    return @sqrt(dist_sum);
}

/// Numerically stable softmax with separate output buffer.
/// softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
/// @param data Input logits (must not be empty)
/// @param out Output probabilities (must have same length as data, caller-owned)
pub fn softmax(data: []const f32, out: []f32) void {
    std.debug.assert(data.len > 0);
    std.debug.assert(data.len == out.len);

    // Step 1: Find max for numerical stability
    const max_val = activations.maxValue(data);

    // Step 2: Compute exp(x - max) into output buffer
    var exp_sum: f32 = 0.0;
    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const max_vec: Vec = @splat(max_val);
        var sum_vec: Vec = @splat(0.0);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const x: Vec = data[i..][0..VectorSize].*;
            const exp_val: Vec = @exp(x - max_vec);
            out[i..][0..VectorSize].* = exp_val;
            sum_vec += exp_val;
        }

        exp_sum += @reduce(.Add, sum_vec);
    }

    while (i < data.len) : (i += 1) {
        const exp_val = @exp(data[i] - max_val);
        out[i] = exp_val;
        exp_sum += exp_val;
    }

    // Step 3: Normalize
    if (exp_sum > 0.0) {
        activations.divideByScalar(out, exp_sum);
    }
}

/// SAXPY: y[i] += a * x[i] (BLAS Level 1)
/// Performs the scalar-alpha-x-plus-y operation in-place on y.
/// @param a Scalar multiplier
/// @param x Input vector (must not be empty)
/// @param y Input/output vector, modified in-place (must have same length as x)
pub fn saxpy(a: f32, x: []const f32, y: []f32) void {
    std.debug.assert(x.len > 0);
    std.debug.assert(x.len == y.len);

    const len = x.len;
    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const va: Vec = @splat(a);

        while (i + VectorSize <= len) : (i += VectorSize) {
            const vx: Vec = x[i..][0..VectorSize].*;
            const vy: Vec = y[i..][0..VectorSize].*;
            y[i..][0..VectorSize].* = @mulAdd(Vec, va, vx, vy);
        }
    }

    while (i < len) : (i += 1) {
        y[i] = @mulAdd(f32, a, x[i], y[i]);
    }
}

/// Reduce sum: sum all elements in a vector.
/// Alias for `activations.sum` -- provided for semantic consistency with `reduceMin`/`reduceMax`.
pub const reduceSum = activations.sum;

/// Reduce min: find the minimum element in a vector.
/// @param data Input vector
/// @return Minimum value, or positive infinity for empty input
pub fn reduceMin(data: []const f32) f32 {
    if (data.len == 0) return std.math.inf(f32);

    var i: usize = 0;
    var result: f32 = data[0];

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        var min_vec: Vec = @splat(data[0]);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const v: Vec = data[i..][0..VectorSize].*;
            min_vec = @min(min_vec, v);
        }

        result = @min(result, @reduce(.Min, min_vec));
    }

    while (i < data.len) : (i += 1) {
        result = @min(result, data[i]);
    }

    return result;
}

/// Reduce max: find the maximum element in a vector.
/// @param data Input vector
/// @return Maximum value, or negative infinity for empty input
pub fn reduceMax(data: []const f32) f32 {
    if (data.len == 0) return -std.math.inf(f32);

    var i: usize = 0;
    var result: f32 = data[0];

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        var max_vec: Vec = @splat(data[0]);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const v: Vec = data[i..][0..VectorSize].*;
            max_vec = @max(max_vec, v);
        }

        result = @max(result, @reduce(.Max, max_vec));
    }

    while (i < data.len) : (i += 1) {
        result = @max(result, data[i]);
    }

    return result;
}

/// Scale vector into a separate output buffer: out[i] = data[i] * scalar
/// Unlike scaleInPlace which modifies in-place, this writes to a distinct output slice.
/// @param data Input vector (must not be empty)
/// @param scalar Scalar multiplier
/// @param out Output buffer (must have same length as data, caller-owned)
pub fn scale(data: []const f32, scalar: f32, out: []f32) void {
    std.debug.assert(data.len > 0);
    std.debug.assert(data.len == out.len);

    const len = data.len;
    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const s: Vec = @splat(scalar);

        while (i + VectorSize <= len) : (i += VectorSize) {
            const v: Vec = data[i..][0..VectorSize].*;
            out[i..][0..VectorSize].* = v * s;
        }
    }

    while (i < len) : (i += 1) {
        out[i] = data[i] * scalar;
    }
}
