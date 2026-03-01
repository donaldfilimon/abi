//! CPU baseline implementations for GPU kernel benchmarks.
//!
//! Scalar and SIMD implementations used for comparison and fallback
//! when GPU hardware is unavailable.

const std = @import("std");

// ============================================================================
// Matrix Operations
// ============================================================================

/// Scalar matrix multiplication: C = A * B
pub fn scalarMatmul(A: []const f32, B: []const f32, C: []f32, M: usize, N: usize, K: usize) void {
    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f32 = 0;
            for (0..K) |k| {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/// Blocked matrix multiplication for better cache utilization
pub fn blockedMatmul(A: []const f32, B: []const f32, C: []f32, M: usize, N: usize, K: usize, block_size: usize) void {
    @memset(C, 0);

    var i: usize = 0;
    while (i < M) : (i += block_size) {
        var j: usize = 0;
        while (j < N) : (j += block_size) {
            var k: usize = 0;
            while (k < K) : (k += block_size) {
                const i_end = @min(i + block_size, M);
                const j_end = @min(j + block_size, N);
                const k_end = @min(k + block_size, K);

                var ii = i;
                while (ii < i_end) : (ii += 1) {
                    var kk = k;
                    while (kk < k_end) : (kk += 1) {
                        const a_val = A[ii * K + kk];
                        var jj = j;
                        while (jj < j_end) : (jj += 1) {
                            C[ii * N + jj] += a_val * B[kk * N + jj];
                        }
                    }
                }
            }
        }
    }
}

/// SIMD matrix multiplication
pub fn simdMatmul(comptime VEC_SIZE: usize, A: []const f32, B: []const f32, C: []f32, M: usize, N: usize, K: usize) void {
    const Vec = @Vector(VEC_SIZE, f32);
    @memset(C, 0);

    for (0..M) |i| {
        for (0..K) |k| {
            const a_val: Vec = @splat(A[i * K + k]);
            var j: usize = 0;

            while (j + VEC_SIZE <= N) : (j += VEC_SIZE) {
                const b_vec: Vec = B[k * N + j ..][0..VEC_SIZE].*;
                const c_vec: Vec = C[i * N + j ..][0..VEC_SIZE].*;
                C[i * N + j ..][0..VEC_SIZE].* = c_vec + a_val * b_vec;
            }

            while (j < N) : (j += 1) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

/// Matrix transpose
pub fn matrixTranspose(src: []const f32, dst: []f32, rows: usize, cols: usize) void {
    for (0..rows) |i| {
        for (0..cols) |j| {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

/// Blocked matrix transpose for better cache utilization
pub fn blockedTranspose(src: []const f32, dst: []f32, rows: usize, cols: usize, block_size: usize) void {
    var i: usize = 0;
    while (i < rows) : (i += block_size) {
        var j: usize = 0;
        while (j < cols) : (j += block_size) {
            const i_end = @min(i + block_size, rows);
            const j_end = @min(j + block_size, cols);

            var ii = i;
            while (ii < i_end) : (ii += 1) {
                var jj = j;
                while (jj < j_end) : (jj += 1) {
                    dst[jj * rows + ii] = src[ii * cols + jj];
                }
            }
        }
    }
}

// ============================================================================
// Vector Operations
// ============================================================================

pub fn scalarVectorAdd(a: []const f32, b: []const f32, result: []f32) void {
    for (a, b, result) |va, vb, *r| r.* = va + vb;
}

pub fn simdVectorAdd(comptime N: usize, a: []const f32, b: []const f32, result: []f32) void {
    const Vec = @Vector(N, f32);
    var i: usize = 0;
    while (i + N <= a.len) : (i += N) {
        const va: Vec = a[i..][0..N].*;
        const vb: Vec = b[i..][0..N].*;
        result[i..][0..N].* = va + vb;
    }
    while (i < a.len) : (i += 1) result[i] = a[i] + b[i];
}

pub fn scalarVectorMul(a: []const f32, b: []const f32, result: []f32) void {
    for (a, b, result) |va, vb, *r| r.* = va * vb;
}

pub fn simdVectorMul(comptime N: usize, a: []const f32, b: []const f32, result: []f32) void {
    const Vec = @Vector(N, f32);
    var i: usize = 0;
    while (i + N <= a.len) : (i += N) {
        const va: Vec = a[i..][0..N].*;
        const vb: Vec = b[i..][0..N].*;
        result[i..][0..N].* = va * vb;
    }
    while (i < a.len) : (i += 1) result[i] = a[i] * b[i];
}

pub fn scalarDotProduct(a: []const f32, b: []const f32) f32 {
    var sum: f32 = 0;
    for (a, b) |va, vb| sum += va * vb;
    return sum;
}

pub fn simdDotProduct(comptime N: usize, a: []const f32, b: []const f32) f32 {
    const Vec = @Vector(N, f32);
    var sum: f32 = 0;
    var i: usize = 0;
    while (i + N <= a.len) : (i += N) {
        const va: Vec = a[i..][0..N].*;
        const vb: Vec = b[i..][0..N].*;
        sum += @reduce(.Add, va * vb);
    }
    while (i < a.len) : (i += 1) sum += a[i] * b[i];
    return sum;
}

pub fn scalarNormalize(v: []const f32, result: []f32) void {
    var sum_sq: f32 = 0;
    for (v) |x| sum_sq += x * x;
    const norm = @sqrt(sum_sq);
    if (norm == 0) {
        @memset(result, 0);
        return;
    }
    const inv_norm = 1.0 / norm;
    for (v, result) |x, *r| r.* = x * inv_norm;
}

pub fn simdNormalize(comptime N: usize, v: []const f32, result: []f32) void {
    const sum_sq = simdDotProduct(N, v, v);
    const norm = @sqrt(sum_sq);
    if (norm == 0) {
        @memset(result, 0);
        return;
    }

    const Vec = @Vector(N, f32);
    const inv_norm: Vec = @splat(1.0 / norm);
    var i: usize = 0;
    while (i + N <= v.len) : (i += N) {
        const vec: Vec = v[i..][0..N].*;
        result[i..][0..N].* = vec * inv_norm;
    }
    const scalar_inv = 1.0 / norm;
    while (i < v.len) : (i += 1) result[i] = v[i] * scalar_inv;
}

// ============================================================================
// Reduction Operations
// ============================================================================

pub fn scalarReduceSum(v: []const f32) f32 {
    var sum: f32 = 0;
    for (v) |x| sum += x;
    return sum;
}

pub fn simdReduceSum(comptime N: usize, v: []const f32) f32 {
    const Vec = @Vector(N, f32);
    var sum: f32 = 0;
    var i: usize = 0;
    while (i + N <= v.len) : (i += N) {
        const vec: Vec = v[i..][0..N].*;
        sum += @reduce(.Add, vec);
    }
    while (i < v.len) : (i += 1) sum += v[i];
    return sum;
}

pub fn scalarReduceMax(v: []const f32) f32 {
    var max_val: f32 = v[0];
    for (v[1..]) |x| {
        if (x > max_val) max_val = x;
    }
    return max_val;
}

pub fn simdReduceMax(comptime N: usize, v: []const f32) f32 {
    const Vec = @Vector(N, f32);
    var max_val: f32 = v[0];
    var i: usize = 0;
    while (i + N <= v.len) : (i += N) {
        const vec: Vec = v[i..][0..N].*;
        const local_max = @reduce(.Max, vec);
        if (local_max > max_val) max_val = local_max;
    }
    while (i < v.len) : (i += 1) {
        if (v[i] > max_val) max_val = v[i];
    }
    return max_val;
}

pub fn scalarReduceMin(v: []const f32) f32 {
    var min_val: f32 = v[0];
    for (v[1..]) |x| {
        if (x < min_val) min_val = x;
    }
    return min_val;
}

pub fn simdReduceMin(comptime N: usize, v: []const f32) f32 {
    const Vec = @Vector(N, f32);
    var min_val: f32 = v[0];
    var i: usize = 0;
    while (i + N <= v.len) : (i += N) {
        const vec: Vec = v[i..][0..N].*;
        const local_min = @reduce(.Min, vec);
        if (local_min < min_val) min_val = local_min;
    }
    while (i < v.len) : (i += 1) {
        if (v[i] < min_val) min_val = v[i];
    }
    return min_val;
}

pub fn scalarArgmax(v: []const f32) usize {
    var max_idx: usize = 0;
    var max_val: f32 = v[0];
    for (v[1..], 1..) |x, i| {
        if (x > max_val) {
            max_val = x;
            max_idx = i;
        }
    }
    return max_idx;
}

// ============================================================================
// Tests
// ============================================================================

test "scalar matmul correctness" {
    const A = [_]f32{ 1, 2, 3, 4 };
    const B = [_]f32{ 1, 0, 0, 1 };
    var C: [4]f32 = undefined;
    scalarMatmul(&A, &B, &C, 2, 2, 2);
    try std.testing.expectApproxEqAbs(@as(f32, 1), C[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2), C[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3), C[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4), C[3], 0.001);
}

test "simd dot product correctness" {
    const a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const b = [_]f32{ 8, 7, 6, 5, 4, 3, 2, 1 };
    const scalar_dot = scalarDotProduct(&a, &b);
    const simd_dot = simdDotProduct(4, &a, &b);
    try std.testing.expectApproxEqAbs(scalar_dot, simd_dot, 0.001);
}

test "reduction operations correctness" {
    const v = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    try std.testing.expectApproxEqAbs(@as(f32, 36), simdReduceSum(4, &v), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 8), simdReduceMax(4, &v), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1), simdReduceMin(4, &v), 0.001);
}

test "normalize correctness" {
    const v = [_]f32{ 3, 4 };
    var result: [2]f32 = undefined;
    simdNormalize(2, &v, &result);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), result[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), result[1], 0.001);
}

test "transpose correctness" {
    const src = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var dst: [6]f32 = undefined;
    matrixTranspose(&src, &dst, 2, 3);
    try std.testing.expectApproxEqAbs(@as(f32, 1), dst[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4), dst[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2), dst[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 5), dst[3], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3), dst[4], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 6), dst[5], 0.001);
}
