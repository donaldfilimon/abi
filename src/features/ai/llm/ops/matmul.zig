//! Optimized matrix multiplication for LLM inference.
//!
//! Implements blocked, SIMD-accelerated matrix multiplication optimized
//! for the access patterns common in transformer models.

const std = @import("std");
const builtin = @import("builtin");

/// Matrix multiplication: C = A @ B
/// A: [M, K], B: [K, N], C: [M, N]
pub fn matrixMultiply(
    a: []const f32,
    b: []const f32,
    c: []f32,
    m: u32,
    k: u32,
    n: u32,
) void {
    // Use blocked algorithm for cache efficiency
    const BLOCK_SIZE: u32 = 64;

    // Zero output
    @memset(c, 0);

    // Blocked multiplication
    var i: u32 = 0;
    while (i < m) : (i += BLOCK_SIZE) {
        const i_end = @min(i + BLOCK_SIZE, m);

        var j: u32 = 0;
        while (j < n) : (j += BLOCK_SIZE) {
            const j_end = @min(j + BLOCK_SIZE, n);

            var kk: u32 = 0;
            while (kk < k) : (kk += BLOCK_SIZE) {
                const k_end = @min(kk + BLOCK_SIZE, k);

                // Micro-kernel for this block
                blockMultiply(a, b, c, i, i_end, j, j_end, kk, k_end, k, n);
            }
        }
    }
}

/// Inner block multiplication kernel.
/// Note: For non-transposed B, elements aren't contiguous, so SIMD vectorization
/// on the k-dimension is limited. Consider using matrixMultiplyTransposed for
/// better SIMD efficiency when possible.
fn blockMultiply(
    a: []const f32,
    b: []const f32,
    c: []f32,
    i_start: u32,
    i_end: u32,
    j_start: u32,
    j_end: u32,
    k_start: u32,
    k_end: u32,
    k: u32,
    n: u32,
) void {
    var ii = i_start;
    while (ii < i_end) : (ii += 1) {
        var jj = j_start;
        while (jj < j_end) : (jj += 1) {
            var sum: f32 = c[@as(usize, ii) * n + jj];

            var kk = k_start;
            // Manual unrolling for non-contiguous access pattern
            // (B is strided by n, so @Vector can't help here)
            while (kk + 4 <= k_end) : (kk += 4) {
                const a_base = @as(usize, ii) * k + kk;
                const b_base = @as(usize, kk) * n + jj;

                sum += a[a_base + 0] * b[b_base + 0 * n];
                sum += a[a_base + 1] * b[b_base + 1 * n];
                sum += a[a_base + 2] * b[b_base + 2 * n];
                sum += a[a_base + 3] * b[b_base + 3 * n];
            }

            // Handle remaining elements
            while (kk < k_end) : (kk += 1) {
                sum += a[@as(usize, ii) * k + kk] * b[@as(usize, kk) * n + jj];
            }

            c[@as(usize, ii) * n + jj] = sum;
        }
    }
}

/// Matrix multiplication with transposed B: C = A @ B^T
/// A: [M, K], B: [N, K] (transposed), C: [M, N]
pub fn matrixMultiplyTransposed(
    a: []const f32,
    b_transposed: []const f32,
    c: []f32,
    m: u32,
    k: u32,
    n: u32,
) void {
    const BLOCK_SIZE: u32 = 64;

    @memset(c, 0);

    var i: u32 = 0;
    while (i < m) : (i += BLOCK_SIZE) {
        const i_end = @min(i + BLOCK_SIZE, m);

        var j: u32 = 0;
        while (j < n) : (j += BLOCK_SIZE) {
            const j_end = @min(j + BLOCK_SIZE, n);

            // With transposed B, each row of B is contiguous
            blockMultiplyTransposed(a, b_transposed, c, i, i_end, j, j_end, k, n);
        }
    }
}

fn blockMultiplyTransposed(
    a: []const f32,
    b_t: []const f32,
    c: []f32,
    i_start: u32,
    i_end: u32,
    j_start: u32,
    j_end: u32,
    k: u32,
    n: u32,
) void {
    var ii = i_start;
    while (ii < i_end) : (ii += 1) {
        var jj = j_start;
        while (jj < j_end) : (jj += 1) {
            // Dot product of row ii of A with row jj of B_T
            const sum = dotProduct(
                a[@as(usize, ii) * k .. @as(usize, ii) * k + k],
                b_t[@as(usize, jj) * k .. @as(usize, jj) * k + k],
            );
            c[@as(usize, ii) * n + jj] = sum;
        }
    }
}

/// SIMD vector length for this platform (auto-detected).
const VectorSize = std.simd.suggestVectorLength(f32) orelse 4;

/// SIMD-optimized dot product using @Vector.
pub fn dotProduct(a: []const f32, b: []const f32) f32 {
    const len = @min(a.len, b.len);
    var sum: f32 = 0;
    var i: usize = 0;

    // Use @Vector for true SIMD acceleration
    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        var vec_sum: Vec = @splat(0.0);

        while (i + VectorSize <= len) : (i += VectorSize) {
            const va: Vec = a[i..][0..VectorSize].*;
            const vb: Vec = b[i..][0..VectorSize].*;
            vec_sum += va * vb;
        }

        // Horizontal sum using @reduce (efficient on modern CPUs)
        sum = @reduce(.Add, vec_sum);
    }

    // Handle remaining elements
    while (i < len) : (i += 1) {
        sum += a[i] * b[i];
    }

    return sum;
}

/// Matrix-vector multiplication: y = A @ x
/// A: [M, K], x: [K], y: [M]
pub fn matrixVectorMultiply(a: []const f32, x: []const f32, y: []f32, m: u32, k: u32) void {
    for (0..m) |i| {
        const row_start = i * k;
        y[i] = dotProduct(a[row_start .. row_start + k], x);
    }
}

/// Matrix-vector multiplication with transposed matrix: y = A^T @ x
/// A: [K, M] stored, but treated as [M, K] transposed, x: [K], y: [M]
/// This computes y[j] = sum_i A[i, j] * x[i] for transposed semantics.
/// In practice for LLM: weights are [vocab_size, hidden_dim] and we want hidden @ weights^T
pub fn matrixVectorMultiplyTransposed(
    a: []const f32, // [M, K] original, but treated as transposed [K, M]
    x: []const f32, // [K]
    y: []f32, // [M]
    m: u32, // output dimension
    k: u32, // input dimension
) void {
    @memset(y, 0);
    // For transposed: output[j] = sum_i input[i] * weight[j, i]
    // where weight is stored as [M, K] row-major
    for (0..m) |j| {
        const row_start = j * k;
        y[j] = dotProduct(a[row_start .. row_start + k], x);
    }
}

/// Add bias to each row: y[i] += bias[i % bias.len]
pub fn addBias(y: []f32, bias: []const f32) void {
    for (y, 0..) |*val, i| {
        val.* += bias[i % bias.len];
    }
}

// Note: hasSimd() removed - using comptime VectorSize check instead

test "matrix multiplication basic" {
    // 2x3 @ 3x2 = 2x2
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b = [_]f32{ 7, 8, 9, 10, 11, 12 };
    var c: [4]f32 = undefined;

    matrixMultiply(&a, &b, &c, 2, 3, 2);

    // Expected: [[58, 64], [139, 154]]
    try std.testing.expectApproxEqAbs(@as(f32, 58), c[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 64), c[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 139), c[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 154), c[3], 0.001);
}

test "dot product" {
    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 5, 6, 7, 8 };

    const result = dotProduct(&a, &b);
    // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    try std.testing.expectApproxEqAbs(@as(f32, 70), result, 0.001);
}

test "matrix vector multiply" {
    // 2x3 matrix times 3-vector
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const x = [_]f32{ 1, 2, 3 };
    var y: [2]f32 = undefined;

    matrixVectorMultiply(&a, &x, &y, 2, 3);

    // Row 0: 1*1 + 2*2 + 3*3 = 14
    // Row 1: 4*1 + 5*2 + 6*3 = 32
    try std.testing.expectApproxEqAbs(@as(f32, 14), y[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 32), y[1], 0.001);
}
