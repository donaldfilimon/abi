//! Backward pass for matrix multiplication.
//!
//! For forward: C = A @ B where A: [M, K], B: [K, N], C: [M, N]
//! Backward:
//!   dA = dC @ B^T  (gradient w.r.t. A)
//!   dB = A^T @ dC  (gradient w.r.t. B)

const std = @import("std");
const matmul = @import("../matmul.zig");

/// Compute gradients for matrix multiplication.
/// C = A @ B
/// Given dC (gradient of loss w.r.t. C), compute dA and dB.
///
/// Args:
///   dC: [M, N] - gradient from upstream
///   A: [M, K] - input A (cached from forward)
///   B: [K, N] - input B (cached from forward)
///   dA: [M, K] - gradient output for A (accumulated)
///   dB: [K, N] - gradient output for B (accumulated)
pub fn matmulBackward(
    dC: []const f32,
    A: []const f32,
    B: []const f32,
    dA: []f32,
    dB: []f32,
    m: u32,
    k: u32,
    n: u32,
) void {
    // dA = dC @ B^T
    // dC: [M, N], B: [K, N], B^T: [N, K], result: [M, K]
    matmulBT(dC, B, dA, m, n, k);

    // dB = A^T @ dC
    // A: [M, K], A^T: [K, M], dC: [M, N], result: [K, N]
    matmulATB(A, dC, dB, m, k, n);
}

/// Compute dC @ B^T: dA = dC @ B^T
/// dC: [M, N], B: [K, N], result: [M, K]
fn matmulBT(dC: []const f32, B: []const f32, dA: []f32, m: u32, n: u32, k: u32) void {
    // For each element dA[i, j]:
    // dA[i, j] = sum over l of dC[i, l] * B[j, l]  (B transposed)
    for (0..m) |i| {
        for (0..k) |j| {
            var sum: f32 = 0;
            for (0..n) |l| {
                // dC[i, l] = dC[i * n + l]
                // B[j, l] = B[j * n + l]
                sum += dC[i * n + l] * B[j * n + l];
            }
            dA[i * k + j] += sum; // Accumulate
        }
    }
}

/// Compute A^T @ dC: dB = A^T @ dC
/// A: [M, K], dC: [M, N], result: [K, N]
fn matmulATB(A: []const f32, dC: []const f32, dB: []f32, m: u32, k: u32, n: u32) void {
    // For each element dB[i, j]:
    // dB[i, j] = sum over l of A[l, i] * dC[l, j]  (A transposed)
    for (0..k) |i| {
        for (0..n) |j| {
            var sum: f32 = 0;
            for (0..m) |l| {
                // A[l, i] = A[l * k + i]
                // dC[l, j] = dC[l * n + j]
                sum += A[l * k + i] * dC[l * n + j];
            }
            dB[i * n + j] += sum; // Accumulate
        }
    }
}

/// Backward pass for matrix-vector multiplication.
/// Forward: y = A @ x where A: [M, K], x: [K], y: [M]
/// Backward:
///   dA[i, j] = dy[i] * x[j]  (outer product)
///   dx[j] = sum_i A[i, j] * dy[i]  (A^T @ dy)
pub fn matrixVectorBackward(
    dy: []const f32, // [M] - gradient from upstream
    A: []const f32, // [M, K] - weight matrix (cached)
    x: []const f32, // [K] - input vector (cached)
    dA: []f32, // [M, K] - gradient for A (accumulated)
    dx: []f32, // [K] - gradient for x (accumulated)
    m: u32,
    k: u32,
) void {
    // dA = outer(dy, x): dA[i, j] = dy[i] * x[j]
    for (0..m) |i| {
        for (0..k) |j| {
            dA[i * k + j] += dy[i] * x[j];
        }
    }

    // dx = A^T @ dy: dx[j] = sum_i A[i, j] * dy[i]
    for (0..k) |j| {
        var sum: f32 = 0;
        for (0..m) |i| {
            sum += A[i * k + j] * dy[i];
        }
        dx[j] += sum;
    }
}

/// Backward for batched matrix multiplication.
/// Forward: C[b] = A[b] @ B[b] for each batch
pub fn batchedMatmulBackward(
    dC: []const f32, // [batch, M, N]
    A: []const f32, // [batch, M, K]
    B: []const f32, // [batch, K, N]
    dA: []f32, // [batch, M, K]
    dB: []f32, // [batch, K, N]
    batch: u32,
    m: u32,
    k: u32,
    n: u32,
) void {
    const a_stride = @as(usize, m) * k;
    const b_stride = @as(usize, k) * n;
    const c_stride = @as(usize, m) * n;

    for (0..batch) |b| {
        const a_off = b * a_stride;
        const b_off = b * b_stride;
        const c_off = b * c_stride;

        matmulBackward(
            dC[c_off..][0..c_stride],
            A[a_off..][0..a_stride],
            B[b_off..][0..b_stride],
            dA[a_off..][0..a_stride],
            dB[b_off..][0..b_stride],
            m,
            k,
            n,
        );
    }
}

/// Compute gradient for bias addition.
/// Forward: y = x + bias (broadcast over batch)
/// Backward: d_bias[j] = sum_i dy[i, j]
pub fn biasBackward(
    dy: []const f32, // [batch, dim]
    d_bias: []f32, // [dim]
    batch: u32,
    dim: u32,
) void {
    for (0..dim) |j| {
        var sum: f32 = 0;
        for (0..batch) |i| {
            sum += dy[i * dim + j];
        }
        d_bias[j] += sum;
    }
}

test "matmul backward basic" {
    // Forward: C = A @ B
    // A = [[1, 2], [3, 4]]  (2x2)
    // B = [[5, 6], [7, 8]]  (2x2)
    // C = [[19, 22], [43, 50]]  (2x2)

    const A = [_]f32{ 1, 2, 3, 4 };
    const B = [_]f32{ 5, 6, 7, 8 };
    var dC = [_]f32{ 1, 0, 0, 1 }; // Identity gradient
    var dA = [_]f32{ 0, 0, 0, 0 };
    var dB = [_]f32{ 0, 0, 0, 0 };

    matmulBackward(&dC, &A, &B, &dA, &dB, 2, 2, 2);

    // dA = dC @ B^T = [[1,0],[0,1]] @ [[5,7],[6,8]] = [[5,7],[6,8]]
    try std.testing.expectApproxEqAbs(@as(f32, 5), dA[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 7), dA[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 6), dA[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 8), dA[3], 0.001);

    // dB = A^T @ dC = [[1,3],[2,4]] @ [[1,0],[0,1]] = [[1,3],[2,4]]
    try std.testing.expectApproxEqAbs(@as(f32, 1), dB[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3), dB[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2), dB[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4), dB[3], 0.001);
}

test "matrix vector backward" {
    // Forward: y = A @ x
    // A = [[1, 2], [3, 4]]  (2x2)
    // x = [1, 1]
    // y = [3, 7]

    const A = [_]f32{ 1, 2, 3, 4 };
    const x = [_]f32{ 1, 1 };
    const dy = [_]f32{ 1, 1 };
    var dA = [_]f32{ 0, 0, 0, 0 };
    var dx = [_]f32{ 0, 0 };

    matrixVectorBackward(&dy, &A, &x, &dA, &dx, 2, 2);

    // dA = outer(dy, x) = [[1,1],[1,1]]
    try std.testing.expectApproxEqAbs(@as(f32, 1), dA[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1), dA[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1), dA[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1), dA[3], 0.001);

    // dx = A^T @ dy = [[1,3],[2,4]] @ [1,1] = [4, 6]
    try std.testing.expectApproxEqAbs(@as(f32, 4), dx[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 6), dx[1], 0.001);
}

test "bias backward" {
    // dy: 2x3 batch
    const dy = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var d_bias = [_]f32{ 0, 0, 0 };

    biasBackward(&dy, &d_bias, 2, 3);

    // d_bias[j] = sum over batch of dy[:, j]
    try std.testing.expectApproxEqAbs(@as(f32, 5), d_bias[0], 0.001); // 1 + 4
    try std.testing.expectApproxEqAbs(@as(f32, 7), d_bias[1], 0.001); // 2 + 5
    try std.testing.expectApproxEqAbs(@as(f32, 9), d_bias[2], 0.001); // 3 + 6
}
