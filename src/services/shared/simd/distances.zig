//! Distance functions, matrix multiplication, and SIMD capability detection
//!
//! Includes L2 distance (squared and actual), inner product alias,
//! SIMD-accelerated matrix multiply with cache-friendly tiling,
//! and compile-time SIMD capability detection.

const std = @import("std");

const VectorSize = std.simd.suggestVectorLength(f32) orelse 4;

const vector_ops = @import("vector_ops.zig");

// ============================================================================
// Distance Calculations (SIMD-accelerated)
// ============================================================================

/// L2 (Euclidean) squared distance using SIMD
/// Returns sum((a - b)^2) - use sqrt for actual distance
pub fn l2DistanceSquared(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    if (a.len == 0) return 0.0;

    var i: usize = 0;
    var total: f32 = 0.0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        var sum_vec: Vec = @splat(0.0);

        while (i + VectorSize <= a.len) : (i += VectorSize) {
            const va: Vec = a[i..][0..VectorSize].*;
            const vb: Vec = b[i..][0..VectorSize].*;
            const diff = va - vb;
            sum_vec += diff * diff;
        }

        total = @reduce(.Add, sum_vec);
    }

    // Scalar tail
    while (i < a.len) : (i += 1) {
        const diff = a[i] - b[i];
        total += diff * diff;
    }

    return total;
}

/// L2 distance (actual Euclidean distance)
pub fn l2Distance(a: []const f32, b: []const f32) f32 {
    return @sqrt(l2DistanceSquared(a, b));
}

/// Inner product (dot product) - already have vectorDot, this is an alias
pub const innerProduct = vector_ops.vectorDot;

// ============================================================================
// SIMD Capabilities
// ============================================================================

/// SIMD instruction set capabilities detected at compile time
pub const SimdCapabilities = struct {
    vector_size: usize,
    has_simd: bool,
    arch: Arch,

    pub const Arch = enum {
        x86_64,
        aarch64,
        wasm,
        generic,
    };
};

/// Get SIMD capabilities for current platform
pub fn getSimdCapabilities() SimdCapabilities {
    const builtin = @import("builtin");
    const arch: SimdCapabilities.Arch = switch (builtin.cpu.arch) {
        .x86_64 => .x86_64,
        .aarch64 => .aarch64,
        .wasm32, .wasm64 => .wasm,
        else => .generic,
    };

    return .{
        .vector_size = VectorSize,
        .has_simd = VectorSize > 1,
        .arch = arch,
    };
}

// ============================================================================
// Matrix Multiplication
// ============================================================================

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

test "l2DistanceSquared identical vectors" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), l2DistanceSquared(&a, &a), 0.001);
}

test "l2Distance known result" {
    const a = [_]f32{ 0.0, 0.0 };
    const b = [_]f32{ 3.0, 4.0 };
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), l2Distance(&a, &b), 0.001);
}

test "l2DistanceSquared empty" {
    const empty: []const f32 = &.{};
    try std.testing.expectEqual(@as(f32, 0.0), l2DistanceSquared(empty, empty));
}

test "getSimdCapabilities returns valid" {
    const caps = getSimdCapabilities();
    try std.testing.expect(caps.vector_size >= 1);
    // has_simd should be consistent with vector_size
    try std.testing.expectEqual(caps.vector_size > 1, caps.has_simd);
}

test "matrixMultiply 2x2" {
    // [1 2] * [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
    // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 5, 6, 7, 8 };
    var result: [4]f32 = undefined;
    matrixMultiply(&a, &b, &result, 2, 2, 2);
    try std.testing.expectApproxEqAbs(@as(f32, 19.0), result[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 22.0), result[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 43.0), result[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 50.0), result[3], 0.001);
}

test "matrixMultiply identity" {
    // A * I = A
    const a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const identity = [_]f32{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    var result: [9]f32 = undefined;
    matrixMultiply(&a, &identity, &result, 3, 3, 3);
    for (a, result) |expected, actual| {
        try std.testing.expectApproxEqAbs(expected, actual, 0.001);
    }
}

test "matrixMultiply non-square" {
    // [1 2 3] * [7  8 ]   [1*7+2*9+3*11  1*8+2*10+3*12] = [58 64]
    // [4 5 6]   [9  10]   [4*7+5*9+6*11  4*8+5*10+6*12]   [139 154]
    //           [11 12]
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b = [_]f32{ 7, 8, 9, 10, 11, 12 };
    var result: [4]f32 = undefined;
    matrixMultiply(&a, &b, &result, 2, 2, 3);
    try std.testing.expectApproxEqAbs(@as(f32, 58.0), result[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 64.0), result[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 139.0), result[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 154.0), result[3], 0.001);
}

test "innerProduct alias" {
    const a = [_]f32{ 1.0, 2.0 };
    const b = [_]f32{ 3.0, 4.0 };
    // innerProduct should be the same as vectorDot
    try std.testing.expectApproxEqAbs(
        vector_ops.vectorDot(&a, &b),
        innerProduct(&a, &b),
        0.001,
    );
}

/// Matrix multiplication with blocking/tiling for cache efficiency and SIMD acceleration
/// Computes result[m][n] = a[m][k] * b[k][n]
/// @param a Matrix A (size m x k, row-major order)
/// @param b Matrix B (size k x n, row-major order)
/// @param result Output matrix (size m x n, caller-owned, must be pre-zeroed or will be overwritten)
/// @param m Number of rows in A and result
/// @param n Number of columns in B and result
/// @param k Number of columns in A, rows in B (must match)
pub fn matrixMultiply(
    a: []const f32,
    b: []const f32,
    result: []f32,
    m: usize,
    n: usize,
    k: usize,
) void {
    std.debug.assert(m > 0 and n > 0 and k > 0);
    std.debug.assert(a.len == m * k);
    std.debug.assert(b.len == k * n);
    std.debug.assert(result.len == m * n);

    @memset(result, 0);

    const BLOCK_SIZE = 32;
    var i: usize = 0;

    while (i < m) : (i += BLOCK_SIZE) {
        const i_end = @min(i + BLOCK_SIZE, m);
        var j: usize = 0;
        while (j < n) : (j += BLOCK_SIZE) {
            const j_end = @min(j + BLOCK_SIZE, n);
            var l: usize = 0;
            while (l < k) : (l += BLOCK_SIZE) {
                const l_end = @min(l + BLOCK_SIZE, k);

                var ii: usize = i;
                while (ii < i_end) : (ii += 1) {
                    // SIMD-vectorized inner loop: process VectorSize columns at a time
                    var jj: usize = j;
                    if (comptime VectorSize > 1) {
                        const Vec = @Vector(VectorSize, f32);
                        while (jj + VectorSize <= j_end) : (jj += VectorSize) {
                            var vec_sum: Vec = result[ii * n + jj ..][0..VectorSize].*;
                            var ll: usize = l;
                            while (ll < l_end) : (ll += 1) {
                                const a_val: Vec = @splat(a[ii * k + ll]);
                                const b_vec: Vec = b[ll * n + jj ..][0..VectorSize].*;
                                vec_sum += a_val * b_vec;
                            }
                            result[ii * n + jj ..][0..VectorSize].* = vec_sum;
                        }
                    }
                    // Scalar fallback for remaining columns
                    while (jj < j_end) : (jj += 1) {
                        var acc: f32 = result[ii * n + jj];
                        var ll: usize = l;
                        while (ll < l_end) : (ll += 1) {
                            acc += a[ii * k + ll] * b[ll * n + jj];
                        }
                        result[ii * n + jj] = acc;
                    }
                }
            }
        }
    }
}
