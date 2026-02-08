//! SIMD Kernel Validation Tests
//!
//! Verifies that SIMD-accelerated kernels produce correct results by comparing
//! against scalar reference implementations. Tests various input sizes to
//! exercise both the SIMD fast path and the scalar tail loop.

const std = @import("std");
const abi = @import("abi");
const simd = abi.shared.simd;

// ============================================================================
// Scalar Reference Implementations
// ============================================================================

fn scalarEuclideanDistance(a: []const f32, b: []const f32) f32 {
    var sum: f32 = 0.0;
    for (a, b) |av, bv| {
        const diff = av - bv;
        sum += diff * diff;
    }
    return @sqrt(sum);
}

fn scalarSoftmax(data: []const f32, out: []f32) void {
    var max_val: f32 = data[0];
    for (data[1..]) |v| max_val = @max(max_val, v);

    var exp_sum: f32 = 0.0;
    for (data, 0..) |v, i| {
        out[i] = @exp(v - max_val);
        exp_sum += out[i];
    }
    for (out) |*v| v.* /= exp_sum;
}

fn scalarSaxpy(a: f32, x: []const f32, y: []f32) void {
    for (x, 0..) |xv, i| y[i] += a * xv;
}

fn scalarReduceSum(data: []const f32) f32 {
    var s: f32 = 0.0;
    for (data) |v| s += v;
    return s;
}

fn scalarReduceMin(data: []const f32) f32 {
    var m: f32 = data[0];
    for (data[1..]) |v| m = @min(m, v);
    return m;
}

fn scalarReduceMax(data: []const f32) f32 {
    var m: f32 = data[0];
    for (data[1..]) |v| m = @max(m, v);
    return m;
}

fn scalarScale(data: []const f32, scalar: f32, out: []f32) void {
    for (data, 0..) |v, i| out[i] = v * scalar;
}

// ============================================================================
// Test Helpers
// ============================================================================

fn expectApproxEq(expected: f32, actual: f32) !void {
    try std.testing.expectApproxEqAbs(expected, actual, 1e-4);
}

fn expectSlicesApproxEq(expected: []const f32, actual: []const f32) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual, 0..) |e, a, i| {
        std.testing.expectApproxEqAbs(e, a, 1e-4) catch |err| {
            std.log.warn("Mismatch at index {d}: expected {d:.6}, got {d:.6}", .{ i, e, a });
            return err;
        };
    }
}

// Test sizes: small (< VectorSize), exact VectorSize, odd (with tail), large
const test_sizes = [_]usize{ 1, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 100, 256 };

fn makeTestData(comptime n: usize) [n]f32 {
    var data: [n]f32 = undefined;
    for (&data, 0..) |*v, i| {
        // Deterministic pseudo-random values in [-2, 2]
        v.* = @sin(@as(f32, @floatFromInt(i)) * 0.7 + 0.3) * 2.0;
    }
    return data;
}

// ============================================================================
// Euclidean Distance Tests
// ============================================================================

test "SIMD euclideanDistance matches scalar" {
    inline for (test_sizes) |n| {
        if (n == 0) continue;
        const a = makeTestData(n);
        var b: [n]f32 = undefined;
        for (&b, 0..) |*v, i| {
            v.* = @cos(@as(f32, @floatFromInt(i)) * 1.3 + 0.7) * 1.5;
        }

        const simd_result = simd.euclideanDistance(&a, &b);
        const scalar_result = scalarEuclideanDistance(&a, &b);
        try expectApproxEq(scalar_result, simd_result);
    }
}

test "SIMD euclideanDistance zero distance" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    try expectApproxEq(0.0, simd.euclideanDistance(&a, &a));
}

test "SIMD euclideanDistance known values" {
    const a = [_]f32{ 0.0, 0.0, 0.0 };
    const b = [_]f32{ 3.0, 4.0, 0.0 };
    try expectApproxEq(5.0, simd.euclideanDistance(&a, &b));
}

// ============================================================================
// Softmax Tests
// ============================================================================

test "SIMD softmax matches scalar" {
    inline for (test_sizes) |n| {
        if (n == 0) continue;
        const data = makeTestData(n);
        var simd_out: [n]f32 = undefined;
        var scalar_out: [n]f32 = undefined;

        simd.softmax(&data, &simd_out);
        scalarSoftmax(&data, &scalar_out);
        try expectSlicesApproxEq(&scalar_out, &simd_out);
    }
}

test "SIMD softmax probabilities sum to 1" {
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var out: [5]f32 = undefined;
    simd.softmax(&data, &out);

    var prob_sum: f32 = 0.0;
    for (out) |v| prob_sum += v;
    try expectApproxEq(1.0, prob_sum);
}

test "SIMD softmax all equal inputs" {
    const data = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    var out: [4]f32 = undefined;
    simd.softmax(&data, &out);

    for (out) |v| try expectApproxEq(0.25, v);
}

test "SIMD softmax numerical stability with large values" {
    const data = [_]f32{ 1000.0, 1001.0, 1002.0, 1003.0 };
    var out: [4]f32 = undefined;
    simd.softmax(&data, &out);

    // Should not produce NaN or Inf
    for (out) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
        try std.testing.expect(v > 0.0);
    }
}

// ============================================================================
// SAXPY Tests
// ============================================================================

test "SIMD saxpy matches scalar" {
    inline for (test_sizes) |n| {
        if (n == 0) continue;
        const x = makeTestData(n);
        var simd_y: [n]f32 = undefined;
        var scalar_y: [n]f32 = undefined;
        for (&simd_y, &scalar_y, 0..) |*sy, *scaly, i| {
            const val = @as(f32, @floatFromInt(i)) * 0.5;
            sy.* = val;
            scaly.* = val;
        }

        const alpha: f32 = 2.5;
        simd.saxpy(alpha, &x, &simd_y);
        scalarSaxpy(alpha, &x, &scalar_y);
        try expectSlicesApproxEq(&scalar_y, &simd_y);
    }
}

test "SIMD saxpy alpha=0 is identity" {
    const x = [_]f32{ 100.0, 200.0, 300.0, 400.0 };
    var y = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const original_y = y;
    simd.saxpy(0.0, &x, &y);
    try expectSlicesApproxEq(&original_y, &y);
}

test "SIMD saxpy alpha=1 is addition" {
    const x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var y = [_]f32{ 10.0, 20.0, 30.0, 40.0 };
    simd.saxpy(1.0, &x, &y);
    const expected = [_]f32{ 11.0, 22.0, 33.0, 44.0 };
    try expectSlicesApproxEq(&expected, &y);
}

// ============================================================================
// Reduce Tests
// ============================================================================

test "SIMD reduceSum matches scalar" {
    inline for (test_sizes) |n| {
        if (n == 0) continue;
        const data = makeTestData(n);
        const simd_result = simd.reduceSum(&data);
        const scalar_result = scalarReduceSum(&data);
        try expectApproxEq(scalar_result, simd_result);
    }
}

test "SIMD reduceSum known values" {
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    try expectApproxEq(15.0, simd.reduceSum(&data));
}

test "SIMD reduceMin matches scalar" {
    inline for (test_sizes) |n| {
        if (n == 0) continue;
        const data = makeTestData(n);
        const simd_result = simd.reduceMin(&data);
        const scalar_result = scalarReduceMin(&data);
        try expectApproxEq(scalar_result, simd_result);
    }
}

test "SIMD reduceMin known values" {
    const data = [_]f32{ 5.0, 3.0, 8.0, 1.0, 9.0 };
    try expectApproxEq(1.0, simd.reduceMin(&data));
}

test "SIMD reduceMin negative values" {
    const data = [_]f32{ -1.0, -5.0, -2.0, -3.0 };
    try expectApproxEq(-5.0, simd.reduceMin(&data));
}

test "SIMD reduceMax matches scalar" {
    inline for (test_sizes) |n| {
        if (n == 0) continue;
        const data = makeTestData(n);
        const simd_result = simd.reduceMax(&data);
        const scalar_result = scalarReduceMax(&data);
        try expectApproxEq(scalar_result, simd_result);
    }
}

test "SIMD reduceMax known values" {
    const data = [_]f32{ 5.0, 3.0, 8.0, 1.0, 9.0 };
    try expectApproxEq(9.0, simd.reduceMax(&data));
}

// ============================================================================
// Scale Tests
// ============================================================================

test "SIMD scale matches scalar" {
    inline for (test_sizes) |n| {
        if (n == 0) continue;
        const data = makeTestData(n);
        var simd_out: [n]f32 = undefined;
        var scalar_out: [n]f32 = undefined;

        const scalar_val: f32 = 3.14;
        simd.scale(&data, scalar_val, &simd_out);
        scalarScale(&data, scalar_val, &scalar_out);
        try expectSlicesApproxEq(&scalar_out, &simd_out);
    }
}

test "SIMD scale by 0 produces zeros" {
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var out: [8]f32 = undefined;
    simd.scale(&data, 0.0, &out);
    for (out) |v| try expectApproxEq(0.0, v);
}

test "SIMD scale by 1 is identity" {
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var out: [4]f32 = undefined;
    simd.scale(&data, 1.0, &out);
    try expectSlicesApproxEq(&data, &out);
}

test "SIMD scale by -1 negates" {
    const data = [_]f32{ 1.0, -2.0, 3.0, -4.0 };
    var out: [4]f32 = undefined;
    simd.scale(&data, -1.0, &out);
    const expected = [_]f32{ -1.0, 2.0, -3.0, 4.0 };
    try expectSlicesApproxEq(&expected, &out);
}
