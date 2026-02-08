//! SIMD module tests — migrated from the original monolithic simd.zig.
//! Tests exercise all submodules through the re-export hub (mod.zig).

const std = @import("std");

const simd = @import("mod.zig");

// Pull in individual submodules for direct access where needed
const vector_ops = simd.vector_ops;
const activations = simd.activations;
const distances = simd.distances;
const integer_ops = simd.integer_ops;
const extras = simd.extras;

// ============================================================================
// Vector Operations Tests
// ============================================================================

test "vector addition works" {
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var b = [_]f32{ 0.5, 1.5, 2.5, 3.5 };
    var result: [4]f32 = undefined;

    simd.vectorAdd(&a, &b, &result);

    try std.testing.expectEqual(@as(f32, 1.5), result[0]);
    try std.testing.expectEqual(@as(f32, 3.5), result[1]);
    try std.testing.expectEqual(@as(f32, 5.5), result[2]);
    try std.testing.expectEqual(@as(f32, 7.5), result[3]);
}

test "vector dot product works" {
    var a = [_]f32{ 1.0, 2.0, 3.0 };
    var b = [_]f32{ 4.0, 5.0, 6.0 };

    const result = simd.vectorDot(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), result, 1e-6);
}

test "vector L2 norm works" {
    var v = [_]f32{ 3.0, 4.0 };

    const result = simd.vectorL2Norm(&v);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), result, 1e-6);
}

test "cosine similarity works" {
    var a = [_]f32{ 1.0, 0.0 };
    var b = [_]f32{ 0.0, 1.0 };

    const result = simd.cosineSimilarity(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result, 1e-6);
}

// ============================================================================
// Matrix Multiplication Tests
// ============================================================================

test "matrix multiplication works" {
    var a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b = [_]f32{ 7, 8, 9, 10, 11, 12 };
    var result: [4]f32 = undefined;

    simd.matrixMultiply(&a, &b, &result, 2, 2, 3);

    try std.testing.expectApproxEqAbs(@as(f32, 58.0), result[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 64.0), result[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 139.0), result[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 154.0), result[3], 1e-5);
}

test "matrix multiplication larger" {
    const size = 8;
    var a: [size * size]f32 = undefined;
    var b: [size * size]f32 = undefined;
    var result: [size * size]f32 = undefined;

    for (0..size) |i| {
        for (0..size) |j| {
            a[i * size + j] = @floatFromInt(i + j);
            b[i * size + j] = @floatFromInt(i * j);
        }
    }

    simd.matrixMultiply(&a, &b, &result, size, size, size);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 168.0), result[size + 1], 1e-4);
}

// ============================================================================
// Activation Function Tests
// ============================================================================

test "siluInPlace works" {
    var data = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    simd.siluInPlace(&data);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data[2], 1e-6);
    try std.testing.expect(data[3] > 0);
    try std.testing.expect(data[4] > 0);
    try std.testing.expect(data[0] < 0);
    try std.testing.expect(data[1] < 0);
}

test "geluInPlace works" {
    var data = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    simd.geluInPlace(&data);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data[2], 1e-6);
    try std.testing.expect(data[3] > 0);
    try std.testing.expect(data[4] > 0);
}

test "reluInPlace works" {
    var data = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    simd.reluInPlace(&data);

    try std.testing.expectEqual(@as(f32, 0.0), data[0]);
    try std.testing.expectEqual(@as(f32, 0.0), data[1]);
    try std.testing.expectEqual(@as(f32, 0.0), data[2]);
    try std.testing.expectEqual(@as(f32, 1.0), data[3]);
    try std.testing.expectEqual(@as(f32, 2.0), data[4]);
}

test "leakyReluInPlace works" {
    var data = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    simd.leakyReluInPlace(&data, 0.1);

    try std.testing.expectApproxEqAbs(@as(f32, -0.2), data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.1), data[1], 1e-6);
    try std.testing.expectEqual(@as(f32, 0.0), data[2]);
    try std.testing.expectEqual(@as(f32, 1.0), data[3]);
    try std.testing.expectEqual(@as(f32, 2.0), data[4]);
}

// ============================================================================
// Softmax Tests
// ============================================================================

test "softmaxInPlace works" {
    var data = [_]f32{ 1.0, 2.0, 3.0 };
    simd.softmaxInPlace(&data);

    var total: f32 = 0.0;
    for (data) |v| total += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), total, 1e-5);

    try std.testing.expect(data[0] < data[1]);
    try std.testing.expect(data[1] < data[2]);

    for (data) |v| {
        try std.testing.expect(v > 0);
    }
}

test "maxValue works" {
    const data = [_]f32{ 1.0, 5.0, 3.0, 2.0, 4.0 };
    const max_val = simd.maxValue(&data);
    try std.testing.expectEqual(@as(f32, 5.0), max_val);
}

test "sum works" {
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const total = simd.sum(&data);
    try std.testing.expectApproxEqAbs(@as(f32, 15.0), total, 1e-6);
}

// ============================================================================
// Normalization Tests
// ============================================================================

test "rmsNormInPlace works" {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const original_sq_sum = simd.squaredSum(&data);
    simd.rmsNormInPlace(&data, null, 1e-6);

    const new_sq_sum = simd.squaredSum(&data);
    const new_rms = @sqrt(new_sq_sum / 4.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), new_rms, 1e-4);

    try std.testing.expect(original_sq_sum > 1.0);
}

test "squaredSum works" {
    const data = [_]f32{ 1.0, 2.0, 3.0 };
    const result = simd.squaredSum(&data);
    try std.testing.expectApproxEqAbs(@as(f32, 14.0), result, 1e-6);
}

// ============================================================================
// Distance Tests
// ============================================================================

test "l2DistanceSquared works" {
    const a = [_]f32{ 0.0, 0.0, 0.0 };
    const b = [_]f32{ 1.0, 2.0, 2.0 };
    const dist_sq = simd.l2DistanceSquared(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), dist_sq, 1e-6);
}

test "l2Distance works" {
    const a = [_]f32{ 0.0, 0.0, 0.0 };
    const b = [_]f32{ 1.0, 2.0, 2.0 };
    const dist = simd.l2Distance(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), dist, 1e-6);
}

test "SIMD activation large array" {
    var data: [64]f32 = undefined;
    for (0..64) |i| {
        data[i] = @as(f32, @floatFromInt(i)) / 32.0 - 1.0;
    }

    var silu_data = data;
    simd.siluInPlace(&silu_data);

    var gelu_data = data;
    simd.geluInPlace(&gelu_data);

    var relu_data = data;
    simd.reluInPlace(&relu_data);

    for (silu_data) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }
    for (gelu_data) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }
    for (relu_data) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }
}

// ============================================================================
// Integer SIMD Tests
// ============================================================================

test "integer addition works" {
    const a = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const b = [_]i32{ 10, 20, 30, 40, 50, 60, 70, 80 };
    var result: [8]i32 = undefined;

    simd.vectorAddI32(&a, &b, &result);

    try std.testing.expectEqual(@as(i32, 11), result[0]);
    try std.testing.expectEqual(@as(i32, 22), result[1]);
    try std.testing.expectEqual(@as(i32, 88), result[7]);
}

test "integer sum works" {
    const data = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const total = simd.sumI32(&data);
    try std.testing.expectEqual(@as(i64, 36), total);
}

test "integer max works" {
    const data = [_]i32{ 1, 5, 3, 9, 2, 8, 4, 7 };
    const max_val = simd.maxI32(&data);
    try std.testing.expectEqual(@as(i32, 9), max_val);
}

test "integer min works" {
    const data = [_]i32{ 5, 3, 9, 1, 8, 4, 7, 2 };
    const min_val = simd.minI32(&data);
    try std.testing.expectEqual(@as(i32, 1), min_val);
}

// ============================================================================
// FMA Tests
// ============================================================================

test "fma works" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 2.0, 3.0, 4.0, 5.0 };
    const c = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    var result: [4]f32 = undefined;

    simd.fma(&a, &b, &c, &result);

    try std.testing.expectApproxEqAbs(@as(f32, 2.5), result[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.5), result[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 12.5), result[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 20.5), result[3], 1e-6);
}

test "fmaScalar works" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    var result: [4]f32 = undefined;

    simd.fmaScalar(2.0, &a, &b, &result);

    try std.testing.expectApproxEqAbs(@as(f32, 2.5), result[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.5), result[1], 1e-6);
}

// ============================================================================
// Scaling Tests
// ============================================================================

test "scaleInPlace works" {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    simd.scaleInPlace(&data, 2.0);

    try std.testing.expectEqual(@as(f32, 2.0), data[0]);
    try std.testing.expectEqual(@as(f32, 4.0), data[1]);
    try std.testing.expectEqual(@as(f32, 6.0), data[2]);
    try std.testing.expectEqual(@as(f32, 8.0), data[3]);
}

test "addScalarInPlace works" {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    simd.addScalarInPlace(&data, 10.0);

    try std.testing.expectEqual(@as(f32, 11.0), data[0]);
    try std.testing.expectEqual(@as(f32, 12.0), data[1]);
}

// ============================================================================
// Element-wise Tests
// ============================================================================

test "hadamard product works" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 2.0, 3.0, 4.0, 5.0 };
    var result: [4]f32 = undefined;

    simd.hadamard(&a, &b, &result);

    try std.testing.expectEqual(@as(f32, 2.0), result[0]);
    try std.testing.expectEqual(@as(f32, 6.0), result[1]);
    try std.testing.expectEqual(@as(f32, 12.0), result[2]);
    try std.testing.expectEqual(@as(f32, 20.0), result[3]);
}

test "absInPlace works" {
    var data = [_]f32{ -1.0, 2.0, -3.0, 4.0 };
    simd.absInPlace(&data);

    try std.testing.expectEqual(@as(f32, 1.0), data[0]);
    try std.testing.expectEqual(@as(f32, 2.0), data[1]);
    try std.testing.expectEqual(@as(f32, 3.0), data[2]);
    try std.testing.expectEqual(@as(f32, 4.0), data[3]);
}

test "clampInPlace works" {
    var data = [_]f32{ -5.0, 0.5, 1.5, 10.0 };
    simd.clampInPlace(&data, 0.0, 1.0);

    try std.testing.expectEqual(@as(f32, 0.0), data[0]);
    try std.testing.expectEqual(@as(f32, 0.5), data[1]);
    try std.testing.expectEqual(@as(f32, 1.0), data[2]);
    try std.testing.expectEqual(@as(f32, 1.0), data[3]);
}

test "countGreaterThan works" {
    const data = [_]f32{ 0.1, 0.6, 0.3, 0.8, 0.2, 0.9, 0.4, 0.7 };
    const count = simd.countGreaterThan(&data, 0.5);
    try std.testing.expectEqual(@as(usize, 4), count);
}

// ============================================================================
// Memory Operation Tests
// ============================================================================

test "copyF32 works" {
    const src = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var dst: [4]f32 = undefined;

    simd.copyF32(&src, &dst);

    try std.testing.expectEqual(@as(f32, 1.0), dst[0]);
    try std.testing.expectEqual(@as(f32, 4.0), dst[3]);
}

test "fillF32 works" {
    var data: [8]f32 = undefined;
    simd.fillF32(&data, 3.14);

    for (data) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 3.14), v, 1e-6);
    }
}

// ============================================================================
// v2.0 SIMD Kernel Tests
// ============================================================================

test "euclideanDistance works" {
    const a = [_]f32{ 0.0, 0.0, 0.0 };
    const b = [_]f32{ 1.0, 2.0, 2.0 };
    const dist = simd.euclideanDistance(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), dist, 1e-6);
}

test "euclideanDistance identical vectors is zero" {
    const a = [_]f32{ 3.0, 4.0, 5.0, 6.0 };
    const dist = simd.euclideanDistance(&a, &a);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), dist, 1e-6);
}

test "euclideanDistance large array" {
    var a: [64]f32 = undefined;
    var b: [64]f32 = undefined;
    for (0..64) |i| {
        a[i] = @floatFromInt(i);
        b[i] = @as(f32, @floatFromInt(i)) + 1.0;
    }
    const dist = simd.euclideanDistance(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), dist, 1e-5);
}

test "softmax output sums to one" {
    const data = [_]f32{ 1.0, 2.0, 3.0 };
    var out: [3]f32 = undefined;
    simd.softmax(&data, &out);

    var total: f32 = 0.0;
    for (out) |v| total += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), total, 1e-5);

    try std.testing.expect(out[0] < out[1]);
    try std.testing.expect(out[1] < out[2]);

    for (out) |v| {
        try std.testing.expect(v > 0);
    }
}

test "softmax numerical stability with large values" {
    const data = [_]f32{ 1000.0, 1001.0, 1002.0 };
    var out: [3]f32 = undefined;
    simd.softmax(&data, &out);

    var total: f32 = 0.0;
    for (out) |v| total += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), total, 1e-5);

    for (out) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }
}

test "saxpy basic operation" {
    var y = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const x = [_]f32{ 10.0, 20.0, 30.0, 40.0 };
    simd.saxpy(2.0, &x, &y);

    try std.testing.expectApproxEqAbs(@as(f32, 21.0), y[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 42.0), y[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 63.0), y[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 84.0), y[3], 1e-6);
}

test "saxpy with zero alpha" {
    const original = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var y = original;
    const x = [_]f32{ 10.0, 20.0, 30.0, 40.0 };
    simd.saxpy(0.0, &x, &y);

    for (y, original) |actual, expected| {
        try std.testing.expectEqual(expected, actual);
    }
}

test "saxpy large array" {
    var y: [64]f32 = undefined;
    var x: [64]f32 = undefined;
    for (0..64) |i| {
        x[i] = 1.0;
        y[i] = @floatFromInt(i);
    }
    simd.saxpy(3.0, &x, &y);

    for (0..64) |i| {
        const expected = @as(f32, @floatFromInt(i)) + 3.0;
        try std.testing.expectApproxEqAbs(expected, y[i], 1e-5);
    }
}

test "reduceSum works" {
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const total = simd.reduceSum(&data);
    try std.testing.expectApproxEqAbs(@as(f32, 15.0), total, 1e-6);
}

test "reduceSum empty returns zero" {
    const data = [_]f32{};
    const total = simd.reduceSum(&data);
    try std.testing.expectEqual(@as(f32, 0.0), total);
}

test "reduceSum large array" {
    var data: [64]f32 = undefined;
    for (&data) |*d| {
        d.* = 1.0;
    }
    const total = simd.reduceSum(&data);
    try std.testing.expectApproxEqAbs(@as(f32, 64.0), total, 1e-4);
}

test "reduceMin works" {
    const data = [_]f32{ 5.0, 3.0, 9.0, 1.0, 7.0 };
    const result = simd.reduceMin(&data);
    try std.testing.expectEqual(@as(f32, 1.0), result);
}

test "reduceMin empty returns positive infinity" {
    const data = [_]f32{};
    const result = simd.reduceMin(&data);
    try std.testing.expect(std.math.isInf(result));
    try std.testing.expect(result > 0);
}

test "reduceMin negative values" {
    const data = [_]f32{ -1.0, -5.0, -2.0, -3.0 };
    const result = simd.reduceMin(&data);
    try std.testing.expectEqual(@as(f32, -5.0), result);
}

test "reduceMax works" {
    const data = [_]f32{ 5.0, 3.0, 9.0, 1.0, 7.0 };
    const result = simd.reduceMax(&data);
    try std.testing.expectEqual(@as(f32, 9.0), result);
}

test "reduceMax empty returns negative infinity" {
    const data = [_]f32{};
    const result = simd.reduceMax(&data);
    try std.testing.expect(std.math.isInf(result));
    try std.testing.expect(result < 0);
}

test "reduceMax negative values" {
    const data = [_]f32{ -1.0, -5.0, -2.0, -3.0 };
    const result = simd.reduceMax(&data);
    try std.testing.expectEqual(@as(f32, -1.0), result);
}

test "scale out-of-place works" {
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var out: [4]f32 = undefined;
    simd.scale(&data, 3.0, &out);

    try std.testing.expectEqual(@as(f32, 3.0), out[0]);
    try std.testing.expectEqual(@as(f32, 6.0), out[1]);
    try std.testing.expectEqual(@as(f32, 9.0), out[2]);
    try std.testing.expectEqual(@as(f32, 12.0), out[3]);
}

test "scale preserves original" {
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var out: [4]f32 = undefined;
    simd.scale(&data, 5.0, &out);

    try std.testing.expectEqual(@as(f32, 1.0), data[0]);
    try std.testing.expectEqual(@as(f32, 2.0), data[1]);
}

test "scale large array" {
    var data: [64]f32 = undefined;
    var out: [64]f32 = undefined;
    for (0..64) |i| {
        data[i] = @floatFromInt(i);
    }
    simd.scale(&data, 2.0, &out);

    for (0..64) |i| {
        const expected = @as(f32, @floatFromInt(i)) * 2.0;
        try std.testing.expectApproxEqAbs(expected, out[i], 1e-5);
    }
}
