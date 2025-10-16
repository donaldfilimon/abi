const std = @import("std");
// SIMD module import - requires shared/simd.zig integration
// const simd = @import("../../../shared/simd.zig");

const SELU_ALPHA = 1.6732632423543772848170429916717;
const SELU_SCALE = 1.0507009873554804934193349852946;
const LEAKY_RELU_SLOPE = 0.01;
const GELU_SQRT_2 = 0.7978845608028654; // sqrt(2/pi)
const EPSILON = 1e-8;

/// High-performance activation function utilities shared across AI components.
pub const ActivationUtils = struct {
    /// Inline fast approximation functions for better performance
    pub inline fn fastSigmoid(x: f32) f32 {
        return 0.5 * (std.math.tanh(0.5 * x) + 1.0);
    }

    pub inline fn fastTanh(x: f32) f32 {
        if (x > 3.0) return 1.0;
        if (x < -3.0) return -1.0;
        const x2 = x * x;
        return x * (27.0 + x2) / (27.0 + 9.0 * x2);
    }

    pub inline fn fastExp(x: f32) f32 {
        if (x > 10.0) return std.math.exp(10.0);
        if (x < -10.0) return std.math.exp(-10.0);
        return std.math.exp(x);
    }

    pub inline fn fastGelu(x: f32) f32 {
        return 0.5 * x * (1.0 + fastTanh(GELU_SQRT_2 * (x + 0.044715 * x * x * x)));
    }

    pub inline fn fastSqrt(x: f32) f32 {
        if (x <= 0.0) return 0.0;
        return @sqrt(x);
    }

    /// Vectorized ReLU activation with SIMD optimization
    pub inline fn vectorizedRelu(data: []f32) void {
        if (comptime std.simd.suggestVectorLength(f32)) |simd_len| {
            if (simd_len >= 4 and data.len >= 8) {
                // SIMD activation - requires VectorOps integration
                // if (simd.VectorOps.shouldUseSimd(data.len)) {
                //     simd.VectorOps.vectorizedRelu(data);
                //     return;
                // }
                if (false) { // Placeholder
                    return;
                }
            }
        }

        var i: usize = 0;
        while (i + 4 <= data.len) : (i += 4) {
            data[i] = @max(0.0, data[i]);
            data[i + 1] = @max(0.0, data[i + 1]);
            data[i + 2] = @max(0.0, data[i + 2]);
            data[i + 3] = @max(0.0, data[i + 3]);
        }
        while (i < data.len) : (i += 1) {
            data[i] = @max(0.0, data[i]);
        }
    }

    pub inline fn vectorizedSigmoid(data: []f32) void {
        var i: usize = 0;
        while (i + 4 <= data.len) : (i += 4) {
            data[i] = fastSigmoid(data[i]);
            data[i + 1] = fastSigmoid(data[i + 1]);
            data[i + 2] = fastSigmoid(data[i + 2]);
            data[i + 3] = fastSigmoid(data[i + 3]);
        }
        while (i < data.len) : (i += 1) {
            data[i] = fastSigmoid(data[i]);
        }
    }

    pub inline fn vectorizedTanh(data: []f32) void {
        var i: usize = 0;
        while (i + 4 <= data.len) : (i += 4) {
            data[i] = fastTanh(data[i]);
            data[i + 1] = fastTanh(data[i + 1]);
            data[i + 2] = fastTanh(data[i + 2]);
            data[i + 3] = fastTanh(data[i + 3]);
        }
        while (i < data.len) : (i += 1) {
            data[i] = fastTanh(data[i]);
        }
    }

    /// Vectorized Leaky ReLU activation with SIMD optimization
    pub inline fn vectorizedLeakyRelu(data: []f32) void {
        if (comptime std.simd.suggestVectorLength(f32)) |simd_len| {
            if (simd_len >= 4 and data.len >= 8) {
                // SIMD activation - requires VectorOps integration
                // if (simd.VectorOps.shouldUseSimd(data.len)) {
                //     simd.VectorOps.vectorizedLeakyRelu(data, LEAKY_RELU_SLOPE);
                //     return;
                // }
                if (false) { // Placeholder
                    return;
                }
            }
        }

        var i: usize = 0;
        while (i + 4 <= data.len) : (i += 4) {
            data[i] = if (data[i] > 0.0) data[i] else LEAKY_RELU_SLOPE * data[i];
            data[i + 1] = if (data[i + 1] > 0.0) data[i + 1] else LEAKY_RELU_SLOPE * data[i + 1];
            data[i + 2] = if (data[i + 2] > 0.0) data[i + 2] else LEAKY_RELU_SLOPE * data[i + 2];
            data[i + 3] = if (data[i + 3] > 0.0) data[i + 3] else LEAKY_RELU_SLOPE * data[i + 3];
        }
        while (i < data.len) : (i += 1) {
            data[i] = if (data[i] > 0.0) data[i] else LEAKY_RELU_SLOPE * data[i];
        }
    }

    pub inline fn vectorizedGelu(data: []f32) void {
        var i: usize = 0;
        while (i + 4 <= data.len) : (i += 4) {
            data[i] = fastGelu(data[i]);
            data[i + 1] = fastGelu(data[i + 1]);
            data[i + 2] = fastGelu(data[i + 2]);
            data[i + 3] = fastGelu(data[i + 3]);
        }
        while (i < data.len) : (i += 1) {
            data[i] = fastGelu(data[i]);
        }
    }

    /// Optimized softmax with numerical stability
    pub inline fn stableSoftmax(data: []f32) void {
        if (data.len == 0) return;

        var max_val = data[0];
        for (data[1..]) |val| {
            max_val = @max(max_val, val);
        }

        var sum: f32 = 0.0;
        for (data) |*val| {
            val.* = fastExp(val.* - max_val);
            sum += val.*;
        }

        const inv_sum = 1.0 / (sum + EPSILON);
        for (data) |*val| {
            val.* *= inv_sum;
        }
    }

    /// Optimized log softmax with numerical stability
    pub inline fn stableLogSoftmax(data: []f32) void {
        if (data.len == 0) return;

        var max_val = data[0];
        for (data[1..]) |val| {
            max_val = @max(max_val, val);
        }

        var sum: f32 = 0.0;
        for (data) |val| {
            sum += fastExp(val - max_val);
        }

        const log_sum = @log(sum + EPSILON) + max_val;
        for (data) |*val| {
            val.* -= log_sum;
        }
    }
};

/// Provide quick access to default constants used in activation utilities.
pub const ActivationConstants = struct {
    pub const selu_alpha = SELU_ALPHA;
    pub const selu_scale = SELU_SCALE;
    pub const leaky_relu_slope = LEAKY_RELU_SLOPE;
};

// Basic regression tests for the activation utilities.
test "activation utils stable softmax" {
    var values = [_]f32{ 1.0, 2.0, 3.0 };
    ActivationUtils.stableSoftmax(values[0..]);

    var sum: f32 = 0.0;
    for (values) |value| sum += value;
    try std.testing.expectApproxEqAbs(1.0, sum, 1e-5);
}

test "activation utils relu fallback" {
    var values = [_]f32{ -1.0, 0.0, 1.0, 2.0 };
    ActivationUtils.vectorizedRelu(values[0..]);
    try std.testing.expect(values[0] == 0.0);
    try std.testing.expect(values[1] == 0.0);
    try std.testing.expect(values[2] == 1.0);
    try std.testing.expect(values[3] == 2.0);
}
