//! Activation functions for neural networks.
//!
//! Implements common activation functions used in LLMs:
//! - SiLU (Swish): x * sigmoid(x)
//! - GELU: Gaussian Error Linear Unit
//! - Softmax: exp(x) / sum(exp(x))
//!
//! Vector operations use SIMD acceleration when available.

const std = @import("std");
const simd = @import("../../../../shared/simd.zig");

/// SiLU (Swish) activation: x * sigmoid(x)
/// Used in LLaMA, Mistral, and other modern LLMs.
pub fn silu(x: f32) f32 {
    return x * sigmoid(x);
}

/// Apply SiLU to a vector in-place (SIMD accelerated).
pub fn siluInPlace(x: []f32) void {
    simd.siluInPlace(x);
}

/// Apply SiLU element-wise.
pub fn siluVector(input: []const f32, output: []f32) void {
    for (input, 0..) |v, i| {
        output[i] = silu(v);
    }
}

/// Sigmoid activation: 1 / (1 + exp(-x))
pub fn sigmoid(x: f32) f32 {
    if (x >= 0) {
        const exp_neg = @exp(-x);
        return 1.0 / (1.0 + exp_neg);
    } else {
        const exp_x = @exp(x);
        return exp_x / (1.0 + exp_x);
    }
}

/// GELU activation (tanh approximation).
/// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu(x: f32) f32 {
    const sqrt_2_pi = 0.7978845608028654; // sqrt(2/pi)
    const coeff = 0.044715;

    const inner = sqrt_2_pi * (x + coeff * x * x * x);
    return 0.5 * x * (1.0 + std.math.tanh(inner));
}

/// Apply GELU to a vector in-place (SIMD accelerated).
pub fn geluInPlace(x: []f32) void {
    simd.geluInPlace(x);
}

/// Error function approximation (Horner's method polynomial approximation)
fn erf(x: f32) f32 {
    // Abramowitz and Stegun approximation 7.1.26
    const a1: f32 = 0.254829592;
    const a2: f32 = -0.284496736;
    const a3: f32 = 1.421413741;
    const a4: f32 = -1.453152027;
    const a5: f32 = 1.061405429;
    const p: f32 = 0.3275911;

    const sign: f32 = if (x < 0) -1.0 else 1.0;
    const abs_x = @abs(x);

    const t = 1.0 / (1.0 + p * abs_x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * @exp(-abs_x * abs_x);

    return sign * y;
}

/// GELU exact (slower but more accurate).
pub fn geluExact(x: f32) f32 {
    const sqrt_2 = 1.4142135623730951;
    return 0.5 * x * (1.0 + erf(x / sqrt_2));
}

/// ReLU activation: max(0, x)
pub fn relu(x: f32) f32 {
    return @max(0, x);
}

/// Apply ReLU to a vector in-place (SIMD accelerated).
pub fn reluInPlace(x: []f32) void {
    simd.reluInPlace(x);
}

/// Leaky ReLU: max(alpha * x, x)
pub fn leakyRelu(x: f32, alpha: f32) f32 {
    return if (x > 0) x else alpha * x;
}

/// Softmax over a vector (numerically stable).
pub fn softmax(input: []const f32, output: []f32) void {
    @memcpy(output, input);
    softmaxInPlace(output);
}

/// In-place softmax with numerical stability (SIMD accelerated).
pub fn softmaxInPlace(x: []f32) void {
    simd.softmaxInPlace(x);
}

/// Softmax with temperature scaling.
pub fn softmaxWithTemperature(input: []const f32, output: []f32, temperature: f32) void {
    const inv_temp = 1.0 / temperature;

    var max_val: f32 = -std.math.inf(f32);
    for (input) |v| {
        const scaled = v * inv_temp;
        if (scaled > max_val) max_val = scaled;
    }

    var sum: f32 = 0;
    for (input, 0..) |v, i| {
        output[i] = @exp(v * inv_temp - max_val);
        sum += output[i];
    }

    const inv_sum = 1.0 / sum;
    for (output) |*v| {
        v.* *= inv_sum;
    }
}

/// Log-softmax for numerical stability in loss computation.
pub fn logSoftmax(input: []const f32, output: []f32) void {
    var max_val: f32 = -std.math.inf(f32);
    for (input) |v| {
        if (v > max_val) max_val = v;
    }

    var log_sum_exp: f32 = 0;
    for (input) |v| {
        log_sum_exp += @exp(v - max_val);
    }
    log_sum_exp = @log(log_sum_exp) + max_val;

    for (input, 0..) |v, i| {
        output[i] = v - log_sum_exp;
    }
}

/// Tanh activation.
pub fn tanh(x: f32) f32 {
    return std.math.tanh(x);
}

/// Apply tanh to a vector in-place.
pub fn tanhInPlace(x: []f32) void {
    for (x) |*v| {
        v.* = std.math.tanh(v.*);
    }
}

/// Hardswish activation (efficient approximation of swish).
pub fn hardswish(x: f32) f32 {
    if (x <= -3) return 0;
    if (x >= 3) return x;
    return x * (x + 3) / 6;
}

/// Mish activation: x * tanh(softplus(x))
pub fn mish(x: f32) f32 {
    const sp = std.math.log1p(@exp(x));
    return x * std.math.tanh(sp);
}

test "silu basic" {
    // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), silu(0), 0.001);

    // SiLU(1) ≈ 1 * 0.731 = 0.731
    try std.testing.expectApproxEqAbs(@as(f32, 0.731), silu(1), 0.01);

    // SiLU(-1) ≈ -1 * 0.269 = -0.269
    try std.testing.expectApproxEqAbs(@as(f32, -0.269), silu(-1), 0.01);
}

test "gelu basic" {
    // GELU(0) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), gelu(0), 0.001);

    // GELU(1) ≈ 0.841
    try std.testing.expectApproxEqAbs(@as(f32, 0.841), gelu(1), 0.01);

    // GELU(-1) ≈ -0.159
    try std.testing.expectApproxEqAbs(@as(f32, -0.159), gelu(-1), 0.01);
}

test "softmax basic" {
    var x = [_]f32{ 1.0, 2.0, 3.0 };
    softmaxInPlace(&x);

    // Sum should be 1
    var sum: f32 = 0;
    for (x) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.001);

    // Values should be monotonically increasing
    try std.testing.expect(x[0] < x[1]);
    try std.testing.expect(x[1] < x[2]);
}

test "softmax numerical stability" {
    // Large values should not overflow
    var x = [_]f32{ 1000.0, 1001.0, 1002.0 };
    softmaxInPlace(&x);

    var sum: f32 = 0;
    for (x) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.001);

    // All values should be valid (not NaN or Inf)
    for (x) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }
}

test "sigmoid bounds" {
    // Sigmoid output should always be in (0, 1)
    try std.testing.expect(sigmoid(-100) > 0);
    try std.testing.expect(sigmoid(-100) < 1);
    try std.testing.expect(sigmoid(100) > 0);
    try std.testing.expect(sigmoid(100) < 1);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), sigmoid(0), 0.001);
}
