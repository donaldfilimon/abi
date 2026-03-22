//! Activation functions and normalization operations (SIMD-accelerated)
//!
//! Includes SiLU, GELU, ReLU, Leaky ReLU, softmax, log-softmax,
//! RMSNorm, LayerNorm, and supporting helper functions.

const std = @import("std");

const VectorSize = std.simd.suggestVectorLength(f32) orelse 4;

// ============================================================================
// Activation Functions (SIMD-accelerated)
// ============================================================================

/// SiLU (Swish) activation: x * sigmoid(x)
/// In-place modification for memory efficiency
pub fn siluInPlace(data: []f32) void {
    if (data.len == 0) return;

    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const one: Vec = @splat(1.0);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const x: Vec = data[i..][0..VectorSize].*;
            // sigmoid(x) = 1 / (1 + exp(-x))
            const neg_x = -x;
            const exp_neg = @exp(neg_x);
            const sigmoid = one / (one + exp_neg);
            data[i..][0..VectorSize].* = x * sigmoid;
        }
    }

    // Scalar tail
    while (i < data.len) : (i += 1) {
        const x = data[i];
        const sig = 1.0 / (1.0 + @exp(-x));
        data[i] = x * sig;
    }
}

/// GELU activation (approximate): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
/// In-place modification for memory efficiency
pub fn geluInPlace(data: []f32) void {
    if (data.len == 0) return;

    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const sqrt_2_pi: Vec = @splat(0.7978845608); // sqrt(2/pi)
        const coeff: Vec = @splat(0.044715);
        const half: Vec = @splat(0.5);
        const one: Vec = @splat(1.0);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const x: Vec = data[i..][0..VectorSize].*;
            const x3 = x * x * x;
            const inner = sqrt_2_pi * (x + coeff * x3);
            // tanh approximation or use @tanh
            const tanh_val = tanhVec(inner);
            data[i..][0..VectorSize].* = half * x * (one + tanh_val);
        }
    }

    // Scalar tail
    while (i < data.len) : (i += 1) {
        const x = data[i];
        const x3 = x * x * x;
        const inner = 0.7978845608 * (x + 0.044715 * x3);
        const tanh_val = std.math.tanh(inner);
        data[i] = 0.5 * x * (1.0 + tanh_val);
    }
}

/// ReLU activation: max(0, x)
/// In-place modification
pub fn reluInPlace(data: []f32) void {
    if (data.len == 0) return;

    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const zero: Vec = @splat(0.0);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const x: Vec = data[i..][0..VectorSize].*;
            data[i..][0..VectorSize].* = @max(zero, x);
        }
    }

    // Scalar tail
    while (i < data.len) : (i += 1) {
        data[i] = @max(0.0, data[i]);
    }
}

/// Leaky ReLU activation: x if x > 0 else alpha * x
/// In-place modification
pub fn leakyReluInPlace(data: []f32, alpha: f32) void {
    if (data.len == 0) return;

    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const zero: Vec = @splat(0.0);
        const alpha_vec: Vec = @splat(alpha);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const x: Vec = data[i..][0..VectorSize].*;
            // leaky_relu = x if x > 0 else alpha * x
            // = max(x, alpha * x) when alpha < 1
            const scaled = alpha_vec * x;
            const mask = x > zero;
            data[i..][0..VectorSize].* = @select(f32, mask, x, scaled);
        }
    }

    // Scalar tail
    while (i < data.len) : (i += 1) {
        const x = data[i];
        data[i] = if (x > 0) x else alpha * x;
    }
}

// ============================================================================
// Softmax Operations (SIMD-accelerated)
// ============================================================================

/// Find maximum value in array using SIMD
pub fn maxValue(data: []const f32) f32 {
    if (data.len == 0) return -std.math.inf(f32);

    var i: usize = 0;
    var max_val: f32 = data[0];

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        var max_vec: Vec = @splat(data[0]);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const v: Vec = data[i..][0..VectorSize].*;
            max_vec = @max(max_vec, v);
        }

        // Horizontal max
        max_val = @reduce(.Max, max_vec);
    }

    // Scalar tail
    while (i < data.len) : (i += 1) {
        max_val = @max(max_val, data[i]);
    }

    return max_val;
}

/// Compute exp(x - max) in-place for numerical stability
pub fn expSubtractMax(data: []f32, max_val: f32) void {
    if (data.len == 0) return;

    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const max_vec: Vec = @splat(max_val);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const x: Vec = data[i..][0..VectorSize].*;
            const diff: Vec = x - max_vec;
            const exp_result: Vec = @exp(diff);
            data[i..][0..VectorSize].* = exp_result;
        }
    }

    // Scalar tail
    while (i < data.len) : (i += 1) {
        data[i] = @exp(data[i] - max_val);
    }
}

/// Sum all values in array using SIMD
pub fn sum(data: []const f32) f32 {
    if (data.len == 0) return 0.0;

    var i: usize = 0;
    var total: f32 = 0.0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        var sum_vec: Vec = @splat(0.0);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const v: Vec = data[i..][0..VectorSize].*;
            sum_vec += v;
        }

        total = @reduce(.Add, sum_vec);
    }

    // Scalar tail
    while (i < data.len) : (i += 1) {
        total += data[i];
    }

    return total;
}

/// Divide all values by a scalar using SIMD
pub fn divideByScalar(data: []f32, divisor: f32) void {
    if (data.len == 0 or divisor == 0.0) return;

    const inv = 1.0 / divisor;
    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const inv_vec: Vec = @splat(inv);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const v: Vec = data[i..][0..VectorSize].*;
            data[i..][0..VectorSize].* = v * inv_vec;
        }
    }

    // Scalar tail
    while (i < data.len) : (i += 1) {
        data[i] *= inv;
    }
}

/// Complete softmax operation in-place
/// softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
pub fn softmaxInPlace(data: []f32) void {
    if (data.len == 0) return;

    // Step 1: Find max for numerical stability
    const max_val = maxValue(data);

    // Step 2: Compute exp(x - max) in-place
    expSubtractMax(data, max_val);

    // Step 3: Compute sum
    const total = sum(data);

    // Step 4: Normalize
    if (total > 0.0) {
        divideByScalar(data, total);
    }
}

/// Log-softmax for cross-entropy loss: log(softmax(x))
/// More numerically stable than log(softmax(x))
pub fn logSoftmaxInPlace(data: []f32) void {
    if (data.len == 0) return;

    // log_softmax(x)_i = x_i - max(x) - log(sum(exp(x - max(x))))
    const max_val = maxValue(data);

    // Compute sum of exp(x - max) using SIMD
    var exp_sum: f32 = 0.0;
    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const max_vec: Vec = @splat(max_val);
        var sum_vec: Vec = @splat(0.0);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const x: Vec = data[i..][0..VectorSize].*;
            sum_vec += @exp(x - max_vec);
        }

        exp_sum = @reduce(.Add, sum_vec);
    }

    // Scalar tail for exp sum
    while (i < data.len) : (i += 1) {
        exp_sum += @exp(data[i] - max_val);
    }

    const log_sum = @log(exp_sum);

    // Apply log-softmax in-place using SIMD
    i = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const max_vec: Vec = @splat(max_val);
        const log_sum_vec: Vec = @splat(log_sum);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const x: Vec = data[i..][0..VectorSize].*;
            data[i..][0..VectorSize].* = x - max_vec - log_sum_vec;
        }
    }

    // Scalar tail
    while (i < data.len) : (i += 1) {
        data[i] = data[i] - max_val - log_sum;
    }
}

// ============================================================================
// Normalization Operations (SIMD-accelerated)
// ============================================================================

/// Compute sum of squares using SIMD (for RMSNorm)
pub fn squaredSum(data: []const f32) f32 {
    if (data.len == 0) return 0.0;

    var i: usize = 0;
    var total: f32 = 0.0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        var sum_vec: Vec = @splat(0.0);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const v: Vec = data[i..][0..VectorSize].*;
            sum_vec += v * v;
        }

        total = @reduce(.Add, sum_vec);
    }

    // Scalar tail
    while (i < data.len) : (i += 1) {
        total += data[i] * data[i];
    }

    return total;
}

/// RMSNorm: x / sqrt(mean(x^2) + eps)
/// Modifies data in-place and optionally applies weights
pub fn rmsNormInPlace(data: []f32, weights: ?[]const f32, eps: f32) void {
    if (data.len == 0) return;

    // Compute RMS
    const sq_sum = squaredSum(data);
    const mean_sq = sq_sum / @as(f32, @floatFromInt(data.len));
    const rms = @sqrt(mean_sq + eps);
    const inv_rms = 1.0 / rms;

    var i: usize = 0;

    if (weights) |w| {
        std.debug.assert(w.len == data.len);

        if (comptime VectorSize > 1) {
            const Vec = @Vector(VectorSize, f32);
            const inv_rms_vec: Vec = @splat(inv_rms);

            while (i + VectorSize <= data.len) : (i += VectorSize) {
                const x: Vec = data[i..][0..VectorSize].*;
                const weight: Vec = w[i..][0..VectorSize].*;
                data[i..][0..VectorSize].* = x * inv_rms_vec * weight;
            }
        }

        // Scalar tail
        while (i < data.len) : (i += 1) {
            data[i] = data[i] * inv_rms * w[i];
        }
    } else {
        if (comptime VectorSize > 1) {
            const Vec = @Vector(VectorSize, f32);
            const inv_rms_vec: Vec = @splat(inv_rms);

            while (i + VectorSize <= data.len) : (i += VectorSize) {
                const x: Vec = data[i..][0..VectorSize].*;
                data[i..][0..VectorSize].* = x * inv_rms_vec;
            }
        }

        // Scalar tail
        while (i < data.len) : (i += 1) {
            data[i] *= inv_rms;
        }
    }
}

/// LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta
/// Modifies data in-place
pub fn layerNormInPlace(data: []f32, gamma: ?[]const f32, beta: ?[]const f32, eps: f32) void {
    if (data.len == 0) return;

    // Compute mean
    const mean = sum(data) / @as(f32, @floatFromInt(data.len));

    // Compute variance
    var variance: f32 = 0.0;
    for (data) |x| {
        const diff = x - mean;
        variance += diff * diff;
    }
    variance /= @as(f32, @floatFromInt(data.len));

    const inv_std = 1.0 / @sqrt(variance + eps);

    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const mean_vec: Vec = @splat(mean);
        const inv_std_vec: Vec = @splat(inv_std);

        if (gamma != null and beta != null) {
            const g = gamma.?;
            const b = beta.?;
            while (i + VectorSize <= data.len) : (i += VectorSize) {
                const x: Vec = data[i..][0..VectorSize].*;
                const g_vec: Vec = g[i..][0..VectorSize].*;
                const b_vec: Vec = b[i..][0..VectorSize].*;
                data[i..][0..VectorSize].* = (x - mean_vec) * inv_std_vec * g_vec + b_vec;
            }
        } else {
            while (i + VectorSize <= data.len) : (i += VectorSize) {
                const x: Vec = data[i..][0..VectorSize].*;
                data[i..][0..VectorSize].* = (x - mean_vec) * inv_std_vec;
            }
        }
    }

    // Scalar tail
    if (gamma != null and beta != null) {
        const g = gamma.?;
        const b = beta.?;
        while (i < data.len) : (i += 1) {
            data[i] = (data[i] - mean) * inv_std * g[i] + b[i];
        }
    } else {
        while (i < data.len) : (i += 1) {
            data[i] = (data[i] - mean) * inv_std;
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

test "reluInPlace zeros negatives" {
    var data = [_]f32{ -3.0, -1.0, 0.0, 1.0, 5.0 };
    reluInPlace(&data);
    try std.testing.expectEqual(@as(f32, 0.0), data[0]);
    try std.testing.expectEqual(@as(f32, 0.0), data[1]);
    try std.testing.expectEqual(@as(f32, 0.0), data[2]);
    try std.testing.expectEqual(@as(f32, 1.0), data[3]);
    try std.testing.expectEqual(@as(f32, 5.0), data[4]);
}

test "leakyReluInPlace scales negatives" {
    var data = [_]f32{ -10.0, 0.0, 5.0 };
    leakyReluInPlace(&data, 0.1);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), data[2], 0.001);
}

test "siluInPlace at zero" {
    var data = [_]f32{0.0};
    siluInPlace(&data);
    // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data[0], 0.001);
}

test "siluInPlace positive value" {
    var data = [_]f32{2.0};
    siluInPlace(&data);
    // SiLU(2) = 2 * sigmoid(2) = 2 * 0.8808 ≈ 1.7616
    try std.testing.expectApproxEqAbs(@as(f32, 1.7616), data[0], 0.01);
}

test "geluInPlace at zero" {
    var data = [_]f32{0.0};
    geluInPlace(&data);
    // GELU(0) = 0.5 * 0 * (1 + tanh(0)) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data[0], 0.001);
}

test "softmaxInPlace sums to 1" {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    softmaxInPlace(&data);
    var total: f32 = 0;
    for (data) |v| total += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), total, 0.001);
    // Largest input should have largest probability
    try std.testing.expect(data[3] > data[2]);
    try std.testing.expect(data[2] > data[1]);
}

test "logSoftmaxInPlace values are negative" {
    var data = [_]f32{ 1.0, 2.0, 3.0 };
    logSoftmaxInPlace(&data);
    // All log-softmax values should be <= 0
    for (data) |v| try std.testing.expect(v <= 0.001);
    // exp(log-softmax) should sum to 1
    var total: f32 = 0;
    for (data) |v| total += @exp(v);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), total, 0.01);
}

test "maxValue finds correct max" {
    const data = [_]f32{ 1.0, 5.0, 3.0, 2.0 };
    try std.testing.expectEqual(@as(f32, 5.0), maxValue(&data));
}

test "sum computes correct total" {
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), sum(&data), 0.001);
}

test "divideByScalar normalizes" {
    var data = [_]f32{ 10.0, 20.0, 30.0 };
    divideByScalar(&data, 10.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), data[2], 0.001);
}

test "squaredSum" {
    const data = [_]f32{ 3.0, 4.0 };
    // 9 + 16 = 25
    try std.testing.expectApproxEqAbs(@as(f32, 25.0), squaredSum(&data), 0.001);
}

test "rmsNormInPlace without weights" {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    rmsNormInPlace(&data, null, 1e-6);
    // After RMSNorm, squared sum / N should ≈ 1 (within rounding)
    var sq: f32 = 0;
    for (data) |v| sq += v * v;
    const rms = @sqrt(sq / 4.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), rms, 0.1);
}

test "layerNormInPlace centers data" {
    var data = [_]f32{ 2.0, 4.0, 6.0, 8.0 };
    layerNormInPlace(&data, null, null, 1e-6);
    // After layernorm without gamma/beta, mean should be ~0
    var mean: f32 = 0;
    for (data) |v| mean += v;
    mean /= 4.0;
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mean, 0.01);
}

test "softmaxInPlace empty is safe" {
    var empty: [0]f32 = .{};
    softmaxInPlace(&empty); // should not crash
}

/// Fast vectorized tanh approximation
/// Uses the identity: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
fn tanhVec(x: @Vector(VectorSize, f32)) @Vector(VectorSize, f32) {
    const Vec = @Vector(VectorSize, f32);
    const one: Vec = @splat(1.0);
    const two: Vec = @splat(2.0);

    const exp2x = @exp(two * x);
    return (exp2x - one) / (exp2x + one);
}

test {
    std.testing.refAllDecls(@This());
}
