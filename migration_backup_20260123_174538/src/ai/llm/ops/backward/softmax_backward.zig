//! Backward pass for softmax activation.
//!
//! Forward: softmax(x)[i] = exp(x[i]) / sum(exp(x))
//!
//! Backward:
//!   d_input[i] = softmax[i] * (d_output[i] - sum_j(softmax[j] * d_output[j]))
//!
//! The Jacobian of softmax is: J[i,j] = softmax[i] * (delta[i,j] - softmax[j])
//! So: d_input = softmax * (d_output - sum(softmax * d_output))

const std = @import("std");

/// Backward pass for softmax.
/// Given softmax output (from forward) and upstream gradient, compute input gradient.
///
/// Args:
///   d_output: [dim] - gradient from upstream
///   softmax_output: [dim] - cached softmax output from forward pass
///   d_input: [dim] - gradient output for input (accumulated)
pub fn softmaxBackward(
    d_output: []const f32,
    softmax_output: []const f32,
    d_input: []f32,
) void {
    const dim = softmax_output.len;

    // Compute dot product: sum(softmax * d_output)
    var dot_sum: f32 = 0;
    for (0..dim) |i| {
        dot_sum += softmax_output[i] * d_output[i];
    }

    // d_input[i] = softmax[i] * (d_output[i] - dot_sum)
    for (0..dim) |i| {
        d_input[i] += softmax_output[i] * (d_output[i] - dot_sum);
    }
}

/// Backward pass for softmax (non-accumulating version).
/// Writes directly to d_input instead of accumulating.
pub fn softmaxBackwardDirect(
    d_output: []const f32,
    softmax_output: []const f32,
    d_input: []f32,
) void {
    const dim = softmax_output.len;

    // Compute dot product: sum(softmax * d_output)
    var dot_sum: f32 = 0;
    for (0..dim) |i| {
        dot_sum += softmax_output[i] * d_output[i];
    }

    // d_input[i] = softmax[i] * (d_output[i] - dot_sum)
    for (0..dim) |i| {
        d_input[i] = softmax_output[i] * (d_output[i] - dot_sum);
    }
}

/// Backward pass for batched softmax (row-wise softmax).
/// Each row is treated as an independent softmax.
pub fn batchSoftmaxBackward(
    d_output: []const f32, // [batch, dim]
    softmax_output: []const f32, // [batch, dim]
    d_input: []f32, // [batch, dim]
    batch_size: u32,
    dim: u32,
) void {
    for (0..batch_size) |b| {
        const offset = b * dim;
        softmaxBackward(
            d_output[offset .. offset + dim],
            softmax_output[offset .. offset + dim],
            d_input[offset .. offset + dim],
        );
    }
}

/// Backward pass for softmax with temperature.
/// Forward: softmax(x / T)
/// Backward needs to account for temperature scaling.
pub fn softmaxWithTemperatureBackward(
    d_output: []const f32,
    softmax_output: []const f32,
    d_input: []f32,
    temperature: f32,
) void {
    const dim = softmax_output.len;
    const inv_temp = 1.0 / temperature;

    // Compute dot product: sum(softmax * d_output)
    var dot_sum: f32 = 0;
    for (0..dim) |i| {
        dot_sum += softmax_output[i] * d_output[i];
    }

    // d_input[i] = (softmax[i] * (d_output[i] - dot_sum)) / temperature
    for (0..dim) |i| {
        d_input[i] += softmax_output[i] * (d_output[i] - dot_sum) * inv_temp;
    }
}

/// Backward pass for log-softmax.
/// Forward: log_softmax(x)[i] = x[i] - log(sum(exp(x)))
/// Backward: d_input[i] = d_output[i] - softmax[i] * sum(d_output)
pub fn logSoftmaxBackward(
    d_output: []const f32,
    softmax_output: []const f32, // exp of log_softmax = softmax
    d_input: []f32,
) void {
    const dim = softmax_output.len;

    // Compute sum of d_output
    var d_sum: f32 = 0;
    for (d_output) |d| {
        d_sum += d;
    }

    // d_input[i] = d_output[i] - softmax[i] * sum(d_output)
    for (0..dim) |i| {
        d_input[i] += d_output[i] - softmax_output[i] * d_sum;
    }
}

/// Backward pass for cross-entropy loss combined with softmax.
/// This is more numerically stable than computing them separately.
/// Forward: loss = -sum(target * log(softmax(x)))
/// Backward: d_input[i] = softmax[i] - target[i]
pub fn crossEntropySoftmaxBackward(
    softmax_output: []const f32,
    target: []const f32, // One-hot or soft labels
    d_input: []f32,
) void {
    for (softmax_output, target, 0..) |s, t, i| {
        d_input[i] += s - t;
    }
}

/// Backward pass for cross-entropy with label index (sparse targets).
/// target_idx is the index of the correct class.
pub fn crossEntropySoftmaxBackwardSparse(
    softmax_output: []const f32,
    target_idx: u32,
    d_input: []f32,
) void {
    for (softmax_output, 0..) |s, i| {
        if (i == target_idx) {
            d_input[i] += s - 1.0;
        } else {
            d_input[i] += s;
        }
    }
}

test "softmax backward basic" {
    // Forward: softmax([1, 2, 3]) ≈ [0.09, 0.24, 0.67]
    const softmax_out = [_]f32{ 0.09003057, 0.24472847, 0.66524096 };
    const d_output = [_]f32{ 1.0, 0.0, 0.0 };
    var d_input = [_]f32{ 0, 0, 0 };

    softmaxBackward(&d_output, &softmax_out, &d_input);

    // d_input[0] = softmax[0] * (1 - sum) = 0.09 * (1 - 0.09) ≈ 0.082
    // d_input[1] = softmax[1] * (0 - sum) = 0.24 * (0 - 0.09) ≈ -0.022
    // d_input[2] = softmax[2] * (0 - sum) = 0.67 * (0 - 0.09) ≈ -0.060
    try std.testing.expectApproxEqAbs(@as(f32, 0.082), d_input[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, -0.022), d_input[1], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, -0.060), d_input[2], 0.01);
}

test "softmax backward sum check" {
    // Gradient of softmax should sum to 0 (due to normalization)
    const softmax_out = [_]f32{ 0.2, 0.3, 0.5 };
    const d_output = [_]f32{ 1.0, 2.0, 3.0 };
    var d_input = [_]f32{ 0, 0, 0 };

    softmaxBackward(&d_output, &softmax_out, &d_input);

    var sum: f32 = 0;
    for (d_input) |v| {
        sum += v;
    }
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sum, 0.001);
}

test "cross entropy softmax backward" {
    const softmax_out = [_]f32{ 0.2, 0.3, 0.5 };
    const target = [_]f32{ 0.0, 0.0, 1.0 }; // One-hot
    var d_input = [_]f32{ 0, 0, 0 };

    crossEntropySoftmaxBackward(&softmax_out, &target, &d_input);

    // d_input = softmax - target
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), d_input[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.3), d_input[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), d_input[2], 0.001);
}

test "cross entropy softmax backward sparse" {
    const softmax_out = [_]f32{ 0.2, 0.3, 0.5 };
    var d_input = [_]f32{ 0, 0, 0 };

    crossEntropySoftmaxBackwardSparse(&softmax_out, 2, &d_input);

    // Same result as dense version with one-hot target
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), d_input[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.3), d_input[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), d_input[2], 0.001);
}
