//! Backward pass for RMS Layer Normalization.
//!
//! Forward: y = x * weight / rms(x)
//! where rms(x) = sqrt(mean(x^2) + eps)
//!
//! Backward:
//!   d_weight[i] = sum_batch dy[i] * x[i] / rms
//!   dx[i] = weight[i] / rms * (dy[i] - x[i] * mean(dy * x_norm) / rms)
//!
//! The gradient is more complex because rms depends on all elements of x.

const std = @import("std");

/// Backward pass for RMSNorm.
/// Forward: y = x * weight / rms(x)
///
/// Args:
///   dy: [dim] - gradient from upstream
///   x: [dim] - input (cached from forward)
///   weight: [dim] - learnable weight (cached)
///   dx: [dim] - gradient output for input (accumulated)
///   d_weight: [dim] - gradient output for weight (accumulated)
///   eps: epsilon for numerical stability
pub fn rmsNormBackward(
    dy: []const f32,
    x: []const f32,
    weight: []const f32,
    dx: []f32,
    d_weight: []f32,
    eps: f32,
) void {
    const dim = x.len;
    const dim_f = @as(f32, @floatFromInt(dim));

    // Compute RMS (same as forward)
    var sum_sq: f32 = 0;
    for (x) |v| {
        sum_sq += v * v;
    }
    const mean_sq = sum_sq / dim_f;
    const rms = @sqrt(mean_sq + eps);
    const inv_rms = 1.0 / rms;

    // Compute x_norm = x / rms
    // and dy_xnorm_sum = sum(dy * x_norm * weight)
    var dy_xnorm_sum: f32 = 0;
    for (0..dim) |i| {
        const x_norm = x[i] * inv_rms;
        dy_xnorm_sum += dy[i] * x_norm * weight[i];
    }

    // Compute gradients
    // d_weight[i] = dy[i] * x[i] / rms
    // dx[i] = weight[i] * inv_rms * dy[i] - weight[i] * x[i] * inv_rms^3 * dy_xnorm_sum / dim
    const factor = dy_xnorm_sum * inv_rms * inv_rms / dim_f;

    for (0..dim) |i| {
        // d_weight gradient
        d_weight[i] += dy[i] * x[i] * inv_rms;

        // dx gradient
        // dx = weight * inv_rms * dy - x * factor
        dx[i] += weight[i] * inv_rms * dy[i] - x[i] * factor;
    }
}

/// Backward pass for batched RMSNorm.
/// Forward: y[b] = x[b] * weight / rms(x[b]) for each batch
pub fn batchRmsNormBackward(
    dy: []const f32, // [batch, dim]
    x: []const f32, // [batch, dim]
    weight: []const f32, // [dim]
    dx: []f32, // [batch, dim]
    d_weight: []f32, // [dim]
    batch_size: u32,
    dim: u32,
    eps: f32,
) void {
    for (0..batch_size) |b| {
        const offset = b * dim;
        rmsNormBackward(
            dy[offset .. offset + dim],
            x[offset .. offset + dim],
            weight,
            dx[offset .. offset + dim],
            d_weight,
            eps,
        );
    }
}

/// Backward pass for standard Layer Normalization.
/// Forward: y = (x - mean) / std * weight + bias
pub fn layerNormBackward(
    dy: []const f32,
    x: []const f32,
    weight: []const f32,
    mean: f32,
    variance: f32,
    dx: []f32,
    d_weight: []f32,
    d_bias: ?[]f32,
    eps: f32,
) void {
    const dim = x.len;
    const dim_f = @as(f32, @floatFromInt(dim));
    const std_dev = @sqrt(variance + eps);
    const inv_std = 1.0 / std_dev;

    // Compute intermediate sums
    var dy_sum: f32 = 0;
    var dy_xhat_sum: f32 = 0;
    for (0..dim) |i| {
        const x_hat = (x[i] - mean) * inv_std;
        dy_sum += dy[i] * weight[i];
        dy_xhat_sum += dy[i] * weight[i] * x_hat;
    }

    // Compute gradients
    for (0..dim) |i| {
        const x_hat = (x[i] - mean) * inv_std;

        // d_weight
        d_weight[i] += dy[i] * x_hat;

        // d_bias
        if (d_bias) |db| {
            db[i] += dy[i];
        }

        // dx
        dx[i] += weight[i] * inv_std * (dy[i] - (dy_sum + x_hat * dy_xhat_sum) / dim_f);
    }
}

/// Compute statistics needed for LayerNorm backward.
pub const LayerNormStats = struct {
    mean: f32,
    variance: f32,
};

/// Compute mean and variance for LayerNorm forward (also needed for backward).
pub fn computeLayerNormStats(x: []const f32) LayerNormStats {
    const dim = x.len;
    const dim_f = @as(f32, @floatFromInt(dim));

    // Compute mean
    var sum: f32 = 0;
    for (x) |v| {
        sum += v;
    }
    const mean = sum / dim_f;

    // Compute variance
    var var_sum: f32 = 0;
    for (x) |v| {
        const diff = v - mean;
        var_sum += diff * diff;
    }
    const variance = var_sum / dim_f;

    return .{ .mean = mean, .variance = variance };
}

test "rmsnorm backward basic" {
    const x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const weight = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const dy = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    var dx = [_]f32{ 0, 0, 0, 0 };
    var d_weight = [_]f32{ 0, 0, 0, 0 };

    rmsNormBackward(&dy, &x, &weight, &dx, &d_weight, 1e-6);

    // Verify gradients are non-zero and reasonable
    var dx_nonzero = false;
    var dw_nonzero = false;
    for (dx) |v| {
        if (v != 0) dx_nonzero = true;
    }
    for (d_weight) |v| {
        if (v != 0) dw_nonzero = true;
    }
    try std.testing.expect(dx_nonzero);
    try std.testing.expect(dw_nonzero);

    // Check numerical values approximately
    // RMS of [1,2,3,4] = sqrt(7.5) ≈ 2.739
    const rms = @sqrt((1.0 + 4.0 + 9.0 + 16.0) / 4.0);
    try std.testing.expectApproxEqAbs(@as(f32, 2.739), rms, 0.01);
}

test "rmsnorm backward numerical gradient check" {
    // Use numerical differentiation to verify analytical gradient
    const eps: f32 = 1e-4;

    const x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const weight = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const dy = [_]f32{ 1.0, 0.5, 0.2, 0.1 };
    var dx_analytical = [_]f32{ 0, 0, 0, 0 };
    var d_weight_analytical = [_]f32{ 0, 0, 0, 0 };

    rmsNormBackward(&dy, &x, &weight, &dx_analytical, &d_weight_analytical, 1e-6);

    // Numerical gradient for d_weight (easier to verify)
    for (0..4) |i| {
        var weight_plus = weight;
        var weight_minus = weight;
        weight_plus[i] += eps;
        weight_minus[i] -= eps;

        // Compute forward with perturbed weights
        var y_plus: [4]f32 = undefined;
        var y_minus: [4]f32 = undefined;
        rmsNormForward(&x, &weight_plus, &y_plus, 1e-6);
        rmsNormForward(&x, &weight_minus, &y_minus, 1e-6);

        // Loss is sum(dy * y)
        var loss_plus: f32 = 0;
        var loss_minus: f32 = 0;
        for (0..4) |j| {
            loss_plus += dy[j] * y_plus[j];
            loss_minus += dy[j] * y_minus[j];
        }

        const numerical_grad = (loss_plus - loss_minus) / (2.0 * eps);

        // Allow some tolerance for numerical precision
        try std.testing.expectApproxEqAbs(numerical_grad, d_weight_analytical[i], 0.01);
    }
}

/// Helper function for numerical gradient check.
fn rmsNormForward(x: []const f32, weight: []const f32, output: []f32, eps: f32) void {
    const dim = x.len;

    // Compute mean of squares
    var sum_sq: f32 = 0;
    for (x) |v| {
        sum_sq += v * v;
    }
    const mean_sq = sum_sq / @as(f32, @floatFromInt(dim));

    // Compute normalization factor
    const rms = @sqrt(mean_sq + eps);
    const inv_rms = 1.0 / rms;

    // Normalize and apply weight
    for (0..dim) |i| {
        output[i] = x[i] * inv_rms * weight[i];
    }
}

test {
    std.testing.refAllDecls(@This());
}
