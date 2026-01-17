//! Backward pass operations for LLM training.
//!
//! Provides gradient computation for all forward operations:
//! - Matrix multiplication backward
//! - RMSNorm backward
//! - Softmax backward
//! - RoPE backward
//! - Attention backward
//! - FFN/SwiGLU backward

const std = @import("std");

pub const matmul_backward = @import("matmul_backward.zig");
pub const rmsnorm_backward = @import("rmsnorm_backward.zig");
pub const softmax_backward = @import("softmax_backward.zig");
pub const rope_backward = @import("rope_backward.zig");
pub const attention_backward = @import("attention_backward.zig");
pub const ffn_backward = @import("ffn_backward.zig");

// Re-exports for convenience
pub const matmulBackward = matmul_backward.matmulBackward;
pub const matrixVectorBackward = matmul_backward.matrixVectorBackward;
pub const rmsNormBackward = rmsnorm_backward.rmsNormBackward;
pub const softmaxBackward = softmax_backward.softmaxBackward;
pub const ropeBackward = rope_backward.ropeBackward;
pub const attentionBackward = attention_backward.attentionBackward;
pub const swigluBackward = ffn_backward.swigluBackward;

/// Gradient accumulation helper.
pub fn accumulateGradient(grad: []f32, delta: []const f32) void {
    for (grad, delta) |*g, d| {
        g.* += d;
    }
}

/// Zero gradients.
pub fn zeroGradients(grad: []f32) void {
    @memset(grad, 0);
}

/// Gradient clipping by global norm.
pub fn clipGradientsByNorm(grads: [][]f32, max_norm: f32) f32 {
    // Compute global norm
    var total_norm_sq: f32 = 0;
    for (grads) |grad| {
        for (grad) |g| {
            total_norm_sq += g * g;
        }
    }
    const total_norm = @sqrt(total_norm_sq);

    // Clip if exceeds max_norm
    if (total_norm > max_norm) {
        const scale = max_norm / (total_norm + 1e-6);
        for (grads) |grad| {
            for (grad) |*g| {
                g.* *= scale;
            }
        }
    }

    return total_norm;
}

/// Gradient clipping by value.
pub fn clipGradientsByValue(grad: []f32, min_val: f32, max_val: f32) void {
    for (grad) |*g| {
        g.* = @max(min_val, @min(max_val, g.*));
    }
}

test "backward module imports" {
    _ = matmul_backward;
    _ = rmsnorm_backward;
    _ = softmax_backward;
    _ = rope_backward;
    _ = attention_backward;
    _ = ffn_backward;
}

test "gradient accumulation" {
    var grad = [_]f32{ 1.0, 2.0, 3.0 };
    const delta = [_]f32{ 0.5, 0.5, 0.5 };

    accumulateGradient(&grad, &delta);

    try std.testing.expectApproxEqAbs(@as(f32, 1.5), grad[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), grad[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.5), grad[2], 0.001);
}

test "gradient clipping by value" {
    var grad = [_]f32{ -5.0, 0.5, 10.0 };

    clipGradientsByValue(&grad, -1.0, 1.0);

    try std.testing.expectApproxEqAbs(@as(f32, -1.0), grad[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), grad[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), grad[2], 0.001);
}
