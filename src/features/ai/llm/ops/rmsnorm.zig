//! RMS Layer Normalization implementation.
//!
//! RMSNorm is a simplified normalization technique used in LLaMA and other
//! modern LLMs. Unlike LayerNorm, it doesn't center the activations.

const std = @import("std");

/// Compute RMS normalization: x = x / sqrt(mean(x^2) + eps) * weight
/// x: [dim], weight: [dim], output: [dim]
pub fn rmsNorm(x: []const f32, weight: []const f32, output: []f32, eps: f32) void {
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

/// In-place RMS normalization.
pub fn rmsNormInPlace(x: []f32, weight: []const f32, eps: f32) void {
    rmsNorm(x, weight, x, eps);
}

/// Compute RMS normalization with SIMD acceleration.
pub fn rmsNormSimd(x: []const f32, weight: []const f32, output: []f32, eps: f32) void {
    const dim = x.len;

    // Compute sum of squares with unrolling
    var sum_sq: f32 = 0;
    var i: usize = 0;

    // Process 8 elements at a time
    while (i + 8 <= dim) : (i += 8) {
        sum_sq += x[i + 0] * x[i + 0];
        sum_sq += x[i + 1] * x[i + 1];
        sum_sq += x[i + 2] * x[i + 2];
        sum_sq += x[i + 3] * x[i + 3];
        sum_sq += x[i + 4] * x[i + 4];
        sum_sq += x[i + 5] * x[i + 5];
        sum_sq += x[i + 6] * x[i + 6];
        sum_sq += x[i + 7] * x[i + 7];
    }

    // Handle remainder
    while (i < dim) : (i += 1) {
        sum_sq += x[i] * x[i];
    }

    const mean_sq = sum_sq / @as(f32, @floatFromInt(dim));
    const inv_rms = 1.0 / @sqrt(mean_sq + eps);

    // Apply normalization with unrolling
    i = 0;
    while (i + 8 <= dim) : (i += 8) {
        output[i + 0] = x[i + 0] * inv_rms * weight[i + 0];
        output[i + 1] = x[i + 1] * inv_rms * weight[i + 1];
        output[i + 2] = x[i + 2] * inv_rms * weight[i + 2];
        output[i + 3] = x[i + 3] * inv_rms * weight[i + 3];
        output[i + 4] = x[i + 4] * inv_rms * weight[i + 4];
        output[i + 5] = x[i + 5] * inv_rms * weight[i + 5];
        output[i + 6] = x[i + 6] * inv_rms * weight[i + 6];
        output[i + 7] = x[i + 7] * inv_rms * weight[i + 7];
    }

    while (i < dim) : (i += 1) {
        output[i] = x[i] * inv_rms * weight[i];
    }
}

/// Compute just the RMS value (for debugging).
pub fn computeRms(x: []const f32) f32 {
    var sum_sq: f32 = 0;
    for (x) |v| {
        sum_sq += v * v;
    }
    return @sqrt(sum_sq / @as(f32, @floatFromInt(x.len)));
}

/// Standard Layer Normalization (for comparison/compatibility).
/// x = (x - mean) / sqrt(var + eps) * weight + bias
pub fn layerNorm(
    x: []const f32,
    weight: []const f32,
    bias: ?[]const f32,
    output: []f32,
    eps: f32,
) void {
    const dim = x.len;

    // Compute mean
    var sum: f32 = 0;
    for (x) |v| {
        sum += v;
    }
    const mean = sum / @as(f32, @floatFromInt(dim));

    // Compute variance
    var var_sum: f32 = 0;
    for (x) |v| {
        const diff = v - mean;
        var_sum += diff * diff;
    }
    const variance = var_sum / @as(f32, @floatFromInt(dim));

    // Normalize
    const inv_std = 1.0 / @sqrt(variance + eps);

    for (0..dim) |i| {
        var y = (x[i] - mean) * inv_std * weight[i];
        if (bias) |b| {
            y += b[i];
        }
        output[i] = y;
    }
}

/// Batch RMS normalization for multiple vectors.
pub fn batchRmsNorm(
    x: []const f32, // [batch, dim]
    weight: []const f32, // [dim]
    output: []f32, // [batch, dim]
    batch_size: u32,
    dim: u32,
    eps: f32,
) void {
    for (0..batch_size) |b| {
        const offset = b * dim;
        rmsNorm(
            x[offset .. offset + dim],
            weight,
            output[offset .. offset + dim],
            eps,
        );
    }
}

test "rms norm basic" {
    const x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const weight = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    var output: [4]f32 = undefined;

    rmsNorm(&x, &weight, &output, 1e-6);

    // RMS of [1,2,3,4] = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.739
    // Output should be x / rms
    const rms = computeRms(&x);
    try std.testing.expectApproxEqAbs(@as(f32, 2.739), rms, 0.01);

    // First element should be 1.0 / 2.739 ≈ 0.365
    try std.testing.expectApproxEqAbs(@as(f32, 0.365), output[0], 0.01);
}

test "rms norm with weight" {
    const x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const weight = [_]f32{ 2.0, 2.0, 2.0, 2.0 };
    var output: [4]f32 = undefined;

    rmsNorm(&x, &weight, &output, 1e-6);

    // With weight=2, output should be 2x the unit-weight case
    const rms = computeRms(&x);
    try std.testing.expectApproxEqAbs(x[0] / rms * 2.0, output[0], 0.01);
}

test "rms norm simd matches basic" {
    const x = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    const weight = [_]f32{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    var output_basic: [16]f32 = undefined;
    var output_simd: [16]f32 = undefined;

    rmsNorm(&x, &weight, &output_basic, 1e-6);
    rmsNormSimd(&x, &weight, &output_simd, 1e-6);

    for (0..16) |i| {
        try std.testing.expectApproxEqAbs(output_basic[i], output_simd[i], 1e-5);
    }
}
