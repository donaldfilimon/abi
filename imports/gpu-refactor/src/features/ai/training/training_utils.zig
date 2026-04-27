//! Shared Training Utilities
//!
//! Common math primitives used across LLM, Vision, and Multimodal trainers:
//! - Weight initialization (Xavier, sinusoidal positional)
//! - Layer normalization
//! - Gradient operations (scaling, update application, L2 normalization)
//!
//! Extracted to eliminate duplication between llm_trainer.zig,
//! vision_trainer.zig, and multimodal_trainer.zig.

const std = @import("std");

/// Xavier (Glorot) uniform initialization.
/// Scales weights by sqrt(2 / fan_in) using a deterministic PRNG.
pub fn initializeXavier(data: []f32) void {
    initializeXavierWithSeed(data, 0x12345678);
}

/// Xavier initialization with a custom seed.
/// Useful when different modules need distinct initializations.
pub fn initializeXavierWithSeed(data: []f32, seed: u64) void {
    const scale = @sqrt(2.0 / @as(f32, @floatFromInt(data.len)));
    var rng = std.Random.DefaultPrng.init(seed);
    for (data) |*val| {
        val.* = (rng.random().float(f32) * 2.0 - 1.0) * scale;
    }
}

/// Sinusoidal position embedding initialization.
/// Produces interleaved sin/cos embeddings following the Transformer paper.
pub fn initializePositional(data: []f32, seq_len: u32, hidden: u32) void {
    for (0..seq_len) |pos| {
        for (0..hidden) |i| {
            const position = @as(f32, @floatFromInt(pos));
            const div_term = @exp(-@as(f32, @floatFromInt(i)) * @log(@as(f32, 10000.0)) / @as(f32, @floatFromInt(hidden)));

            if (i % 2 == 0) {
                data[pos * hidden + i] = @sin(position * div_term);
            } else {
                data[pos * hidden + i] = @cos(position * div_term);
            }
        }
    }
}

/// In-place layer normalization: x = (x - mean) / sqrt(var + eps) * weight + bias.
pub fn layerNorm(data: []f32, weight: []const f32, bias: []const f32) void {
    const dim = data.len;
    const dim_f = @as(f32, @floatFromInt(dim));

    // Compute mean
    var mean: f32 = 0;
    for (data) |v| mean += v;
    mean /= dim_f;

    // Compute variance
    var variance: f32 = 0;
    for (data) |v| {
        const diff = v - mean;
        variance += diff * diff;
    }
    variance /= dim_f;

    // Normalize
    const inv_std = 1.0 / @sqrt(variance + 1e-6);
    for (0..dim) |i| {
        data[i] = (data[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

/// Scale every element of a slice by a constant factor.
pub fn scaleSlice(data: []f32, scale: f32) void {
    for (data) |*v| {
        v.* *= scale;
    }
}

/// Apply gradient update: weight -= learning_rate * gradient.
pub fn applyUpdate(weights: []f32, gradients: []const f32, learning_rate: f32) void {
    for (weights, gradients) |*w, g| {
        w.* -= learning_rate * g;
    }
}

/// L2-normalize a vector in place. No-op if the vector is zero.
pub fn l2Normalize(data: []f32) void {
    var norm_sq: f32 = 0;
    for (data) |v| norm_sq += v * v;

    if (norm_sq > 0) {
        const inv_norm = 1.0 / @sqrt(norm_sq);
        for (data) |*v| {
            v.* *= inv_norm;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

test "xavier initialization produces non-zero values" {
    var data = [_]f32{0} ** 16;
    initializeXavier(&data);

    var any_nonzero = false;
    for (data) |v| {
        if (v != 0) {
            any_nonzero = true;
            break;
        }
    }
    try std.testing.expect(any_nonzero);
}

test "xavier with different seeds produces different values" {
    var data_a = [_]f32{0} ** 8;
    var data_b = [_]f32{0} ** 8;
    initializeXavierWithSeed(&data_a, 0x12345678);
    initializeXavierWithSeed(&data_b, 0x87654321);

    var any_different = false;
    for (data_a, data_b) |a, b| {
        if (a != b) {
            any_different = true;
            break;
        }
    }
    try std.testing.expect(any_different);
}

test "positional initialization produces alternating sin/cos" {
    var data = [_]f32{0} ** 8; // 2 positions, 4 hidden
    initializePositional(&data, 2, 4);

    // Position 0 should be all zeros for sin, non-trivial for cos at i=1
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data[0], 0.001); // sin(0) = 0
}

test "layer norm produces zero mean" {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const weight = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const bias = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    layerNorm(&data, &weight, &bias);

    var mean: f32 = 0;
    for (data) |v| mean += v;
    mean /= 4.0;

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mean, 0.001);
}

test "scale slice" {
    var data = [_]f32{ 1.0, 2.0, 3.0 };
    scaleSlice(&data, 0.5);

    try std.testing.expectApproxEqAbs(@as(f32, 0.5), data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), data[2], 0.001);
}

test "apply update" {
    var weights = [_]f32{ 1.0, 2.0, 3.0 };
    const grads = [_]f32{ 0.1, 0.2, 0.3 };
    applyUpdate(&weights, &grads, 1.0);

    try std.testing.expectApproxEqAbs(@as(f32, 0.9), weights[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.8), weights[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.7), weights[2], 0.001);
}

test "l2 normalize" {
    var data = [_]f32{ 3.0, 4.0 };
    l2Normalize(&data);

    var norm_sq: f32 = 0;
    for (data) |v| norm_sq += v * v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), norm_sq, 0.001);
}

test "l2 normalize zero vector is no-op" {
    var data = [_]f32{ 0.0, 0.0 };
    l2Normalize(&data);

    try std.testing.expectEqual(@as(f32, 0.0), data[0]);
    try std.testing.expectEqual(@as(f32, 0.0), data[1]);
}

test {
    std.testing.refAllDecls(@This());
}
