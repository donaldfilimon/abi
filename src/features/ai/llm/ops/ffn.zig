//! Feed-Forward Network (FFN) implementations.
//!
//! Implements the FFN blocks used in transformer models:
//! - Standard FFN: Linear -> Activation -> Linear
//! - SwiGLU FFN: Gate * Up -> SiLU -> Down (used in LLaMA)

const std = @import("std");
const matmul = @import("matmul.zig");
const activations = @import("activations.zig");

/// SwiGLU FFN configuration.
pub const SwigluConfig = struct {
    hidden_dim: u32,
    intermediate_dim: u32,
};

/// SwiGLU FFN: output = down(silu(gate(x)) * up(x))
/// This is the FFN architecture used in LLaMA and similar models.
///
/// gate: [hidden_dim, intermediate_dim]
/// up: [hidden_dim, intermediate_dim]
/// down: [intermediate_dim, hidden_dim]
pub fn swiglu(
    allocator: std.mem.Allocator,
    x: []const f32, // [hidden_dim]
    gate_weight: []const f32, // [intermediate_dim, hidden_dim] (transposed for matmul)
    up_weight: []const f32, // [intermediate_dim, hidden_dim]
    down_weight: []const f32, // [hidden_dim, intermediate_dim]
    output: []f32, // [hidden_dim]
    hidden_dim: u32,
    intermediate_dim: u32,
) !void {
    // Compute gate(x) and up(x)
    const gate_out = try allocator.alloc(f32, intermediate_dim);
    defer allocator.free(gate_out);
    const up_out = try allocator.alloc(f32, intermediate_dim);
    defer allocator.free(up_out);

    // gate_out = x @ gate_weight^T
    matmul.matrixVectorMultiply(gate_weight, x, gate_out, intermediate_dim, hidden_dim);

    // up_out = x @ up_weight^T
    matmul.matrixVectorMultiply(up_weight, x, up_out, intermediate_dim, hidden_dim);

    // Apply SiLU to gate and multiply with up
    for (0..intermediate_dim) |i| {
        gate_out[i] = activations.silu(gate_out[i]) * up_out[i];
    }

    // output = intermediate @ down_weight^T
    matmul.matrixVectorMultiply(down_weight, gate_out, output, hidden_dim, intermediate_dim);
}

/// Standard FFN: output = down(activation(up(x)))
pub fn feedForward(
    allocator: std.mem.Allocator,
    x: []const f32, // [hidden_dim]
    up_weight: []const f32, // [intermediate_dim, hidden_dim]
    down_weight: []const f32, // [hidden_dim, intermediate_dim]
    output: []f32, // [hidden_dim]
    hidden_dim: u32,
    intermediate_dim: u32,
    activation: ActivationType,
) !void {
    // Compute up(x)
    const hidden = try allocator.alloc(f32, intermediate_dim);
    defer allocator.free(hidden);

    matmul.matrixVectorMultiply(up_weight, x, hidden, intermediate_dim, hidden_dim);

    // Apply activation
    switch (activation) {
        .relu => activations.reluInPlace(hidden),
        .gelu => activations.geluInPlace(hidden),
        .silu => activations.siluInPlace(hidden),
    }

    // output = activation(up(x)) @ down^T
    matmul.matrixVectorMultiply(down_weight, hidden, output, hidden_dim, intermediate_dim);
}

pub const ActivationType = enum {
    relu,
    gelu,
    silu,
};

/// Batched SwiGLU for processing multiple positions.
pub fn swigluBatch(
    allocator: std.mem.Allocator,
    x: []const f32, // [batch, hidden_dim]
    gate_weight: []const f32,
    up_weight: []const f32,
    down_weight: []const f32,
    output: []f32, // [batch, hidden_dim]
    batch_size: u32,
    hidden_dim: u32,
    intermediate_dim: u32,
) !void {
    for (0..batch_size) |b| {
        const x_offset = b * hidden_dim;
        const out_offset = b * hidden_dim;

        try swiglu(
            allocator,
            x[x_offset .. x_offset + hidden_dim],
            gate_weight,
            up_weight,
            down_weight,
            output[out_offset .. out_offset + hidden_dim],
            hidden_dim,
            intermediate_dim,
        );
    }
}

/// Fused gate and up projection for efficiency.
/// Computes both projections in one pass when possible.
pub fn fusedGateUp(
    x: []const f32, // [hidden_dim]
    gate_up_weight: []const f32, // [2 * intermediate_dim, hidden_dim]
    gate_out: []f32, // [intermediate_dim]
    up_out: []f32, // [intermediate_dim]
    hidden_dim: u32,
    intermediate_dim: u32,
) void {
    // Some models pack gate and up weights together
    // gate_up_weight layout: [gate_weights..., up_weights...]

    const gate_weight = gate_up_weight[0 .. intermediate_dim * hidden_dim];
    const up_weight = gate_up_weight[intermediate_dim * hidden_dim ..];

    matmul.matrixVectorMultiply(gate_weight, x, gate_out, intermediate_dim, hidden_dim);
    matmul.matrixVectorMultiply(up_weight, x, up_out, intermediate_dim, hidden_dim);
}

/// Expert-mixture FFN for MoE (Mixture of Experts) models.
pub fn expertFFN(
    allocator: std.mem.Allocator,
    x: []const f32,
    expert_weights: []const []const f32, // Array of expert weight sets
    router_logits: []const f32, // [num_experts]
    output: []f32,
    hidden_dim: u32,
    intermediate_dim: u32,
    num_experts: u32,
    top_k: u32,
) !void {
    _ = allocator;
    _ = x;
    _ = expert_weights;
    _ = intermediate_dim;
    _ = num_experts;

    // Find top-k experts
    var expert_scores: [8]struct { idx: u32, score: f32 } = undefined;
    for (0..@min(router_logits.len, 8)) |i| {
        expert_scores[i] = .{ .idx = @intCast(i), .score = router_logits[i] };
    }

    // Simple selection of top-k (should use proper top-k algorithm for larger k)
    const ExpertScore = struct { idx: u32, score: f32 };
    std.mem.sort(
        ExpertScore,
        expert_scores[0..@min(router_logits.len, 8)],
        {},
        struct {
            fn lessThan(_: void, a: ExpertScore, b: ExpertScore) bool {
                return a.score > b.score;
            }
        }.lessThan,
    );

    // Initialize output to zeros
    @memset(output, 0);

    // Compute softmax over selected experts
    var total_weight: f32 = 0;
    for (0..top_k) |i| {
        total_weight += @exp(expert_scores[i].score);
    }

    // Weighted sum of expert outputs
    for (0..top_k) |i| {
        const weight = @exp(expert_scores[i].score) / total_weight;

        // Add weighted expert output
        // In real implementation, would compute expert FFN here
        for (0..hidden_dim) |j| {
            output[j] += weight * 0; // Placeholder
        }
    }
}

test "swiglu basic" {
    const allocator = std.testing.allocator;

    const hidden_dim: u32 = 4;
    const intermediate_dim: u32 = 8;

    // Create identity-ish weights for testing
    const gate = try allocator.alloc(f32, intermediate_dim * hidden_dim);
    defer allocator.free(gate);
    const up = try allocator.alloc(f32, intermediate_dim * hidden_dim);
    defer allocator.free(up);
    const down = try allocator.alloc(f32, hidden_dim * intermediate_dim);
    defer allocator.free(down);

    @memset(gate, 0.1);
    @memset(up, 0.1);
    @memset(down, 0.1);

    const x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var output: [4]f32 = undefined;

    try swiglu(allocator, &x, gate, up, down, &output, hidden_dim, intermediate_dim);

    // Output should be non-zero
    var has_nonzero = false;
    for (output) |v| {
        if (v != 0) has_nonzero = true;
    }
    try std.testing.expect(has_nonzero);
}

test "feed forward basic" {
    const allocator = std.testing.allocator;

    const hidden_dim: u32 = 4;
    const intermediate_dim: u32 = 8;

    const up = try allocator.alloc(f32, intermediate_dim * hidden_dim);
    defer allocator.free(up);
    const down = try allocator.alloc(f32, hidden_dim * intermediate_dim);
    defer allocator.free(down);

    @memset(up, 0.1);
    @memset(down, 0.1);

    const x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var output: [4]f32 = undefined;

    try feedForward(allocator, &x, up, down, &output, hidden_dim, intermediate_dim, .relu);

    // Output should be non-negative with ReLU
    for (output) |v| {
        try std.testing.expect(v >= 0);
    }
}
