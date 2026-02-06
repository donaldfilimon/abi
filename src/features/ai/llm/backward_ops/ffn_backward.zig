//! Backward pass for Feed-Forward Networks (FFN).
//!
//! SwiGLU FFN (used in LLaMA):
//!   Forward: output = down(silu(gate(x)) * up(x))
//!
//! Backward:
//!   d_down = outer(d_output, intermediate)
//!   d_intermediate = down^T @ d_output
//!   d_gate = d_intermediate * up_out * silu'(gate_out)
//!   d_up = d_intermediate * silu(gate_out)
//!   d_gate_weight = outer(d_gate, x)
//!   d_up_weight = outer(d_up, x)
//!   d_x = gate^T @ d_gate + up^T @ d_up

const std = @import("std");
const matmul_backward = @import("matmul_backward.zig");
const activations = @import("../ops/activations.zig");

/// Cached activations needed for SwiGLU backward pass.
pub const SwigluCache = struct {
    allocator: std.mem.Allocator,
    /// Input: [hidden_dim]
    x: []f32,
    /// Gate projection output (before SiLU): [intermediate_dim]
    gate_out: []f32,
    /// Up projection output: [intermediate_dim]
    up_out: []f32,
    /// Intermediate (after SiLU * up): [intermediate_dim]
    intermediate: []f32,
    /// Dimensions
    hidden_dim: u32,
    intermediate_dim: u32,

    pub fn init(
        allocator: std.mem.Allocator,
        hidden_dim: u32,
        intermediate_dim: u32,
    ) !SwigluCache {
        const x = try allocator.alloc(f32, hidden_dim);
        errdefer allocator.free(x);
        const gate_out = try allocator.alloc(f32, intermediate_dim);
        errdefer allocator.free(gate_out);
        const up_out = try allocator.alloc(f32, intermediate_dim);
        errdefer allocator.free(up_out);
        const intermediate = try allocator.alloc(f32, intermediate_dim);

        return .{
            .allocator = allocator,
            .x = x,
            .gate_out = gate_out,
            .up_out = up_out,
            .intermediate = intermediate,
            .hidden_dim = hidden_dim,
            .intermediate_dim = intermediate_dim,
        };
    }

    pub fn deinit(self: *SwigluCache) void {
        self.allocator.free(self.x);
        self.allocator.free(self.gate_out);
        self.allocator.free(self.up_out);
        self.allocator.free(self.intermediate);
        self.* = undefined;
    }
};

/// SiLU derivative: d/dx[x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
///                                        = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
///                                        = silu(x)/x + sigmoid(x) * (1 - sigmoid(x)) * x
/// Simplified: silu'(x) = sigmoid(x) + silu(x) * (1 - sigmoid(x))
fn siluDerivative(x: f32) f32 {
    const sig = activations.sigmoid(x);
    return sig * (1.0 + x * (1.0 - sig));
}

/// Backward pass for SwiGLU FFN.
///
/// Forward: output = down(silu(gate(x)) * up(x))
///
/// Args:
///   d_output: [hidden_dim] - gradient from upstream
///   cache: cached forward activations
///   gate_weight: [intermediate_dim, hidden_dim]
///   up_weight: [intermediate_dim, hidden_dim]
///   down_weight: [hidden_dim, intermediate_dim]
///   d_gate_weight: gradient for gate_weight (accumulated)
///   d_up_weight: gradient for up_weight (accumulated)
///   d_down_weight: gradient for down_weight (accumulated)
///   d_x: gradient for input (accumulated)
pub fn swigluBackward(
    d_output: []const f32,
    cache: *const SwigluCache,
    gate_weight: []const f32,
    up_weight: []const f32,
    down_weight: []const f32,
    d_gate_weight: []f32,
    d_up_weight: []f32,
    d_down_weight: []f32,
    d_x: []f32,
) void {
    const hidden_dim = cache.hidden_dim;
    const intermediate_dim = cache.intermediate_dim;

    // Step 1: Backprop through down projection
    // d_intermediate = down^T @ d_output
    // d_down_weight += outer(d_output, intermediate)

    // d_intermediate[i] = sum_j down_weight[j, i] * d_output[j]
    var d_intermediate: [4096]f32 = undefined; // Stack allocation for common sizes
    const d_inter_slice = d_intermediate[0..intermediate_dim];

    for (0..intermediate_dim) |i| {
        var sum: f32 = 0;
        for (0..hidden_dim) |j| {
            // down_weight layout: [hidden_dim, intermediate_dim]
            sum += down_weight[j * intermediate_dim + i] * d_output[j];
        }
        d_inter_slice[i] = sum;
    }

    // d_down_weight[j, i] += d_output[j] * intermediate[i]
    for (0..hidden_dim) |j| {
        for (0..intermediate_dim) |i| {
            d_down_weight[j * intermediate_dim + i] += d_output[j] * cache.intermediate[i];
        }
    }

    // Step 2: Backprop through element-wise multiply (silu(gate) * up)
    // intermediate = silu(gate_out) * up_out
    // d_silu_gate = d_intermediate * up_out
    // d_up_out = d_intermediate * silu(gate_out)

    var d_gate_pre: [4096]f32 = undefined;
    var d_up_out: [4096]f32 = undefined;
    const d_gate_slice = d_gate_pre[0..intermediate_dim];
    const d_up_slice = d_up_out[0..intermediate_dim];

    for (0..intermediate_dim) |i| {
        const silu_val = activations.silu(cache.gate_out[i]);
        d_up_slice[i] = d_inter_slice[i] * silu_val;
        // d_gate_pre is gradient of silu(gate_out)
        // Need to multiply by silu derivative
        d_gate_slice[i] = d_inter_slice[i] * cache.up_out[i] * siluDerivative(cache.gate_out[i]);
    }

    // Step 3: Backprop through gate and up projections
    // gate_out = gate_weight @ x (treating as matrix-vector)
    // up_out = up_weight @ x

    // d_gate_weight[i, j] += d_gate[i] * x[j]
    // d_up_weight[i, j] += d_up[i] * x[j]
    for (0..intermediate_dim) |i| {
        for (0..hidden_dim) |j| {
            d_gate_weight[i * hidden_dim + j] += d_gate_slice[i] * cache.x[j];
            d_up_weight[i * hidden_dim + j] += d_up_slice[i] * cache.x[j];
        }
    }

    // d_x += gate_weight^T @ d_gate + up_weight^T @ d_up
    for (0..hidden_dim) |j| {
        var sum: f32 = 0;
        for (0..intermediate_dim) |i| {
            sum += gate_weight[i * hidden_dim + j] * d_gate_slice[i];
            sum += up_weight[i * hidden_dim + j] * d_up_slice[i];
        }
        d_x[j] += sum;
    }
}

/// Backward pass for standard FFN: output = down(activation(up(x)))
pub fn feedForwardBackward(
    d_output: []const f32,
    x: []const f32, // Cached input
    up_out: []const f32, // Cached up projection output (before activation)
    up_weight: []const f32,
    down_weight: []const f32,
    d_up_weight: []f32,
    d_down_weight: []f32,
    d_x: []f32,
    hidden_dim: u32,
    intermediate_dim: u32,
    activation: ActivationType,
) void {
    // Step 1: Backprop through down projection
    var d_activated: [4096]f32 = undefined;
    const d_act_slice = d_activated[0..intermediate_dim];

    for (0..intermediate_dim) |i| {
        var sum: f32 = 0;
        for (0..hidden_dim) |j| {
            sum += down_weight[j * intermediate_dim + i] * d_output[j];
        }
        d_act_slice[i] = sum;
    }

    // Step 2: Backprop through activation
    var d_up_out: [4096]f32 = undefined;
    const d_up_slice = d_up_out[0..intermediate_dim];

    for (0..intermediate_dim) |i| {
        const deriv = switch (activation) {
            .relu => if (up_out[i] > 0) @as(f32, 1.0) else @as(f32, 0.0),
            .gelu => geluDerivative(up_out[i]),
            .silu => siluDerivative(up_out[i]),
        };
        d_up_slice[i] = d_act_slice[i] * deriv;
    }

    // Compute activated output for d_down_weight
    var activated: [4096]f32 = undefined;
    const act_slice = activated[0..intermediate_dim];
    for (0..intermediate_dim) |i| {
        act_slice[i] = switch (activation) {
            .relu => activations.relu(up_out[i]),
            .gelu => activations.gelu(up_out[i]),
            .silu => activations.silu(up_out[i]),
        };
    }

    // d_down_weight
    for (0..hidden_dim) |j| {
        for (0..intermediate_dim) |i| {
            d_down_weight[j * intermediate_dim + i] += d_output[j] * act_slice[i];
        }
    }

    // Step 3: Backprop through up projection
    // d_up_weight
    for (0..intermediate_dim) |i| {
        for (0..hidden_dim) |j| {
            d_up_weight[i * hidden_dim + j] += d_up_slice[i] * x[j];
        }
    }

    // d_x
    for (0..hidden_dim) |j| {
        var sum: f32 = 0;
        for (0..intermediate_dim) |i| {
            sum += up_weight[i * hidden_dim + j] * d_up_slice[i];
        }
        d_x[j] += sum;
    }
}

pub const ActivationType = enum {
    relu,
    gelu,
    silu,
};

/// GELU derivative (tanh approximation).
fn geluDerivative(x: f32) f32 {
    const sqrt_2_pi = 0.7978845608028654;
    const coeff = 0.044715;

    const inner = sqrt_2_pi * (x + coeff * x * x * x);
    const tanh_inner = std.math.tanh(inner);
    const sech2 = 1.0 - tanh_inner * tanh_inner;

    // d/dx[0.5 * x * (1 + tanh(inner))]
    // = 0.5 * (1 + tanh(inner)) + 0.5 * x * sech^2(inner) * d_inner/dx
    const d_inner = sqrt_2_pi * (1.0 + 3.0 * coeff * x * x);

    return 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner;
}

test "silu derivative" {
    // Numerical check
    const eps: f32 = 1e-4;

    const x: f32 = 1.5;
    const analytical = siluDerivative(x);
    const numerical = (activations.silu(x + eps) - activations.silu(x - eps)) / (2.0 * eps);

    try std.testing.expectApproxEqAbs(numerical, analytical, 0.001);
}

test "swiglu backward basic" {
    const allocator = std.testing.allocator;

    const hidden_dim: u32 = 4;
    const intermediate_dim: u32 = 8;

    // Create cache
    var cache = try SwigluCache.init(allocator, hidden_dim, intermediate_dim);
    defer cache.deinit();

    // Fill with simple values
    for (0..hidden_dim) |i| {
        cache.x[i] = 0.1 * @as(f32, @floatFromInt(i + 1));
    }
    for (0..intermediate_dim) |i| {
        cache.gate_out[i] = 0.1 * @as(f32, @floatFromInt(i + 1));
        cache.up_out[i] = 0.1 * @as(f32, @floatFromInt(i + 1));
        cache.intermediate[i] = activations.silu(cache.gate_out[i]) * cache.up_out[i];
    }

    // Create weights
    const gate_weight = try allocator.alloc(f32, intermediate_dim * hidden_dim);
    defer allocator.free(gate_weight);
    const up_weight = try allocator.alloc(f32, intermediate_dim * hidden_dim);
    defer allocator.free(up_weight);
    const down_weight = try allocator.alloc(f32, hidden_dim * intermediate_dim);
    defer allocator.free(down_weight);

    @memset(gate_weight, 0.1);
    @memset(up_weight, 0.1);
    @memset(down_weight, 0.1);

    // Gradients
    const d_gate = try allocator.alloc(f32, intermediate_dim * hidden_dim);
    defer allocator.free(d_gate);
    const d_up = try allocator.alloc(f32, intermediate_dim * hidden_dim);
    defer allocator.free(d_up);
    const d_down = try allocator.alloc(f32, hidden_dim * intermediate_dim);
    defer allocator.free(d_down);
    const d_x = try allocator.alloc(f32, hidden_dim);
    defer allocator.free(d_x);

    @memset(d_gate, 0);
    @memset(d_up, 0);
    @memset(d_down, 0);
    @memset(d_x, 0);

    // Upstream gradient
    var d_output: [4]f32 = undefined;
    @memset(&d_output, 1.0);

    swigluBackward(
        &d_output,
        &cache,
        gate_weight,
        up_weight,
        down_weight,
        d_gate,
        d_up,
        d_down,
        d_x,
    );

    // Check gradients are non-zero
    var has_nonzero = false;
    for (d_x) |v| {
        if (v != 0) has_nonzero = true;
    }
    try std.testing.expect(has_nonzero);
}

test "gelu derivative" {
    const eps: f32 = 1e-4;

    const x: f32 = 0.5;
    const analytical = geluDerivative(x);
    const numerical = (activations.gelu(x + eps) - activations.gelu(x - eps)) / (2.0 * eps);

    try std.testing.expectApproxEqAbs(numerical, analytical, 0.01);
}
