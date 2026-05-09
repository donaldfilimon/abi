//! Transformer Layers
//!
//! Core building blocks for the Vision Transformer: feed-forward MLP,
//! layer normalization, transformer encoder blocks, and activation functions.

const std = @import("std");
const math = std.math;
const ViTConfig = @import("../vit.zig").ViTConfig;
const MultiHeadAttention = @import("attention.zig").MultiHeadAttention;

// ============================================================================
// Activation Functions
// ============================================================================

/// GELU activation function (Gaussian Error Linear Unit)
pub fn gelu(x: f32) f32 {
    // Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const sqrt_2_over_pi: f32 = 0.7978845608;
    const coeff: f32 = 0.044715;
    const inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    return 0.5 * x * (1.0 + math.tanh(inner));
}

/// Apply GELU activation to a slice
pub fn geluSlice(data: []f32) void {
    for (data) |*v| {
        v.* = gelu(v.*);
    }
}

// ============================================================================
// Feed-Forward Network (MLP)
// ============================================================================

/// MLP block used in transformer
pub const MLP = struct {
    allocator: std.mem.Allocator,
    hidden_size: u32,
    mlp_dim: u32,
    use_gelu: bool,

    /// First linear [mlp_dim, hidden_size]
    w1: []f32,
    b1: []f32,

    /// Second linear [hidden_size, mlp_dim]
    w2: []f32,
    b2: []f32,

    pub fn init(allocator: std.mem.Allocator, hidden_size: u32, mlp_dim: u32, use_gelu_act: bool) !MLP {
        const w1 = try allocator.alloc(f32, mlp_dim * hidden_size);
        const b1 = try allocator.alloc(f32, mlp_dim);
        const w2 = try allocator.alloc(f32, hidden_size * mlp_dim);
        const b2 = try allocator.alloc(f32, hidden_size);

        // Xavier initialization
        const scale1 = @sqrt(2.0 / @as(f32, @floatFromInt(hidden_size + mlp_dim)));
        const scale2 = @sqrt(2.0 / @as(f32, @floatFromInt(mlp_dim + hidden_size)));
        var prng = std.Random.DefaultPrng.init(456);

        for (w1) |*w| w.* = (prng.random().float(f32) * 2.0 - 1.0) * scale1;
        for (w2) |*w| w.* = (prng.random().float(f32) * 2.0 - 1.0) * scale2;

        @memset(b1, 0.0);
        @memset(b2, 0.0);

        return .{
            .allocator = allocator,
            .hidden_size = hidden_size,
            .mlp_dim = mlp_dim,
            .use_gelu = use_gelu_act,
            .w1 = w1,
            .b1 = b1,
            .w2 = w2,
            .b2 = b2,
        };
    }

    pub fn deinit(self: *MLP) void {
        self.allocator.free(self.w1);
        self.allocator.free(self.b1);
        self.allocator.free(self.w2);
        self.allocator.free(self.b2);
    }

    /// Forward pass: [seq_len, hidden] -> [seq_len, hidden]
    pub fn forward(self: *const MLP, x: []const f32, seq_len: usize) ![]f32 {
        const hidden = self.hidden_size;
        const mlp_dim = self.mlp_dim;

        // First linear + activation
        const intermediate = try self.allocator.alloc(f32, seq_len * mlp_dim);
        defer self.allocator.free(intermediate);

        for (0..seq_len) |s| {
            for (0..mlp_dim) |m| {
                var sum: f32 = self.b1[m];
                for (0..hidden) |h| {
                    sum += x[s * hidden + h] * self.w1[m * hidden + h];
                }
                intermediate[s * mlp_dim + m] = if (self.use_gelu) gelu(sum) else @max(sum, 0.0);
            }
        }

        // Second linear
        const output = try self.allocator.alloc(f32, seq_len * hidden);
        for (0..seq_len) |s| {
            for (0..hidden) |h| {
                var sum: f32 = self.b2[h];
                for (0..mlp_dim) |m| {
                    sum += intermediate[s * mlp_dim + m] * self.w2[h * mlp_dim + m];
                }
                output[s * hidden + h] = sum;
            }
        }

        return output;
    }
};

// ============================================================================
// Layer Normalization
// ============================================================================

/// Layer normalization
pub const LayerNorm = struct {
    allocator: std.mem.Allocator,
    hidden_size: u32,
    eps: f32,

    /// Scale parameter (gamma)
    gamma: []f32,
    /// Shift parameter (beta)
    beta: []f32,

    pub fn init(allocator: std.mem.Allocator, hidden_size: u32, eps: f32) !LayerNorm {
        const gamma = try allocator.alloc(f32, hidden_size);
        const beta = try allocator.alloc(f32, hidden_size);

        // Initialize gamma to 1, beta to 0
        for (gamma) |*g| g.* = 1.0;
        @memset(beta, 0.0);

        return .{
            .allocator = allocator,
            .hidden_size = hidden_size,
            .eps = eps,
            .gamma = gamma,
            .beta = beta,
        };
    }

    pub fn deinit(self: *LayerNorm) void {
        self.allocator.free(self.gamma);
        self.allocator.free(self.beta);
    }

    /// Forward pass: normalizes the last dimension
    pub fn forward(self: *const LayerNorm, x: []const f32, seq_len: usize) ![]f32 {
        const hidden = self.hidden_size;
        const output = try self.allocator.alloc(f32, seq_len * hidden);

        for (0..seq_len) |s| {
            // Compute mean
            var mean: f32 = 0.0;
            for (0..hidden) |h| {
                mean += x[s * hidden + h];
            }
            mean /= @floatFromInt(hidden);

            // Compute variance
            var variance: f32 = 0.0;
            for (0..hidden) |h| {
                const diff = x[s * hidden + h] - mean;
                variance += diff * diff;
            }
            variance /= @floatFromInt(hidden);

            // Normalize
            const std_dev = @sqrt(variance + self.eps);
            for (0..hidden) |h| {
                const normalized = (x[s * hidden + h] - mean) / std_dev;
                output[s * hidden + h] = normalized * self.gamma[h] + self.beta[h];
            }
        }

        return output;
    }
};

// ============================================================================
// Transformer Encoder Block
// ============================================================================

/// Single transformer encoder block
pub const TransformerBlock = struct {
    allocator: std.mem.Allocator,
    hidden_size: u32,
    pre_norm: bool,

    attention: MultiHeadAttention,
    mlp: MLP,
    norm1: LayerNorm,
    norm2: LayerNorm,

    pub fn init(allocator: std.mem.Allocator, config: ViTConfig) !TransformerBlock {
        return .{
            .allocator = allocator,
            .hidden_size = config.hidden_size,
            .pre_norm = config.pre_norm,
            .attention = try MultiHeadAttention.init(allocator, config.hidden_size, config.num_heads),
            .mlp = try MLP.init(allocator, config.hidden_size, config.mlp_dim, config.use_gelu),
            .norm1 = try LayerNorm.init(allocator, config.hidden_size, config.layer_norm_eps),
            .norm2 = try LayerNorm.init(allocator, config.hidden_size, config.layer_norm_eps),
        };
    }

    pub fn deinit(self: *TransformerBlock) void {
        self.attention.deinit();
        self.mlp.deinit();
        self.norm1.deinit();
        self.norm2.deinit();
    }

    /// Forward pass with residual connections
    pub fn forward(self: *const TransformerBlock, x: []const f32, seq_len: usize) ![]f32 {
        const hidden = self.hidden_size;

        if (self.pre_norm) {
            // Pre-norm: norm -> attention -> residual -> norm -> mlp -> residual
            const normed1 = try self.norm1.forward(x, seq_len);
            defer self.allocator.free(normed1);

            const attn_out = try self.attention.forward(normed1, seq_len);
            defer self.allocator.free(attn_out);

            // Add residual
            const residual1 = try self.allocator.alloc(f32, seq_len * hidden);
            for (0..seq_len * hidden) |i| {
                residual1[i] = x[i] + attn_out[i];
            }

            const normed2 = try self.norm2.forward(residual1, seq_len);
            defer self.allocator.free(normed2);

            const mlp_out = try self.mlp.forward(normed2, seq_len);
            defer self.allocator.free(mlp_out);

            // Add residual
            for (0..seq_len * hidden) |i| {
                residual1[i] = residual1[i] + mlp_out[i];
            }

            return residual1;
        } else {
            // Post-norm: attention -> residual -> norm -> mlp -> residual -> norm
            const attn_out = try self.attention.forward(x, seq_len);
            defer self.allocator.free(attn_out);

            // Add residual and norm
            const residual1 = try self.allocator.alloc(f32, seq_len * hidden);
            for (0..seq_len * hidden) |i| {
                residual1[i] = x[i] + attn_out[i];
            }

            const normed1 = try self.norm1.forward(residual1, seq_len);
            self.allocator.free(residual1);
            defer self.allocator.free(normed1);

            const mlp_out = try self.mlp.forward(normed1, seq_len);
            defer self.allocator.free(mlp_out);

            // Add residual and norm
            const residual2 = try self.allocator.alloc(f32, seq_len * hidden);
            for (0..seq_len * hidden) |i| {
                residual2[i] = normed1[i] + mlp_out[i];
            }

            const output = try self.norm2.forward(residual2, seq_len);
            self.allocator.free(residual2);

            return output;
        }
    }
};
