//! Transformer Module
//!
//! Implements transformer architectures for sequence processing tasks.
//!
//! ## Architecture Components
//!
//! - **Multi-head Attention**: Allows model to attend to different parts of input simultaneously
//! - **Feed-forward Networks**: Position-wise fully connected layers with ReLU activation
//! - **Positional Encoding**: Adds sequence position information to embeddings
//! - **Layer Normalization**: Stabilizes training with normalized activations
//!
//! ## Key Features
//!
//! - **Self-Attention**: Computes attention weights between all positions in sequence
//! - **Scaled Dot-Product Attention**: Efficient attention computation with scaling
//! - **Multi-Head Mechanism**: Parallel attention computations with different projections
//! - **Residual Connections**: Improves gradient flow through deep networks
//!
//! ## Usage Example
//!
//! ```zig
//! const config = TransformerConfig{
//!     .vocab_size = 30000,
//!     .d_model = 512,
//!     .n_heads = 8,
//!     .d_ff = 2048,
//!     .n_layers = 6,
//! };
//!
//! var encoder = try TransformerEncoderLayer.init(allocator, config);
//! defer encoder.deinit();
//!
//! const output = try encoder.forward(input_embeddings, attention_mask);
//! ```
//!
//! ## References
//!
//! - "Attention Is All You Need" (Vaswani et al., 2017)
//! - Original transformer paper introducing self-attention mechanisms

const std = @import("std");

/// Configuration parameters for transformer models
pub const TransformerConfig = struct {
    /// Size of vocabulary (number of unique tokens)
    vocab_size: usize,
    /// Model dimension (embedding size and hidden state size)
    d_model: usize,
    /// Number of attention heads in multi-head attention
    n_heads: usize,
    /// Dimension of feed-forward network inner layer
    d_ff: usize,
    /// Number of transformer encoder/decoder layers
    n_layers: usize,
    /// Dropout probability for regularization (0.0 to 1.0)
    dropout: f32 = 0.1,
    /// Maximum sequence length for positional encoding
    max_seq_len: usize = 512,
};

/// Multi-head attention mechanism
pub const MultiHeadAttention = struct {
    const Self = @This();

    config: TransformerConfig,
    w_q: []f32, // Query weights
    w_k: []f32, // Key weights
    w_v: []f32, // Value weights
    w_o: []f32, // Output weights
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: TransformerConfig) !MultiHeadAttention {
        const w_size = config.d_model * config.d_model;

        const w_q = try allocator.alloc(f32, w_size);
        errdefer allocator.free(w_q);
        // TODO: Replace with proper random initialization (Xavier/Glorot)
        @memset(w_q, 0.1); // Initialize with small values

        const w_k = try allocator.alloc(f32, w_size);
        errdefer allocator.free(w_k);
        @memset(w_k, 0.1);

        const w_v = try allocator.alloc(f32, w_size);
        errdefer allocator.free(w_v);
        @memset(w_v, 0.1);

        const w_o = try allocator.alloc(f32, w_size);
        errdefer allocator.free(w_o);
        @memset(w_o, 0.1);

        return MultiHeadAttention{
            .config = config,
            .w_q = w_q,
            .w_k = w_k,
            .w_v = w_v,
            .w_o = w_o,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.w_q);
        self.allocator.free(self.w_k);
        self.allocator.free(self.w_v);
        self.allocator.free(self.w_o);
    }

    /// Forward pass through multi-head attention
    pub fn forward(self: *Self, query: []const f32, key: []const f32, value: []const f32, mask: ?[]const bool) ![]f32 {
        const seq_len = query.len / self.config.d_model;
        const head_dim = self.config.d_model / self.config.n_heads;

        // Allocate output buffer
        const output = try self.allocator.alloc(f32, query.len);
        errdefer self.allocator.free(output);

        // Temporary buffers for computations
        const q_heads = try self.allocator.alloc(f32, seq_len * self.config.d_model);
        defer self.allocator.free(q_heads);
        const k_heads = try self.allocator.alloc(f32, seq_len * self.config.d_model);
        defer self.allocator.free(k_heads);
        const v_heads = try self.allocator.alloc(f32, seq_len * self.config.d_model);
        defer self.allocator.free(v_heads);

        const attention_scores = try self.allocator.alloc(f32, seq_len * seq_len);
        defer self.allocator.free(attention_scores);

        // Linear transformations for all heads at once
        try self.linearTransform(q_heads, query, self.w_q);
        try self.linearTransform(k_heads, key, self.w_k);
        try self.linearTransform(v_heads, value, self.w_v);

        // Process each attention head
        for (0..self.config.n_heads) |head| {
            const head_offset = head * head_dim;

            // Extract head-specific projections
            const q_head = q_heads[head_offset .. head_offset + seq_len * head_dim];
            const k_head = k_heads[head_offset .. head_offset + seq_len * head_dim];
            const v_head = v_heads[head_offset .. head_offset + seq_len * head_dim];

            // Scaled dot-product attention
            const head_attention = try self.allocator.alloc(f32, seq_len * seq_len);
            defer self.allocator.free(head_attention);
            const head_output = try self.allocator.alloc(f32, seq_len * head_dim);
            defer self.allocator.free(head_output);

            try self.scaledDotProductAttention(head_attention, head_output, q_head, k_head, v_head, seq_len, head_dim, mask);

            // Copy attention output to final output position
            const output_head = output[head_offset .. head_offset + seq_len * head_dim];
            @memcpy(output_head, head_output);
        }

        // Final linear transformation (W_o)
        const final_output = try self.allocator.alloc(f32, query.len);
        defer self.allocator.free(final_output);
        try self.linearTransform(final_output, output, self.w_o);

        return final_output;
    }

    /// Linear transformation: output = input * weights
    fn linearTransform(self: *Self, output: []f32, input: []const f32, weights: []const f32) !void {
        const input_dim = input.len / self.config.d_model;
        const output_dim = self.config.d_model;

        // Matrix multiplication: (seq_len, d_model) * (d_model, d_model) -> (seq_len, d_model)
        for (0..input_dim) |i| {
            for (0..output_dim) |j| {
                var sum: f32 = 0.0;
                for (0..self.config.d_model) |k| {
                    const input_idx = i * self.config.d_model + k;
                    const weight_idx = k * output_dim + j;
                    sum += input[input_idx] * weights[weight_idx];
                }
                output[i * output_dim + j] = sum;
            }
        }
    }

    /// Scaled dot-product attention
    fn scaledDotProductAttention(_: *Self, attention_scores: []f32, output: []f32, query: []const f32, key: []const f32, value: []const f32, seq_len: usize, head_dim: usize, mask: ?[]const bool) !void {
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        // Compute attention scores: Q * K^T / sqrt(d_k)
        for (0..seq_len) |i| {
            for (0..seq_len) |j| {
                var score: f32 = 0.0;
                for (0..head_dim) |k| {
                    const q_idx = i * head_dim + k;
                    const k_idx = j * head_dim + k;
                    score += query[q_idx] * key[k_idx];
                }
                score *= scale;

                // Apply mask if provided
                if (mask) |m| {
                    if (m[i * seq_len + j]) {
                        score = -std.math.inf(f32);
                    }
                }

                attention_scores[i * seq_len + j] = score;
            }
        }

        // Softmax over each row
        for (0..seq_len) |i| {
            const row_start = i * seq_len;
            const row = attention_scores[row_start .. row_start + seq_len];

            // Find max for numerical stability
            var max_val = -std.math.inf(f32);
            for (row) |val| {
                max_val = @max(max_val, val);
            }

            // Compute exp and sum
            var sum: f32 = 0.0;
            for (0..seq_len) |j| {
                const exp_val = std.math.exp(row[j] - max_val);
                attention_scores[row_start + j] = exp_val;
                sum += exp_val;
            }

            // Normalize
            for (0..seq_len) |j| {
                attention_scores[row_start + j] /= sum;
            }
        }

        // Apply attention to values: softmax_scores * V
        for (0..seq_len) |i| {
            for (0..head_dim) |k| {
                var weighted_sum: f32 = 0.0;
                for (0..seq_len) |j| {
                    const attention_weight = attention_scores[i * seq_len + j];
                    const value_idx = j * head_dim + k;
                    weighted_sum += attention_weight * value[value_idx];
                }
                output[i * head_dim + k] = weighted_sum;
            }
        }
    }
};

/// Feed-forward network
pub const FeedForward = struct {
    const Self = @This();

    config: TransformerConfig,
    w1: []f32, // First layer weights
    b1: []f32, // First layer bias
    w2: []f32, // Second layer weights
    b2: []f32, // Second layer bias
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: TransformerConfig) !FeedForward {
        const w1_size = config.d_model * config.d_ff;
        const w2_size = config.d_ff * config.d_model;

        const w1 = try allocator.alloc(f32, w1_size);
        errdefer allocator.free(w1);
        @memset(w1, 0.1);

        const b1 = try allocator.alloc(f32, config.d_ff);
        errdefer allocator.free(b1);
        @memset(b1, 0.0);

        const w2 = try allocator.alloc(f32, w2_size);
        errdefer allocator.free(w2);
        @memset(w2, 0.1);

        const b2 = try allocator.alloc(f32, config.d_model);
        errdefer allocator.free(b2);
        @memset(b2, 0.0);

        return FeedForward{
            .config = config,
            .w1 = w1,
            .b1 = b1,
            .w2 = w2,
            .b2 = b2,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.w1);
        self.allocator.free(self.b1);
        self.allocator.free(self.w2);
        self.allocator.free(self.b2);
    }

    /// Forward pass: FF(x) = max(0, x*W1 + b1) * W2 + b2
    pub fn forward(self: *Self, input: []const f32) ![]f32 {
        var hidden = try self.allocator.alloc(f32, self.config.d_ff);
        defer self.allocator.free(hidden);

        // First linear layer + ReLU
        for (0..self.config.d_ff) |i| {
            var sum: f32 = self.b1[i];
            for (0..self.config.d_model) |j| {
                sum += input[j] * self.w1[i * self.config.d_model + j];
            }
            hidden[i] = @max(0, sum); // ReLU
        }

        // Second linear layer
        var output = try self.allocator.alloc(f32, self.config.d_model);
        errdefer self.allocator.free(output);

        for (0..self.config.d_model) |i| {
            var sum: f32 = self.b2[i];
            for (0..self.config.d_ff) |j| {
                sum += hidden[j] * self.w2[i * self.config.d_ff + j];
            }
            output[i] = sum;
        }

        return output;
    }
};

/// Positional encoding for sequence positions
pub const PositionalEncoding = struct {
    const Self = @This();

    encodings: []f32,
    max_len: usize,
    d_model: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, max_len: usize, d_model: usize) !PositionalEncoding {
        const size = max_len * d_model;
        var encodings = try allocator.alloc(f32, size);

        // Compute positional encodings: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        // PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        for (0..max_len) |pos| {
            for (0..d_model / 2) |i| {
                const angle = @as(f32, @floatFromInt(pos)) / std.math.pow(f32, 10000, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(d_model)));
                const offset = pos * d_model + 2 * i;
                encodings[offset] = @sin(angle);
                encodings[offset + 1] = @cos(angle);
            }
        }

        return PositionalEncoding{
            .encodings = encodings,
            .max_len = max_len,
            .d_model = d_model,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.encodings);
    }

    /// Add positional encoding to input embeddings
    pub fn encode(self: *Self, embeddings: []f32, seq_len: usize) void {
        for (0..seq_len) |pos| {
            for (0..self.d_model) |dim| {
                const emb_idx = pos * self.d_model + dim;
                const enc_idx = pos * self.d_model + dim;
                embeddings[emb_idx] += self.encodings[enc_idx];
            }
        }
    }
};

/// Transformer encoder layer
pub const TransformerEncoderLayer = struct {
    const Self = @This();

    config: TransformerConfig,
    self_attention: MultiHeadAttention,
    feed_forward: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: TransformerConfig) !TransformerEncoderLayer {
        var self_attention = try MultiHeadAttention.init(allocator, config);
        errdefer self_attention.deinit();

        var feed_forward = try FeedForward.init(allocator, config);
        errdefer feed_forward.deinit();

        var norm1 = try LayerNorm.init(allocator, config.d_model);
        errdefer norm1.deinit();

        var norm2 = try LayerNorm.init(allocator, config.d_model);
        errdefer norm2.deinit();

        return TransformerEncoderLayer{
            .config = config,
            .self_attention = self_attention,
            .feed_forward = feed_forward,
            .norm1 = norm1,
            .norm2 = norm2,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.self_attention.deinit();
        self.feed_forward.deinit();
        self.norm1.deinit();
        self.norm2.deinit();
    }

    /// Forward pass through encoder layer
    pub fn forward(self: *Self, input: []const f32, mask: ?[]const bool) ![]f32 {
        // Self-attention with residual connection and layer norm
        const attn_output = try self.self_attention.forward(input, input, input, mask);
        defer self.allocator.free(attn_output);

        const residual1 = try self.addResidual(input, attn_output);
        defer self.allocator.free(residual1);

        self.norm1.normalize(residual1);

        // Feed-forward with residual connection and layer norm
        const ff_output = try self.feed_forward.forward(residual1);
        defer self.allocator.free(ff_output);

        const residual2 = try self.addResidual(residual1, ff_output);
        defer self.allocator.free(residual2);

        self.norm2.normalize(residual2);

        return residual2;
    }

    fn addResidual(self: *Self, a: []const f32, b: []const f32) ![]f32 {
        const result = try self.allocator.dupe(f32, a);
        for (result, b) |*r, bb| r.* += bb;
        return result;
    }
};

/// Complete transformer encoder with multiple layers
pub const TransformerEncoder = struct {
    const Self = @This();

    config: TransformerConfig,
    layers: []TransformerEncoderLayer,
    positional_encoding: PositionalEncoding,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: TransformerConfig) !TransformerEncoder {
        var layers = try allocator.alloc(TransformerEncoderLayer, config.n_layers);
        errdefer allocator.free(layers);

        // Initialize each layer
        var initialized: usize = 0;
        errdefer for (layers[0..initialized]) |*layer| layer.deinit();

        for (layers) |*layer| {
            layer.* = try TransformerEncoderLayer.init(allocator, config);
            initialized += 1;
        }

        var positional_encoding = try PositionalEncoding.init(allocator, config.max_seq_len, config.d_model);
        errdefer positional_encoding.deinit();

        return TransformerEncoder{
            .config = config,
            .layers = layers,
            .positional_encoding = positional_encoding,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.layers) |*layer| layer.deinit();
        self.allocator.free(self.layers);
        self.positional_encoding.deinit();
    }

    /// Forward pass through the complete encoder
    pub fn forward(self: *Self, input_embeddings: []f32, attention_mask: ?[]const bool) ![]f32 {
        // Add positional encoding
        const seq_len = input_embeddings.len / self.config.d_model;
        self.positional_encoding.encode(input_embeddings, seq_len);

        // Pass through each encoder layer
        var current_output = input_embeddings;
        for (self.layers) |*layer| {
            const layer_output = try layer.forward(current_output, attention_mask);
            if (current_output.ptr != input_embeddings.ptr) {
                self.allocator.free(current_output);
            }
            current_output = layer_output;
        }

        return current_output;
    }
};

/// Simple layer normalization
pub const LayerNorm = struct {
    const Self = @This();

    gamma: []f32,
    beta: []f32,
    size: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, size: usize) !LayerNorm {
        const gamma = try allocator.alloc(f32, size);
        @memset(gamma, 1.0); // Initialize to 1

        const beta = try allocator.alloc(f32, size);
        @memset(beta, 0.0); // Initialize to 0

        return LayerNorm{
            .gamma = gamma,
            .beta = beta,
            .size = size,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.gamma);
        self.allocator.free(self.beta);
    }

    /// Apply layer normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    pub fn normalize(self: *Self, x: []f32) void {
        const eps = 1e-6;

        // Compute mean
        var mean: f32 = 0;
        for (x) |val| mean += val;
        mean /= @as(f32, @floatFromInt(x.len));

        // Compute variance
        var variance: f32 = 0;
        for (x) |val| {
            const diff = val - mean;
            variance += diff * diff;
        }
        variance /= @as(f32, @floatFromInt(x.len));

        // Normalize
        const std_dev = @sqrt(variance + eps);
        for (x, 0..) |*val, i| {
            val.* = (val.* - mean) / std_dev * self.gamma[i] + self.beta[i];
        }
    }
};

test "Transformer components initialization" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const config = TransformerConfig{
        .vocab_size = 1000,
        .d_model = 64,
        .n_heads = 4,
        .d_ff = 128,
        .n_layers = 2,
    };

    var attention = try MultiHeadAttention.init(allocator, config);
    defer attention.deinit();

    var feed_forward = try FeedForward.init(allocator, config);
    defer feed_forward.deinit();

    var pos_encoding = try PositionalEncoding.init(allocator, 10, config.d_model);
    defer pos_encoding.deinit();

    try testing.expectEqual(@as(usize, 64 * 64), attention.w_q.len);
    try testing.expectEqual(@as(usize, 64 * 128), feed_forward.w1.len);
}

test "Positional encoding functionality" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var pos_encoding = try PositionalEncoding.init(allocator, 5, 4);
    defer pos_encoding.deinit();

    var embeddings = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0 };
    pos_encoding.encode(&embeddings, 5); // 5 positions, 4 dimensions each

    // Check that embeddings were modified (positional encoding added)
    try testing.expect(embeddings[0] != 1.0);
    try testing.expect(embeddings[4] != 5.0);
    try testing.expect(embeddings[8] != 9.0);
}

test "Feed-forward network forward pass" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const config = TransformerConfig{
        .vocab_size = 100,
        .d_model = 8,
        .n_heads = 2,
        .d_ff = 16,
        .n_layers = 1,
    };

    var ff = try FeedForward.init(allocator, config);
    defer ff.deinit();

    var input = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };
    const output = try ff.forward(&input);
    defer allocator.free(output);

    try testing.expectEqual(@as(usize, 8), output.len);

    // Check that output is different from input (transformation applied)
    var has_difference = false;
    for (input, output) |inp, out| {
        if (@abs(inp - out) > 0.001) has_difference = true;
    }
    try testing.expect(has_difference);
}

test "Layer normalization" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var ln = try LayerNorm.init(allocator, 4);
    defer ln.deinit();

    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    ln.normalize(&x);

    // After normalization, mean should be close to 0
    var mean: f32 = 0;
    for (x) |val| mean += val;
    mean /= 4;
    try testing.expect(@abs(mean) < 0.1);
}
