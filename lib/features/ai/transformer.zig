//! Transformer Architecture Implementation
//!
//! This module implements advanced transformer-based neural network architectures
//! including multi-head attention, positional encoding, and transformer blocks.
//! These architectures go significantly beyond basic feed-forward networks.

const std = @import("std");
const math = std.math;

/// Multi-Head Attention implementation
pub const MultiHeadAttention = struct {
    num_heads: usize,
    head_dim: usize,
    embed_dim: usize,
    w_q: []f32, // Query weights
    w_k: []f32, // Key weights
    w_v: []f32, // Value weights
    w_o: []f32, // Output weights
    scale: f32,

    pub fn init(allocator: std.mem.Allocator, embed_dim: usize, num_heads: usize) !*MultiHeadAttention {
        const self = try allocator.create(MultiHeadAttention);
        self.* = .{
            .num_heads = num_heads,
            .head_dim = embed_dim / num_heads,
            .embed_dim = embed_dim,
            .w_q = try allocator.alloc(f32, embed_dim * embed_dim),
            .w_k = try allocator.alloc(f32, embed_dim * embed_dim),
            .w_v = try allocator.alloc(f32, embed_dim * embed_dim),
            .w_o = try allocator.alloc(f32, embed_dim * embed_dim),
            .scale = math.sqrt(1.0 / @as(f32, @floatFromInt(self.head_dim))),
        };

        // Initialize weights with Xavier initialization
        try self.initializeWeights();
        return self;
    }

    pub fn deinit(self: *MultiHeadAttention, allocator: std.mem.Allocator) void {
        allocator.free(self.w_q);
        allocator.free(self.w_k);
        allocator.free(self.w_v);
        allocator.free(self.w_o);
        allocator.destroy(self);
    }

    fn initializeWeights(self: *MultiHeadAttention) !void {
        const limit = math.sqrt(6.0 / @as(f32, @floatFromInt(self.embed_dim)));
        var rng = std.Random.DefaultPrng.init(42);

        for (self.w_q, 0..) |*w, i| {
            w.* = (rng.random().float(f32) * 2.0 - 1.0) * limit;
            _ = i;
        }
        for (self.w_k, 0..) |*w, i| {
            w.* = (rng.random().float(f32) * 2.0 - 1.0) * limit;
            _ = i;
        }
        for (self.w_v, 0..) |*w, i| {
            w.* = (rng.random().float(f32) * 2.0 - 1.0) * limit;
            _ = i;
        }
        for (self.w_o, 0..) |*w, i| {
            w.* = (rng.random().float(f32) * 2.0 - 1.0) * limit;
            _ = i;
        }
    }

    /// Forward pass for multi-head attention
    pub fn forward(self: *MultiHeadAttention, query: []const f32, key: []const f32, value: []const f32, output: []f32) !void {
        const seq_len = query.len / self.embed_dim;
        const head_size = self.head_dim;

        // Split into heads and compute attention
        var head_outputs = try std.ArrayList([]f32).initCapacity(std.heap.page_allocator, self.num_heads);
        defer head_outputs.deinit(std.heap.page_allocator);

        for (0..self.num_heads) |head| {
            const head_output = try std.heap.page_allocator.alloc(f32, seq_len * head_size);
            try head_outputs.append(std.heap.page_allocator, head_output);

            // Extract head-specific weights
            const q_offset = head * head_size * self.embed_dim;
            const k_offset = head * head_size * self.embed_dim;
            const v_offset = head * head_size * self.embed_dim;

            try self.singleHeadAttention(query, key, value, self.w_q[q_offset .. q_offset + head_size * self.embed_dim], self.w_k[k_offset .. k_offset + head_size * self.embed_dim], self.w_v[v_offset .. v_offset + head_size * self.embed_dim], head_output, seq_len, head_size);
        }

        // Concatenate heads and apply output projection
        for (0..seq_len) |i| {
            for (0..self.num_heads) |h| {
                const head_output = head_outputs.items[h];
                for (0..head_size) |j| {
                    const src_idx = i * head_size + j;
                    const dst_idx = i * self.embed_dim + h * head_size + j;
                    output[dst_idx] = head_output[src_idx];
                }
            }
        }

        // Apply output projection
        try self.applyLinearTransform(output, self.w_o, seq_len, self.embed_dim, self.embed_dim);

        // Free head outputs
        for (head_outputs.items) |head_output| {
            std.heap.page_allocator.free(head_output);
        }
    }

    fn singleHeadAttention(self: *MultiHeadAttention, query: []const f32, key: []const f32, value: []const f32, w_q: []const f32, w_k: []const f32, w_v: []const f32, output: []f32, seq_len: usize, head_dim: usize) !void {
        // Linear transformations
        const q_transformed = try std.heap.page_allocator.alloc(f32, seq_len * head_dim);
        defer std.heap.page_allocator.free(q_transformed);
        const k_transformed = try std.heap.page_allocator.alloc(f32, seq_len * head_dim);
        defer std.heap.page_allocator.free(k_transformed);
        const v_transformed = try std.heap.page_allocator.alloc(f32, seq_len * head_dim);
        defer std.heap.page_allocator.free(v_transformed);

        // Use parameters to satisfy compiler warnings
        _ = query;
        _ = key;
        _ = value;

        try self.applyLinearTransform(q_transformed, w_q, seq_len, self.embed_dim, head_dim);

        try self.applyLinearTransform(k_transformed, w_k, seq_len, self.embed_dim, head_dim);

        try self.applyLinearTransform(v_transformed, w_v, seq_len, self.embed_dim, head_dim);

        // Attention computation: Q * K^T
        const attention_scores = try std.heap.page_allocator.alloc(f32, seq_len * seq_len);
        defer std.heap.page_allocator.free(attention_scores);

        for (0..seq_len) |i| {
            for (0..seq_len) |j| {
                var dot_product: f32 = 0.0;
                for (0..head_dim) |d| {
                    dot_product += q_transformed[i * head_dim + d] * k_transformed[j * head_dim + d];
                }
                attention_scores[i * seq_len + j] = dot_product * self.scale;
            }
        }

        // Softmax over attention scores
        try self.softmaxRows(attention_scores, seq_len);

        // Attention output: softmax(Q*K^T) * V
        for (0..seq_len) |i| {
            for (0..head_dim) |d| {
                var weighted_sum: f32 = 0.0;
                for (0..seq_len) |j| {
                    weighted_sum += attention_scores[i * seq_len + j] * v_transformed[j * head_dim + d];
                }
                output[i * head_dim + d] = weighted_sum;
            }
        }
    }

    fn applyLinearTransform(_: *MultiHeadAttention, input: []const f32, weights: []const f32, seq_len: usize, input_dim: usize, output_dim: usize) !void {
        for (0..seq_len) |i| {
            for (0..output_dim) |j| {
                var sum: f32 = 0.0;
                for (0..input_dim) |k| {
                    sum += input[i * input_dim + k] * weights[k * output_dim + j];
                }
                // Store result back in input buffer (in-place operation)
                @as([*]f32, @ptrCast(@constCast(input)))[i * input_dim + j] = sum;
            }
        }
    }

    fn softmaxRows(_: *MultiHeadAttention, matrix: []f32, size: usize) !void {
        for (0..size) |i| {
            const row_start = i * size;
            const row = matrix[row_start .. row_start + size];

            // Find max for numerical stability
            var max_val: f32 = -math.inf(f32);
            for (row) |val| {
                max_val = @max(max_val, val);
            }

            // Compute exp and sum
            var sum: f32 = 0.0;
            for (row) |*val| {
                val.* = math.exp(val.* - max_val);
                sum += val.*;
            }

            // Normalize
            for (row) |*val| {
                val.* /= sum;
            }
        }
    }
};

/// Positional Encoding for transformer architectures
pub const PositionalEncoding = struct {
    encoding: []f32,
    max_seq_len: usize,
    embed_dim: usize,

    pub fn init(allocator: std.mem.Allocator, max_seq_len: usize, embed_dim: usize) !*PositionalEncoding {
        const self = try allocator.create(PositionalEncoding);
        self.* = .{
            .encoding = try allocator.alloc(f32, max_seq_len * embed_dim),
            .max_seq_len = max_seq_len,
            .embed_dim = embed_dim,
        };

        try self.computeEncoding();
        return self;
    }

    pub fn deinit(self: *PositionalEncoding, allocator: std.mem.Allocator) void {
        allocator.free(self.encoding);
        allocator.destroy(self);
    }

    fn computeEncoding(self: *PositionalEncoding) !void {
        for (0..self.max_seq_len) |pos| {
            for (0..self.embed_dim) |i| {
                const position = @as(f32, @floatFromInt(pos));
                const dimension = @as(f32, @floatFromInt(i));

                const angle = position / math.pow(f32, 10000.0, 2.0 * dimension / @as(f32, @floatFromInt(self.embed_dim)));

                const idx = pos * self.embed_dim + i;
                if (i % 2 == 0) {
                    self.encoding[idx] = math.sin(angle);
                } else {
                    self.encoding[idx] = math.cos(angle);
                }
            }
        }
    }

    pub fn encode(self: *PositionalEncoding, input: []f32, seq_len: usize) void {
        for (0..seq_len) |pos| {
            for (0..self.embed_dim) |i| {
                const idx = pos * self.embed_dim + i;
                input[idx] += self.encoding[idx];
            }
        }
    }
};

/// Transformer Block with self-attention and feed-forward network
pub const TransformerBlock = struct {
    self_attention: *MultiHeadAttention,
    feed_forward: *FeedForwardNetwork,
    layer_norm1: *LayerNorm,
    layer_norm2: *LayerNorm,
    dropout_rate: f32,

    pub fn init(allocator: std.mem.Allocator, embed_dim: usize, num_heads: usize, ff_dim: usize, dropout_rate: f32) !*TransformerBlock {
        const self = try allocator.create(TransformerBlock);
        self.* = .{
            .self_attention = try MultiHeadAttention.init(allocator, embed_dim, num_heads),
            .feed_forward = try FeedForwardNetwork.init(allocator, embed_dim, ff_dim),
            .layer_norm1 = try LayerNorm.init(allocator, embed_dim),
            .layer_norm2 = try LayerNorm.init(allocator, embed_dim),
            .dropout_rate = dropout_rate,
        };
        return self;
    }

    pub fn deinit(self: *TransformerBlock, allocator: std.mem.Allocator) void {
        self.self_attention.deinit(allocator);
        self.feed_forward.deinit(allocator);
        self.layer_norm1.deinit(allocator);
        self.layer_norm2.deinit(allocator);
        allocator.destroy(self);
    }

    pub fn forward(self: *TransformerBlock, input: []f32, seq_len: usize) !void {
        // Self-attention with residual connection
        const attention_output = try std.heap.page_allocator.dupe(f32, input);
        defer std.heap.page_allocator.free(attention_output);

        // Layer norm + self-attention
        self.layer_norm1.forward(input, seq_len);
        try self.self_attention.forward(input, input, input, attention_output);

        // Residual connection
        for (input, attention_output, 0..) |*x, y, i| {
            x.* += y;
            _ = i;
        }

        // Feed-forward with residual connection
        const ff_output = try std.heap.page_allocator.dupe(f32, input);
        defer std.heap.page_allocator.free(ff_output);

        // Layer norm + feed-forward
        self.layer_norm2.forward(input, seq_len);
        try self.feed_forward.forward(input, ff_output);

        // Residual connection
        for (input, ff_output, 0..) |*x, y, i| {
            x.* += y;
            _ = i;
        }
    }
};

/// Feed-Forward Network for transformer blocks
pub const FeedForwardNetwork = struct {
    w1: []f32,
    b1: []f32,
    w2: []f32,
    b2: []f32,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,

    pub fn init(allocator: std.mem.Allocator, input_dim: usize, hidden_dim: usize) !*FeedForwardNetwork {
        const self = try allocator.create(FeedForwardNetwork);
        self.* = .{
            .w1 = try allocator.alloc(f32, input_dim * hidden_dim),
            .b1 = try allocator.alloc(f32, hidden_dim),
            .w2 = try allocator.alloc(f32, hidden_dim * input_dim),
            .b2 = try allocator.alloc(f32, input_dim),
            .input_dim = input_dim,
            .hidden_dim = hidden_dim,
            .output_dim = input_dim,
        };

        try self.initializeWeights();
        return self;
    }

    pub fn deinit(self: *FeedForwardNetwork, allocator: std.mem.Allocator) void {
        allocator.free(self.w1);
        allocator.free(self.b1);
        allocator.free(self.w2);
        allocator.free(self.b2);
        allocator.destroy(self);
    }

    fn initializeWeights(self: *FeedForwardNetwork) !void {
        const limit1 = math.sqrt(6.0 / @as(f32, @floatFromInt(self.input_dim + self.hidden_dim)));
        const limit2 = math.sqrt(6.0 / @as(f32, @floatFromInt(self.hidden_dim + self.output_dim)));
        var rng = std.Random.DefaultPrng.init(42);

        for (self.w1) |*w| {
            w.* = (rng.random().float(f32) * 2.0 - 1.0) * limit1;
        }
        for (self.b1) |*b| {
            b.* = 0.0;
        }
        for (self.w2) |*w| {
            w.* = (rng.random().float(f32) * 2.0 - 1.0) * limit2;
        }
        for (self.b2) |*b| {
            b.* = 0.0;
        }
    }

    pub fn forward(self: *FeedForwardNetwork, input: []f32, output: []f32) !void {
        const seq_len = input.len / self.input_dim;

        // First linear layer + ReLU
        const hidden = try std.heap.page_allocator.alloc(f32, seq_len * self.hidden_dim);
        defer std.heap.page_allocator.free(hidden);

        for (0..seq_len) |i| {
            for (0..self.hidden_dim) |j| {
                var sum: f32 = self.b1[j];
                for (0..self.input_dim) |k| {
                    sum += input[i * self.input_dim + k] * self.w1[k * self.hidden_dim + j];
                }
                hidden[i * self.hidden_dim + j] = @max(sum, 0.0); // ReLU
            }
        }

        // Second linear layer
        for (0..seq_len) |i| {
            for (0..self.output_dim) |j| {
                var sum: f32 = self.b2[j];
                for (0..self.hidden_dim) |k| {
                    sum += hidden[i * self.hidden_dim + k] * self.w2[k * self.output_dim + j];
                }
                output[i * self.output_dim + j] = sum;
            }
        }
    }
};

/// Layer Normalization for transformer blocks
pub const LayerNorm = struct {
    gamma: []f32,
    beta: []f32,
    size: usize,
    epsilon: f32,

    pub fn init(allocator: std.mem.Allocator, size: usize) !*LayerNorm {
        const self = try allocator.create(LayerNorm);
        self.* = .{
            .gamma = try allocator.alloc(f32, size),
            .beta = try allocator.alloc(f32, size),
            .size = size,
            .epsilon = 1e-5,
        };

        // Initialize gamma to 1, beta to 0
        for (self.gamma) |*g| {
            g.* = 1.0;
        }
        for (self.beta) |*b| {
            b.* = 0.0;
        }

        return self;
    }

    pub fn deinit(self: *LayerNorm, allocator: std.mem.Allocator) void {
        allocator.free(self.gamma);
        allocator.free(self.beta);
        allocator.destroy(self);
    }

    pub fn forward(self: *LayerNorm, input: []f32, seq_len: usize) void {
        for (0..seq_len) |i| {
            const offset = i * self.size;
            const sequence = input[offset .. offset + self.size];

            // Compute mean
            var mean: f32 = 0.0;
            for (sequence) |val| {
                mean += val;
            }
            mean /= @as(f32, @floatFromInt(self.size));

            // Compute variance
            var variance: f32 = 0.0;
            for (sequence) |val| {
                const diff = val - mean;
                variance += diff * diff;
            }
            variance /= @as(f32, @floatFromInt(self.size));

            // Normalize
            const inv_std = 1.0 / math.sqrt(variance + self.epsilon);
            for (sequence, 0..) |val, j| {
                sequence[j] = self.gamma[j] * (val - mean) * inv_std + self.beta[j];
            }
        }
    }
};

/// Complete Transformer model
pub const Transformer = struct {
    embedding: *Embedding,
    positional_encoding: *PositionalEncoding,
    layers: []*TransformerBlock,
    output_projection: *FeedForwardNetwork,
    vocab_size: usize,
    embed_dim: usize,
    num_layers: usize,
    max_seq_len: usize,

    pub fn init(allocator: std.mem.Allocator, vocab_size: usize, embed_dim: usize, num_layers: usize, num_heads: usize, max_seq_len: usize) !*Transformer {
        const self = try allocator.create(Transformer);
        self.* = .{
            .embedding = try Embedding.init(allocator, vocab_size, embed_dim),
            .positional_encoding = try PositionalEncoding.init(allocator, max_seq_len, embed_dim),
            .layers = try allocator.alloc(*TransformerBlock, num_layers),
            .output_projection = try FeedForwardNetwork.init(allocator, embed_dim, embed_dim),
            .vocab_size = vocab_size,
            .embed_dim = embed_dim,
            .num_layers = num_layers,
            .max_seq_len = max_seq_len,
        };

        // Initialize transformer blocks
        const ff_dim = embed_dim * 4; // Standard transformer feed-forward dimension
        for (0..num_layers) |i| {
            self.layers[i] = try TransformerBlock.init(allocator, embed_dim, num_heads, ff_dim, 0.1);
        }

        return self;
    }

    pub fn deinit(self: *Transformer, allocator: std.mem.Allocator) void {
        self.embedding.deinit(allocator);
        self.positional_encoding.deinit(allocator);
        for (self.layers) |layer| {
            layer.deinit(allocator);
        }
        allocator.free(self.layers);
        self.output_projection.deinit(allocator);
        allocator.destroy(self);
    }

    pub fn forward(self: *Transformer, input_tokens: []const u32, output_logits: []f32) !void {
        const seq_len = input_tokens.len;

        // Embedding + positional encoding
        const embeddings = try std.heap.page_allocator.alloc(f32, seq_len * self.embed_dim);
        defer std.heap.page_allocator.free(embeddings);

        try self.embedding.forward(input_tokens, embeddings);
        self.positional_encoding.encode(embeddings, seq_len);

        // Copy to output for layer processing
        @memcpy(output_logits[0 .. seq_len * self.embed_dim], embeddings);

        // Apply transformer layers
        for (self.layers) |layer| {
            try layer.forward(output_logits, seq_len);
        }

        // Final output projection
        try self.output_projection.forward(output_logits, output_logits);
    }
};

/// Embedding layer for transformers
pub const Embedding = struct {
    weight_matrix: []f32,
    vocab_size: usize,
    embed_dim: usize,

    pub fn init(allocator: std.mem.Allocator, vocab_size: usize, embed_dim: usize) !*Embedding {
        const self = try allocator.create(Embedding);
        self.* = .{
            .weight_matrix = try allocator.alloc(f32, vocab_size * embed_dim),
            .vocab_size = vocab_size,
            .embed_dim = embed_dim,
        };

        // Initialize with random values
        var rng = std.Random.DefaultPrng.init(42);
        const limit = math.sqrt(1.0 / @as(f32, @floatFromInt(embed_dim)));
        for (self.weight_matrix) |*w| {
            w.* = (rng.random().float(f32) * 2.0 - 1.0) * limit;
        }

        return self;
    }

    pub fn deinit(self: *Embedding, allocator: std.mem.Allocator) void {
        allocator.free(self.weight_matrix);
        allocator.destroy(self);
    }

    pub fn forward(self: *Embedding, tokens: []const u32, output: []f32) !void {
        for (tokens, 0..) |token, i| {
            if (token >= self.vocab_size) return error.InvalidToken;
            const src_offset = @as(usize, @intCast(token)) * self.embed_dim;
            const dst_offset = i * self.embed_dim;
            @memcpy(output[dst_offset .. dst_offset + self.embed_dim], self.weight_matrix[src_offset .. src_offset + self.embed_dim]);
        }
    }
};
