//! Multi-Head Self-Attention
//!
//! Implements scaled dot-product multi-head self-attention as used in the
//! Vision Transformer (ViT) architecture.

const std = @import("std");

/// Softmax over a slice (in-place)
pub fn softmax(data: []f32) void {
    if (data.len == 0) return;

    // Find max for numerical stability
    var max_val: f32 = data[0];
    for (data[1..]) |v| {
        if (v > max_val) max_val = v;
    }

    // Compute exp and sum
    var sum: f32 = 0.0;
    for (data) |*v| {
        v.* = @exp(v.* - max_val);
        sum += v.*;
    }

    // Normalize
    if (sum > 0.0) {
        for (data) |*v| {
            v.* /= sum;
        }
    }
}

/// Multi-head self-attention layer
pub const MultiHeadAttention = struct {
    allocator: std.mem.Allocator,
    hidden_size: u32,
    num_heads: u32,
    head_dim: u32,

    /// Query projection [hidden_size, hidden_size]
    wq: []f32,
    /// Key projection [hidden_size, hidden_size]
    wk: []f32,
    /// Value projection [hidden_size, hidden_size]
    wv: []f32,
    /// Output projection [hidden_size, hidden_size]
    wo: []f32,

    /// Biases
    bq: []f32,
    bk: []f32,
    bv: []f32,
    bo: []f32,

    pub fn init(allocator: std.mem.Allocator, hidden_size: u32, num_heads: u32) !MultiHeadAttention {
        const head_dim = hidden_size / num_heads;
        const size = hidden_size * hidden_size;

        const wq = try allocator.alloc(f32, size);
        const wk = try allocator.alloc(f32, size);
        const wv = try allocator.alloc(f32, size);
        const wo = try allocator.alloc(f32, size);

        const bq = try allocator.alloc(f32, hidden_size);
        const bk = try allocator.alloc(f32, hidden_size);
        const bv = try allocator.alloc(f32, hidden_size);
        const bo = try allocator.alloc(f32, hidden_size);

        // Xavier initialization
        const scale = @sqrt(2.0 / @as(f32, @floatFromInt(hidden_size * 2)));
        var prng = std.Random.DefaultPrng.init(123);

        for (wq) |*w| w.* = (prng.random().float(f32) * 2.0 - 1.0) * scale;
        for (wk) |*w| w.* = (prng.random().float(f32) * 2.0 - 1.0) * scale;
        for (wv) |*w| w.* = (prng.random().float(f32) * 2.0 - 1.0) * scale;
        for (wo) |*w| w.* = (prng.random().float(f32) * 2.0 - 1.0) * scale;

        @memset(bq, 0.0);
        @memset(bk, 0.0);
        @memset(bv, 0.0);
        @memset(bo, 0.0);

        return .{
            .allocator = allocator,
            .hidden_size = hidden_size,
            .num_heads = num_heads,
            .head_dim = head_dim,
            .wq = wq,
            .wk = wk,
            .wv = wv,
            .wo = wo,
            .bq = bq,
            .bk = bk,
            .bv = bv,
            .bo = bo,
        };
    }

    pub fn deinit(self: *MultiHeadAttention) void {
        self.allocator.free(self.wq);
        self.allocator.free(self.wk);
        self.allocator.free(self.wv);
        self.allocator.free(self.wo);
        self.allocator.free(self.bq);
        self.allocator.free(self.bk);
        self.allocator.free(self.bv);
        self.allocator.free(self.bo);
    }

    /// Forward pass: [seq_len, hidden] -> [seq_len, hidden]
    pub fn forward(self: *const MultiHeadAttention, x: []const f32, seq_len: usize) ![]f32 {
        const hidden = self.hidden_size;
        const num_heads = self.num_heads;
        const head_dim = self.head_dim;

        // Allocate Q, K, V
        const q = try self.allocator.alloc(f32, seq_len * hidden);
        defer self.allocator.free(q);
        const k = try self.allocator.alloc(f32, seq_len * hidden);
        defer self.allocator.free(k);
        const v = try self.allocator.alloc(f32, seq_len * hidden);
        defer self.allocator.free(v);

        // Linear projections
        for (0..seq_len) |s| {
            for (0..hidden) |h| {
                var q_sum: f32 = self.bq[h];
                var k_sum: f32 = self.bk[h];
                var v_sum: f32 = self.bv[h];

                for (0..hidden) |i| {
                    const x_val = x[s * hidden + i];
                    q_sum += x_val * self.wq[h * hidden + i];
                    k_sum += x_val * self.wk[h * hidden + i];
                    v_sum += x_val * self.wv[h * hidden + i];
                }

                q[s * hidden + h] = q_sum;
                k[s * hidden + h] = k_sum;
                v[s * hidden + h] = v_sum;
            }
        }

        // Allocate attention output
        const attn_out = try self.allocator.alloc(f32, seq_len * hidden);
        errdefer self.allocator.free(attn_out);
        @memset(attn_out, 0.0);

        // Compute attention for each head
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        // Temporary attention scores
        const scores = try self.allocator.alloc(f32, seq_len);
        defer self.allocator.free(scores);

        for (0..num_heads) |head| {
            const offset = head * head_dim;

            for (0..seq_len) |i| {
                // Compute attention scores for position i
                for (0..seq_len) |j| {
                    var dot: f32 = 0.0;
                    for (0..head_dim) |d| {
                        dot += q[i * hidden + offset + d] * k[j * hidden + offset + d];
                    }
                    scores[j] = dot * scale;
                }

                // Softmax
                softmax(scores);

                // Weighted sum of values
                for (0..head_dim) |d| {
                    var sum: f32 = 0.0;
                    for (0..seq_len) |j| {
                        sum += scores[j] * v[j * hidden + offset + d];
                    }
                    attn_out[i * hidden + offset + d] = sum;
                }
            }
        }

        // Output projection
        const output = try self.allocator.alloc(f32, seq_len * hidden);
        for (0..seq_len) |s| {
            for (0..hidden) |h| {
                var sum: f32 = self.bo[h];
                for (0..hidden) |i| {
                    sum += attn_out[s * hidden + i] * self.wo[h * hidden + i];
                }
                output[s * hidden + h] = sum;
            }
        }

        self.allocator.free(attn_out);
        return output;
    }
};
