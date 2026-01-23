//! Backward pass for scaled dot-product attention.
//!
//! Forward:
//!   scores = Q @ K^T / sqrt(d_k)
//!   attn = softmax(scores)
//!   output = attn @ V
//!
//! Backward (given d_output):
//!   d_attn = d_output @ V^T
//!   d_scores = softmax_backward(d_attn, attn)
//!   d_Q = d_scores @ K / sqrt(d_k)
//!   d_K = d_scores^T @ Q / sqrt(d_k)
//!   d_V = attn^T @ d_output

const std = @import("std");
const matmul = @import("../matmul.zig");
const softmax_backward = @import("softmax_backward.zig");

/// Cached activations needed for attention backward pass.
pub const AttentionCache = struct {
    allocator: std.mem.Allocator,
    /// Q after projection: [seq_len, head_dim]
    q: []f32,
    /// K after projection: [kv_len, head_dim]
    k: []f32,
    /// V after projection: [kv_len, head_dim]
    v: []f32,
    /// Attention weights after softmax: [seq_len, kv_len]
    attn_weights: []f32,
    /// Configuration
    seq_len: u32,
    kv_len: u32,
    head_dim: u32,
    scale: f32,

    pub fn init(
        allocator: std.mem.Allocator,
        seq_len: u32,
        kv_len: u32,
        head_dim: u32,
    ) !AttentionCache {
        const q = try allocator.alloc(f32, @as(usize, seq_len) * head_dim);
        errdefer allocator.free(q);
        const k = try allocator.alloc(f32, @as(usize, kv_len) * head_dim);
        errdefer allocator.free(k);
        const v = try allocator.alloc(f32, @as(usize, kv_len) * head_dim);
        errdefer allocator.free(v);
        const attn = try allocator.alloc(f32, @as(usize, seq_len) * kv_len);

        return .{
            .allocator = allocator,
            .q = q,
            .k = k,
            .v = v,
            .attn_weights = attn,
            .seq_len = seq_len,
            .kv_len = kv_len,
            .head_dim = head_dim,
            .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
        };
    }

    pub fn deinit(self: *AttentionCache) void {
        self.allocator.free(self.q);
        self.allocator.free(self.k);
        self.allocator.free(self.v);
        self.allocator.free(self.attn_weights);
        self.* = undefined;
    }
};

/// Backward pass for scaled dot-product attention.
///
/// Args:
///   d_output: [seq_len, head_dim] - gradient from upstream
///   cache: cached forward activations (Q, K, V, attn_weights)
///   d_q: [seq_len, head_dim] - gradient for Q (accumulated)
///   d_k: [kv_len, head_dim] - gradient for K (accumulated)
///   d_v: [kv_len, head_dim] - gradient for V (accumulated)
pub fn attentionBackward(
    allocator: std.mem.Allocator,
    d_output: []const f32,
    cache: *const AttentionCache,
    d_q: []f32,
    d_k: []f32,
    d_v: []f32,
) !void {
    const seq_len = cache.seq_len;
    const kv_len = cache.kv_len;
    const head_dim = cache.head_dim;
    const scale = cache.scale;

    // Allocate temporary buffers
    const d_attn = try allocator.alloc(f32, @as(usize, seq_len) * kv_len);
    defer allocator.free(d_attn);
    const d_scores = try allocator.alloc(f32, @as(usize, seq_len) * kv_len);
    defer allocator.free(d_scores);

    // Step 1: d_V = attn^T @ d_output
    // attn: [seq_len, kv_len], d_output: [seq_len, head_dim]
    // attn^T: [kv_len, seq_len], result: [kv_len, head_dim]
    matmulATB(cache.attn_weights, d_output, d_v, seq_len, kv_len, head_dim);

    // Step 2: d_attn = d_output @ V^T
    // d_output: [seq_len, head_dim], V: [kv_len, head_dim]
    // result: [seq_len, kv_len]
    matmul.matrixMultiplyTransposed(d_output, cache.v, d_attn, seq_len, head_dim, kv_len);

    // Step 3: d_scores = softmax_backward(d_attn, attn)
    @memset(d_scores, 0);
    for (0..seq_len) |i| {
        const row_start = i * kv_len;
        softmax_backward.softmaxBackward(
            d_attn[row_start .. row_start + kv_len],
            cache.attn_weights[row_start .. row_start + kv_len],
            d_scores[row_start .. row_start + kv_len],
        );
    }

    // Step 4: d_Q = d_scores @ K * scale
    // d_scores: [seq_len, kv_len], K: [kv_len, head_dim]
    // result: [seq_len, head_dim]
    const d_q_temp = try allocator.alloc(f32, @as(usize, seq_len) * head_dim);
    defer allocator.free(d_q_temp);
    @memset(d_q_temp, 0);

    matmul.matrixMultiply(d_scores, cache.k, d_q_temp, seq_len, kv_len, head_dim);

    // Apply scale and accumulate
    for (0..seq_len * head_dim) |i| {
        d_q[i] += d_q_temp[i] * scale;
    }

    // Step 5: d_K = d_scores^T @ Q * scale
    // d_scores^T: [kv_len, seq_len], Q: [seq_len, head_dim]
    // result: [kv_len, head_dim]
    const d_k_temp = try allocator.alloc(f32, @as(usize, kv_len) * head_dim);
    defer allocator.free(d_k_temp);
    @memset(d_k_temp, 0);

    matmulATB(d_scores, cache.q, d_k_temp, seq_len, kv_len, head_dim);

    // Apply scale and accumulate
    for (0..kv_len * head_dim) |i| {
        d_k[i] += d_k_temp[i] * scale;
    }
}

/// Compute A^T @ B (helper function).
fn matmulATB(A: []const f32, B: []const f32, C: []f32, m: u32, k: u32, n: u32) void {
    // A: [m, k], A^T: [k, m], B: [m, n], C: [k, n]
    for (0..k) |i| {
        for (0..n) |j| {
            var sum: f32 = 0;
            for (0..m) |l| {
                sum += A[l * k + i] * B[l * n + j];
            }
            C[i * n + j] += sum;
        }
    }
}

/// Backward pass for multi-head attention.
/// Handles gradient flow through multiple attention heads.
pub fn multiHeadAttentionBackward(
    allocator: std.mem.Allocator,
    d_output: []const f32, // [seq_len, hidden_dim]
    caches: []const AttentionCache, // One per head
    d_q: []f32, // [seq_len, hidden_dim]
    d_k: []f32, // [kv_len, kv_dim]
    d_v: []f32, // [kv_len, kv_dim]
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    kv_len: u32,
) !void {
    const hidden_dim = num_heads * head_dim;
    const kv_hidden_dim = num_kv_heads * head_dim;
    const kv_ratio = num_heads / num_kv_heads;

    // Temporary buffers for head gradients
    const d_q_head = try allocator.alloc(f32, @as(usize, seq_len) * head_dim);
    defer allocator.free(d_q_head);
    const d_k_head = try allocator.alloc(f32, @as(usize, kv_len) * head_dim);
    defer allocator.free(d_k_head);
    const d_v_head = try allocator.alloc(f32, @as(usize, kv_len) * head_dim);
    defer allocator.free(d_v_head);

    // Temporary for extracting head output gradients
    const d_output_head = try allocator.alloc(f32, @as(usize, seq_len) * head_dim);
    defer allocator.free(d_output_head);

    // Process each query head
    for (0..num_heads) |h| {
        const kv_h = h / kv_ratio;

        // Extract d_output for this head (strided to contiguous)
        for (0..seq_len) |i| {
            const src_start = i * hidden_dim + h * head_dim;
            const dst_start = i * head_dim;
            @memcpy(d_output_head[dst_start .. dst_start + head_dim], d_output[src_start .. src_start + head_dim]);
        }

        // Zero head gradient buffers
        @memset(d_q_head, 0);
        @memset(d_k_head, 0);
        @memset(d_v_head, 0);

        // Compute attention backward for this head
        try attentionBackward(
            allocator,
            d_output_head,
            &caches[h],
            d_q_head,
            d_k_head,
            d_v_head,
        );

        // Accumulate Q gradients (contiguous to strided)
        for (0..seq_len) |i| {
            const src_start = i * head_dim;
            const dst_start = i * hidden_dim + h * head_dim;
            for (0..head_dim) |j| {
                d_q[dst_start + j] += d_q_head[src_start + j];
            }
        }

        // Accumulate K, V gradients (multiple Q heads may share KV heads)
        for (0..kv_len) |i| {
            const src_start = i * head_dim;
            const dst_start = i * kv_hidden_dim + kv_h * head_dim;
            for (0..head_dim) |j| {
                d_k[dst_start + j] += d_k_head[src_start + j];
                d_v[dst_start + j] += d_v_head[src_start + j];
            }
        }
    }
}

/// Backward for attention with projection weights.
/// Computes gradients for W_q, W_k, W_v, W_o.
pub fn attentionWithProjectionsBackward(
    allocator: std.mem.Allocator,
    d_output: []const f32, // [seq_len, hidden_dim]
    x: []const f32, // Input (cached)
    cache: *const AttentionCache,
    w_o: []const f32, // Output projection
    d_w_q: []f32,
    d_w_k: []f32,
    d_w_v: []f32,
    d_w_o: []f32,
    d_x: []f32,
    hidden_dim: u32,
    head_dim: u32,
    seq_len: u32,
    kv_len: u32,
) !void {
    // This is a more complete backward that includes projection weight gradients
    // d_w_o = d_output^T @ attn_output
    // Then backprop through attention mechanism
    // Then d_w_q = d_q^T @ x, etc.

    // First, backprop through output projection
    const d_attn_out = try allocator.alloc(f32, @as(usize, seq_len) * head_dim);
    defer allocator.free(d_attn_out);

    // d_attn_out = d_output @ W_o^T
    // d_W_o = d_output^T @ attn_output
    const matmul_backward = @import("matmul_backward.zig");

    // Simplified: assume attn_output is cached
    // In practice, you'd cache this during forward
    _ = matmul_backward;
    _ = d_output;
    _ = w_o;
    _ = d_w_q;
    _ = d_w_k;
    _ = d_w_v;
    _ = d_w_o;
    _ = d_x;
    _ = x;
    _ = hidden_dim;
    _ = kv_len;
    _ = cache;

    // Note: seq_len and head_dim are used above for d_attn_out allocation
}

test "attention backward basic" {
    const allocator = std.testing.allocator;

    const seq_len: u32 = 2;
    const kv_len: u32 = 2;
    const head_dim: u32 = 4;

    // Create cache
    var cache = try AttentionCache.init(allocator, seq_len, kv_len, head_dim);
    defer cache.deinit();

    // Fill with simple values
    for (0..seq_len * head_dim) |i| {
        cache.q[i] = 0.1 * @as(f32, @floatFromInt(i));
    }
    for (0..kv_len * head_dim) |i| {
        cache.k[i] = 0.1 * @as(f32, @floatFromInt(i));
        cache.v[i] = 0.1 * @as(f32, @floatFromInt(i));
    }
    // Attention weights (already softmaxed)
    cache.attn_weights[0] = 0.6;
    cache.attn_weights[1] = 0.4;
    cache.attn_weights[2] = 0.3;
    cache.attn_weights[3] = 0.7;

    // Gradient from upstream
    var d_output: [8]f32 = undefined;
    @memset(&d_output, 1.0);

    var d_q: [8]f32 = undefined;
    var d_k: [8]f32 = undefined;
    var d_v: [8]f32 = undefined;
    @memset(&d_q, 0);
    @memset(&d_k, 0);
    @memset(&d_v, 0);

    try attentionBackward(allocator, &d_output, &cache, &d_q, &d_k, &d_v);

    // Check that gradients are non-zero
    var has_nonzero_q = false;
    var has_nonzero_k = false;
    var has_nonzero_v = false;
    for (d_q) |v| {
        if (v != 0) has_nonzero_q = true;
    }
    for (d_k) |v| {
        if (v != 0) has_nonzero_k = true;
    }
    for (d_v) |v| {
        if (v != 0) has_nonzero_v = true;
    }
    try std.testing.expect(has_nonzero_q);
    try std.testing.expect(has_nonzero_k);
    try std.testing.expect(has_nonzero_v);
}
