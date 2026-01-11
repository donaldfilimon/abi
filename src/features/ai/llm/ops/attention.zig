//! Self-attention implementation for transformer models.
//!
//! Implements scaled dot-product attention with causal masking,
//! multi-head attention, and grouped-query attention (GQA).

const std = @import("std");
const matmul = @import("matmul.zig");
const activations = @import("activations.zig");

/// Self-attention configuration.
pub const AttentionConfig = struct {
    /// Hidden dimension
    hidden_dim: u32,
    /// Number of attention heads
    num_heads: u32,
    /// Number of key-value heads (for GQA)
    num_kv_heads: u32,
    /// Head dimension (hidden_dim / num_heads)
    head_dim: u32,
    /// Whether to use causal masking
    causal: bool = true,
    /// Attention dropout (0 = disabled)
    dropout: f32 = 0.0,

    pub fn fromModel(hidden_dim: u32, num_heads: u32, num_kv_heads: u32) AttentionConfig {
        return .{
            .hidden_dim = hidden_dim,
            .num_heads = num_heads,
            .num_kv_heads = num_kv_heads,
            .head_dim = hidden_dim / num_heads,
            .causal = true,
            .dropout = 0.0,
        };
    }
};

/// Scaled dot-product attention.
/// Q: [seq_len, head_dim]
/// K: [seq_len, head_dim] or [kv_len, head_dim] for KV cache
/// V: [seq_len, head_dim] or [kv_len, head_dim]
/// Output: [seq_len, head_dim]
pub fn scaledDotProductAttention(
    allocator: std.mem.Allocator,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    output: []f32,
    seq_len: u32,
    kv_len: u32,
    head_dim: u32,
    causal: bool,
) !void {
    // Scaling factor
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    // Compute attention scores: QK^T / sqrt(d)
    const scores = try allocator.alloc(f32, @as(usize, seq_len) * kv_len);
    defer allocator.free(scores);

    // Q @ K^T
    matmul.matrixMultiplyTransposed(q, k, scores, seq_len, head_dim, kv_len);

    // Scale scores
    for (scores) |*s| {
        s.* *= scale;
    }

    // Apply causal mask (set future positions to -inf)
    if (causal) {
        applyCausalMask(scores, seq_len, kv_len);
    }

    // Softmax over last dimension (per row)
    for (0..seq_len) |i| {
        const row_start = i * kv_len;
        activations.softmaxInPlace(scores[row_start .. row_start + kv_len]);
    }

    // Output = Attention @ V
    matmul.matrixMultiply(scores, v, output, seq_len, kv_len, head_dim);
}

/// Apply causal mask to attention scores.
fn applyCausalMask(scores: []f32, seq_len: u32, kv_len: u32) void {
    const neg_inf = -std.math.inf(f32);

    // For each query position, mask out future key positions
    for (0..seq_len) |q_pos| {
        // Keys at positions > q_pos + (kv_len - seq_len) are in the future
        const last_valid_k = q_pos + (kv_len - seq_len);

        for (last_valid_k + 1..kv_len) |k_pos| {
            scores[q_pos * kv_len + k_pos] = neg_inf;
        }
    }
}

/// Multi-head attention.
pub fn multiHeadAttention(
    allocator: std.mem.Allocator,
    q: []const f32, // [seq_len, num_heads * head_dim]
    k: []const f32, // [kv_len, num_kv_heads * head_dim]
    v: []const f32, // [kv_len, num_kv_heads * head_dim]
    output: []f32, // [seq_len, num_heads * head_dim]
    config: AttentionConfig,
    seq_len: u32,
    kv_len: u32,
) !void {
    const head_dim = config.head_dim;
    const num_heads = config.num_heads;
    const num_kv_heads = config.num_kv_heads;

    // Ratio for grouped-query attention
    const kv_ratio = num_heads / num_kv_heads;

    // Temporary storage for one head's attention
    const head_output = try allocator.alloc(f32, @as(usize, seq_len) * head_dim);
    defer allocator.free(head_output);

    // Process each query head
    for (0..num_heads) |h| {
        // Determine which KV head to use
        const kv_h = h / kv_ratio;

        // Extract Q slice for this head
        const q_offset = h * head_dim;
        const k_offset = kv_h * head_dim;
        const v_offset = kv_h * head_dim;

        // Create views (strided access)
        var q_head = try allocator.alloc(f32, @as(usize, seq_len) * head_dim);
        defer allocator.free(q_head);
        var k_head = try allocator.alloc(f32, @as(usize, kv_len) * head_dim);
        defer allocator.free(k_head);
        var v_head = try allocator.alloc(f32, @as(usize, kv_len) * head_dim);
        defer allocator.free(v_head);

        // Copy head slices (strided to contiguous)
        for (0..seq_len) |i| {
            const src_start = i * config.hidden_dim + q_offset;
            const dst_start = i * head_dim;
            @memcpy(q_head[dst_start .. dst_start + head_dim], q[src_start .. src_start + head_dim]);
        }

        for (0..kv_len) |i| {
            const k_src = i * (num_kv_heads * head_dim) + k_offset;
            const v_src = i * (num_kv_heads * head_dim) + v_offset;
            const dst = i * head_dim;
            @memcpy(k_head[dst .. dst + head_dim], k[k_src .. k_src + head_dim]);
            @memcpy(v_head[dst .. dst + head_dim], v[v_src .. v_src + head_dim]);
        }

        // Compute attention for this head
        try scaledDotProductAttention(
            allocator,
            q_head,
            k_head,
            v_head,
            head_output,
            seq_len,
            kv_len,
            head_dim,
            config.causal,
        );

        // Copy result to output (contiguous to strided)
        for (0..seq_len) |i| {
            const src_start = i * head_dim;
            const dst_start = i * config.hidden_dim + q_offset;
            @memcpy(output[dst_start .. dst_start + head_dim], head_output[src_start .. src_start + head_dim]);
        }
    }
}

/// Self-attention for a single position (incremental decoding).
pub fn selfAttention(
    allocator: std.mem.Allocator,
    q: []const f32, // [1, hidden_dim]
    k_cache: []const f32, // [cache_len, kv_dim]
    v_cache: []const f32, // [cache_len, kv_dim]
    output: []f32, // [1, hidden_dim]
    config: AttentionConfig,
    cache_len: u32,
) !void {
    // For single-position attention, seq_len = 1
    try multiHeadAttention(
        allocator,
        q,
        k_cache,
        v_cache,
        output,
        config,
        1,
        cache_len,
    );
}

/// Compute attention scores for visualization/debugging.
pub fn computeAttentionScores(
    allocator: std.mem.Allocator,
    q: []const f32,
    k: []const f32,
    head_dim: u32,
    seq_len: u32,
    kv_len: u32,
) ![]f32 {
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    const scores = try allocator.alloc(f32, @as(usize, seq_len) * kv_len);
    matmul.matrixMultiplyTransposed(q, k, scores, seq_len, head_dim, kv_len);

    for (scores) |*s| {
        s.* *= scale;
    }

    return scores;
}

test "scaled dot product attention" {
    const allocator = std.testing.allocator;

    // Simple 2x4 Q, K, V
    const q = [_]f32{ 1, 0, 0, 0, 0, 1, 0, 0 };
    const k = [_]f32{ 1, 0, 0, 0, 0, 1, 0, 0 };
    const v = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var output: [8]f32 = undefined;

    try scaledDotProductAttention(
        allocator,
        &q,
        &k,
        &v,
        &output,
        2,
        2,
        4,
        false,
    );

    // With identity-like Q, K, output should be weighted sum of V rows
    try std.testing.expect(output[0] > 0);
}

test "causal mask" {
    var scores = [_]f32{ 1, 1, 1, 1, 1, 1, 1, 1, 1 }; // 3x3

    applyCausalMask(&scores, 3, 3);

    // Row 0: only position 0 visible
    try std.testing.expect(scores[1] == -std.math.inf(f32));
    try std.testing.expect(scores[2] == -std.math.inf(f32));

    // Row 1: positions 0, 1 visible
    try std.testing.expect(scores[3] != -std.math.inf(f32));
    try std.testing.expect(scores[4] != -std.math.inf(f32));
    try std.testing.expect(scores[5] == -std.math.inf(f32));

    // Row 2: all visible
    try std.testing.expect(scores[6] != -std.math.inf(f32));
}
