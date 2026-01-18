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

// ============================================================================
// Flash Attention Implementation
// ============================================================================
//
// Flash Attention is a memory-efficient attention algorithm that computes
// attention without materializing the full N×N attention matrix. Instead,
// it processes Q, K, V in blocks using online softmax normalization.
//
// Memory complexity: O(N) instead of O(N²)
// References: "FlashAttention: Fast and Memory-Efficient Exact Attention"
//             (Dao et al., 2022)

/// Flash Attention configuration.
pub const FlashAttentionConfig = struct {
    /// Block size for Q (rows processed per iteration)
    block_size_q: u32 = 64,
    /// Block size for KV (columns processed per iteration)
    block_size_kv: u32 = 64,
    /// Whether to use causal masking
    causal: bool = true,
    /// Scaling factor (typically 1/sqrt(head_dim))
    scale: ?f32 = null,
};

/// Flash Attention: Memory-efficient scaled dot-product attention.
///
/// Computes attention without materializing the full N×N attention matrix.
/// Uses tiled computation with online softmax normalization.
///
/// Q: [seq_len, head_dim]
/// K: [kv_len, head_dim]
/// V: [kv_len, head_dim]
/// Output: [seq_len, head_dim]
pub fn flashAttention(
    allocator: std.mem.Allocator,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    output: []f32,
    seq_len: u32,
    kv_len: u32,
    head_dim: u32,
    config: FlashAttentionConfig,
) !void {
    const scale = config.scale orelse (1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))));
    const block_q = @min(config.block_size_q, seq_len);
    const block_kv = @min(config.block_size_kv, kv_len);

    // Allocate working buffers for one block
    const scores_block = try allocator.alloc(f32, @as(usize, block_q) * block_kv);
    defer allocator.free(scores_block);

    // Per-row running statistics for online softmax
    const row_max = try allocator.alloc(f32, seq_len);
    defer allocator.free(row_max);
    const row_sum = try allocator.alloc(f32, seq_len);
    defer allocator.free(row_sum);

    // Initialize output to zero, max to -inf, sum to 0
    @memset(output, 0);
    for (row_max) |*m| m.* = -std.math.inf(f32);
    @memset(row_sum, 0);

    // Process K,V in blocks
    var kv_start: u32 = 0;
    while (kv_start < kv_len) : (kv_start += block_kv) {
        const kv_end = @min(kv_start + block_kv, kv_len);

        // Process Q in blocks
        var q_start: u32 = 0;
        while (q_start < seq_len) : (q_start += block_q) {
            const q_end = @min(q_start + block_q, seq_len);

            // Skip if causal and all KV positions are in the future
            if (config.causal) {
                const last_valid_kv = q_start + (kv_len - seq_len);
                if (kv_start > last_valid_kv) continue;
            }

            // Compute S_ij = Q_i @ K_j^T * scale for this block
            computeBlockScores(
                q,
                k,
                scores_block,
                q_start,
                q_end,
                kv_start,
                kv_end,
                head_dim,
                scale,
                seq_len,
                kv_len,
                config.causal,
            );

            // Update output using online softmax algorithm
            updateOutputOnlineSoftmax(
                output,
                v,
                scores_block,
                row_max,
                row_sum,
                q_start,
                q_end,
                kv_start,
                kv_end,
                head_dim,
                kv_len,
            );
        }
    }

    // Final normalization: output /= row_sum
    for (0..seq_len) |i| {
        const inv_sum = if (row_sum[i] > 0) 1.0 / row_sum[i] else 0.0;
        const row_start = i * head_dim;
        for (output[row_start .. row_start + head_dim]) |*val| {
            val.* *= inv_sum;
        }
    }
}

/// Compute block attention scores: S = Q @ K^T * scale
fn computeBlockScores(
    q: []const f32,
    k: []const f32,
    scores: []f32,
    q_start: u32,
    q_end: u32,
    kv_start: u32,
    kv_end: u32,
    head_dim: u32,
    scale: f32,
    seq_len: u32,
    kv_len: u32,
    causal: bool,
) void {
    const neg_inf = -std.math.inf(f32);
    const curr_q_len = q_end - q_start;
    const curr_kv_len = kv_end - kv_start;

    for (0..curr_q_len) |qi| {
        const global_q = q_start + @as(u32, @intCast(qi));
        const q_row = (q_start + @as(u32, @intCast(qi))) * head_dim;

        for (0..curr_kv_len) |ki| {
            const global_k = kv_start + @as(u32, @intCast(ki));
            const k_row = (kv_start + @as(u32, @intCast(ki))) * head_dim;

            // Check causal mask
            if (causal) {
                const last_valid_k = global_q + (kv_len - seq_len);
                if (global_k > last_valid_k) {
                    scores[qi * curr_kv_len + ki] = neg_inf;
                    continue;
                }
            }

            // Dot product Q[qi] @ K[ki]
            var dot: f32 = 0;
            for (0..head_dim) |d| {
                dot += q[q_row + d] * k[k_row + d];
            }
            scores[qi * curr_kv_len + ki] = dot * scale;
        }
    }
}

/// Update output using online softmax (Flash Attention algorithm).
/// This allows computing softmax incrementally without storing the full matrix.
fn updateOutputOnlineSoftmax(
    output: []f32,
    v: []const f32,
    scores: []const f32,
    row_max: []f32,
    row_sum: []f32,
    q_start: u32,
    q_end: u32,
    kv_start: u32,
    kv_end: u32,
    head_dim: u32,
    kv_len: u32,
) void {
    _ = kv_len;
    const curr_q_len = q_end - q_start;
    const curr_kv_len = kv_end - kv_start;

    for (0..curr_q_len) |qi| {
        const global_q = q_start + @as(u32, @intCast(qi));
        const out_row = global_q * head_dim;

        // Find max in this block for this row
        var block_max: f32 = -std.math.inf(f32);
        for (0..curr_kv_len) |ki| {
            block_max = @max(block_max, scores[qi * curr_kv_len + ki]);
        }

        // Compute new global max
        const prev_max = row_max[global_q];
        const new_max = @max(prev_max, block_max);

        // Rescale previous accumulated values if max changed
        if (prev_max != -std.math.inf(f32) and new_max > prev_max) {
            const rescale = @exp(prev_max - new_max);
            row_sum[global_q] *= rescale;
            for (output[out_row .. out_row + head_dim]) |*val| {
                val.* *= rescale;
            }
        }

        // Accumulate exp(score - new_max) * V for this block
        var block_sum: f32 = 0;
        for (0..curr_kv_len) |ki| {
            const score = scores[qi * curr_kv_len + ki];
            if (score == -std.math.inf(f32)) continue;

            const exp_score = @exp(score - new_max);
            block_sum += exp_score;

            // Accumulate V weighted by attention
            const v_row = (kv_start + @as(u32, @intCast(ki))) * head_dim;
            for (0..head_dim) |d| {
                output[out_row + d] += exp_score * v[v_row + d];
            }
        }

        row_max[global_q] = new_max;
        row_sum[global_q] += block_sum;
    }
}

/// Flash Attention for multi-head attention.
/// More memory efficient than standard multiHeadAttention for long sequences.
pub fn flashMultiHeadAttention(
    allocator: std.mem.Allocator,
    q: []const f32, // [seq_len, num_heads * head_dim]
    k: []const f32, // [kv_len, num_kv_heads * head_dim]
    v: []const f32, // [kv_len, num_kv_heads * head_dim]
    output: []f32, // [seq_len, num_heads * head_dim]
    config: AttentionConfig,
    seq_len: u32,
    kv_len: u32,
    flash_config: FlashAttentionConfig,
) !void {
    const head_dim = config.head_dim;
    const num_heads = config.num_heads;
    const num_kv_heads = config.num_kv_heads;
    const kv_ratio = num_heads / num_kv_heads;

    // Temporary storage for one head
    const head_output = try allocator.alloc(f32, @as(usize, seq_len) * head_dim);
    defer allocator.free(head_output);

    var q_head = try allocator.alloc(f32, @as(usize, seq_len) * head_dim);
    defer allocator.free(q_head);
    var k_head = try allocator.alloc(f32, @as(usize, kv_len) * head_dim);
    defer allocator.free(k_head);
    var v_head = try allocator.alloc(f32, @as(usize, kv_len) * head_dim);
    defer allocator.free(v_head);

    for (0..num_heads) |h| {
        const kv_h = h / kv_ratio;
        const q_offset = h * head_dim;
        const k_offset = kv_h * head_dim;

        // Extract head slices
        for (0..seq_len) |i| {
            const src = i * config.hidden_dim + q_offset;
            const dst = i * head_dim;
            @memcpy(q_head[dst .. dst + head_dim], q[src .. src + head_dim]);
        }

        for (0..kv_len) |i| {
            const k_src = i * (num_kv_heads * head_dim) + k_offset;
            const v_src = i * (num_kv_heads * head_dim) + k_offset;
            const dst = i * head_dim;
            @memcpy(k_head[dst .. dst + head_dim], k[k_src .. k_src + head_dim]);
            @memcpy(v_head[dst .. dst + head_dim], v[v_src .. v_src + head_dim]);
        }

        // Use Flash Attention for this head
        try flashAttention(
            allocator,
            q_head,
            k_head,
            v_head,
            head_output,
            seq_len,
            kv_len,
            head_dim,
            flash_config,
        );

        // Copy back to output
        for (0..seq_len) |i| {
            const src = i * head_dim;
            const dst = i * config.hidden_dim + q_offset;
            @memcpy(output[dst .. dst + head_dim], head_output[src .. src + head_dim]);
        }
    }
}

test "flash attention basic" {
    const allocator = std.testing.allocator;

    // Simple 2x4 Q, K, V (same as standard attention test)
    const q = [_]f32{ 1, 0, 0, 0, 0, 1, 0, 0 };
    const k = [_]f32{ 1, 0, 0, 0, 0, 1, 0, 0 };
    const v = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var output: [8]f32 = undefined;

    try flashAttention(
        allocator,
        &q,
        &k,
        &v,
        &output,
        2,
        2,
        4,
        .{ .causal = false, .block_size_q = 1, .block_size_kv = 1 },
    );

    // Output should match standard attention
    try std.testing.expect(output[0] > 0);
}

test "flash attention matches standard" {
    const allocator = std.testing.allocator;

    // Test with larger matrices
    const seq_len: u32 = 4;
    const head_dim: u32 = 8;
    const size = @as(usize, seq_len) * head_dim;

    var q = try allocator.alloc(f32, size);
    defer allocator.free(q);
    var k = try allocator.alloc(f32, size);
    defer allocator.free(k);
    var v = try allocator.alloc(f32, size);
    defer allocator.free(v);
    const output_std = try allocator.alloc(f32, size);
    defer allocator.free(output_std);
    const output_flash = try allocator.alloc(f32, size);
    defer allocator.free(output_flash);

    // Initialize with test data
    for (0..size) |i| {
        q[i] = @as(f32, @floatFromInt(i % 7)) * 0.1;
        k[i] = @as(f32, @floatFromInt((i + 3) % 7)) * 0.1;
        v[i] = @as(f32, @floatFromInt((i + 5) % 7)) * 0.1;
    }

    // Standard attention
    try scaledDotProductAttention(allocator, q, k, v, output_std, seq_len, seq_len, head_dim, false);

    // Flash attention
    try flashAttention(allocator, q, k, v, output_flash, seq_len, seq_len, head_dim, .{
        .causal = false,
        .block_size_q = 2,
        .block_size_kv = 2,
    });

    // Results should be close (within floating point tolerance)
    for (0..size) |i| {
        const diff = @abs(output_std[i] - output_flash[i]);
        try std.testing.expect(diff < 1e-4);
    }
}
