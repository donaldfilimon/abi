//! Abbey Attention Mechanisms
//!
//! Self-attention and cross-attention for dynamic context understanding.
//! Supports adaptive attention patterns that evolve during conversations.

const std = @import("std");
const tensor = @import("tensor.zig");
const layer = @import("layer.zig");
const types = @import("../core/types.zig");

const F32Tensor = tensor.F32Tensor;
const LinearLayer = layer.LinearLayer;

// ============================================================================
// Scaled Dot-Product Attention
// ============================================================================

/// Compute attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
pub fn scaledDotProductAttention(
    _: std.mem.Allocator,
    query: *const F32Tensor,
    key: *const F32Tensor,
    value: *const F32Tensor,
    mask: ?*const F32Tensor,
) !AttentionOutput {
    const d_k = query.shape[query.shape.len - 1];
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d_k)));

    // Compute QK^T
    var key_t = try key.transpose();
    defer key_t.deinit();

    var scores = try query.matmul(&key_t);
    defer scores.deinit();

    // Scale
    scores.scaleInPlace(scale);

    // Apply mask if provided
    if (mask) |m| {
        for (0..scores.data.len) |i| {
            if (m.data[i] == 0) {
                scores.data[i] = -1e9; // Large negative for softmax
            }
        }
    }

    // Softmax
    var attention_weights = try scores.softmax();
    errdefer attention_weights.deinit();

    // Apply attention to values
    const output = try attention_weights.matmul(value);

    return AttentionOutput{
        .output = output,
        .attention_weights = attention_weights,
    };
}

pub const AttentionOutput = struct {
    output: F32Tensor,
    attention_weights: F32Tensor,

    pub fn deinit(self: *AttentionOutput) void {
        self.output.deinit();
        self.attention_weights.deinit();
    }
};

// ============================================================================
// Multi-Head Helper Functions
// ============================================================================

/// Extract a slice of features corresponding to one attention head
/// Input: [batch_size, d_model], Output: [batch_size, d_k]
fn extractHeadSlice(
    allocator: std.mem.Allocator,
    tensor_in: *const F32Tensor,
    batch_size: usize,
    start_dim: usize,
    end_dim: usize,
) !F32Tensor {
    const d_k = end_dim - start_dim;
    const d_model = tensor_in.shape[1];

    var head_data = try allocator.alloc(f32, batch_size * d_k);
    errdefer allocator.free(head_data);

    // Copy data for this head's dimensions
    for (0..batch_size) |b| {
        const src_offset = b * d_model + start_dim;
        const dst_offset = b * d_k;
        @memcpy(head_data[dst_offset..][0..d_k], tensor_in.data[src_offset..][0..d_k]);
    }

    return F32Tensor.fromSlice(allocator, head_data, &.{ batch_size, d_k });
}

/// Merge outputs from all heads back into full d_model dimension
/// Input: array of [batch_size, d_k] tensors, Output: [batch_size, d_model]
fn mergeHeadOutputs(
    allocator: std.mem.Allocator,
    head_outputs: []const F32Tensor,
    batch_size: usize,
    d_model: usize,
) !F32Tensor {
    const num_heads = head_outputs.len;
    const d_k = d_model / num_heads;

    var merged_data = try allocator.alloc(f32, batch_size * d_model);
    errdefer allocator.free(merged_data);

    // Copy each head's output into the corresponding dimension range
    for (head_outputs, 0..) |head_out, h| {
        const head_start = h * d_k;
        for (0..batch_size) |b| {
            const src_offset = b * d_k;
            const dst_offset = b * d_model + head_start;
            @memcpy(merged_data[dst_offset..][0..d_k], head_out.data[src_offset..][0..d_k]);
        }
    }

    return F32Tensor.fromSlice(allocator, merged_data, &.{ batch_size, d_model });
}

// ============================================================================
// Multi-Head Attention
// ============================================================================

/// Multi-head attention mechanism
pub const MultiHeadAttention = struct {
    allocator: std.mem.Allocator,
    d_model: usize,
    num_heads: usize,
    d_k: usize,
    d_v: usize,

    // Projections
    w_q: LinearLayer,
    w_k: LinearLayer,
    w_v: LinearLayer,
    w_o: LinearLayer,

    // For adaptive learning
    attention_history: std.ArrayListUnmanaged(F32Tensor),
    adapt_weights: bool = false,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, d_model: usize, num_heads: usize) !Self {
        if (d_model % num_heads != 0) return error.InvalidDimensions;

        const d_k = d_model / num_heads;

        var w_q = try LinearLayer.init(allocator, d_model, d_model);
        errdefer w_q.deinit();

        var w_k = try LinearLayer.init(allocator, d_model, d_model);
        errdefer w_k.deinit();

        var w_v = try LinearLayer.init(allocator, d_model, d_model);
        errdefer w_v.deinit();

        var w_o = try LinearLayer.init(allocator, d_model, d_model);
        errdefer w_o.deinit();

        return Self{
            .allocator = allocator,
            .d_model = d_model,
            .num_heads = num_heads,
            .d_k = d_k,
            .d_v = d_k,
            .w_q = w_q,
            .w_k = w_k,
            .w_v = w_v,
            .w_o = w_o,
            .attention_history = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        self.w_q.deinit();
        self.w_k.deinit();
        self.w_v.deinit();
        self.w_o.deinit();
        for (self.attention_history.items) |*h| {
            h.deinit();
        }
        self.attention_history.deinit(self.allocator);
    }

    pub fn forward(
        self: *Self,
        query: *const F32Tensor,
        key: *const F32Tensor,
        value: *const F32Tensor,
        mask: ?*const F32Tensor,
    ) !AttentionOutput {
        const batch_size = query.shape[0];
        const d_model = query.shape[1];

        // Project Q, K, V
        var q_proj = try self.w_q.forward(query);
        defer q_proj.deinit();

        var k_proj = try self.w_k.forward(key);
        defer k_proj.deinit();

        var v_proj = try self.w_v.forward(value);
        defer v_proj.deinit();

        // Multi-head attention: split projections into heads and process each
        // For input shape [batch_size, d_model], we treat it as [batch_size, 1, d_model]
        // and split d_model into [num_heads, d_k]
        var head_outputs = try self.allocator.alloc(F32Tensor, self.num_heads);
        errdefer {
            for (head_outputs) |*ho| {
                ho.deinit();
            }
            self.allocator.free(head_outputs);
        }
        var heads_initialized: usize = 0;

        // Allocate combined attention weights tensor
        var combined_weights = try F32Tensor.zeros(self.allocator, &.{ batch_size, batch_size });
        errdefer combined_weights.deinit();

        // Process each head
        for (0..self.num_heads) |h| {
            const head_start = h * self.d_k;
            const head_end = head_start + self.d_k;

            // Extract head slice from Q, K, V projections
            var q_head = try extractHeadSlice(self.allocator, &q_proj, batch_size, head_start, head_end);
            defer q_head.deinit();

            var k_head = try extractHeadSlice(self.allocator, &k_proj, batch_size, head_start, head_end);
            defer k_head.deinit();

            var v_head = try extractHeadSlice(self.allocator, &v_proj, batch_size, head_start, head_end);
            defer v_head.deinit();

            // Compute attention for this head
            var head_attn = try scaledDotProductAttention(
                self.allocator,
                &q_head,
                &k_head,
                &v_head,
                mask,
            );
            defer head_attn.attention_weights.deinit();

            // Accumulate attention weights (average across heads)
            for (0..combined_weights.data.len) |i| {
                combined_weights.data[i] += head_attn.attention_weights.data[i] / @as(f32, @floatFromInt(self.num_heads));
            }

            head_outputs[h] = head_attn.output;
            heads_initialized = h + 1;
        }

        // Concatenate head outputs back to d_model dimension
        var concat_output = try mergeHeadOutputs(self.allocator, head_outputs, batch_size, d_model);
        defer concat_output.deinit();

        // Clean up individual head outputs
        for (head_outputs) |*ho| {
            ho.deinit();
        }
        self.allocator.free(head_outputs);

        // Store attention for adaptive learning
        if (self.adapt_weights and self.attention_history.items.len < 100) {
            try self.attention_history.append(self.allocator, try combined_weights.clone());
        }

        // Project output
        const output = try self.w_o.forward(&concat_output);

        return AttentionOutput{
            .output = output,
            .attention_weights = combined_weights,
        };
    }

    /// Enable adaptive attention weight learning
    pub fn enableAdaptation(self: *Self) void {
        self.adapt_weights = true;
    }

    /// Adapt attention based on history
    pub fn adaptFromHistory(self: *Self, learning_rate: f32) !void {
        if (self.attention_history.items.len < 10) return;

        // Analyze attention patterns
        var avg_attention = try F32Tensor.zeros(
            self.allocator,
            self.attention_history.items[0].shape,
        );
        defer avg_attention.deinit();

        for (self.attention_history.items) |*hist| {
            try avg_attention.addInPlace(hist);
        }
        avg_attention.scaleInPlace(1.0 / @as(f32, @floatFromInt(self.attention_history.items.len)));

        // Use average pattern to influence weights (simplified)
        // In full implementation, would compute proper gradients
        _ = learning_rate;

        // Clear history after adaptation
        for (self.attention_history.items) |*h| {
            h.deinit();
        }
        self.attention_history.clearRetainingCapacity();
    }

    pub fn zeroGrad(self: *Self) void {
        self.w_q.zeroGrad();
        self.w_k.zeroGrad();
        self.w_v.zeroGrad();
        self.w_o.zeroGrad();
    }
};

// ============================================================================
// Self-Attention Layer
// ============================================================================

/// Self-attention where Q, K, V come from same source
pub const SelfAttention = struct {
    mha: MultiHeadAttention,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, d_model: usize, num_heads: usize) !Self {
        return Self{
            .mha = try MultiHeadAttention.init(allocator, d_model, num_heads),
        };
    }

    pub fn deinit(self: *Self) void {
        self.mha.deinit();
    }

    pub fn forward(self: *Self, x: *const F32Tensor, mask: ?*const F32Tensor) !AttentionOutput {
        return self.mha.forward(x, x, x, mask);
    }
};

// ============================================================================
// Cross-Attention Layer
// ============================================================================

/// Cross-attention between two sequences
pub const CrossAttention = struct {
    mha: MultiHeadAttention,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, d_model: usize, num_heads: usize) !Self {
        return Self{
            .mha = try MultiHeadAttention.init(allocator, d_model, num_heads),
        };
    }

    pub fn deinit(self: *Self) void {
        self.mha.deinit();
    }

    /// x attends to context
    pub fn forward(
        self: *Self,
        x: *const F32Tensor,
        context: *const F32Tensor,
        mask: ?*const F32Tensor,
    ) !AttentionOutput {
        return self.mha.forward(x, context, context, mask);
    }
};

// ============================================================================
// Adaptive Attention (Abbey's unique feature)
// ============================================================================

/// Attention that adapts based on emotional and contextual signals
pub const AdaptiveAttention = struct {
    allocator: std.mem.Allocator,
    base_attention: MultiHeadAttention,

    // Emotional modulation
    emotion_weights: F32Tensor,
    emotion_bias: F32Tensor,

    // Context-aware gating
    gate_weights: F32Tensor,

    // Learning statistics
    queries_seen: usize = 0,
    adaptations: usize = 0,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, d_model: usize, num_heads: usize, num_emotions: usize) !Self {
        var base = try MultiHeadAttention.init(allocator, d_model, num_heads);
        errdefer base.deinit();

        // Emotion-to-attention modulation
        var emotion_weights = try F32Tensor.random(allocator, &.{ num_emotions, d_model }, -0.1, 0.1);
        errdefer emotion_weights.deinit();

        var emotion_bias = try F32Tensor.zeros(allocator, &.{d_model});
        errdefer emotion_bias.deinit();

        // Gating mechanism
        var gate_weights = try F32Tensor.random(allocator, &.{ d_model, 1 }, -0.1, 0.1);
        errdefer gate_weights.deinit();

        return Self{
            .allocator = allocator,
            .base_attention = base,
            .emotion_weights = emotion_weights,
            .emotion_bias = emotion_bias,
            .gate_weights = gate_weights,
        };
    }

    pub fn deinit(self: *Self) void {
        self.base_attention.deinit();
        self.emotion_weights.deinit();
        self.emotion_bias.deinit();
        self.gate_weights.deinit();
    }

    /// Forward with emotional context
    pub fn forward(
        self: *Self,
        query: *const F32Tensor,
        key: *const F32Tensor,
        value: *const F32Tensor,
        emotion_idx: usize,
        mask: ?*const F32Tensor,
    ) !AttentionOutput {
        self.queries_seen += 1;

        // Get emotion modulation vector
        const d_model = self.emotion_weights.shape[1];
        const emb_start = emotion_idx * d_model;

        // Modulate query with emotion (simplified)
        var modulated_query = try query.clone();
        defer modulated_query.deinit();

        for (0..modulated_query.data.len) |i| {
            const dim = i % d_model;
            modulated_query.data[i] += self.emotion_weights.data[emb_start + dim] + self.emotion_bias.data[dim];
        }

        // Compute base attention
        return self.base_attention.forward(&modulated_query, key, value, mask);
    }

    /// Adapt based on feedback
    pub fn adapt(self: *Self, feedback_score: f32, learning_rate: f32) void {
        // Simple adaptation: scale emotion weights based on feedback
        const adjustment = learning_rate * feedback_score;
        for (self.emotion_weights.data) |*w| {
            w.* += adjustment * 0.01; // Small adjustment
        }
        self.adaptations += 1;
    }

    pub fn getStats(self: *const Self) struct { queries: usize, adaptations: usize } {
        return .{
            .queries = self.queries_seen,
            .adaptations = self.adaptations,
        };
    }
};

// ============================================================================
// Causal Mask Generation
// ============================================================================

/// Generate a causal (lower triangular) mask
pub fn createCausalMask(allocator: std.mem.Allocator, seq_len: usize) !F32Tensor {
    var mask = try F32Tensor.init(allocator, &.{ seq_len, seq_len });

    for (0..seq_len) |i| {
        for (0..seq_len) |j| {
            mask.data[i * seq_len + j] = if (j <= i) 1.0 else 0.0;
        }
    }

    return mask;
}

/// Generate a padding mask from lengths
pub fn createPaddingMask(allocator: std.mem.Allocator, lengths: []const usize, max_len: usize) !F32Tensor {
    const batch_size = lengths.len;
    var mask = try F32Tensor.init(allocator, &.{ batch_size, max_len });

    for (0..batch_size) |b| {
        for (0..max_len) |i| {
            mask.data[b * max_len + i] = if (i < lengths[b]) 1.0 else 0.0;
        }
    }

    return mask;
}

// ============================================================================
// Tests
// ============================================================================

test "scaled dot-product attention" {
    const allocator = std.testing.allocator;

    var q = try F32Tensor.random(allocator, &.{ 2, 4 }, -1, 1);
    defer q.deinit();
    var k = try F32Tensor.random(allocator, &.{ 2, 4 }, -1, 1);
    defer k.deinit();
    var v = try F32Tensor.random(allocator, &.{ 2, 4 }, -1, 1);
    defer v.deinit();

    var attn = try scaledDotProductAttention(allocator, &q, &k, &v, null);
    defer attn.deinit();

    try std.testing.expectEqual(@as(usize, 2), attn.output.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), attn.output.shape[1]);

    // Check attention weights sum to 1 (per row)
    for (0..2) |i| {
        var row_sum: f32 = 0;
        for (0..2) |j| {
            row_sum += attn.attention_weights.data[i * 2 + j];
        }
        try std.testing.expect(@abs(row_sum - 1.0) < 0.01);
    }
}

test "causal mask" {
    const allocator = std.testing.allocator;

    var mask = try createCausalMask(allocator, 4);
    defer mask.deinit();

    // Should be lower triangular
    try std.testing.expectEqual(@as(f32, 1.0), mask.data[0]); // (0,0)
    try std.testing.expectEqual(@as(f32, 0.0), mask.data[1]); // (0,1)
    try std.testing.expectEqual(@as(f32, 1.0), mask.data[4]); // (1,0)
    try std.testing.expectEqual(@as(f32, 1.0), mask.data[5]); // (1,1)
}

test "multi-head attention" {
    const allocator = std.testing.allocator;

    var mha = try MultiHeadAttention.init(allocator, 16, 4);
    defer mha.deinit();

    var x = try F32Tensor.random(allocator, &.{ 2, 16 }, -1, 1);
    defer x.deinit();

    var attn = try mha.forward(&x, &x, &x, null);
    defer attn.deinit();

    try std.testing.expectEqual(@as(usize, 2), attn.output.shape[0]);
    try std.testing.expectEqual(@as(usize, 16), attn.output.shape[1]);
}
