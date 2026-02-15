const std = @import("std");
const ops = @import("../../llm/ops/mod.zig");
const backward_ops = ops.backward;

/// Execute a forward pass through the model and write logits.
pub fn run(
    model: anytype,
    input_ids: []const u32,
    logits_out: []f32,
) !void {
    const seq_len: u32 = @intCast(input_ids.len);
    const hidden_dim = model.config.hidden_dim;
    const vocab_size = model.config.vocab_size;

    // Ensure activation cache is initialized
    if (model.activations == null) {
        try model.prepareForTraining(seq_len);
    }
    const cache = model.activations.?;

    // Allocate working buffers
    var hidden = try model.allocator.alloc(f32, seq_len * hidden_dim);
    defer model.allocator.free(hidden);
    const residual = try model.allocator.alloc(f32, seq_len * hidden_dim);
    defer model.allocator.free(residual);
    var norm_out = try model.allocator.alloc(f32, seq_len * hidden_dim);
    defer model.allocator.free(norm_out);

    // Step 1: Token embedding lookup
    for (0..seq_len) |pos| {
        const token_id = input_ids[pos];
        if (token_id >= vocab_size) return error.InvalidTokenId;

        const emb_offset = @as(usize, token_id) * hidden_dim;
        const hidden_offset = pos * hidden_dim;

        @memcpy(
            hidden[hidden_offset .. hidden_offset + hidden_dim],
            model.weights.token_embedding[emb_offset .. emb_offset + hidden_dim],
        );
    }

    // Cache embeddings for backward pass
    @memcpy(cache.embeddings, hidden);

    // Step 2: Process through transformer layers
    for (model.weights.layers, 0..) |*layer, layer_idx| {
        var layer_cache = &cache.layer_caches[layer_idx];

        // Save pre-norm input for backward pass
        @memcpy(layer_cache.pre_attn_norm, hidden);

        // Attention block: hidden = hidden + attention(norm(hidden))
        // Apply attention normalization per position
        for (0..seq_len) |pos| {
            const offset = pos * hidden_dim;
            ops.rmsNorm(
                hidden[offset .. offset + hidden_dim],
                layer.attn_norm,
                norm_out[offset .. offset + hidden_dim],
                model.config.norm_eps,
            );
        }

        // Compute attention with caching
        try computeAttentionLayer(
            model,
            norm_out,
            layer,
            seq_len,
            residual,
            &layer_cache.attn_cache,
        );

        // Residual connection
        for (hidden, residual) |*h, r| {
            h.* += r;
        }

        // Cache post-attention hidden states
        @memcpy(layer_cache.post_attn, hidden);

        // FFN block: hidden = hidden + ffn(norm(hidden))
        // Save pre-FFN norm input
        @memcpy(layer_cache.pre_ffn_norm, hidden);

        // Apply FFN normalization per position
        for (0..seq_len) |pos| {
            const offset = pos * hidden_dim;
            ops.rmsNorm(
                hidden[offset .. offset + hidden_dim],
                layer.ffn_norm,
                norm_out[offset .. offset + hidden_dim],
                model.config.norm_eps,
            );
        }

        // Compute SwiGLU FFN per position
        try computeFFNLayer(
            model,
            norm_out,
            layer,
            seq_len,
            residual,
            &layer_cache.ffn_cache,
        );

        // Residual connection
        for (hidden, residual) |*h, r| {
            h.* += r;
        }

        // Cache post-FFN hidden states
        @memcpy(layer_cache.post_ffn, hidden);
    }

    // Step 3: Final layer normalization
    for (0..seq_len) |pos| {
        const offset = pos * hidden_dim;
        ops.rmsNorm(
            hidden[offset .. offset + hidden_dim],
            model.weights.final_norm,
            norm_out[offset .. offset + hidden_dim],
            model.config.norm_eps,
        );
    }

    // Cache final hidden states
    @memcpy(cache.final_hidden, norm_out);

    // Step 4: Project to vocabulary logits
    const output_weights = model.weights.output_proj orelse model.weights.token_embedding;

    for (0..seq_len) |pos| {
        const hidden_offset = pos * hidden_dim;
        const logit_offset = pos * vocab_size;

        // logits = hidden @ output_weights^T
        ops.matmul.matrixVectorMultiplyTransposed(
            output_weights,
            norm_out[hidden_offset .. hidden_offset + hidden_dim],
            logits_out[logit_offset .. logit_offset + vocab_size],
            vocab_size,
            hidden_dim,
        );
    }
}

fn computeAttentionLayer(
    model: anytype,
    input: []const f32,
    layer: anytype,
    seq_len: u32,
    output: []f32,
    attn_cache: *backward_ops.attention_backward.AttentionCache,
) !void {
    const hidden_dim = model.config.hidden_dim;
    const head_dim = model.config.headDim();
    const num_heads = model.config.num_heads;
    const num_kv_heads = model.config.num_kv_heads;
    const kv_dim = num_kv_heads * head_dim;

    // Allocate projections
    var q_proj = try model.allocator.alloc(f32, seq_len * hidden_dim);
    defer model.allocator.free(q_proj);
    var k_proj = try model.allocator.alloc(f32, seq_len * kv_dim);
    defer model.allocator.free(k_proj);
    var v_proj = try model.allocator.alloc(f32, seq_len * kv_dim);
    defer model.allocator.free(v_proj);
    var attn_out = try model.allocator.alloc(f32, seq_len * hidden_dim);
    defer model.allocator.free(attn_out);

    // Q, K, V projections for each position
    for (0..seq_len) |pos| {
        const in_offset = pos * hidden_dim;
        const q_offset = pos * hidden_dim;
        const kv_offset = pos * kv_dim;

        // Q = input @ W_q^T
        ops.matmul.matrixVectorMultiply(
            layer.w_q,
            input[in_offset .. in_offset + hidden_dim],
            q_proj[q_offset .. q_offset + hidden_dim],
            hidden_dim,
            hidden_dim,
        );

        // K = input @ W_k^T
        ops.matmul.matrixVectorMultiply(
            layer.w_k,
            input[in_offset .. in_offset + hidden_dim],
            k_proj[kv_offset .. kv_offset + kv_dim],
            kv_dim,
            hidden_dim,
        );

        // V = input @ W_v^T
        ops.matmul.matrixVectorMultiply(
            layer.w_v,
            input[in_offset .. in_offset + hidden_dim],
            v_proj[kv_offset .. kv_offset + kv_dim],
            kv_dim,
            hidden_dim,
        );
    }

    // Apply RoPE to Q and K
    // Only apply if we have a RoPE cache and positions are within bounds
    if (model.rope_cache) |rc| {
        const max_rope_pos = rc.config.max_seq_len;
        for (0..seq_len) |pos| {
            // Clamp position to max_seq_len to avoid out-of-bounds
            const rope_pos: u32 = @intCast(@min(pos, max_rope_pos - 1));

            const q_offset = pos * hidden_dim;
            for (0..num_heads) |h| {
                const head_offset = h * head_dim;
                ops.applyRope(
                    q_proj[q_offset + head_offset .. q_offset + head_offset + head_dim],
                    rope_pos,
                    rc,
                );
            }

            const kv_offset = pos * kv_dim;
            for (0..num_kv_heads) |h| {
                const head_offset = h * head_dim;
                ops.applyRope(
                    k_proj[kv_offset + head_offset .. kv_offset + head_offset + head_dim],
                    rope_pos,
                    rc,
                );
            }
        }
    }

    // Cache Q, K, V for backward pass (store first head for simplicity)
    @memcpy(attn_cache.q[0..@min(attn_cache.q.len, seq_len * head_dim)], q_proj[0..@min(attn_cache.q.len, seq_len * head_dim)]);
    @memcpy(attn_cache.k[0..@min(attn_cache.k.len, seq_len * head_dim)], k_proj[0..@min(attn_cache.k.len, seq_len * head_dim)]);
    @memcpy(attn_cache.v[0..@min(attn_cache.v.len, seq_len * head_dim)], v_proj[0..@min(attn_cache.v.len, seq_len * head_dim)]);

    // Multi-head attention
    // For GQA: each KV head serves multiple Q heads
    const heads_per_kv = num_heads / num_kv_heads;

    var head_output = try model.allocator.alloc(f32, seq_len * head_dim);
    defer model.allocator.free(head_output);

    @memset(attn_out, 0);

    for (0..num_heads) |h| {
        const kv_head = h / heads_per_kv;
        const q_head_offset = h * head_dim;
        const kv_head_offset = kv_head * head_dim;

        // Extract Q, K, V for this head across all positions
        var q_head = try model.allocator.alloc(f32, seq_len * head_dim);
        defer model.allocator.free(q_head);
        var k_head = try model.allocator.alloc(f32, seq_len * head_dim);
        defer model.allocator.free(k_head);
        var v_head = try model.allocator.alloc(f32, seq_len * head_dim);
        defer model.allocator.free(v_head);

        for (0..seq_len) |pos| {
            const src_q = pos * hidden_dim + q_head_offset;
            const src_kv = pos * kv_dim + kv_head_offset;
            const dst = pos * head_dim;

            @memcpy(q_head[dst .. dst + head_dim], q_proj[src_q .. src_q + head_dim]);
            @memcpy(k_head[dst .. dst + head_dim], k_proj[src_kv .. src_kv + head_dim]);
            @memcpy(v_head[dst .. dst + head_dim], v_proj[src_kv .. src_kv + head_dim]);
        }

        // Scaled dot-product attention for this head
        try ops.scaledDotProductAttention(
            model.allocator,
            q_head,
            k_head,
            v_head,
            head_output,
            seq_len,
            seq_len,
            head_dim,
            true, // causal
        );

        // Store attention weights in cache (only for first head)
        if (h == 0) {
            // Recompute attention weights for caching (needed for backward)
            const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
            for (0..seq_len) |qi| {
                for (0..seq_len) |ki| {
                    if (ki <= qi) {
                        var dot: f32 = 0;
                        for (0..head_dim) |d| {
                            dot += q_head[qi * head_dim + d] * k_head[ki * head_dim + d];
                        }
                        attn_cache.attn_weights[qi * seq_len + ki] = dot * scale;
                    } else {
                        attn_cache.attn_weights[qi * seq_len + ki] = -std.math.inf(f32);
                    }
                }
                // Apply softmax to this row
                ops.softmaxInPlace(attn_cache.attn_weights[qi * seq_len .. (qi + 1) * seq_len]);
            }
        }

        // Concatenate head outputs
        for (0..seq_len) |pos| {
            const src = pos * head_dim;
            const dst = pos * hidden_dim + q_head_offset;
            @memcpy(attn_out[dst .. dst + head_dim], head_output[src .. src + head_dim]);
        }
    }

    // Output projection: output = attn_out @ W_o^T
    for (0..seq_len) |pos| {
        const offset = pos * hidden_dim;
        ops.matmul.matrixVectorMultiply(
            layer.w_o,
            attn_out[offset .. offset + hidden_dim],
            output[offset .. offset + hidden_dim],
            hidden_dim,
            hidden_dim,
        );
    }
}

fn computeFFNLayer(
    model: anytype,
    input: []const f32,
    layer: anytype,
    seq_len: u32,
    output: []f32,
    ffn_cache: *backward_ops.ffn_backward.SwigluCache,
) !void {
    const hidden_dim = model.config.hidden_dim;
    const intermediate_dim = model.config.intermediate_dim;

    // Process each position
    for (0..seq_len) |pos| {
        const in_offset = pos * hidden_dim;
        const out_offset = pos * hidden_dim;

        // Cache input for backward (use first position for simplicity)
        if (pos == 0) {
            @memcpy(ffn_cache.x, input[in_offset .. in_offset + hidden_dim]);
        }

        // SwiGLU: output = down(silu(gate(x)) * up(x))
        try ops.swiglu(
            model.allocator,
            input[in_offset .. in_offset + hidden_dim],
            layer.w_gate,
            layer.w_up,
            layer.w_down,
            output[out_offset .. out_offset + hidden_dim],
            hidden_dim,
            intermediate_dim,
        );

        // Cache gate/up outputs for backward (first position)
        if (pos == 0) {
            // Compute and cache gate_out, up_out, intermediate
            ops.matmul.matrixVectorMultiply(
                layer.w_gate,
                input[in_offset .. in_offset + hidden_dim],
                ffn_cache.gate_out,
                intermediate_dim,
                hidden_dim,
            );
            ops.matmul.matrixVectorMultiply(
                layer.w_up,
                input[in_offset .. in_offset + hidden_dim],
                ffn_cache.up_out,
                intermediate_dim,
                hidden_dim,
            );
            for (0..intermediate_dim) |i| {
                ffn_cache.intermediate[i] = ops.activations.silu(ffn_cache.gate_out[i]) * ffn_cache.up_out[i];
            }
        }
    }
}
