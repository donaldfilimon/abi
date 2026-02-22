const std = @import("std");
const ops = @import("../../llm/ops/mod.zig");
const backward_ops = ops.backward;
const training_bridge = @import("../../../gpu/training_bridge.zig");

pub fn backward(model: anytype, d_logits: []const f32, input_ids: []const u32) !void {
    const seq_len: u32 = @intCast(input_ids.len);
    const hidden_dim = model.config.hidden_dim;
    const vocab_size = model.config.vocab_size;

    const cache = model.activations orelse return error.NoActivationCache;

    // Working buffers for gradients
    var d_hidden = try model.allocator.alloc(f32, seq_len * hidden_dim);
    defer model.allocator.free(d_hidden);
    @memset(d_hidden, 0);

    var d_norm_out = try model.allocator.alloc(f32, seq_len * hidden_dim);
    defer model.allocator.free(d_norm_out);

    var d_residual = try model.allocator.alloc(f32, seq_len * hidden_dim);
    defer model.allocator.free(d_residual);

    // Step 1: Backward through output projection
    // logits = norm_out @ output_weights^T
    // d_norm_out = d_logits @ output_weights
    // d_output_weights += d_logits^T @ norm_out
    const output_weights = model.weights.output_proj orelse model.weights.token_embedding;
    const d_output_weights = model.weights.d_output_proj orelse model.weights.d_token_embedding;

    for (0..seq_len) |pos| {
        const hidden_offset = pos * hidden_dim;
        const logit_offset = pos * vocab_size;

        // d_norm_out[pos] = d_logits[pos] @ output_weights
        for (0..hidden_dim) |h| {
            var grad: f32 = 0;
            for (0..vocab_size) |v| {
                grad += d_logits[logit_offset + v] * output_weights[v * hidden_dim + h];
            }
            d_norm_out[hidden_offset + h] = grad;
        }

        // d_output_weights += outer(d_logits[pos], norm_out[pos])
        for (0..vocab_size) |v| {
            for (0..hidden_dim) |h| {
                d_output_weights[v * hidden_dim + h] += d_logits[logit_offset + v] * cache.final_hidden[hidden_offset + h];
            }
        }
    }

    // Step 2: Backward through final norm
    for (0..seq_len) |pos| {
        const offset = pos * hidden_dim;
        // Get the pre-norm input from last layer's post_ffn
        const pre_norm = if (model.config.num_layers > 0)
            cache.layer_caches[model.config.num_layers - 1].post_ffn[offset .. offset + hidden_dim]
        else
            cache.embeddings[offset .. offset + hidden_dim];

        backward_ops.rmsNormBackward(
            d_norm_out[offset .. offset + hidden_dim],
            pre_norm,
            model.weights.final_norm,
            d_hidden[offset .. offset + hidden_dim],
            model.weights.d_final_norm,
            model.config.norm_eps,
        );
    }

    // Step 3: Backward through transformer layers (reverse order)
    var layer_idx: usize = model.config.num_layers;
    while (layer_idx > 0) {
        layer_idx -= 1;
        const layer = &model.weights.layers[layer_idx];
        const layer_cache = &cache.layer_caches[layer_idx];

        // Backward through FFN residual: hidden = pre_ffn + ffn_out
        // d_pre_ffn += d_hidden
        // d_ffn_out = d_hidden
        @memcpy(d_residual, d_hidden);

        // Backward through FFN norm
        var d_ffn_in = try model.allocator.alloc(f32, seq_len * hidden_dim);
        defer model.allocator.free(d_ffn_in);
        @memset(d_ffn_in, 0);

        for (0..seq_len) |pos| {
            const offset = pos * hidden_dim;

            // Backward through SwiGLU (simplified - use cached values)
            if (pos == 0) {
                const d_x = try model.allocator.alloc(f32, hidden_dim);
                defer model.allocator.free(d_x);
                @memset(d_x, 0);

                backward_ops.swigluBackward(
                    d_residual[offset .. offset + hidden_dim],
                    &layer_cache.ffn_cache,
                    layer.w_gate,
                    layer.w_up,
                    layer.w_down,
                    layer.d_w_gate,
                    layer.d_w_up,
                    layer.d_w_down,
                    d_x,
                );

                // Backward through FFN norm
                backward_ops.rmsNormBackward(
                    d_x,
                    layer_cache.pre_ffn_norm[offset .. offset + hidden_dim],
                    layer.ffn_norm,
                    d_ffn_in[offset .. offset + hidden_dim],
                    layer.d_ffn_norm,
                    model.config.norm_eps,
                );
            }
        }

        // Accumulate gradient from FFN path
        for (d_hidden, d_ffn_in) |*dh, dfi| {
            dh.* = dfi;
        }

        // Backward through attention residual
        @memcpy(d_residual, d_hidden);

        // Backward through attention (simplified)
        var d_attn_in = try model.allocator.alloc(f32, seq_len * hidden_dim);
        defer model.allocator.free(d_attn_in);
        @memset(d_attn_in, 0);

        // Backward through output projection W_o
        for (0..seq_len) |pos| {
            const offset = pos * hidden_dim;

            // d_attn_out = d_residual @ W_o^T (already have d_residual)
            // d_W_o += outer(d_residual, attn_out) - approximated

            // Backward through attention norm
            backward_ops.rmsNormBackward(
                d_residual[offset .. offset + hidden_dim],
                layer_cache.pre_attn_norm[offset .. offset + hidden_dim],
                layer.attn_norm,
                d_attn_in[offset .. offset + hidden_dim],
                layer.d_attn_norm,
                model.config.norm_eps,
            );
        }

        // Propagate gradient to next layer (or embeddings)
        for (d_hidden, d_attn_in) |*dh, dai| {
            dh.* = dai;
        }
    }

    // Step 4: Backward through token embeddings
    for (0..seq_len) |pos| {
        const token_id = input_ids[pos];
        const emb_offset = @as(usize, token_id) * hidden_dim;
        const grad_offset = pos * hidden_dim;

        // d_embedding[token_id] += d_hidden[pos]
        for (0..hidden_dim) |h| {
            model.weights.d_token_embedding[emb_offset + h] += d_hidden[grad_offset + h];
        }
    }
}

/// Backward pass with optional GPU bridge acceleration.
/// Currently delegates to CPU backward (GPU backward ops not yet implemented).
pub fn backwardWithBridge(model: anytype, d_logits: []const f32, input_ids: []const u32, _: ?*training_bridge.GpuTrainingBridge) !void {
    return backward(model, d_logits, input_ids);
}

pub fn computeCrossEntropyLoss(
    logits: []const f32,
    targets: []const u32,
    d_logits: []f32,
    vocab_size: u32,
) f32 {
    const seq_len = targets.len;
    var total_loss: f32 = 0;

    for (0..seq_len) |pos| {
        const logit_offset = pos * vocab_size;
        const target = targets[pos];

        // Compute softmax
        var probs = d_logits[logit_offset .. logit_offset + vocab_size];

        // Find max for numerical stability
        var max_logit: f32 = logits[logit_offset];
        for (logits[logit_offset .. logit_offset + vocab_size]) |l| {
            max_logit = @max(max_logit, l);
        }

        // Compute exp and sum
        var sum_exp: f32 = 0;
        for (0..vocab_size) |v| {
            probs[v] = @exp(logits[logit_offset + v] - max_logit);
            sum_exp += probs[v];
        }

        // Normalize to get probabilities
        for (probs) |*p| {
            p.* /= sum_exp;
        }

        // Cross-entropy loss: -log(prob[target])
        const target_prob = probs[target];
        total_loss -= @log(target_prob + 1e-10);

        // Gradient: probs - one_hot(target)
        probs[target] -= 1.0;
    }

    return total_loss / @as(f32, @floatFromInt(seq_len));
}

pub fn trainStep(model: anytype, input_ids: []const u32, target_ids: []const u32) !f32 {
    const seq_len: u32 = @intCast(input_ids.len);
    const vocab_size = model.config.vocab_size;

    // Allocate logits and gradient buffer
    const logits = try model.allocator.alloc(f32, seq_len * vocab_size);
    defer model.allocator.free(logits);
    const d_logits = try model.allocator.alloc(f32, seq_len * vocab_size);
    defer model.allocator.free(d_logits);

    // Forward pass
    try model.forward(input_ids, logits);

    // Compute loss and gradient
    const loss = computeCrossEntropyLoss(logits, target_ids, d_logits, vocab_size);

    // Backward pass
    try backward(model, d_logits, input_ids);

    return loss;
}
