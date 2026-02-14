//! Trainable LLM model wrapper for training.
//!
//! Provides a wrapper around LLM weights that enables:
//! - Mutable weights for gradient updates
//! - Gradient storage for backpropagation
//! - Activation caching for backward pass
//! - Integration with optimizers
//! - GGUF weight loading with dequantization

const std = @import("std");
const ops = @import("../llm/ops/mod.zig");
const backward_ops = ops.backward;
const gguf = @import("../llm/io/gguf.zig");
const gguf_writer = @import("../llm/io/gguf_writer.zig");
const tensor_loader = @import("../llm/io/tensor_loader.zig");
const quantized = @import("../llm/tensor/quantized.zig");
const model_config = @import("model/config.zig");
const checkpoint = @import("checkpoint.zig");

// Re-exports from weights.zig
const weights_mod = @import("weights.zig");
pub const TrainableLayerWeights = weights_mod.TrainableLayerWeights;
pub const TrainableWeights = weights_mod.TrainableWeights;
pub const ActivationCache = weights_mod.ActivationCache;

// Re-exports from trainable_checkpoint.zig
const trainable_ckpt = @import("trainable_checkpoint.zig");
pub const GradientCheckpointer = trainable_ckpt.GradientCheckpointer;
pub const ModelCheckpoint = trainable_ckpt.ModelCheckpoint;
pub const LoadError = trainable_ckpt.LoadError;

/// Gradient checkpointing strategy.
pub const CheckpointingStrategy = model_config.CheckpointingStrategy;

/// Configuration for a trainable model.
pub const TrainableModelConfig = model_config.TrainableModelConfig;

/// Trainable LLM model.
pub const TrainableModel = struct {
    allocator: std.mem.Allocator,
    config: TrainableModelConfig,
    weights: TrainableWeights,
    activations: ?*ActivationCache,
    rope_cache: ?*ops.rope.RopeCache,

    pub fn init(allocator: std.mem.Allocator, config: TrainableModelConfig) !TrainableModel {
        var weights = try TrainableWeights.init(allocator, config);
        errdefer weights.deinit();

        const rope_cache = try allocator.create(ops.rope.RopeCache);
        errdefer allocator.destroy(rope_cache);
        rope_cache.* = try ops.rope.RopeCache.init(allocator, .{
            .head_dim = config.headDim(),
            .theta_base = config.rope_theta,
            .max_seq_len = config.max_seq_len,
        });

        return .{
            .allocator = allocator,
            .config = config,
            .weights = weights,
            .activations = null,
            .rope_cache = rope_cache,
        };
    }

    pub fn deinit(self: *TrainableModel) void {
        if (self.activations) |act| {
            act.deinit();
            self.allocator.destroy(act);
        }
        if (self.rope_cache) |rc| {
            rc.deinit();
            self.allocator.destroy(rc);
        }
        self.weights.deinit();
        self.* = undefined;
    }

    /// Zero all gradients.
    pub fn zeroGradients(self: *TrainableModel) void {
        self.weights.zeroGradients();
    }

    /// Get number of parameters.
    pub fn numParams(self: *const TrainableModel) usize {
        return self.config.numParams();
    }

    /// Prepare activation cache for training.
    pub fn prepareForTraining(self: *TrainableModel, max_seq_len: u32) !void {
        if (self.activations) |act| {
            act.deinit();
            self.allocator.destroy(act);
        }
        const act = try self.allocator.create(ActivationCache);
        act.* = try ActivationCache.init(self.allocator, self.config, max_seq_len);
        self.activations = act;
    }

    /// Forward pass through the model.
    /// Returns logits: [seq_len, vocab_size]
    ///
    /// This method:
    /// 1. Looks up token embeddings
    /// 2. Processes through all transformer layers (attention + FFN)
    /// 3. Applies final normalization
    /// 4. Projects to vocabulary logits
    ///
    /// Intermediate activations are cached for the backward pass.
    pub fn forward(
        self: *TrainableModel,
        input_ids: []const u32,
        logits_out: []f32,
    ) !void {
        const seq_len: u32 = @intCast(input_ids.len);
        const hidden_dim = self.config.hidden_dim;
        const vocab_size = self.config.vocab_size;

        // Ensure activation cache is initialized
        if (self.activations == null) {
            try self.prepareForTraining(seq_len);
        }
        const cache = self.activations.?;

        // Allocate working buffers
        var hidden = try self.allocator.alloc(f32, seq_len * hidden_dim);
        defer self.allocator.free(hidden);
        const residual = try self.allocator.alloc(f32, seq_len * hidden_dim);
        defer self.allocator.free(residual);
        var norm_out = try self.allocator.alloc(f32, seq_len * hidden_dim);
        defer self.allocator.free(norm_out);

        // Step 1: Token embedding lookup
        for (0..seq_len) |pos| {
            const token_id = input_ids[pos];
            if (token_id >= vocab_size) return error.InvalidTokenId;

            const emb_offset = @as(usize, token_id) * hidden_dim;
            const hidden_offset = pos * hidden_dim;

            @memcpy(
                hidden[hidden_offset .. hidden_offset + hidden_dim],
                self.weights.token_embedding[emb_offset .. emb_offset + hidden_dim],
            );
        }

        // Cache embeddings for backward pass
        @memcpy(cache.embeddings, hidden);

        // Step 2: Process through transformer layers
        for (self.weights.layers, 0..) |*layer, layer_idx| {
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
                    self.config.norm_eps,
                );
            }

            // Compute attention with caching
            try self.computeAttentionLayer(
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
                    self.config.norm_eps,
                );
            }

            // Compute SwiGLU FFN per position
            try self.computeFFNLayer(
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
                self.weights.final_norm,
                norm_out[offset .. offset + hidden_dim],
                self.config.norm_eps,
            );
        }

        // Cache final hidden states
        @memcpy(cache.final_hidden, norm_out);

        // Step 4: Project to vocabulary logits
        const output_weights = self.weights.output_proj orelse self.weights.token_embedding;

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

    /// Compute attention layer for all positions.
    fn computeAttentionLayer(
        self: *TrainableModel,
        input: []const f32,
        layer: *TrainableLayerWeights,
        seq_len: u32,
        output: []f32,
        attn_cache: *backward_ops.attention_backward.AttentionCache,
    ) !void {
        const hidden_dim = self.config.hidden_dim;
        const head_dim = self.config.headDim();
        const num_heads = self.config.num_heads;
        const num_kv_heads = self.config.num_kv_heads;
        const kv_dim = num_kv_heads * head_dim;

        // Allocate projections
        var q_proj = try self.allocator.alloc(f32, seq_len * hidden_dim);
        defer self.allocator.free(q_proj);
        var k_proj = try self.allocator.alloc(f32, seq_len * kv_dim);
        defer self.allocator.free(k_proj);
        var v_proj = try self.allocator.alloc(f32, seq_len * kv_dim);
        defer self.allocator.free(v_proj);
        var attn_out = try self.allocator.alloc(f32, seq_len * hidden_dim);
        defer self.allocator.free(attn_out);

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
        if (self.rope_cache) |rc| {
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

        var head_output = try self.allocator.alloc(f32, seq_len * head_dim);
        defer self.allocator.free(head_output);

        @memset(attn_out, 0);

        for (0..num_heads) |h| {
            const kv_head = h / heads_per_kv;
            const q_head_offset = h * head_dim;
            const kv_head_offset = kv_head * head_dim;

            // Extract Q, K, V for this head across all positions
            var q_head = try self.allocator.alloc(f32, seq_len * head_dim);
            defer self.allocator.free(q_head);
            var k_head = try self.allocator.alloc(f32, seq_len * head_dim);
            defer self.allocator.free(k_head);
            var v_head = try self.allocator.alloc(f32, seq_len * head_dim);
            defer self.allocator.free(v_head);

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
                self.allocator,
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

    /// Compute SwiGLU FFN layer for all positions.
    fn computeFFNLayer(
        self: *TrainableModel,
        input: []const f32,
        layer: *TrainableLayerWeights,
        seq_len: u32,
        output: []f32,
        ffn_cache: *backward_ops.ffn_backward.SwigluCache,
    ) !void {
        const hidden_dim = self.config.hidden_dim;
        const intermediate_dim = self.config.intermediate_dim;

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
                self.allocator,
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

    /// Backward pass through the model.
    /// Computes gradients for all weights given the upstream gradient.
    ///
    /// Args:
    ///   d_logits: [seq_len, vocab_size] - gradient of loss w.r.t. logits
    ///   input_ids: original input tokens (for embedding gradient)
    ///
    /// After calling this, gradients are accumulated in self.weights.d_*
    pub fn backward(
        self: *TrainableModel,
        d_logits: []const f32,
        input_ids: []const u32,
    ) !void {
        const seq_len: u32 = @intCast(input_ids.len);
        const hidden_dim = self.config.hidden_dim;
        const vocab_size = self.config.vocab_size;

        const cache = self.activations orelse return error.NoActivationCache;

        // Working buffers for gradients
        var d_hidden = try self.allocator.alloc(f32, seq_len * hidden_dim);
        defer self.allocator.free(d_hidden);
        @memset(d_hidden, 0);

        var d_norm_out = try self.allocator.alloc(f32, seq_len * hidden_dim);
        defer self.allocator.free(d_norm_out);

        var d_residual = try self.allocator.alloc(f32, seq_len * hidden_dim);
        defer self.allocator.free(d_residual);

        // Step 1: Backward through output projection
        // logits = norm_out @ output_weights^T
        // d_norm_out = d_logits @ output_weights
        // d_output_weights += d_logits^T @ norm_out
        const output_weights = self.weights.output_proj orelse self.weights.token_embedding;
        const d_output_weights = self.weights.d_output_proj orelse self.weights.d_token_embedding;

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
            const pre_norm = if (self.config.num_layers > 0)
                cache.layer_caches[self.config.num_layers - 1].post_ffn[offset .. offset + hidden_dim]
            else
                cache.embeddings[offset .. offset + hidden_dim];

            backward_ops.rmsNormBackward(
                d_norm_out[offset .. offset + hidden_dim],
                pre_norm,
                self.weights.final_norm,
                d_hidden[offset .. offset + hidden_dim],
                self.weights.d_final_norm,
                self.config.norm_eps,
            );
        }

        // Step 3: Backward through transformer layers (reverse order)
        var layer_idx: usize = self.config.num_layers;
        while (layer_idx > 0) {
            layer_idx -= 1;
            const layer = &self.weights.layers[layer_idx];
            const layer_cache = &cache.layer_caches[layer_idx];

            // Backward through FFN residual: hidden = pre_ffn + ffn_out
            // d_pre_ffn += d_hidden
            // d_ffn_out = d_hidden
            @memcpy(d_residual, d_hidden);

            // Backward through FFN norm
            var d_ffn_in = try self.allocator.alloc(f32, seq_len * hidden_dim);
            defer self.allocator.free(d_ffn_in);
            @memset(d_ffn_in, 0);

            for (0..seq_len) |pos| {
                const offset = pos * hidden_dim;

                // Backward through SwiGLU (simplified - use cached values)
                if (pos == 0) {
                    const d_x = try self.allocator.alloc(f32, hidden_dim);
                    defer self.allocator.free(d_x);
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
                        self.config.norm_eps,
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
            var d_attn_in = try self.allocator.alloc(f32, seq_len * hidden_dim);
            defer self.allocator.free(d_attn_in);
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
                    self.config.norm_eps,
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
                self.weights.d_token_embedding[emb_offset + h] += d_hidden[grad_offset + h];
            }
        }
    }

    /// Compute cross-entropy loss and its gradient.
    /// Returns the loss value and populates d_logits with gradients.
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

    /// Complete training step: forward, loss, backward.
    /// Returns the loss value.
    pub fn trainStep(
        self: *TrainableModel,
        input_ids: []const u32,
        target_ids: []const u32,
    ) !f32 {
        const seq_len: u32 = @intCast(input_ids.len);
        const vocab_size = self.config.vocab_size;

        // Allocate logits and gradient buffer
        const logits = try self.allocator.alloc(f32, seq_len * vocab_size);
        defer self.allocator.free(logits);
        const d_logits = try self.allocator.alloc(f32, seq_len * vocab_size);
        defer self.allocator.free(d_logits);

        // Forward pass
        try self.forward(input_ids, logits);

        // Compute loss and gradient
        const loss = computeCrossEntropyLoss(logits, target_ids, d_logits, vocab_size);

        // Backward pass
        try self.backward(d_logits, input_ids);

        return loss;
    }

    /// Load weights from a GGUF file.
    /// Dequantizes quantized tensors to f32 for training.
    pub fn loadFromGguf(self: *TrainableModel, path: []const u8) !void {
        var gguf_file = try gguf.GgufFile.open(self.allocator, path);
        defer gguf_file.deinit();

        // Verify config matches
        const gguf_hidden = gguf_file.getEmbeddingLength() orelse return error.MissingMetadata;
        const gguf_layers = gguf_file.getBlockCount() orelse return error.MissingMetadata;
        const gguf_heads = gguf_file.getHeadCount() orelse return error.MissingMetadata;

        if (gguf_hidden != self.config.hidden_dim) return error.ConfigMismatch;
        if (gguf_layers != self.config.num_layers) return error.ConfigMismatch;
        if (gguf_heads != self.config.num_heads) return error.ConfigMismatch;

        // Load token embedding
        try self.loadTensor(&gguf_file, "token_embd.weight", self.weights.token_embedding);

        // Load layer weights
        for (self.weights.layers, 0..) |*layer, i| {
            try self.loadLayerWeights(&gguf_file, layer, @intCast(i));
        }

        // Load final norm
        try self.loadTensor(&gguf_file, "output_norm.weight", self.weights.final_norm);

        // Load output projection if not tied
        if (self.weights.output_proj) |out_proj| {
            try self.loadTensor(&gguf_file, "output.weight", out_proj);
        }

        std.log.info("Loaded weights from GGUF: {s}", .{path});
    }

    /// Load a single tensor from GGUF, dequantizing if necessary.
    fn loadTensor(self: *TrainableModel, gguf_file: *gguf.GgufFile, name: []const u8, dest: []f32) !void {
        _ = self;
        const info = gguf_file.getTensor(name) orelse {
            std.log.warn("Tensor not found: {s}", .{name});
            return error.TensorNotFound;
        };

        const data = gguf_file.getTensorData(name) orelse return error.TensorNotFound;

        // Check destination size
        const elem_count = info.elementCount();
        if (elem_count > dest.len) return error.BufferTooSmall;

        // Dequantize based on tensor type
        switch (info.tensor_type) {
            .f32 => {
                const src = std.mem.bytesAsSlice(f32, data);
                @memcpy(dest[0..src.len], src);
            },
            .f16 => {
                const src = std.mem.bytesAsSlice(f16, data);
                for (dest[0..src.len], src) |*d, s| {
                    d.* = @floatCast(s);
                }
            },
            .bf16 => {
                const src = std.mem.bytesAsSlice(u16, data);
                for (dest[0..src.len], src) |*d, s| {
                    d.* = tensor_loader.bf16ToF32(s);
                }
            },
            .q4_0 => {
                try tensor_loader.dequantizeQ4_0(data, dest[0..@intCast(elem_count)]);
            },
            .q8_0 => {
                try tensor_loader.dequantizeQ8_0(data, dest[0..@intCast(elem_count)]);
            },
            .mxfp4 => {
                try tensor_loader.dequantizeMXFP4(data, dest[0..@intCast(elem_count)]);
            },
            else => {
                std.log.warn("Unsupported tensor type for training: {t}", .{info.tensor_type});
                return error.UnsupportedTensorType;
            },
        }
    }

    /// Load weights for a single transformer layer.
    fn loadLayerWeights(self: *TrainableModel, gguf_file: *gguf.GgufFile, layer: *TrainableLayerWeights, layer_idx: u32) !void {
        var name_buf: [128]u8 = undefined;

        // Attention weights
        const attn_q = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_q.weight", .{layer_idx}) catch return error.NameTooLong;
        self.loadTensor(gguf_file, attn_q, layer.w_q) catch |err| {
            if (err != error.TensorNotFound) return err;
        };

        const attn_k = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_k.weight", .{layer_idx}) catch return error.NameTooLong;
        self.loadTensor(gguf_file, attn_k, layer.w_k) catch |err| {
            if (err != error.TensorNotFound) return err;
        };

        const attn_v = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_v.weight", .{layer_idx}) catch return error.NameTooLong;
        self.loadTensor(gguf_file, attn_v, layer.w_v) catch |err| {
            if (err != error.TensorNotFound) return err;
        };

        const attn_output = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_output.weight", .{layer_idx}) catch return error.NameTooLong;
        self.loadTensor(gguf_file, attn_output, layer.w_o) catch |err| {
            if (err != error.TensorNotFound) return err;
        };

        const attn_norm = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_norm.weight", .{layer_idx}) catch return error.NameTooLong;
        self.loadTensor(gguf_file, attn_norm, layer.attn_norm) catch |err| {
            if (err != error.TensorNotFound) return err;
        };

        // FFN weights
        const ffn_gate = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_gate.weight", .{layer_idx}) catch return error.NameTooLong;
        self.loadTensor(gguf_file, ffn_gate, layer.w_gate) catch |err| {
            if (err != error.TensorNotFound) return err;
        };

        const ffn_up = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_up.weight", .{layer_idx}) catch return error.NameTooLong;
        self.loadTensor(gguf_file, ffn_up, layer.w_up) catch |err| {
            if (err != error.TensorNotFound) return err;
        };

        const ffn_down = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_down.weight", .{layer_idx}) catch return error.NameTooLong;
        self.loadTensor(gguf_file, ffn_down, layer.w_down) catch |err| {
            if (err != error.TensorNotFound) return err;
        };

        const ffn_norm = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_norm.weight", .{layer_idx}) catch return error.NameTooLong;
        self.loadTensor(gguf_file, ffn_norm, layer.ffn_norm) catch |err| {
            if (err != error.TensorNotFound) return err;
        };
    }

    /// Create a TrainableModel from a GGUF file.
    /// Extracts config from metadata and loads weights.
    pub fn fromGguf(allocator: std.mem.Allocator, path: []const u8) !TrainableModel {
        var gguf_file = try gguf.GgufFile.open(allocator, path);
        defer gguf_file.deinit();

        // Extract config from GGUF metadata
        const config = TrainableModelConfig{
            .hidden_dim = gguf_file.getEmbeddingLength() orelse return error.MissingMetadata,
            .num_layers = gguf_file.getBlockCount() orelse return error.MissingMetadata,
            .num_heads = gguf_file.getHeadCount() orelse return error.MissingMetadata,
            .num_kv_heads = gguf_file.getHeadCountKV() orelse gguf_file.getHeadCount() orelse return error.MissingMetadata,
            .intermediate_dim = getIntermediateDim(&gguf_file) orelse return error.MissingMetadata,
            .vocab_size = gguf_file.getVocabSize() orelse return error.MissingMetadata,
            .max_seq_len = gguf_file.getContextLength() orelse 2048,
            .rope_theta = getRopeTheta(&gguf_file) orelse 10000.0,
            .norm_eps = getNormEps(&gguf_file) orelse 1e-5,
        };

        var model = try TrainableModel.init(allocator, config);
        errdefer model.deinit();

        // Re-open and load weights
        var gguf_file2 = try gguf.GgufFile.open(allocator, path);
        defer gguf_file2.deinit();

        try model.loadFromGgufInternal(&gguf_file2);

        return model;
    }

    /// Internal weight loading (for use after config is set).
    fn loadFromGgufInternal(self: *TrainableModel, gguf_file: *gguf.GgufFile) !void {
        // Load token embedding
        self.loadTensor(gguf_file, "token_embd.weight", self.weights.token_embedding) catch |err| {
            std.log.warn("Failed to load token embedding: {t}", .{err});
        };

        // Load layer weights
        for (self.weights.layers, 0..) |*layer, i| {
            self.loadLayerWeights(gguf_file, layer, @intCast(i)) catch |err| {
                std.log.warn("Failed to load layer {d} weights: {t}", .{ i, err });
            };
        }

        // Load final norm
        self.loadTensor(gguf_file, "output_norm.weight", self.weights.final_norm) catch |err| {
            std.log.warn("Failed to load final norm: {t}", .{err});
        };

        // Load output projection if not tied
        if (self.weights.output_proj) |out_proj| {
            self.loadTensor(gguf_file, "output.weight", out_proj) catch |err| {
                std.log.warn("Failed to load output projection: {t}", .{err});
            };
        }
    }

    /// Save model checkpoint to a file.
    /// This flattens all weights and saves them with config metadata.
    pub fn saveCheckpoint(self: *const TrainableModel, path: []const u8, step: u64) !void {
        // Collect all weights into a flat array
        const weights = try self.collectWeights();
        defer self.allocator.free(weights);

        const view = checkpoint.CheckpointView{
            .step = step,
            .timestamp = 0, // Will be set by checkpoint.saveCheckpoint
            .weights = weights,
        };

        try checkpoint.saveCheckpoint(self.allocator, path, view);
        std.log.info("Saved checkpoint to {s} (step {d}, {d} params)", .{ path, step, weights.len });
    }

    /// Load model checkpoint from a file.
    /// The model config must match the checkpoint.
    pub fn loadCheckpointFile(self: *TrainableModel, path: []const u8) !u64 {
        var ckpt = try checkpoint.loadCheckpoint(self.allocator, path);
        defer ckpt.deinit(self.allocator);

        // Verify weight count matches
        const expected = self.config.numParams();
        if (ckpt.weights.len != expected) {
            std.log.err("Checkpoint weight count mismatch: expected {d}, got {d}", .{ expected, ckpt.weights.len });
            return error.ConfigMismatch;
        }

        // Distribute weights back to model
        try self.distributeWeights(ckpt.weights);

        std.log.info("Loaded checkpoint from {s} (step {d})", .{ path, ckpt.step });
        return ckpt.step;
    }

    /// Collect all weights into a flat array for checkpointing.
    pub fn collectWeights(self: *const TrainableModel) ![]f32 {
        const total_params = self.config.numParams();
        const weights = try self.allocator.alloc(f32, total_params);
        errdefer self.allocator.free(weights);

        var offset: usize = 0;

        // Token embedding
        @memcpy(weights[offset..][0..self.weights.token_embedding.len], self.weights.token_embedding);
        offset += self.weights.token_embedding.len;

        // Per-layer weights
        for (self.weights.layers) |layer| {
            @memcpy(weights[offset..][0..layer.w_q.len], layer.w_q);
            offset += layer.w_q.len;
            @memcpy(weights[offset..][0..layer.w_k.len], layer.w_k);
            offset += layer.w_k.len;
            @memcpy(weights[offset..][0..layer.w_v.len], layer.w_v);
            offset += layer.w_v.len;
            @memcpy(weights[offset..][0..layer.w_o.len], layer.w_o);
            offset += layer.w_o.len;
            @memcpy(weights[offset..][0..layer.attn_norm.len], layer.attn_norm);
            offset += layer.attn_norm.len;
            @memcpy(weights[offset..][0..layer.w_gate.len], layer.w_gate);
            offset += layer.w_gate.len;
            @memcpy(weights[offset..][0..layer.w_up.len], layer.w_up);
            offset += layer.w_up.len;
            @memcpy(weights[offset..][0..layer.w_down.len], layer.w_down);
            offset += layer.w_down.len;
            @memcpy(weights[offset..][0..layer.ffn_norm.len], layer.ffn_norm);
            offset += layer.ffn_norm.len;
        }

        // Final norm
        @memcpy(weights[offset..][0..self.weights.final_norm.len], self.weights.final_norm);
        offset += self.weights.final_norm.len;

        // Output projection (if not tied)
        if (self.weights.output_proj) |op| {
            @memcpy(weights[offset..][0..op.len], op);
            offset += op.len;
        }

        return weights;
    }

    /// Distribute a flat weight array back to model weights.
    pub fn distributeWeights(self: *TrainableModel, weights: []const f32) !void {
        var offset: usize = 0;

        // Token embedding
        @memcpy(self.weights.token_embedding, weights[offset..][0..self.weights.token_embedding.len]);
        offset += self.weights.token_embedding.len;

        // Per-layer weights
        for (self.weights.layers) |*layer| {
            @memcpy(layer.w_q, weights[offset..][0..layer.w_q.len]);
            offset += layer.w_q.len;
            @memcpy(layer.w_k, weights[offset..][0..layer.w_k.len]);
            offset += layer.w_k.len;
            @memcpy(layer.w_v, weights[offset..][0..layer.w_v.len]);
            offset += layer.w_v.len;
            @memcpy(layer.w_o, weights[offset..][0..layer.w_o.len]);
            offset += layer.w_o.len;
            @memcpy(layer.attn_norm, weights[offset..][0..layer.attn_norm.len]);
            offset += layer.attn_norm.len;
            @memcpy(layer.w_gate, weights[offset..][0..layer.w_gate.len]);
            offset += layer.w_gate.len;
            @memcpy(layer.w_up, weights[offset..][0..layer.w_up.len]);
            offset += layer.w_up.len;
            @memcpy(layer.w_down, weights[offset..][0..layer.w_down.len]);
            offset += layer.w_down.len;
            @memcpy(layer.ffn_norm, weights[offset..][0..layer.ffn_norm.len]);
            offset += layer.ffn_norm.len;
        }

        // Final norm
        @memcpy(self.weights.final_norm, weights[offset..][0..self.weights.final_norm.len]);
        offset += self.weights.final_norm.len;

        // Output projection (if not tied)
        if (self.weights.output_proj) |op| {
            @memcpy(op, weights[offset..][0..op.len]);
        }
    }

    /// Create a checkpoint from current model state.
    pub fn createCheckpoint(self: *const TrainableModel, step: u64) !ModelCheckpoint {
        const weights = try self.collectWeights();
        return .{
            .allocator = self.allocator,
            .config = self.config,
            .weights = weights,
            .step = step,
            .timestamp = 0,
        };
    }

    /// Load model state from a checkpoint.
    pub fn loadFromCheckpoint(self: *TrainableModel, ckpt: *const ModelCheckpoint) !void {
        // Verify config matches
        if (ckpt.config.hidden_dim != self.config.hidden_dim or
            ckpt.config.num_layers != self.config.num_layers or
            ckpt.config.num_heads != self.config.num_heads)
        {
            return error.ConfigMismatch;
        }
        try self.distributeWeights(ckpt.weights);
    }

    // =========================================================================
    // Gradient Management Methods
    // =========================================================================

    /// Compute the global gradient norm (L2 norm across all parameters).
    pub fn computeGradientNorm(self: *const TrainableModel) f32 {
        var sum_sq: f32 = 0;

        // Token embedding gradients
        for (self.weights.d_token_embedding) |g| {
            sum_sq += g * g;
        }

        // Per-layer gradients
        for (self.weights.layers) |layer| {
            for (layer.d_w_q) |g| sum_sq += g * g;
            for (layer.d_w_k) |g| sum_sq += g * g;
            for (layer.d_w_v) |g| sum_sq += g * g;
            for (layer.d_w_o) |g| sum_sq += g * g;
            for (layer.d_attn_norm) |g| sum_sq += g * g;
            for (layer.d_w_gate) |g| sum_sq += g * g;
            for (layer.d_w_up) |g| sum_sq += g * g;
            for (layer.d_w_down) |g| sum_sq += g * g;
            for (layer.d_ffn_norm) |g| sum_sq += g * g;
        }

        // Final norm gradients
        for (self.weights.d_final_norm) |g| {
            sum_sq += g * g;
        }

        // Output projection gradients (if not tied)
        if (self.weights.d_output_proj) |d_op| {
            for (d_op) |g| {
                sum_sq += g * g;
            }
        }

        return @sqrt(sum_sq);
    }

    /// Clip gradients by global norm.
    /// If the global norm exceeds max_norm, gradients are scaled down.
    /// Returns the original gradient norm before clipping.
    pub fn clipGradients(self: *TrainableModel, max_norm: f32) f32 {
        const grad_norm = self.computeGradientNorm();

        if (grad_norm > max_norm and grad_norm > 0) {
            const scale = max_norm / grad_norm;

            // Scale token embedding gradients
            for (self.weights.d_token_embedding) |*g| {
                g.* *= scale;
            }

            // Scale per-layer gradients
            for (self.weights.layers) |*layer| {
                for (layer.d_w_q) |*g| g.* *= scale;
                for (layer.d_w_k) |*g| g.* *= scale;
                for (layer.d_w_v) |*g| g.* *= scale;
                for (layer.d_w_o) |*g| g.* *= scale;
                for (layer.d_attn_norm) |*g| g.* *= scale;
                for (layer.d_w_gate) |*g| g.* *= scale;
                for (layer.d_w_up) |*g| g.* *= scale;
                for (layer.d_w_down) |*g| g.* *= scale;
                for (layer.d_ffn_norm) |*g| g.* *= scale;
            }

            // Scale final norm gradients
            for (self.weights.d_final_norm) |*g| {
                g.* *= scale;
            }

            // Scale output projection gradients (if not tied)
            if (self.weights.d_output_proj) |d_op| {
                for (d_op) |*g| {
                    g.* *= scale;
                }
            }
        }

        return grad_norm;
    }

    /// Check if gradients contain any non-finite values (NaN or Inf).
    pub fn hasNonFiniteGradients(self: *const TrainableModel) bool {
        for (self.weights.d_token_embedding) |g| {
            if (!std.math.isFinite(g)) return true;
        }

        for (self.weights.layers) |layer| {
            for (layer.d_w_q) |g| if (!std.math.isFinite(g)) return true;
            for (layer.d_w_k) |g| if (!std.math.isFinite(g)) return true;
            for (layer.d_w_v) |g| if (!std.math.isFinite(g)) return true;
            for (layer.d_w_o) |g| if (!std.math.isFinite(g)) return true;
            for (layer.d_attn_norm) |g| if (!std.math.isFinite(g)) return true;
            for (layer.d_w_gate) |g| if (!std.math.isFinite(g)) return true;
            for (layer.d_w_up) |g| if (!std.math.isFinite(g)) return true;
            for (layer.d_w_down) |g| if (!std.math.isFinite(g)) return true;
            for (layer.d_ffn_norm) |g| if (!std.math.isFinite(g)) return true;
        }

        for (self.weights.d_final_norm) |g| {
            if (!std.math.isFinite(g)) return true;
        }

        if (self.weights.d_output_proj) |d_op| {
            for (d_op) |g| {
                if (!std.math.isFinite(g)) return true;
            }
        }

        return false;
    }

    /// Apply SGD update to all weights.
    /// weights = weights - learning_rate * gradients
    pub fn applySgdUpdate(self: *TrainableModel, learning_rate: f32) void {
        // Token embedding
        for (self.weights.token_embedding, self.weights.d_token_embedding) |*w, g| {
            w.* -= learning_rate * g;
        }

        // Per-layer weights
        for (self.weights.layers) |*layer| {
            for (layer.w_q, layer.d_w_q) |*w, g| w.* -= learning_rate * g;
            for (layer.w_k, layer.d_w_k) |*w, g| w.* -= learning_rate * g;
            for (layer.w_v, layer.d_w_v) |*w, g| w.* -= learning_rate * g;
            for (layer.w_o, layer.d_w_o) |*w, g| w.* -= learning_rate * g;
            for (layer.attn_norm, layer.d_attn_norm) |*w, g| w.* -= learning_rate * g;
            for (layer.w_gate, layer.d_w_gate) |*w, g| w.* -= learning_rate * g;
            for (layer.w_up, layer.d_w_up) |*w, g| w.* -= learning_rate * g;
            for (layer.w_down, layer.d_w_down) |*w, g| w.* -= learning_rate * g;
            for (layer.ffn_norm, layer.d_ffn_norm) |*w, g| w.* -= learning_rate * g;
        }

        // Final norm
        for (self.weights.final_norm, self.weights.d_final_norm) |*w, g| {
            w.* -= learning_rate * g;
        }

        // Output projection (if not tied)
        if (self.weights.output_proj) |op| {
            if (self.weights.d_output_proj) |d_op| {
                for (op, d_op) |*w, g| {
                    w.* -= learning_rate * g;
                }
            }
        }
    }

    /// Scale all gradients by a factor (for mixed precision unscaling).
    pub fn scaleGradients(self: *TrainableModel, scale: f32) void {
        for (self.weights.d_token_embedding) |*g| g.* *= scale;

        for (self.weights.layers) |*layer| {
            for (layer.d_w_q) |*g| g.* *= scale;
            for (layer.d_w_k) |*g| g.* *= scale;
            for (layer.d_w_v) |*g| g.* *= scale;
            for (layer.d_w_o) |*g| g.* *= scale;
            for (layer.d_attn_norm) |*g| g.* *= scale;
            for (layer.d_w_gate) |*g| g.* *= scale;
            for (layer.d_w_up) |*g| g.* *= scale;
            for (layer.d_w_down) |*g| g.* *= scale;
            for (layer.d_ffn_norm) |*g| g.* *= scale;
        }

        for (self.weights.d_final_norm) |*g| g.* *= scale;

        if (self.weights.d_output_proj) |d_op| {
            for (d_op) |*g| g.* *= scale;
        }
    }

    /// Training step with gradient clipping and optional mixed precision.
    /// Returns the loss value and gradient norm before clipping.
    pub fn trainStepWithClipping(
        self: *TrainableModel,
        input_ids: []const u32,
        target_ids: []const u32,
        learning_rate: f32,
        max_grad_norm: f32,
        loss_scale: ?f32,
    ) !TrainStepResult {
        const seq_len: u32 = @intCast(input_ids.len);
        const vocab_size = self.config.vocab_size;

        // Allocate logits and gradient buffer
        const logits = try self.allocator.alloc(f32, seq_len * vocab_size);
        defer self.allocator.free(logits);
        const d_logits = try self.allocator.alloc(f32, seq_len * vocab_size);
        defer self.allocator.free(d_logits);

        // Forward pass
        try self.forward(input_ids, logits);

        // Compute loss and gradient
        const loss = computeCrossEntropyLoss(logits, target_ids, d_logits, vocab_size);

        // Scale loss for mixed precision (if enabled)
        if (loss_scale) |scale| {
            for (d_logits) |*g| {
                g.* *= scale;
            }
        }

        // Backward pass
        try self.backward(d_logits, input_ids);

        // Unscale gradients (if mixed precision)
        if (loss_scale) |scale| {
            self.scaleGradients(1.0 / scale);
        }

        // Check for NaN/Inf gradients
        const has_nan = self.hasNonFiniteGradients();
        if (has_nan) {
            // Zero out gradients on overflow
            self.zeroGradients();
            return .{
                .loss = loss,
                .grad_norm = 0,
                .grad_norm_clipped = 0,
                .skipped = true,
            };
        }

        // Clip gradients
        const grad_norm = self.clipGradients(max_grad_norm);
        const grad_norm_clipped = self.computeGradientNorm();

        // Apply update
        self.applySgdUpdate(learning_rate);

        // Zero gradients for next step
        self.zeroGradients();

        return .{
            .loss = loss,
            .grad_norm = grad_norm,
            .grad_norm_clipped = grad_norm_clipped,
            .skipped = false,
        };
    }

    /// Result of a training step with clipping.
    pub const TrainStepResult = struct {
        loss: f32,
        grad_norm: f32,
        grad_norm_clipped: f32,
        skipped: bool,
    };

    /// Export trainable weights to GGUF format (weights only).
    pub fn exportToGguf(
        self: *const TrainableModel,
        allocator: std.mem.Allocator,
        path: []const u8,
        config: struct {
            name: []const u8 = "abi-llama",
            tokenizer: ?gguf_writer.TokenizerConfig = null,
        },
    ) !void {
        const layer_count: usize = @intCast(self.config.num_layers);
        const layers = try allocator.alloc(gguf_writer.LayerWeights, layer_count);
        defer allocator.free(layers);

        for (self.weights.layers, 0..) |layer, i| {
            layers[i] = .{
                .attn_norm = layer.attn_norm,
                .ffn_norm = layer.ffn_norm,
                .wq = layer.w_q,
                .wk = layer.w_k,
                .wv = layer.w_v,
                .wo = layer.w_o,
                .w_gate = layer.w_gate,
                .w_up = layer.w_up,
                .w_down = layer.w_down,
            };
        }

        const export_config = gguf_writer.ExportConfig{
            .name = config.name,
            .vocab_size = self.config.vocab_size,
            .context_length = self.config.max_seq_len,
            .embedding_length = self.config.hidden_dim,
            .block_count = self.config.num_layers,
            .head_count = self.config.num_heads,
            .head_count_kv = self.config.num_kv_heads,
            .ffn_hidden_dim = self.config.intermediate_dim,
            .rope_freq_base = self.config.rope_theta,
            .layer_norm_rms_epsilon = self.config.norm_eps,
            .tokenizer = config.tokenizer,
        };

        const export_weights = gguf_writer.ExportWeights{
            .token_embedding = self.weights.token_embedding,
            .output_weight = self.weights.output_proj,
            .output_norm = self.weights.final_norm,
            .layers = layers,
        };

        try gguf_writer.exportToGguf(allocator, path, export_config, export_weights);
    }
};

/// Get intermediate dimension from GGUF metadata.
fn getIntermediateDim(gguf_file: *gguf.GgufFile) ?u32 {
    const arch = gguf_file.getArchitecture() orelse "llama";
    var buf: [64]u8 = undefined;
    const key = std.fmt.bufPrint(&buf, "{s}.feed_forward_length", .{arch}) catch return null;
    const val = gguf_file.getMetadata(key) orelse return null;
    return val.asU32();
}

/// Get RoPE theta from GGUF metadata.
fn getRopeTheta(gguf_file: *gguf.GgufFile) ?f32 {
    const arch = gguf_file.getArchitecture() orelse "llama";
    var buf: [64]u8 = undefined;
    const key = std.fmt.bufPrint(&buf, "{s}.rope.freq_base", .{arch}) catch return null;
    const val = gguf_file.getMetadata(key) orelse return null;
    return val.asF32();
}

/// Get norm epsilon from GGUF metadata.
fn getNormEps(gguf_file: *gguf.GgufFile) ?f32 {
    const arch = gguf_file.getArchitecture() orelse "llama";
    var buf: [64]u8 = undefined;
    const key = std.fmt.bufPrint(&buf, "{s}.attention.layer_norm_rms_epsilon", .{arch}) catch return null;
    const val = gguf_file.getMetadata(key) orelse return null;
    return val.asF32();
}
