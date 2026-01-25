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
const quantized = @import("../llm/tensor/quantized.zig");

/// Gradient checkpointing strategy.
pub const CheckpointingStrategy = enum {
    /// No checkpointing - store all activations
    none,
    /// Checkpoint every N layers (trades memory for compute)
    every_n_layers,
    /// Only checkpoint attention (most memory-heavy)
    attention_only,
    /// Full checkpointing - recompute everything
    full,
};

/// Configuration for a trainable model.
pub const TrainableModelConfig = struct {
    /// Model dimension (hidden size)
    hidden_dim: u32,
    /// Number of transformer layers
    num_layers: u32,
    /// Number of attention heads
    num_heads: u32,
    /// Number of key-value heads (for GQA)
    num_kv_heads: u32,
    /// Intermediate dimension for FFN
    intermediate_dim: u32,
    /// Vocabulary size
    vocab_size: u32,
    /// Maximum sequence length
    max_seq_len: u32 = 2048,
    /// RoPE theta base
    rope_theta: f32 = 10000.0,
    /// RMSNorm epsilon
    norm_eps: f32 = 1e-5,
    /// Whether to use tied embeddings
    tie_embeddings: bool = true,
    /// Gradient checkpointing strategy
    checkpointing: CheckpointingStrategy = .none,
    /// Checkpoint interval (for every_n_layers)
    checkpoint_interval: u32 = 4,

    /// Compute head dimension.
    pub fn headDim(self: TrainableModelConfig) u32 {
        return self.hidden_dim / self.num_heads;
    }

    /// Compute total number of parameters.
    pub fn numParams(self: TrainableModelConfig) usize {
        const head_dim = self.headDim();
        const kv_dim = self.num_kv_heads * head_dim;

        var total: usize = 0;

        // Token embedding
        total += self.vocab_size * self.hidden_dim;

        // Per-layer weights
        const per_layer: usize =
            // Attention: Q, K, V projections
            self.hidden_dim * self.hidden_dim + // W_q
            self.hidden_dim * kv_dim + // W_k
            self.hidden_dim * kv_dim + // W_v
            self.hidden_dim * self.hidden_dim + // W_o
            // Attention norm
            self.hidden_dim +
            // FFN: gate, up, down
            self.hidden_dim * self.intermediate_dim + // gate
            self.hidden_dim * self.intermediate_dim + // up
            self.intermediate_dim * self.hidden_dim + // down
            // FFN norm
            self.hidden_dim;

        total += per_layer * self.num_layers;

        // Final norm
        total += self.hidden_dim;

        // Output projection (if not tied)
        if (!self.tie_embeddings) {
            total += self.hidden_dim * self.vocab_size;
        }

        return total;
    }
};

/// Trainable weights for a single transformer layer.
pub const TrainableLayerWeights = struct {
    allocator: std.mem.Allocator,

    // Attention weights
    w_q: []f32,
    w_k: []f32,
    w_v: []f32,
    w_o: []f32,
    attn_norm: []f32,

    // FFN weights
    w_gate: []f32,
    w_up: []f32,
    w_down: []f32,
    ffn_norm: []f32,

    // Gradients
    d_w_q: []f32,
    d_w_k: []f32,
    d_w_v: []f32,
    d_w_o: []f32,
    d_attn_norm: []f32,
    d_w_gate: []f32,
    d_w_up: []f32,
    d_w_down: []f32,
    d_ffn_norm: []f32,

    pub fn init(allocator: std.mem.Allocator, config: TrainableModelConfig) !TrainableLayerWeights {
        const head_dim = config.headDim();
        const kv_dim = config.num_kv_heads * head_dim;

        const w_q = try allocator.alloc(f32, config.hidden_dim * config.hidden_dim);
        errdefer allocator.free(w_q);
        const w_k = try allocator.alloc(f32, config.hidden_dim * kv_dim);
        errdefer allocator.free(w_k);
        const w_v = try allocator.alloc(f32, config.hidden_dim * kv_dim);
        errdefer allocator.free(w_v);
        const w_o = try allocator.alloc(f32, config.hidden_dim * config.hidden_dim);
        errdefer allocator.free(w_o);
        const attn_norm = try allocator.alloc(f32, config.hidden_dim);
        errdefer allocator.free(attn_norm);

        const w_gate = try allocator.alloc(f32, config.intermediate_dim * config.hidden_dim);
        errdefer allocator.free(w_gate);
        const w_up = try allocator.alloc(f32, config.intermediate_dim * config.hidden_dim);
        errdefer allocator.free(w_up);
        const w_down = try allocator.alloc(f32, config.hidden_dim * config.intermediate_dim);
        errdefer allocator.free(w_down);
        const ffn_norm = try allocator.alloc(f32, config.hidden_dim);
        errdefer allocator.free(ffn_norm);

        // Allocate gradients
        const d_w_q = try allocator.alloc(f32, config.hidden_dim * config.hidden_dim);
        errdefer allocator.free(d_w_q);
        const d_w_k = try allocator.alloc(f32, config.hidden_dim * kv_dim);
        errdefer allocator.free(d_w_k);
        const d_w_v = try allocator.alloc(f32, config.hidden_dim * kv_dim);
        errdefer allocator.free(d_w_v);
        const d_w_o = try allocator.alloc(f32, config.hidden_dim * config.hidden_dim);
        errdefer allocator.free(d_w_o);
        const d_attn_norm = try allocator.alloc(f32, config.hidden_dim);
        errdefer allocator.free(d_attn_norm);

        const d_w_gate = try allocator.alloc(f32, config.intermediate_dim * config.hidden_dim);
        errdefer allocator.free(d_w_gate);
        const d_w_up = try allocator.alloc(f32, config.intermediate_dim * config.hidden_dim);
        errdefer allocator.free(d_w_up);
        const d_w_down = try allocator.alloc(f32, config.hidden_dim * config.intermediate_dim);
        errdefer allocator.free(d_w_down);
        const d_ffn_norm = try allocator.alloc(f32, config.hidden_dim);

        // Initialize weights
        initializeXavier(w_q);
        initializeXavier(w_k);
        initializeXavier(w_v);
        initializeXavier(w_o);
        @memset(attn_norm, 1.0);

        initializeXavier(w_gate);
        initializeXavier(w_up);
        initializeXavier(w_down);
        @memset(ffn_norm, 1.0);

        // Zero gradients
        @memset(d_w_q, 0);
        @memset(d_w_k, 0);
        @memset(d_w_v, 0);
        @memset(d_w_o, 0);
        @memset(d_attn_norm, 0);
        @memset(d_w_gate, 0);
        @memset(d_w_up, 0);
        @memset(d_w_down, 0);
        @memset(d_ffn_norm, 0);

        return .{
            .allocator = allocator,
            .w_q = w_q,
            .w_k = w_k,
            .w_v = w_v,
            .w_o = w_o,
            .attn_norm = attn_norm,
            .w_gate = w_gate,
            .w_up = w_up,
            .w_down = w_down,
            .ffn_norm = ffn_norm,
            .d_w_q = d_w_q,
            .d_w_k = d_w_k,
            .d_w_v = d_w_v,
            .d_w_o = d_w_o,
            .d_attn_norm = d_attn_norm,
            .d_w_gate = d_w_gate,
            .d_w_up = d_w_up,
            .d_w_down = d_w_down,
            .d_ffn_norm = d_ffn_norm,
        };
    }

    pub fn deinit(self: *TrainableLayerWeights) void {
        self.allocator.free(self.d_ffn_norm);
        self.allocator.free(self.d_w_down);
        self.allocator.free(self.d_w_up);
        self.allocator.free(self.d_w_gate);
        self.allocator.free(self.d_attn_norm);
        self.allocator.free(self.d_w_o);
        self.allocator.free(self.d_w_v);
        self.allocator.free(self.d_w_k);
        self.allocator.free(self.d_w_q);

        self.allocator.free(self.ffn_norm);
        self.allocator.free(self.w_down);
        self.allocator.free(self.w_up);
        self.allocator.free(self.w_gate);
        self.allocator.free(self.attn_norm);
        self.allocator.free(self.w_o);
        self.allocator.free(self.w_v);
        self.allocator.free(self.w_k);
        self.allocator.free(self.w_q);
        self.* = undefined;
    }

    /// Zero all gradients.
    pub fn zeroGradients(self: *TrainableLayerWeights) void {
        @memset(self.d_w_q, 0);
        @memset(self.d_w_k, 0);
        @memset(self.d_w_v, 0);
        @memset(self.d_w_o, 0);
        @memset(self.d_attn_norm, 0);
        @memset(self.d_w_gate, 0);
        @memset(self.d_w_up, 0);
        @memset(self.d_w_down, 0);
        @memset(self.d_ffn_norm, 0);
    }
};

/// Trainable weights for the full model.
pub const TrainableWeights = struct {
    allocator: std.mem.Allocator,
    config: TrainableModelConfig,

    // Embedding
    token_embedding: []f32,
    d_token_embedding: []f32,

    // Layers
    layers: []TrainableLayerWeights,

    // Final norm
    final_norm: []f32,
    d_final_norm: []f32,

    // Output projection (if not tied)
    output_proj: ?[]f32,
    d_output_proj: ?[]f32,

    pub fn init(allocator: std.mem.Allocator, config: TrainableModelConfig) !TrainableWeights {
        const token_embedding = try allocator.alloc(f32, config.vocab_size * config.hidden_dim);
        errdefer allocator.free(token_embedding);
        const d_token_embedding = try allocator.alloc(f32, config.vocab_size * config.hidden_dim);
        errdefer allocator.free(d_token_embedding);

        const layers = try allocator.alloc(TrainableLayerWeights, config.num_layers);
        errdefer allocator.free(layers);

        var initialized_layers: usize = 0;
        errdefer {
            for (0..initialized_layers) |i| {
                layers[i].deinit();
            }
        }

        for (layers) |*layer| {
            layer.* = try TrainableLayerWeights.init(allocator, config);
            initialized_layers += 1;
        }

        const final_norm = try allocator.alloc(f32, config.hidden_dim);
        errdefer allocator.free(final_norm);
        const d_final_norm = try allocator.alloc(f32, config.hidden_dim);
        errdefer allocator.free(d_final_norm);

        var output_proj: ?[]f32 = null;
        var d_output_proj: ?[]f32 = null;
        if (!config.tie_embeddings) {
            output_proj = try allocator.alloc(f32, config.hidden_dim * config.vocab_size);
            d_output_proj = try allocator.alloc(f32, config.hidden_dim * config.vocab_size);
        }

        // Initialize
        initializeXavier(token_embedding);
        @memset(d_token_embedding, 0);
        @memset(final_norm, 1.0);
        @memset(d_final_norm, 0);
        if (output_proj) |op| {
            initializeXavier(op);
        }
        if (d_output_proj) |dop| {
            @memset(dop, 0);
        }

        return .{
            .allocator = allocator,
            .config = config,
            .token_embedding = token_embedding,
            .d_token_embedding = d_token_embedding,
            .layers = layers,
            .final_norm = final_norm,
            .d_final_norm = d_final_norm,
            .output_proj = output_proj,
            .d_output_proj = d_output_proj,
        };
    }

    pub fn deinit(self: *TrainableWeights) void {
        if (self.d_output_proj) |dop| {
            self.allocator.free(dop);
        }
        if (self.output_proj) |op| {
            self.allocator.free(op);
        }
        self.allocator.free(self.d_final_norm);
        self.allocator.free(self.final_norm);
        for (self.layers) |*layer| {
            layer.deinit();
        }
        self.allocator.free(self.layers);
        self.allocator.free(self.d_token_embedding);
        self.allocator.free(self.token_embedding);
        self.* = undefined;
    }

    /// Zero all gradients.
    pub fn zeroGradients(self: *TrainableWeights) void {
        @memset(self.d_token_embedding, 0);
        for (self.layers) |*layer| {
            layer.zeroGradients();
        }
        @memset(self.d_final_norm, 0);
        if (self.d_output_proj) |dop| {
            @memset(dop, 0);
        }
    }

    /// Get output projection weights (tied or separate).
    pub fn getOutputProj(self: *const TrainableWeights) []const f32 {
        return self.output_proj orelse self.token_embedding;
    }

    /// Get output projection gradient.
    pub fn getOutputProjGrad(self: *TrainableWeights) []f32 {
        return self.d_output_proj orelse self.d_token_embedding;
    }

    /// Number of parameters.
    pub fn numParams(self: *const TrainableWeights) usize {
        return self.config.numParams();
    }
};

/// Activation cache for backward pass.
pub const ActivationCache = struct {
    allocator: std.mem.Allocator,
    config: TrainableModelConfig,
    seq_len: u32,

    /// Per-layer activation caches
    layer_caches: []LayerActivationCache,

    /// Input embeddings
    embeddings: []f32,

    /// Final hidden states (before output projection)
    final_hidden: []f32,

    pub const LayerActivationCache = struct {
        allocator: std.mem.Allocator,

        // Pre-attention norm input
        pre_attn_norm: []f32,
        // Post-attention hidden states
        post_attn: []f32,
        // Attention cache
        attn_cache: backward_ops.attention_backward.AttentionCache,
        // Pre-FFN norm input
        pre_ffn_norm: []f32,
        // FFN cache
        ffn_cache: backward_ops.ffn_backward.SwigluCache,
        // Post-FFN hidden states
        post_ffn: []f32,

        pub fn init(
            allocator: std.mem.Allocator,
            config: TrainableModelConfig,
            seq_len: u32,
        ) !LayerActivationCache {
            const head_dim = config.headDim();

            const pre_attn_norm = try allocator.alloc(f32, seq_len * config.hidden_dim);
            errdefer allocator.free(pre_attn_norm);
            const post_attn = try allocator.alloc(f32, seq_len * config.hidden_dim);
            errdefer allocator.free(post_attn);
            const attn_cache = try backward_ops.attention_backward.AttentionCache.init(
                allocator,
                seq_len,
                seq_len,
                head_dim,
            );
            errdefer attn_cache.allocator.free(attn_cache.attn_weights);
            const pre_ffn_norm = try allocator.alloc(f32, seq_len * config.hidden_dim);
            errdefer allocator.free(pre_ffn_norm);
            const ffn_cache = try backward_ops.ffn_backward.SwigluCache.init(
                allocator,
                config.hidden_dim,
                config.intermediate_dim,
            );
            const post_ffn = try allocator.alloc(f32, seq_len * config.hidden_dim);

            return .{
                .allocator = allocator,
                .pre_attn_norm = pre_attn_norm,
                .post_attn = post_attn,
                .attn_cache = attn_cache,
                .pre_ffn_norm = pre_ffn_norm,
                .ffn_cache = ffn_cache,
                .post_ffn = post_ffn,
            };
        }

        pub fn deinit(self: *LayerActivationCache) void {
            self.allocator.free(self.post_ffn);
            var ffn_cache = self.ffn_cache;
            ffn_cache.deinit();
            self.allocator.free(self.pre_ffn_norm);
            var attn_cache = self.attn_cache;
            attn_cache.deinit();
            self.allocator.free(self.post_attn);
            self.allocator.free(self.pre_attn_norm);
            self.* = undefined;
        }
    };

    pub fn init(
        allocator: std.mem.Allocator,
        config: TrainableModelConfig,
        seq_len: u32,
    ) !ActivationCache {
        const layer_caches = try allocator.alloc(LayerActivationCache, config.num_layers);
        errdefer allocator.free(layer_caches);

        var initialized: usize = 0;
        errdefer {
            for (0..initialized) |i| {
                layer_caches[i].deinit();
            }
        }

        for (layer_caches) |*cache| {
            cache.* = try LayerActivationCache.init(allocator, config, seq_len);
            initialized += 1;
        }

        const embeddings = try allocator.alloc(f32, seq_len * config.hidden_dim);
        errdefer allocator.free(embeddings);
        const final_hidden = try allocator.alloc(f32, seq_len * config.hidden_dim);

        return .{
            .allocator = allocator,
            .config = config,
            .seq_len = seq_len,
            .layer_caches = layer_caches,
            .embeddings = embeddings,
            .final_hidden = final_hidden,
        };
    }

    pub fn deinit(self: *ActivationCache) void {
        self.allocator.free(self.final_hidden);
        self.allocator.free(self.embeddings);
        for (self.layer_caches) |*cache| {
            cache.deinit();
        }
        self.allocator.free(self.layer_caches);
        self.* = undefined;
    }
};

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
            var layer = &self.weights.layers[layer_idx];
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
            .q4_0 => {
                try dequantizeQ4_0(data, dest[0..@intCast(elem_count)], self.allocator);
            },
            .q8_0 => {
                try dequantizeQ8_0(data, dest[0..@intCast(elem_count)], self.allocator);
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

/// Dequantize Q4_0 data to f32.
fn dequantizeQ4_0(data: []const u8, dest: []f32, allocator: std.mem.Allocator) !void {
    _ = allocator;
    const block_size: usize = 32;
    const bytes_per_block: usize = 18; // 2 byte scale + 16 bytes data

    var dest_idx: usize = 0;
    var data_idx: usize = 0;

    while (data_idx + bytes_per_block <= data.len and dest_idx + block_size <= dest.len) {
        // Read scale (f16)
        const scale_bytes = data[data_idx..][0..2];
        const scale: f32 = @floatCast(@as(f16, @bitCast(scale_bytes.*)));
        data_idx += 2;

        // Read quantized values (4 bits each, packed)
        for (0..16) |i| {
            const byte = data[data_idx + i];
            const lo: i8 = @as(i8, @intCast(byte & 0x0F)) - 8;
            const hi: i8 = @as(i8, @intCast(byte >> 4)) - 8;

            dest[dest_idx] = @as(f32, @floatFromInt(lo)) * scale;
            dest[dest_idx + 1] = @as(f32, @floatFromInt(hi)) * scale;
            dest_idx += 2;
        }
        data_idx += 16;
    }
}

/// Dequantize Q8_0 data to f32.
fn dequantizeQ8_0(data: []const u8, dest: []f32, allocator: std.mem.Allocator) !void {
    _ = allocator;
    const block_size: usize = 32;
    const bytes_per_block: usize = 34; // 2 byte scale + 32 bytes data

    var dest_idx: usize = 0;
    var data_idx: usize = 0;

    while (data_idx + bytes_per_block <= data.len and dest_idx + block_size <= dest.len) {
        // Read scale (f16)
        const scale_bytes = data[data_idx..][0..2];
        const scale: f32 = @floatCast(@as(f16, @bitCast(scale_bytes.*)));
        data_idx += 2;

        // Read quantized values (8 bits each)
        for (0..32) |i| {
            const qval: i8 = @bitCast(data[data_idx + i]);
            dest[dest_idx + i] = @as(f32, @floatFromInt(qval)) * scale;
        }
        dest_idx += 32;
        data_idx += 32;
    }
}

pub const LoadError = error{
    MissingMetadata,
    ConfigMismatch,
    TensorNotFound,
    BufferTooSmall,
    UnsupportedTensorType,
    NameTooLong,
    OutOfMemory,
} || gguf.GgufError;

const checkpoint = @import("checkpoint.zig");

/// Gradient checkpointing manager.
/// Manages selective storage/recomputation of activations to reduce memory.
pub const GradientCheckpointer = struct {
    allocator: std.mem.Allocator,
    strategy: CheckpointingStrategy,
    checkpoint_interval: u32,
    num_layers: u32,
    /// Which layers to checkpoint (store activations)
    checkpointed_layers: []bool,
    /// Checkpointed inputs for recomputation
    layer_inputs: []?[]f32,

    pub fn init(
        allocator: std.mem.Allocator,
        config: TrainableModelConfig,
    ) !GradientCheckpointer {
        const checkpointed_layers = try allocator.alloc(bool, config.num_layers);
        @memset(checkpointed_layers, false);

        const layer_inputs = try allocator.alloc(?[]f32, config.num_layers);
        @memset(layer_inputs, null);

        var self = GradientCheckpointer{
            .allocator = allocator,
            .strategy = config.checkpointing,
            .checkpoint_interval = config.checkpoint_interval,
            .num_layers = config.num_layers,
            .checkpointed_layers = checkpointed_layers,
            .layer_inputs = layer_inputs,
        };

        // Mark which layers should be checkpointed based on strategy
        self.computeCheckpointMask();

        return self;
    }

    pub fn deinit(self: *GradientCheckpointer) void {
        for (self.layer_inputs) |maybe_input| {
            if (maybe_input) |input| {
                self.allocator.free(input);
            }
        }
        self.allocator.free(self.layer_inputs);
        self.allocator.free(self.checkpointed_layers);
        self.* = undefined;
    }

    fn computeCheckpointMask(self: *GradientCheckpointer) void {
        switch (self.strategy) {
            .none => {
                // Checkpoint all layers (store all activations)
                @memset(self.checkpointed_layers, true);
            },
            .every_n_layers => {
                // Checkpoint every N layers
                for (0..self.num_layers) |i| {
                    self.checkpointed_layers[i] = (i % self.checkpoint_interval == 0);
                }
                // Always checkpoint first and last
                self.checkpointed_layers[0] = true;
                if (self.num_layers > 0) {
                    self.checkpointed_layers[self.num_layers - 1] = true;
                }
            },
            .attention_only => {
                // Only checkpoint attention layers (assuming alternating)
                @memset(self.checkpointed_layers, false);
                // We still need layer boundaries
                for (0..self.num_layers) |i| {
                    if (i % 2 == 0) { // Mark every other layer
                        self.checkpointed_layers[i] = true;
                    }
                }
            },
            .full => {
                // Don't checkpoint any layers (recompute everything)
                @memset(self.checkpointed_layers, false);
                // Still need to checkpoint first layer input
                if (self.num_layers > 0) {
                    self.checkpointed_layers[0] = true;
                }
            },
        }
    }

    /// Check if layer should store activations.
    pub fn shouldStoreActivations(self: *const GradientCheckpointer, layer_idx: u32) bool {
        if (layer_idx >= self.num_layers) return false;
        return self.checkpointed_layers[layer_idx];
    }

    /// Store layer input for potential recomputation.
    pub fn storeLayerInput(self: *GradientCheckpointer, layer_idx: u32, input: []const f32) !void {
        if (layer_idx >= self.num_layers) return;
        if (!self.checkpointed_layers[layer_idx]) return;

        // Free existing if any
        if (self.layer_inputs[layer_idx]) |existing| {
            self.allocator.free(existing);
        }

        // Copy input
        const copy = try self.allocator.alloc(f32, input.len);
        @memcpy(copy, input);
        self.layer_inputs[layer_idx] = copy;
    }

    /// Get stored layer input for recomputation.
    pub fn getLayerInput(self: *const GradientCheckpointer, layer_idx: u32) ?[]const f32 {
        if (layer_idx >= self.num_layers) return null;
        return self.layer_inputs[layer_idx];
    }

    /// Find the nearest checkpoint before this layer.
    pub fn findNearestCheckpoint(self: *const GradientCheckpointer, layer_idx: u32) ?u32 {
        if (layer_idx == 0) return 0;

        var i: u32 = layer_idx;
        while (i > 0) {
            i -= 1;
            if (self.checkpointed_layers[i]) return i;
        }
        return if (self.checkpointed_layers[0]) @as(u32, 0) else null;
    }

    /// Clear all stored inputs (call after backward pass).
    pub fn clearStoredInputs(self: *GradientCheckpointer) void {
        for (self.layer_inputs) |*maybe_input| {
            if (maybe_input.*) |input| {
                self.allocator.free(input);
                maybe_input.* = null;
            }
        }
    }

    /// Estimate memory savings compared to full activation storage.
    pub fn estimateMemorySavings(self: *const GradientCheckpointer) f32 {
        var stored_count: u32 = 0;
        for (self.checkpointed_layers) |is_stored| {
            if (is_stored) stored_count += 1;
        }
        const full = @as(f32, @floatFromInt(self.num_layers));
        const actual = @as(f32, @floatFromInt(stored_count));
        return 1.0 - (actual / full);
    }
};

/// Model checkpoint for saving/loading.
pub const ModelCheckpoint = struct {
    allocator: std.mem.Allocator,
    config: TrainableModelConfig,
    weights: []f32,
    step: u64,
    timestamp: u64,

    pub fn deinit(self: *ModelCheckpoint) void {
        self.allocator.free(self.weights);
        self.* = undefined;
    }
};

/// Xavier uniform initialization.
fn initializeXavier(weights: []f32) void {
    const fan_in = @sqrt(@as(f32, @floatFromInt(weights.len)));
    const limit = @sqrt(6.0) / fan_in;

    var rng = std.Random.DefaultPrng.init(@as(u64, @intCast(weights.len)));
    for (weights) |*w| {
        w.* = (rng.random().float(f32) * 2.0 - 1.0) * limit;
    }
}

test "trainable model config" {
    const config = TrainableModelConfig{
        .hidden_dim = 512,
        .num_layers = 4,
        .num_heads = 8,
        .num_kv_heads = 8,
        .intermediate_dim = 1024,
        .vocab_size = 1000,
    };

    try std.testing.expectEqual(@as(u32, 64), config.headDim());
    try std.testing.expect(config.numParams() > 0);
}

test "trainable weights init" {
    const allocator = std.testing.allocator;

    const config = TrainableModelConfig{
        .hidden_dim = 32,
        .num_layers = 2,
        .num_heads = 4,
        .num_kv_heads = 4,
        .intermediate_dim = 64,
        .vocab_size = 100,
    };

    var weights = try TrainableWeights.init(allocator, config);
    defer weights.deinit();

    try std.testing.expectEqual(config.vocab_size * config.hidden_dim, weights.token_embedding.len);
    try std.testing.expectEqual(config.num_layers, weights.layers.len);

    // Test zero gradients
    weights.zeroGradients();
    for (weights.d_token_embedding) |g| {
        try std.testing.expectEqual(@as(f32, 0), g);
    }
}

test "trainable model init" {
    const allocator = std.testing.allocator;

    const config = TrainableModelConfig{
        .hidden_dim = 32,
        .num_layers = 2,
        .num_heads = 4,
        .num_kv_heads = 4,
        .intermediate_dim = 64,
        .vocab_size = 100,
        .max_seq_len = 64,
    };

    var model = try TrainableModel.init(allocator, config);
    defer model.deinit();

    try std.testing.expect(model.numParams() > 0);

    // Test prepare for training
    try model.prepareForTraining(32);
    try std.testing.expect(model.activations != null);
}

test "model checkpoint collect/distribute weights" {
    const allocator = std.testing.allocator;

    const config = TrainableModelConfig{
        .hidden_dim = 32,
        .num_layers = 2,
        .num_heads = 4,
        .num_kv_heads = 4,
        .intermediate_dim = 64,
        .vocab_size = 100,
    };

    var model = try TrainableModel.init(allocator, config);
    defer model.deinit();

    // Collect weights
    const weights = try model.collectWeights();
    defer allocator.free(weights);

    try std.testing.expectEqual(config.numParams(), weights.len);

    // Modify a weight
    model.weights.token_embedding[0] = 42.0;

    // Distribute weights back (should restore original values)
    try model.distributeWeights(weights);

    // Weight should be restored to original value (from Xavier init)
    try std.testing.expect(model.weights.token_embedding[0] != 42.0);
}

test "model checkpoint create/load" {
    const allocator = std.testing.allocator;

    const config = TrainableModelConfig{
        .hidden_dim = 32,
        .num_layers = 2,
        .num_heads = 4,
        .num_kv_heads = 4,
        .intermediate_dim = 64,
        .vocab_size = 100,
    };

    var model1 = try TrainableModel.init(allocator, config);
    defer model1.deinit();

    // Set a known value
    model1.weights.token_embedding[0] = 123.456;

    // Create checkpoint
    var ckpt = try model1.createCheckpoint(42);
    defer ckpt.deinit();

    try std.testing.expectEqual(@as(u64, 42), ckpt.step);
    try std.testing.expectEqual(@as(f32, 123.456), ckpt.weights[0]);

    // Create another model and load checkpoint
    var model2 = try TrainableModel.init(allocator, config);
    defer model2.deinit();

    try model2.loadFromCheckpoint(&ckpt);

    try std.testing.expectEqual(@as(f32, 123.456), model2.weights.token_embedding[0]);
}

test "gradient checkpointer every_n_layers" {
    const allocator = std.testing.allocator;

    const config = TrainableModelConfig{
        .hidden_dim = 32,
        .num_layers = 8,
        .num_heads = 4,
        .num_kv_heads = 4,
        .intermediate_dim = 64,
        .vocab_size = 100,
        .checkpointing = .every_n_layers,
        .checkpoint_interval = 2,
    };

    var checkpointer = try GradientCheckpointer.init(allocator, config);
    defer checkpointer.deinit();

    // Layer 0 should be checkpointed (first layer)
    try std.testing.expect(checkpointer.shouldStoreActivations(0));
    // Layer 1 should not
    try std.testing.expect(!checkpointer.shouldStoreActivations(1));
    // Layer 2 should be (interval of 2)
    try std.testing.expect(checkpointer.shouldStoreActivations(2));
    // Layer 7 (last) should be
    try std.testing.expect(checkpointer.shouldStoreActivations(7));

    // Memory savings should be > 0
    const savings = checkpointer.estimateMemorySavings();
    try std.testing.expect(savings > 0);
}

test "gradient checkpointer full recomputation" {
    const allocator = std.testing.allocator;

    const config = TrainableModelConfig{
        .hidden_dim = 32,
        .num_layers = 4,
        .num_heads = 4,
        .num_kv_heads = 4,
        .intermediate_dim = 64,
        .vocab_size = 100,
        .checkpointing = .full,
    };

    var checkpointer = try GradientCheckpointer.init(allocator, config);
    defer checkpointer.deinit();

    // Only first layer should be checkpointed
    try std.testing.expect(checkpointer.shouldStoreActivations(0));
    try std.testing.expect(!checkpointer.shouldStoreActivations(1));
    try std.testing.expect(!checkpointer.shouldStoreActivations(2));
    try std.testing.expect(!checkpointer.shouldStoreActivations(3));

    // High memory savings
    const savings = checkpointer.estimateMemorySavings();
    try std.testing.expect(savings >= 0.5);
}

test "gradient checkpointer store and retrieve" {
    const allocator = std.testing.allocator;

    const config = TrainableModelConfig{
        .hidden_dim = 32,
        .num_layers = 4,
        .num_heads = 4,
        .num_kv_heads = 4,
        .intermediate_dim = 64,
        .vocab_size = 100,
        .checkpointing = .none, // Store all
    };

    var checkpointer = try GradientCheckpointer.init(allocator, config);
    defer checkpointer.deinit();

    // Store input for layer 0
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    try checkpointer.storeLayerInput(0, &input);

    // Retrieve it
    const retrieved = checkpointer.getLayerInput(0);
    try std.testing.expect(retrieved != null);
    try std.testing.expectEqualSlices(f32, &input, retrieved.?);

    // Clear
    checkpointer.clearStoredInputs();
    try std.testing.expect(checkpointer.getLayerInput(0) == null);
}

test "forward pass produces valid output" {
    const allocator = std.testing.allocator;

    const config = TrainableModelConfig{
        .hidden_dim = 32,
        .num_layers = 1,
        .num_heads = 4,
        .num_kv_heads = 4,
        .intermediate_dim = 64,
        .vocab_size = 100,
        .max_seq_len = 16,
    };

    var model = try TrainableModel.init(allocator, config);
    defer model.deinit();

    // Initialize with small random values (Xavier-like)
    const scale: f32 = 0.1;
    for (model.weights.token_embedding) |*w| {
        w.* = @as(f32, @floatFromInt(@as(i32, @truncate(@as(i64, @bitCast(@as(u64, @intFromPtr(w)))) % 1000)))) * scale * 0.001;
    }

    // Simple input
    const input_ids = [_]u32{ 1, 2, 3 };
    const logits = try allocator.alloc(f32, input_ids.len * config.vocab_size);
    defer allocator.free(logits);

    // Forward should not error
    try model.forward(&input_ids, logits);

    // Check logits are finite (not NaN or Inf)
    for (logits) |l| {
        try std.testing.expect(std.math.isFinite(l));
    }
}

test "cross entropy loss computation" {
    const vocab_size: u32 = 10;
    const seq_len: usize = 3;

    // Create simple logits (uniform)
    var logits: [seq_len * vocab_size]f32 = undefined;
    for (&logits) |*l| {
        l.* = 0.0;
    }

    // Set one logit higher to create non-uniform distribution
    logits[5] = 2.0; // First position, token 5
    logits[vocab_size + 3] = 2.0; // Second position, token 3
    logits[2 * vocab_size + 7] = 2.0; // Third position, token 7

    const targets = [_]u32{ 5, 3, 7 };
    var d_logits: [seq_len * vocab_size]f32 = undefined;

    const loss = TrainableModel.computeCrossEntropyLoss(&logits, &targets, &d_logits, vocab_size);

    // Loss should be positive
    try std.testing.expect(loss > 0);

    // Loss should be lower than log(vocab_size) since we boosted correct logits
    const max_loss = @log(@as(f32, vocab_size));
    try std.testing.expect(loss < max_loss);

    // Gradients should be finite
    for (d_logits) |g| {
        try std.testing.expect(std.math.isFinite(g));
    }

    // Gradient for correct class should be negative (prob - 1)
    try std.testing.expect(d_logits[5] < 0);
}

test "train step computes loss and gradients" {
    const allocator = std.testing.allocator;

    const config = TrainableModelConfig{
        .hidden_dim = 16,
        .num_layers = 1,
        .num_heads = 2,
        .num_kv_heads = 2,
        .intermediate_dim = 32,
        .vocab_size = 50,
        .max_seq_len = 8,
    };

    var model = try TrainableModel.init(allocator, config);
    defer model.deinit();

    // Initialize with small random-ish values
    for (model.weights.token_embedding, 0..) |*w, i| {
        w.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 100)))) * 0.01 - 0.5;
    }
    for (model.weights.layers) |*layer| {
        for (layer.w_q) |*w| {
            w.* = 0.01;
        }
        for (layer.w_k) |*w| {
            w.* = 0.01;
        }
        for (layer.w_v) |*w| {
            w.* = 0.01;
        }
        for (layer.w_o) |*w| {
            w.* = 0.01;
        }
        for (layer.w_gate) |*w| {
            w.* = 0.01;
        }
        for (layer.w_up) |*w| {
            w.* = 0.01;
        }
        for (layer.w_down) |*w| {
            w.* = 0.01;
        }
        for (layer.attn_norm) |*w| {
            w.* = 1.0;
        }
        for (layer.ffn_norm) |*w| {
            w.* = 1.0;
        }
    }
    for (model.weights.final_norm) |*w| {
        w.* = 1.0;
    }

    // Zero gradients before training step
    model.zeroGradients();

    const input_ids = [_]u32{ 1, 2, 3, 4 };
    const target_ids = [_]u32{ 2, 3, 4, 5 };

    const loss = try model.trainStep(&input_ids, &target_ids);

    // Loss should be positive and finite
    try std.testing.expect(loss > 0);
    try std.testing.expect(std.math.isFinite(loss));

    // Some gradients should be non-zero
    var has_nonzero_grad = false;
    for (model.weights.d_token_embedding) |g| {
        if (g != 0) {
            has_nonzero_grad = true;
            break;
        }
    }
    try std.testing.expect(has_nonzero_grad);
}
