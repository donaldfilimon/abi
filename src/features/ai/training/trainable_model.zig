//! Trainable LLM model wrapper for training.
//!
//! Provides a wrapper around LLM weights that enables:
//! - Mutable weights for gradient updates
//! - Gradient storage for backpropagation
//! - Activation caching for backward pass
//! - Integration with optimizers

const std = @import("std");
const ops = @import("../llm/ops/mod.zig");
const backward = ops.backward;

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
        attn_cache: backward.attention_backward.AttentionCache,
        // Pre-FFN norm input
        pre_ffn_norm: []f32,
        // FFN cache
        ffn_cache: backward.ffn_backward.SwigluCache,
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
            const attn_cache = try backward.attention_backward.AttentionCache.init(
                allocator,
                seq_len,
                seq_len,
                head_dim,
            );
            errdefer attn_cache.allocator.free(attn_cache.attn_weights);
            const pre_ffn_norm = try allocator.alloc(f32, seq_len * config.hidden_dim);
            errdefer allocator.free(pre_ffn_norm);
            const ffn_cache = try backward.ffn_backward.SwigluCache.init(
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
