//! Activation cache for backward pass.

const std = @import("std");
const config_mod = @import("config.zig");
const backward_ops = @import("../../llm/ops/mod.zig").backward;

/// Activation cache for backward pass.
pub const ActivationCache = struct {
    allocator: std.mem.Allocator,
    config: config_mod.TrainableModelConfig,
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
            config: config_mod.TrainableModelConfig,
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
        config: config_mod.TrainableModelConfig,
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

test {
    std.testing.refAllDecls(@This());
}
