//! LLaMA model implementation.
//!
//! Provides the full LLaMA model including loading, forward pass, and generation.

const std = @import("std");
const config_mod = @import("config.zig");
const weights_mod = @import("weights.zig");
const layer_mod = @import("layer.zig");
const cache_mod = @import("../cache/mod.zig");
const ops = @import("../ops/mod.zig");
const tok_mod = @import("../tokenizer/mod.zig");
const generation = @import("../generation/mod.zig");
const gguf = @import("../io/gguf.zig");

/// Full LLaMA model.
pub const LlamaModel = struct {
    allocator: std.mem.Allocator,
    config: config_mod.LlamaConfig,
    weights: weights_mod.LlamaWeights,
    layers: []layer_mod.TransformerLayer,
    kv_cache: cache_mod.KvCache,
    rope_cache: ops.rope.RopeCache,
    tokenizer: ?tok_mod.Tokenizer,

    // Scratch buffers
    logits: []f32,
    hidden: []f32,

    pub fn init(allocator: std.mem.Allocator, llama_config: config_mod.LlamaConfig) !LlamaModel {
        if (!llama_config.supportsLlamaAttentionLayout()) {
            return error.UnsupportedArchitecture;
        }

        // Initialize weights (empty initially)
        var w = try weights_mod.LlamaWeights.init(allocator, llama_config);
        errdefer w.deinit();

        // Initialize layers
        const layers = try allocator.alloc(layer_mod.TransformerLayer, llama_config.n_layers);
        errdefer allocator.free(layers);

        for (0..llama_config.n_layers) |i| {
            layers[i] = try layer_mod.TransformerLayer.init(allocator, llama_config, @intCast(i));
        }

        // Initialize KV cache
        const kv_cache = try cache_mod.KvCache.init(allocator, .{
            .num_layers = llama_config.n_layers,
            .num_kv_heads = llama_config.n_kv_heads,
            .head_dim = llama_config.keyHeadDim(),
            .max_seq_len = llama_config.max_seq_len,
        });

        // Initialize RoPE cache
        const rope_cache = try ops.rope.RopeCache.init(allocator, .{
            .head_dim = llama_config.queryHeadDim(),
            .max_seq_len = llama_config.max_seq_len,
            .theta_base = llama_config.rope_theta,
        });

        // Allocate scratch buffers
        const logits = try allocator.alloc(f32, llama_config.vocab_size);
        const hidden = try allocator.alloc(f32, llama_config.dim);

        return .{
            .allocator = allocator,
            .config = llama_config,
            .weights = w,
            .layers = layers,
            .kv_cache = kv_cache,
            .rope_cache = rope_cache,
            .tokenizer = null,
            .logits = logits,
            .hidden = hidden,
        };
    }

    pub fn deinit(self: *LlamaModel) void {
        self.allocator.free(self.logits);
        self.allocator.free(self.hidden);

        for (self.layers) |*layer| {
            layer.deinit();
        }
        self.allocator.free(self.layers);

        self.kv_cache.deinit();
        @constCast(&self.rope_cache).deinit();
        self.weights.deinit();

        if (self.tokenizer) |*t| {
            t.deinit();
        }

        self.* = undefined;
    }

    /// Load model from a GGUF file.
    pub fn load(allocator: std.mem.Allocator, path: []const u8) !LlamaModel {
        // First, open GGUF to get config
        var temp_gguf = try gguf.GgufFile.open(allocator, path);
        defer temp_gguf.deinit();
        const llama_config = config_mod.LlamaConfig.fromGguf(&temp_gguf);

        // Only LLaMA-compatible attention layouts are supported by the current
        // local forward path.
        if (!llama_config.supportsLlamaAttentionLayout()) {
            return error.UnsupportedArchitecture;
        }

        // Initialize model with detected config
        var model = try init(allocator, llama_config);
        errdefer model.deinit();

        // Load weights
        try model.weights.loadFromGguf(path);

        // Load tokenizer from GGUF metadata.
        model.tokenizer = tok_mod.loadFromGguf(allocator, &temp_gguf) catch null;

        return model;
    }

    /// Forward pass for a single token.
    pub fn forward(self: *LlamaModel, token: u32, pos: u32) ![]f32 {
        // Get token embedding
        const embed_offset = @as(usize, token) * self.config.dim;
        @memcpy(self.hidden, self.weights.token_embedding[embed_offset .. embed_offset + self.config.dim]);

        // Pass through all layers
        for (self.layers, 0..) |*layer, i| {
            const layer_weights = self.weights.getLayer(@intCast(i));
            const kv_layer = self.kv_cache.getLayer(@intCast(i));

            try layer.forward(
                self.hidden,
                pos,
                layer_weights,
                kv_layer,
                &self.rope_cache,
            );
        }

        // Final normalization
        ops.rmsnorm.rmsNormInPlace(self.hidden, self.weights.final_norm, self.config.norm_eps);

        // Compute logits
        if (self.weights.output_proj) |output_proj| {
            ops.matmul.matrixVectorMultiply(
                output_proj,
                self.hidden,
                self.logits,
                self.config.vocab_size,
                self.config.dim,
            );
        } else {
            // Tied embeddings
            ops.matmul.matrixVectorMultiply(
                self.weights.token_embedding,
                self.hidden,
                self.logits,
                self.config.vocab_size,
                self.config.dim,
            );
        }

        return self.logits;
    }

    /// Forward pass for multiple tokens (prefill).
    pub fn forwardBatch(self: *LlamaModel, tokens: []const u32, start_pos: u32) ![]f32 {
        // Process each token
        for (tokens, 0..) |token, i| {
            _ = try self.forward(token, start_pos + @as(u32, @intCast(i)));
        }

        // Return logits for last position
        return self.logits;
    }

    /// Generate text continuation.
    pub fn generate(
        self: *LlamaModel,
        prompt_tokens: []const u32,
        gen_config: generation.GeneratorConfig,
    ) ![]u32 {
        var gen = generation.Generator.init(self.allocator, self, gen_config);
        defer gen.deinit();

        return gen.generateTokens(prompt_tokens);
    }

    /// Create a generator for streaming output.
    pub fn generator(self: *LlamaModel, gen_config: generation.GeneratorConfig) generation.Generator {
        return generation.Generator.init(self.allocator, self, gen_config);
    }

    /// Reset the model state (clear KV cache).
    pub fn reset(self: *LlamaModel) void {
        self.kv_cache.clear();
    }

    /// Get current sequence position.
    pub fn currentPosition(self: *const LlamaModel) u32 {
        return self.kv_cache.sequenceLength();
    }

    /// Encode text to tokens.
    pub fn encode(self: *LlamaModel, text: []const u8) ![]u32 {
        if (self.tokenizer) |*t| {
            return t.encode(self.allocator, text);
        }
        return error.TokenizerNotLoaded;
    }

    /// Decode tokens to text.
    pub fn decode(self: *LlamaModel, tokens: []const u32) ![]u8 {
        if (self.tokenizer) |*t| {
            return t.decode(self.allocator, tokens);
        }
        return error.TokenizerNotLoaded;
    }

    /// Get model info.
    pub fn info(self: *const LlamaModel) ModelInfo {
        return .{
            .architecture = self.config.arch,
            .dim = self.config.dim,
            .n_layers = self.config.n_layers,
            .n_heads = self.config.n_heads,
            .n_kv_heads = self.config.n_kv_heads,
            .vocab_size = self.config.vocab_size,
            .max_seq_len = self.config.max_seq_len,
            .current_pos = self.currentPosition(),
            .kv_cache_memory = self.kv_cache.memoryUsed(),
            .weights_memory = self.weights.memoryBytes(),
        };
    }

    /// Print model summary.
    pub fn printSummary(self: *const LlamaModel, writer: anytype) !void {
        const i = self.info();

        try writer.print("LLaMA Model Summary\n", .{});
        try writer.print("===================\n", .{});
        try writer.print("Architecture: {s}\n", .{i.architecture});
        try writer.print("Parameters: ~{d:.1}B\n", .{@as(f64, @floatFromInt(self.config.estimateParameters())) / 1e9});
        try writer.print("Hidden dim: {d}\n", .{i.dim});
        try writer.print("Layers: {d}\n", .{i.n_layers});
        try writer.print("Heads: {d} (KV: {d})\n", .{ i.n_heads, i.n_kv_heads });
        try writer.print("Vocab: {d}\n", .{i.vocab_size});
        try writer.print("Max context: {d}\n", .{i.max_seq_len});
        try writer.print("Current position: {d}\n", .{i.current_pos});
        try writer.print("KV cache: {B}\n", .{i.kv_cache_memory});
        try writer.print("Weights: {B}\n", .{i.weights_memory});
    }
};

pub const ModelInfo = struct {
    architecture: []const u8,
    dim: u32,
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    vocab_size: u32,
    max_seq_len: u32,
    current_pos: u32,
    kv_cache_memory: u64,
    weights_memory: u64,
};

test "llama model init" {
    const allocator = std.testing.allocator;

    // Use a smaller config for testing
    const llama_config = config_mod.LlamaConfig{
        .dim = 64,
        .n_layers = 2,
        .n_heads = 4,
        .n_kv_heads = 4,
        .vocab_size = 256,
        .max_seq_len = 128,
        .ffn_dim = 128,
        .norm_eps = 1e-6,
        .rope_theta = 10000.0,
        .tie_embeddings = true,
        .arch = "test",
    };

    var model = try LlamaModel.init(allocator, llama_config);
    defer model.deinit();

    try std.testing.expectEqual(@as(u32, 2), model.config.n_layers);
    try std.testing.expectEqual(@as(u32, 0), model.currentPosition());
}

test {
    std.testing.refAllDecls(@This());
}
