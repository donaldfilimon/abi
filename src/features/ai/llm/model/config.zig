//! Model configuration for LLaMA and similar architectures.

const std = @import("std");
const gguf = @import("../io/gguf.zig");

/// LLaMA model configuration.
pub const LlamaConfig = struct {
    /// Hidden dimension
    dim: u32,
    /// Number of transformer layers
    n_layers: u32,
    /// Number of attention heads
    n_heads: u32,
    /// Number of KV heads (for GQA)
    n_kv_heads: u32,
    /// Vocabulary size
    vocab_size: u32,
    /// Maximum sequence length
    max_seq_len: u32,
    /// FFN intermediate dimension (usually 4 * dim or 8/3 * dim for SwiGLU)
    ffn_dim: u32,
    /// RMS norm epsilon
    norm_eps: f32,
    /// RoPE theta base
    rope_theta: f32,
    /// Whether to use tied embeddings
    tie_embeddings: bool,
    /// Architecture name
    arch: []const u8,
    /// Key/query head dimension from GGUF metadata. If zero, derive from dim / n_heads.
    attention_key_length: u32 = 0,
    /// Value head dimension from GGUF metadata. If zero, derive from key head dimension.
    attention_value_length: u32 = 0,

    /// Compute head dimension.
    pub fn headDim(self: LlamaConfig) u32 {
        return self.queryHeadDim();
    }

    /// Compute query/key head dimension.
    pub fn queryHeadDim(self: LlamaConfig) u32 {
        if (self.attention_key_length != 0) return self.attention_key_length;
        if (self.n_heads == 0) return 0;
        return self.dim / self.n_heads;
    }

    /// Compute key head dimension.
    pub fn keyHeadDim(self: LlamaConfig) u32 {
        return self.queryHeadDim();
    }

    /// Compute value head dimension.
    pub fn valueHeadDim(self: LlamaConfig) u32 {
        if (self.attention_value_length != 0) return self.attention_value_length;
        return self.keyHeadDim();
    }

    /// Compute query projection output dimension.
    pub fn queryDim(self: LlamaConfig) u32 {
        return self.queryHeadDim() * self.n_heads;
    }

    /// Compute KV dimension.
    pub fn kvDim(self: LlamaConfig) u32 {
        return self.keyHeadDim() * self.n_kv_heads;
    }

    /// Compute V projection dimension.
    pub fn valueDim(self: LlamaConfig) u32 {
        return self.valueHeadDim() * self.n_kv_heads;
    }

    /// Compute GQA ratio (heads per KV head).
    pub fn gqaRatio(self: LlamaConfig) u32 {
        if (self.n_kv_heads == 0) return 0;
        return self.n_heads / self.n_kv_heads;
    }

    /// Check if attention layout is compatible with the current local LLaMA forward path.
    pub fn supportsLlamaAttentionLayout(self: LlamaConfig) bool {
        if (self.n_heads == 0 or self.n_kv_heads == 0) return false;
        return self.queryDim() == self.dim and
            self.kvDim() == self.valueDim() and
            self.queryHeadDim() == self.keyHeadDim() and
            self.keyHeadDim() == self.valueHeadDim();
    }

    /// Create default LLaMA 7B configuration.
    pub fn llama7B() LlamaConfig {
        return .{
            .dim = 4096,
            .n_layers = 32,
            .n_heads = 32,
            .n_kv_heads = 32,
            .vocab_size = 32000,
            .max_seq_len = 2048,
            .ffn_dim = 11008,
            .norm_eps = 1e-6,
            .rope_theta = 10000.0,
            .tie_embeddings = false,
            .arch = "llama",
        };
    }

    /// Create default LLaMA 13B configuration.
    pub fn llama13B() LlamaConfig {
        return .{
            .dim = 5120,
            .n_layers = 40,
            .n_heads = 40,
            .n_kv_heads = 40,
            .vocab_size = 32000,
            .max_seq_len = 2048,
            .ffn_dim = 13824,
            .norm_eps = 1e-6,
            .rope_theta = 10000.0,
            .tie_embeddings = false,
            .arch = "llama",
        };
    }

    /// Create Mistral 7B configuration.
    pub fn mistral7B() LlamaConfig {
        return .{
            .dim = 4096,
            .n_layers = 32,
            .n_heads = 32,
            .n_kv_heads = 8, // GQA with 4:1 ratio
            .vocab_size = 32000,
            .max_seq_len = 8192,
            .ffn_dim = 14336,
            .norm_eps = 1e-5,
            .rope_theta = 10000.0,
            .tie_embeddings = false,
            .arch = "mistral",
        };
    }

    /// Load configuration from GGUF file.
    pub fn fromGguf(gguf_file: *const gguf.GgufFile) LlamaConfig {
        const arch = gguf_file.getArchitecture() orelse "llama";
        const key_length = gguf_file.getAttentionKeyLength() orelse 0;
        const value_length_raw = gguf_file.getAttentionValueLength() orelse 0;
        const value_length = if (value_length_raw != 0) value_length_raw else key_length;

        return .{
            .dim = gguf_file.getEmbeddingLength() orelse 4096,
            .n_layers = gguf_file.getBlockCount() orelse 32,
            .n_heads = gguf_file.getHeadCount() orelse 32,
            .n_kv_heads = gguf_file.getHeadCountKV() orelse gguf_file.getHeadCount() orelse 32,
            .vocab_size = gguf_file.getVocabSize() orelse 32000,
            .max_seq_len = gguf_file.getContextLength() orelse 2048,
            .ffn_dim = getArchMetadataU32(gguf_file, arch, "feed_forward_length") orelse
                getMetadataU32(gguf_file, "llama.feed_forward_length") orelse
                11008,
            .norm_eps = getArchMetadataF32(gguf_file, arch, "attention.layer_norm_rms_epsilon") orelse
                getMetadataF32(gguf_file, "llama.attention.layer_norm_rms_epsilon") orelse
                1e-6,
            .rope_theta = getArchMetadataF32(gguf_file, arch, "rope.freq_base") orelse
                getMetadataF32(gguf_file, "llama.rope.freq_base") orelse
                10000.0,
            .tie_embeddings = false,
            .arch = arch,
            .attention_key_length = key_length,
            .attention_value_length = value_length,
        };
    }

    /// Print configuration summary.
    pub fn print(self: LlamaConfig, writer: anytype) !void {
        try writer.print("Model Configuration:\n", .{});
        try writer.print("  Architecture: {s}\n", .{self.arch});
        try writer.print("  Hidden dim: {d}\n", .{self.dim});
        try writer.print("  Layers: {d}\n", .{self.n_layers});
        try writer.print("  Attention heads: {d}\n", .{self.n_heads});
        try writer.print("  KV heads: {d}\n", .{self.n_kv_heads});
        try writer.print("  GQA ratio: {d}\n", .{self.gqaRatio()});
        try writer.print("  Q head dim: {d}\n", .{self.queryHeadDim()});
        try writer.print("  KV head dim: {d}\n", .{self.keyHeadDim()});
        try writer.print("  V head dim: {d}\n", .{self.valueHeadDim()});
        try writer.print("  Q dim: {d}\n", .{self.queryDim()});
        try writer.print("  KV dim: {d}\n", .{self.kvDim()});
        try writer.print("  V dim: {d}\n", .{self.valueDim()});
        try writer.print("  Vocab size: {d}\n", .{self.vocab_size});
        try writer.print("  Max seq len: {d}\n", .{self.max_seq_len});
        try writer.print("  FFN dim: {d}\n", .{self.ffn_dim});
        try writer.print("  RoPE theta: {d:.1}\n", .{self.rope_theta});
        try writer.print("  Local LLaMA layout: {s}\n", .{if (self.supportsLlamaAttentionLayout()) "compatible" else "unsupported"});
    }

    /// Estimate model memory requirements in bytes.
    pub fn estimateMemory(self: LlamaConfig) u64 {
        const dim: u64 = self.dim;
        const q_dim: u64 = self.queryDim();
        const kv_dim: u64 = self.kvDim();
        const v_dim: u64 = self.valueDim();
        const ffn_dim: u64 = self.ffn_dim;

        // Embedding: vocab_size * dim
        const embed_bytes = @as(u64, self.vocab_size) * dim * 4;

        // Per layer:
        // - attention: q_proj + k_proj + v_proj + o_proj
        const attn_bytes = (q_dim * dim + // q_proj
            dim * kv_dim + // k_proj
            dim * v_dim + // v_proj
            dim * q_dim) * 4; // o_proj

        // - FFN: gate + up + down
        const ffn_bytes = (dim * ffn_dim * 2 + // gate + up
            ffn_dim * dim) * 4; // down

        // - Norms: 2 per layer
        const norm_bytes = dim * 2 * 4;

        const per_layer = attn_bytes + ffn_bytes + norm_bytes;
        const all_layers = per_layer * @as(u64, self.n_layers);

        // Output projection (if not tied)
        const output_bytes = if (!self.tie_embeddings) @as(u64, self.vocab_size) * self.dim * 4 else 0;

        // KV cache estimate (at max seq len)
        const kv_cache = @as(u64, self.n_layers) * self.max_seq_len * (kv_dim + v_dim) * 4;

        return embed_bytes + all_layers + output_bytes + kv_cache;
    }

    /// Estimate model parameters.
    pub fn estimateParameters(self: LlamaConfig) u64 {
        const dim: u64 = self.dim;
        const q_dim: u64 = self.queryDim();
        const kv_dim: u64 = self.kvDim();
        const v_dim: u64 = self.valueDim();
        const ffn_dim: u64 = self.ffn_dim;

        const embed_params = @as(u64, self.vocab_size) * dim;

        const attn_params_per_layer = dim * q_dim + // q
            dim * kv_dim + // k
            dim * v_dim + // v
            dim * q_dim; // o

        const ffn_params_per_layer = dim * ffn_dim * 2 + // gate + up
            ffn_dim * dim; // down

        const norm_params_per_layer = dim * 2;

        const per_layer = attn_params_per_layer + ffn_params_per_layer + norm_params_per_layer;

        const output_params = if (!self.tie_embeddings) @as(u64, self.vocab_size) * dim else 0;

        return embed_params + per_layer * @as(u64, self.n_layers) + output_params;
    }
};

/// Generic model configuration interface.
pub const ModelConfig = union(enum) {
    llama: LlamaConfig,
    // Future: mistral, phi, etc.
};

fn getMetadataU32(file: *const gguf.GgufFile, key: []const u8) ?u32 {
    const val = file.getMetadata(key) orelse return null;
    return val.asU32();
}

fn getMetadataF32(file: *const gguf.GgufFile, key: []const u8) ?f32 {
    const val = file.getMetadata(key) orelse return null;
    return val.asF32();
}

fn getArchMetadataU32(file: *const gguf.GgufFile, arch: []const u8, suffix: []const u8) ?u32 {
    var key_buf: [128]u8 = undefined;
    const key = std.fmt.bufPrint(&key_buf, "{s}.{s}", .{ arch, suffix }) catch return null;
    return getMetadataU32(file, key);
}

fn getArchMetadataF32(file: *const gguf.GgufFile, arch: []const u8, suffix: []const u8) ?f32 {
    var key_buf: [128]u8 = undefined;
    const key = std.fmt.bufPrint(&key_buf, "{s}.{s}", .{ arch, suffix }) catch return null;
    return getMetadataF32(file, key);
}

test "llama config defaults" {
    const config = LlamaConfig.llama7B();

    try std.testing.expectEqual(@as(u32, 4096), config.dim);
    try std.testing.expectEqual(@as(u32, 32), config.n_layers);
    try std.testing.expectEqual(@as(u32, 128), config.headDim());
    try std.testing.expectEqual(@as(u32, 1), config.gqaRatio());
    try std.testing.expect(config.supportsLlamaAttentionLayout());
}

test "mistral config gqa" {
    const config = LlamaConfig.mistral7B();

    try std.testing.expectEqual(@as(u32, 8), config.n_kv_heads);
    try std.testing.expectEqual(@as(u32, 4), config.gqaRatio());
    try std.testing.expectEqual(@as(u32, 1024), config.kvDim()); // 8 * 128
    try std.testing.expect(config.supportsLlamaAttentionLayout());
}

test "non-llama attention layout is detected" {
    const config = LlamaConfig{
        .dim = 2880,
        .n_layers = 24,
        .n_heads = 64,
        .n_kv_heads = 8,
        .vocab_size = 200000,
        .max_seq_len = 131072,
        .ffn_dim = 2880,
        .norm_eps = 1e-5,
        .rope_theta = 150000.0,
        .tie_embeddings = false,
        .arch = "gptoss",
        .attention_key_length = 64,
        .attention_value_length = 64,
    };

    try std.testing.expectEqual(@as(u32, 4096), config.queryDim());
    try std.testing.expectEqual(@as(u32, 512), config.kvDim());
    try std.testing.expect(!config.supportsLlamaAttentionLayout());
}

test "memory estimation" {
    const config = LlamaConfig.llama7B();
    const memory = config.estimateMemory();

    // Should be in the range of 14-28 GB for 7B model
    try std.testing.expect(memory > 10_000_000_000); // > 10 GB
    try std.testing.expect(memory < 50_000_000_000); // < 50 GB
}

test "parameter estimation" {
    const config = LlamaConfig.llama7B();
    const params = config.estimateParameters();

    // Should be around 7 billion
    try std.testing.expect(params > 6_000_000_000);
    try std.testing.expect(params < 8_000_000_000);
}
