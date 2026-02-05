//! Transformer layer implementation.

const std = @import("std");
const config_mod = @import("config.zig");
const weights_mod = @import("weights.zig");
const ops = @import("../ops/mod.zig");
const cache = @import("../cache/mod.zig");

/// Single transformer layer.
pub const TransformerLayer = struct {
    allocator: std.mem.Allocator,
    config: config_mod.LlamaConfig,
    layer_idx: u32,

    // Scratch buffers
    hidden_scratch: []f32,
    attn_scratch: []f32,
    ffn_scratch: []f32,

    pub fn init(allocator: std.mem.Allocator, llama_config: config_mod.LlamaConfig, layer_idx: u32) !TransformerLayer {
        const hidden_scratch = try allocator.alloc(f32, llama_config.dim);
        errdefer allocator.free(hidden_scratch);

        const attn_scratch = try allocator.alloc(f32, llama_config.dim);
        errdefer allocator.free(attn_scratch);

        const ffn_scratch = try allocator.alloc(f32, llama_config.ffn_dim);

        return .{
            .allocator = allocator,
            .config = llama_config,
            .layer_idx = layer_idx,
            .hidden_scratch = hidden_scratch,
            .attn_scratch = attn_scratch,
            .ffn_scratch = ffn_scratch,
        };
    }

    pub fn deinit(self: *TransformerLayer) void {
        self.allocator.free(self.hidden_scratch);
        self.allocator.free(self.attn_scratch);
        self.allocator.free(self.ffn_scratch);
        self.* = undefined;
    }

    /// Forward pass for a single position.
    pub fn forward(
        self: *TransformerLayer,
        x: []f32, // [dim]
        pos: u32,
        layer_weights: *const weights_mod.LayerWeights,
        kv_cache_layer: *cache.LayerKvCache,
        rope_cache: *const ops.rope.RopeCache,
    ) !void {
        const dim = self.config.dim;
        const head_dim = self.config.headDim();
        const kv_dim = self.config.kvDim();

        // 1. Input normalization
        ops.rmsnorm.rmsNorm(x, layer_weights.input_norm, self.hidden_scratch, self.config.norm_eps);

        // 2. Compute Q, K, V projections
        const q = try self.allocator.alloc(f32, dim);
        defer self.allocator.free(q);
        const k = try self.allocator.alloc(f32, kv_dim);
        defer self.allocator.free(k);
        const v = try self.allocator.alloc(f32, kv_dim);
        defer self.allocator.free(v);

        // Q = x @ W_q
        ops.matmul.matrixVectorMultiply(layer_weights.q_proj, self.hidden_scratch, q, dim, dim);
        // K = x @ W_k
        ops.matmul.matrixVectorMultiply(layer_weights.k_proj, self.hidden_scratch, k, kv_dim, dim);
        // V = x @ W_v
        ops.matmul.matrixVectorMultiply(layer_weights.v_proj, self.hidden_scratch, v, kv_dim, dim);

        // 3. Apply RoPE to Q and K
        for (0..self.config.n_heads) |h| {
            const q_offset = h * head_dim;
            ops.rope.applyRope(q[q_offset .. q_offset + head_dim], pos, rope_cache);
        }

        for (0..self.config.n_kv_heads) |h| {
            const k_offset = h * head_dim;
            ops.rope.applyRope(k[k_offset .. k_offset + head_dim], pos, rope_cache);
        }

        // 4. Update KV cache
        kv_cache_layer.update(k, v, pos);

        // 5. Compute attention
        const attn_config = ops.attention.AttentionConfig.fromModel(dim, self.config.n_heads, self.config.n_kv_heads);

        try ops.attention.selfAttention(
            self.allocator,
            q,
            kv_cache_layer.getK(),
            kv_cache_layer.getV(),
            self.attn_scratch,
            attn_config,
            kv_cache_layer.length(),
        );

        // 6. Output projection
        const attn_out = try self.allocator.alloc(f32, dim);
        defer self.allocator.free(attn_out);
        ops.matmul.matrixVectorMultiply(layer_weights.o_proj, self.attn_scratch, attn_out, dim, dim);

        // 7. Residual connection
        for (0..dim) |i| {
            x[i] += attn_out[i];
        }

        // 8. FFN input norm
        ops.rmsnorm.rmsNorm(x, layer_weights.post_attn_norm, self.hidden_scratch, self.config.norm_eps);

        // 9. FFN (SwiGLU)
        const ffn_out = try self.allocator.alloc(f32, dim);
        defer self.allocator.free(ffn_out);

        try ops.ffn.swiglu(
            self.allocator,
            self.hidden_scratch,
            layer_weights.gate_proj,
            layer_weights.up_proj,
            layer_weights.down_proj,
            ffn_out,
            dim,
            self.config.ffn_dim,
        );

        // 10. Residual connection
        for (0..dim) |i| {
            x[i] += ffn_out[i];
        }
    }

    /// Batch forward pass for prefill with pre-allocated buffers.
    /// This is more efficient than calling forward() repeatedly as it avoids
    /// per-position allocations for Q, K, V, attn_out, and ffn_out buffers.
    pub fn forwardBatch(
        self: *TransformerLayer,
        x: []f32, // [seq_len, dim]
        start_pos: u32,
        seq_len: u32,
        layer_weights: *const weights_mod.LayerWeights,
        kv_cache_layer: *cache.LayerKvCache,
        rope_cache: *const ops.rope.RopeCache,
    ) !void {
        const dim = self.config.dim;
        const head_dim = self.config.headDim();
        const kv_dim = self.config.kvDim();

        // Pre-allocate buffers ONCE outside the loop to avoid repeated allocations
        const q = try self.allocator.alloc(f32, dim);
        defer self.allocator.free(q);
        const k = try self.allocator.alloc(f32, kv_dim);
        defer self.allocator.free(k);
        const v = try self.allocator.alloc(f32, kv_dim);
        defer self.allocator.free(v);
        const attn_out = try self.allocator.alloc(f32, dim);
        defer self.allocator.free(attn_out);
        const ffn_out = try self.allocator.alloc(f32, dim);
        defer self.allocator.free(ffn_out);

        // Process each position reusing the pre-allocated buffers
        for (0..seq_len) |i| {
            const offset = i * dim;
            const pos = start_pos + @as(u32, @intCast(i));
            const x_pos = x[offset .. offset + dim];

            // 1. Input normalization
            ops.rmsnorm.rmsNorm(x_pos, layer_weights.input_norm, self.hidden_scratch, self.config.norm_eps);

            // 2. Compute Q, K, V projections (reusing pre-allocated buffers)
            ops.matmul.matrixVectorMultiply(layer_weights.q_proj, self.hidden_scratch, q, dim, dim);
            ops.matmul.matrixVectorMultiply(layer_weights.k_proj, self.hidden_scratch, k, kv_dim, dim);
            ops.matmul.matrixVectorMultiply(layer_weights.v_proj, self.hidden_scratch, v, kv_dim, dim);

            // 3. Apply RoPE to Q and K
            for (0..self.config.n_heads) |h| {
                const q_offset = h * head_dim;
                ops.rope.applyRope(q[q_offset .. q_offset + head_dim], pos, rope_cache);
            }

            for (0..self.config.n_kv_heads) |h| {
                const k_offset = h * head_dim;
                ops.rope.applyRope(k[k_offset .. k_offset + head_dim], pos, rope_cache);
            }

            // 4. Update KV cache
            kv_cache_layer.update(k, v, pos);

            // 5. Compute attention
            const attn_config = ops.attention.AttentionConfig.fromModel(dim, self.config.n_heads, self.config.n_kv_heads);

            try ops.attention.selfAttention(
                self.allocator,
                q,
                kv_cache_layer.getK(),
                kv_cache_layer.getV(),
                self.attn_scratch,
                attn_config,
                kv_cache_layer.length(),
            );

            // 6. Output projection (reusing pre-allocated attn_out)
            ops.matmul.matrixVectorMultiply(layer_weights.o_proj, self.attn_scratch, attn_out, dim, dim);

            // 7. Residual connection
            for (0..dim) |j| {
                x_pos[j] += attn_out[j];
            }

            // 8. FFN input norm
            ops.rmsnorm.rmsNorm(x_pos, layer_weights.post_attn_norm, self.hidden_scratch, self.config.norm_eps);

            // 9. FFN (SwiGLU) - reusing pre-allocated ffn_out
            try ops.ffn.swiglu(
                self.allocator,
                self.hidden_scratch,
                layer_weights.gate_proj,
                layer_weights.up_proj,
                layer_weights.down_proj,
                ffn_out,
                dim,
                self.config.ffn_dim,
            );

            // 10. Residual connection
            for (0..dim) |j| {
                x_pos[j] += ffn_out[j];
            }
        }
    }
};

test "transformer layer init" {
    const allocator = std.testing.allocator;
    const llama_config = config_mod.LlamaConfig.llama7B();

    var layer = try TransformerLayer.init(allocator, llama_config, 0);
    defer layer.deinit();

    try std.testing.expectEqual(@as(usize, 4096), layer.hidden_scratch.len);
}
