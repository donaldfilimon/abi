//! Vision Transformer (ViT) Implementation
//!
//! This module implements the Vision Transformer architecture for image understanding.
//! Based on "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
//! (Dosovitskiy et al., 2020).
//!
//! ## Architecture
//!
//! 1. **Patch Embedding**: Splits image into fixed-size patches and projects to embeddings
//! 2. **Position Embedding**: Learnable position embeddings for each patch position
//! 3. **Transformer Encoder**: Standard transformer encoder blocks with multi-head attention
//! 4. **Classification Head**: Optional MLP head for classification/embedding extraction
//!
//! ## Submodules
//!
//! - `embedding` — Patch embedding and position encoding
//! - `attention` — Multi-head self-attention
//! - `layers` — Transformer blocks, MLP, layer norm, activation functions
//!
//! ## Usage
//!
//! ```zig
//! const vit = @import("vit.zig");
//!
//! // Create ViT-Base/16 config
//! const config = vit.ViTConfig.base(224, 16);
//!
//! // Initialize model
//! var model = try vit.VisionTransformer.init(allocator, config);
//! defer model.deinit();
//!
//! // Forward pass
//! const embeddings = try model.forward(image_tensor);
//! ```

const std = @import("std");

// Submodule re-exports
pub const embedding = @import("vit/embedding.zig");
pub const attention = @import("vit/attention.zig");
pub const layers = @import("vit/layers.zig");

// Type re-exports for backward compatibility
pub const PatchEmbedding = embedding.PatchEmbedding;
pub const MultiHeadAttention = attention.MultiHeadAttention;
pub const softmax = attention.softmax;
pub const TransformerBlock = layers.TransformerBlock;
pub const MLP = layers.MLP;
pub const LayerNorm = layers.LayerNorm;
pub const gelu = layers.gelu;
pub const geluSlice = layers.geluSlice;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for Vision Transformer
pub const ViTConfig = struct {
    /// Input image size (square, e.g., 224)
    image_size: u32 = 224,

    /// Patch size (e.g., 16 for ViT-16)
    patch_size: u32 = 16,

    /// Number of input channels (3 for RGB)
    in_channels: u32 = 3,

    /// Hidden dimension (embedding size)
    hidden_size: u32 = 768,

    /// Number of transformer layers
    num_layers: u32 = 12,

    /// Number of attention heads
    num_heads: u32 = 12,

    /// MLP intermediate dimension (usually 4x hidden_size)
    mlp_dim: u32 = 3072,

    /// Dropout probability
    dropout: f32 = 0.0,

    /// Attention dropout probability
    attention_dropout: f32 = 0.0,

    /// Whether to use class token
    use_class_token: bool = true,

    /// Number of output classes (0 for embedding only)
    num_classes: u32 = 0,

    /// Layer normalization epsilon
    layer_norm_eps: f32 = 1e-6,

    /// Use GELU activation (vs ReLU)
    use_gelu: bool = true,

    /// Pre-norm vs post-norm (pre-norm is more stable)
    pre_norm: bool = true,

    /// Compute number of patches
    pub fn numPatches(self: ViTConfig) u32 {
        const patches_per_side = self.image_size / self.patch_size;
        return patches_per_side * patches_per_side;
    }

    /// Compute sequence length (patches + optional class token)
    pub fn seqLength(self: ViTConfig) u32 {
        return self.numPatches() + @intFromBool(self.use_class_token);
    }

    /// ViT-Tiny configuration (5.7M params)
    pub fn tiny(image_size: u32, patch_size: u32) ViTConfig {
        return .{
            .image_size = image_size,
            .patch_size = patch_size,
            .hidden_size = 192,
            .num_layers = 12,
            .num_heads = 3,
            .mlp_dim = 768,
        };
    }

    /// ViT-Small configuration (22M params)
    pub fn small(image_size: u32, patch_size: u32) ViTConfig {
        return .{
            .image_size = image_size,
            .patch_size = patch_size,
            .hidden_size = 384,
            .num_layers = 12,
            .num_heads = 6,
            .mlp_dim = 1536,
        };
    }

    /// ViT-Base configuration (86M params)
    pub fn base(image_size: u32, patch_size: u32) ViTConfig {
        return .{
            .image_size = image_size,
            .patch_size = patch_size,
            .hidden_size = 768,
            .num_layers = 12,
            .num_heads = 12,
            .mlp_dim = 3072,
        };
    }

    /// ViT-Large configuration (307M params)
    pub fn large(image_size: u32, patch_size: u32) ViTConfig {
        return .{
            .image_size = image_size,
            .patch_size = patch_size,
            .hidden_size = 1024,
            .num_layers = 24,
            .num_heads = 16,
            .mlp_dim = 4096,
        };
    }

    /// ViT-Huge configuration (632M params)
    pub fn huge(image_size: u32, patch_size: u32) ViTConfig {
        return .{
            .image_size = image_size,
            .patch_size = patch_size,
            .hidden_size = 1280,
            .num_layers = 32,
            .num_heads = 16,
            .mlp_dim = 5120,
        };
    }
};

// ============================================================================
// Vision Transformer Model
// ============================================================================

/// Complete Vision Transformer model
pub const VisionTransformer = struct {
    allocator: std.mem.Allocator,
    config: ViTConfig,

    patch_embed: PatchEmbedding,
    blocks: []TransformerBlock,
    final_norm: LayerNorm,

    /// Classification head (optional)
    cls_head: ?[]f32,
    cls_bias: ?[]f32,

    pub fn init(allocator: std.mem.Allocator, config: ViTConfig) !VisionTransformer {
        var patch_embed = try PatchEmbedding.init(allocator, config);
        errdefer patch_embed.deinit();

        const blocks = try allocator.alloc(TransformerBlock, config.num_layers);
        errdefer allocator.free(blocks);

        var initialized_blocks: usize = 0;
        errdefer {
            for (blocks[0..initialized_blocks]) |*block| {
                block.deinit();
            }
        }

        for (blocks) |*block| {
            block.* = try TransformerBlock.init(allocator, config);
            initialized_blocks += 1;
        }

        const final_norm = try LayerNorm.init(allocator, config.hidden_size, config.layer_norm_eps);

        // Classification head
        var cls_head: ?[]f32 = null;
        var cls_bias: ?[]f32 = null;

        if (config.num_classes > 0) {
            cls_head = try allocator.alloc(f32, config.num_classes * config.hidden_size);
            cls_bias = try allocator.alloc(f32, config.num_classes);

            // Xavier initialization
            const scale = @sqrt(2.0 / @as(f32, @floatFromInt(config.hidden_size + config.num_classes)));
            var prng = std.Random.DefaultPrng.init(789);
            for (cls_head.?) |*w| w.* = (prng.random().float(f32) * 2.0 - 1.0) * scale;
            @memset(cls_bias.?, 0.0);
        }

        return .{
            .allocator = allocator,
            .config = config,
            .patch_embed = patch_embed,
            .blocks = blocks,
            .final_norm = final_norm,
            .cls_head = cls_head,
            .cls_bias = cls_bias,
        };
    }

    pub fn deinit(self: *VisionTransformer) void {
        self.patch_embed.deinit();
        for (self.blocks) |*block| {
            block.deinit();
        }
        self.allocator.free(self.blocks);
        self.final_norm.deinit();
        if (self.cls_head) |head| self.allocator.free(head);
        if (self.cls_bias) |bias| self.allocator.free(bias);
    }

    /// Forward pass returning embeddings
    /// Input: [channels, height, width] flattened image tensor
    /// Output: [hidden_size] embedding (class token or pooled)
    pub fn forward(self: *const VisionTransformer, image: []const f32) ![]f32 {
        const seq_len = self.config.seqLength();

        // Patch embedding
        var x = try self.patch_embed.forward(image);

        // Transformer blocks
        for (self.blocks) |*block| {
            const new_x = try block.forward(x, seq_len);
            self.allocator.free(x);
            x = new_x;
        }

        // Final normalization
        const normed = try self.final_norm.forward(x, seq_len);
        self.allocator.free(x);

        // Extract class token embedding or pool
        const hidden = self.config.hidden_size;
        const result_embedding = try self.allocator.alloc(f32, hidden);

        if (self.config.use_class_token) {
            // Use class token (first position)
            @memcpy(result_embedding, normed[0..hidden]);
        } else {
            // Global average pooling
            @memset(result_embedding, 0.0);
            for (0..seq_len) |s| {
                for (0..hidden) |h| {
                    result_embedding[h] += normed[s * hidden + h];
                }
            }
            for (result_embedding) |*e| {
                e.* /= @floatFromInt(seq_len);
            }
        }

        self.allocator.free(normed);
        return result_embedding;
    }

    /// Forward pass with classification
    /// Returns logits if num_classes > 0, otherwise returns embedding
    pub fn classify(self: *const VisionTransformer, image: []const f32) ![]f32 {
        const result_embedding = try self.forward(image);

        if (self.cls_head) |head| {
            defer self.allocator.free(result_embedding);

            const num_classes = self.config.num_classes;
            const hidden = self.config.hidden_size;
            const logits = try self.allocator.alloc(f32, num_classes);

            for (0..num_classes) |c| {
                var sum: f32 = self.cls_bias.?[c];
                for (0..hidden) |h| {
                    sum += result_embedding[h] * head[c * hidden + h];
                }
                logits[c] = sum;
            }

            return logits;
        }

        return result_embedding;
    }

    /// Get number of parameters
    pub fn numParams(self: *const VisionTransformer) usize {
        const cfg = self.config;
        const patch_dim = cfg.patch_size * cfg.patch_size * cfg.in_channels;
        const seq_len = cfg.seqLength();

        var total: usize = 0;

        // Patch embedding
        total += cfg.hidden_size * patch_dim + cfg.hidden_size; // proj + bias
        if (cfg.use_class_token) total += cfg.hidden_size; // cls token
        total += seq_len * cfg.hidden_size; // pos embed

        // Transformer blocks
        const per_block = blk: {
            var block_params: usize = 0;
            // Attention: Q, K, V, O weights and biases
            block_params += 4 * cfg.hidden_size * cfg.hidden_size;
            block_params += 4 * cfg.hidden_size;
            // MLP
            block_params += cfg.mlp_dim * cfg.hidden_size + cfg.mlp_dim;
            block_params += cfg.hidden_size * cfg.mlp_dim + cfg.hidden_size;
            // Layer norms
            block_params += 4 * cfg.hidden_size;
            break :blk block_params;
        };
        total += cfg.num_layers * per_block;

        // Final norm
        total += 2 * cfg.hidden_size;

        // Classification head
        if (cfg.num_classes > 0) {
            total += cfg.num_classes * cfg.hidden_size + cfg.num_classes;
        }

        return total;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "ViTConfig presets" {
    const base_cfg = ViTConfig.base(224, 16);
    try std.testing.expectEqual(@as(u32, 768), base_cfg.hidden_size);
    try std.testing.expectEqual(@as(u32, 12), base_cfg.num_layers);
    try std.testing.expectEqual(@as(u32, 196), base_cfg.numPatches());
    try std.testing.expectEqual(@as(u32, 197), base_cfg.seqLength());

    const large_cfg = ViTConfig.large(224, 16);
    try std.testing.expectEqual(@as(u32, 1024), large_cfg.hidden_size);
    try std.testing.expectEqual(@as(u32, 24), large_cfg.num_layers);
}

test "gelu activation" {
    const result = gelu(0.0);
    try std.testing.expect(@abs(result) < 0.001);

    const positive = gelu(1.0);
    try std.testing.expect(positive > 0.8);

    const negative = gelu(-1.0);
    try std.testing.expect(negative > -0.2 and negative < 0.0);
}

test "softmax" {
    var data = [_]f32{ 1.0, 2.0, 3.0 };
    softmax(&data);

    var sum: f32 = 0.0;
    for (data) |v| sum += v;
    try std.testing.expect(@abs(sum - 1.0) < 0.001);

    // Highest value should have highest probability
    try std.testing.expect(data[2] > data[1]);
    try std.testing.expect(data[1] > data[0]);
}

test "LayerNorm forward" {
    const allocator = std.testing.allocator;

    var ln = try LayerNorm.init(allocator, 4, 1e-5);
    defer ln.deinit();

    const x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const out = try ln.forward(&x, 1);
    defer allocator.free(out);

    // Check normalized output has mean ~0 and std ~1
    var mean: f32 = 0.0;
    for (out) |v| mean += v;
    mean /= 4.0;
    try std.testing.expect(@abs(mean) < 0.01);
}

test "VisionTransformer tiny init and forward" {
    const allocator = std.testing.allocator;

    // Use a very small config for testing
    const config = ViTConfig{
        .image_size = 32,
        .patch_size = 8,
        .in_channels = 3,
        .hidden_size = 64,
        .num_layers = 2,
        .num_heads = 4,
        .mlp_dim = 128,
    };

    var vit_model = try VisionTransformer.init(allocator, config);
    defer vit_model.deinit();

    // Create dummy image
    const img_size = config.in_channels * config.image_size * config.image_size;
    const image = try allocator.alloc(f32, img_size);
    defer allocator.free(image);
    for (image) |*p| p.* = 0.5;

    // Forward pass
    const result_embedding = try vit_model.forward(image);
    defer allocator.free(result_embedding);

    try std.testing.expectEqual(@as(usize, config.hidden_size), result_embedding.len);
}

test "VisionTransformer parameter count" {
    const config = ViTConfig.base(224, 16);
    const allocator = std.testing.allocator;

    var vit_model = try VisionTransformer.init(allocator, config);
    defer vit_model.deinit();

    const params = vit_model.numParams();
    // ViT-Base should have ~86M parameters
    try std.testing.expect(params > 80_000_000 and params < 90_000_000);
}

test {
    std.testing.refAllDecls(@This());
}
