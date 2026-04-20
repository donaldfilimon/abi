//! Text Encoding Pipelines
//!
//! Token embedding and transformer-based text encoders for multi-modal learning.
//! Provides TextEmbedding (token + position embeddings) and TextEncoder
//! (full transformer stack with projection to shared embedding space).

const std = @import("std");
const vit = @import("../vit.zig");
const preprocessing = @import("preprocessing.zig");

const MultiModalConfig = preprocessing.MultiModalConfig;

// ============================================================================
// Text Embedding Layer
// ============================================================================

/// Text embedding layer
pub const TextEmbedding = struct {
    allocator: std.mem.Allocator,
    vocab_size: u32,
    hidden_size: u32,
    max_length: u32,

    /// Token embeddings [vocab_size, hidden_size]
    token_embed: []f32,

    /// Position embeddings [max_length, hidden_size]
    pos_embed: []f32,

    pub fn init(allocator: std.mem.Allocator, vocab_size: u32, hidden_size: u32, max_length: u32) !TextEmbedding {
        const token_embed = try allocator.alloc(f32, vocab_size * hidden_size);
        const pos_embed = try allocator.alloc(f32, max_length * hidden_size);

        // Initialize with small random values
        var prng = std.Random.DefaultPrng.init(888);
        for (token_embed) |*w| w.* = (prng.random().float(f32) * 2.0 - 1.0) * 0.02;
        for (pos_embed) |*w| w.* = (prng.random().float(f32) * 2.0 - 1.0) * 0.02;

        return .{
            .allocator = allocator,
            .vocab_size = vocab_size,
            .hidden_size = hidden_size,
            .max_length = max_length,
            .token_embed = token_embed,
            .pos_embed = pos_embed,
        };
    }

    pub fn deinit(self: *TextEmbedding) void {
        self.allocator.free(self.token_embed);
        self.allocator.free(self.pos_embed);
    }

    /// Forward pass: token_ids -> embeddings
    pub fn forward(self: *const TextEmbedding, token_ids: []const u32) ![]f32 {
        const seq_len = @min(token_ids.len, self.max_length);
        const hidden = self.hidden_size;

        const output = try self.allocator.alloc(f32, seq_len * hidden);

        for (0..seq_len) |i| {
            const token_id = @min(token_ids[i], self.vocab_size - 1);
            for (0..hidden) |h| {
                output[i * hidden + h] = self.token_embed[token_id * hidden + h] +
                    self.pos_embed[i * hidden + h];
            }
        }

        return output;
    }
};

// ============================================================================
// Text Encoder (Transformer-based)
// ============================================================================

/// Simplified text encoder using transformer blocks
pub const TextEncoder = struct {
    allocator: std.mem.Allocator,
    config: MultiModalConfig,

    embedding: TextEmbedding,
    blocks: []vit.TransformerBlock,
    final_norm: vit.LayerNorm,

    /// Projection to shared embedding space
    projection: []f32,
    projection_bias: []f32,

    pub fn init(allocator: std.mem.Allocator, config: MultiModalConfig) !TextEncoder {
        const embedding = try TextEmbedding.init(
            allocator,
            config.vocab_size,
            config.text_hidden_size,
            config.max_text_length,
        );
        errdefer embedding.deinit();

        // Create transformer config for text
        const text_vit_config = vit.ViTConfig{
            .hidden_size = config.text_hidden_size,
            .num_layers = config.text_num_layers,
            .num_heads = config.text_num_heads,
            .mlp_dim = config.text_hidden_size * 4,
            .layer_norm_eps = 1e-5,
            .pre_norm = true,
            .use_gelu = true,
            .image_size = 1, // Not used for text
            .patch_size = 1, // Not used for text
        };

        const blocks = try allocator.alloc(vit.TransformerBlock, config.text_num_layers);
        var initialized: usize = 0;
        errdefer {
            for (blocks[0..initialized]) |*b| b.deinit();
            allocator.free(blocks);
        }

        for (blocks) |*block| {
            block.* = try vit.TransformerBlock.init(allocator, text_vit_config);
            initialized += 1;
        }

        const final_norm = try vit.LayerNorm.init(allocator, config.text_hidden_size, 1e-5);

        // Projection layer
        const projection = try allocator.alloc(f32, config.projection_dim * config.text_hidden_size);
        const projection_bias = try allocator.alloc(f32, config.projection_dim);

        const scale = @sqrt(2.0 / @as(f32, @floatFromInt(config.text_hidden_size + config.projection_dim)));
        var prng = std.Random.DefaultPrng.init(999);
        for (projection) |*w| w.* = (prng.random().float(f32) * 2.0 - 1.0) * scale;
        @memset(projection_bias, 0.0);

        return .{
            .allocator = allocator,
            .config = config,
            .embedding = embedding,
            .blocks = blocks,
            .final_norm = final_norm,
            .projection = projection,
            .projection_bias = projection_bias,
        };
    }

    pub fn deinit(self: *TextEncoder) void {
        self.embedding.deinit();
        for (self.blocks) |*block| block.deinit();
        self.allocator.free(self.blocks);
        self.final_norm.deinit();
        self.allocator.free(self.projection);
        self.allocator.free(self.projection_bias);
    }

    /// Forward pass: token_ids -> projection_dim embedding
    pub fn forward(self: *const TextEncoder, token_ids: []const u32) ![]f32 {
        const seq_len = @min(token_ids.len, self.config.max_text_length);
        const hidden = self.config.text_hidden_size;
        const proj_dim = self.config.projection_dim;

        // Embed tokens
        var x = try self.embedding.forward(token_ids);
        defer self.allocator.free(x);

        // Pass through transformer blocks
        for (self.blocks) |*block| {
            const new_x = try block.forward(x, seq_len);
            self.allocator.free(x);
            x = new_x;
        }

        // Final normalization
        const normed = try self.final_norm.forward(x, seq_len);
        self.allocator.free(x);
        defer self.allocator.free(normed);

        // Pool: take the embedding at the [EOS] position (last token)
        // For simplicity, we use the last position
        const last_pos = seq_len - 1;

        // Project to shared embedding space
        const output = try self.allocator.alloc(f32, proj_dim);
        for (0..proj_dim) |p| {
            var sum: f32 = self.projection_bias[p];
            for (0..hidden) |h| {
                sum += normed[last_pos * hidden + h] * self.projection[p * hidden + h];
            }
            output[p] = sum;
        }

        return output;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "TextEmbedding forward" {
    const allocator = std.testing.allocator;

    var embed = try TextEmbedding.init(allocator, 1000, 64, 32);
    defer embed.deinit();

    const tokens = [_]u32{ 1, 2, 3, 4, 5 };
    const output = try embed.forward(&tokens);
    defer allocator.free(output);

    try std.testing.expectEqual(@as(usize, 5 * 64), output.len);
}

test {
    std.testing.refAllDecls(@This());
}
