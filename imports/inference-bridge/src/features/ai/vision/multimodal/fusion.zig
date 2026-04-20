//! Cross-Modal Attention and Feature Fusion
//!
//! Implements cross-attention layers for fusing vision and language modalities,
//! contrastive loss (InfoNCE), fusion blocks with bidirectional attention,
//! and the complete CLIP-style contrastive learning model.

const std = @import("std");
const vit = @import("../vit.zig");
const simd = @import("../../../../foundation/mod.zig").simd;
const preprocessing = @import("preprocessing.zig");
const encoders = @import("encoders.zig");

const MultiModalConfig = preprocessing.MultiModalConfig;
const TextEncoder = encoders.TextEncoder;

// ============================================================================
// Contrastive Loss (InfoNCE)
// ============================================================================

/// InfoNCE contrastive loss for aligning embeddings
pub const ContrastiveLoss = struct {
    temperature: f32,
    log_temperature: ?f32, // Learnable log temperature

    pub fn init(temperature: f32, learnable: bool) ContrastiveLoss {
        return .{
            .temperature = temperature,
            .log_temperature = if (learnable) @log(temperature) else null,
        };
    }

    /// Get effective temperature (learnable or fixed)
    pub fn getTemperature(self: *const ContrastiveLoss) f32 {
        if (self.log_temperature) |log_t| {
            return @exp(log_t);
        }
        return self.temperature;
    }

    /// Compute InfoNCE loss between image and text embeddings
    /// image_embeds: [batch, projection_dim]
    /// text_embeds: [batch, projection_dim]
    /// Returns: scalar loss value
    pub fn forward(
        self: *const ContrastiveLoss,
        allocator: std.mem.Allocator,
        image_embeds: []const f32,
        text_embeds: []const f32,
        batch_size: usize,
        embed_dim: usize,
    ) !f32 {
        const temp = self.getTemperature();

        // Normalize embeddings (L2 normalization)
        const norm_images = try allocator.alloc(f32, batch_size * embed_dim);
        defer allocator.free(norm_images);
        const norm_texts = try allocator.alloc(f32, batch_size * embed_dim);
        defer allocator.free(norm_texts);

        for (0..batch_size) |b| {
            // Compute L2 norm
            var img_norm: f32 = 0.0;
            var txt_norm: f32 = 0.0;
            for (0..embed_dim) |d| {
                img_norm += image_embeds[b * embed_dim + d] * image_embeds[b * embed_dim + d];
                txt_norm += text_embeds[b * embed_dim + d] * text_embeds[b * embed_dim + d];
            }
            img_norm = @sqrt(img_norm + 1e-8);
            txt_norm = @sqrt(txt_norm + 1e-8);

            // Normalize
            for (0..embed_dim) |d| {
                norm_images[b * embed_dim + d] = image_embeds[b * embed_dim + d] / img_norm;
                norm_texts[b * embed_dim + d] = text_embeds[b * embed_dim + d] / txt_norm;
            }
        }

        // Compute similarity matrix: [batch, batch]
        const logits_per_image = try allocator.alloc(f32, batch_size * batch_size);
        defer allocator.free(logits_per_image);
        const logits_per_text = try allocator.alloc(f32, batch_size * batch_size);
        defer allocator.free(logits_per_text);

        for (0..batch_size) |i| {
            for (0..batch_size) |j| {
                var dot: f32 = 0.0;
                for (0..embed_dim) |d| {
                    dot += norm_images[i * embed_dim + d] * norm_texts[j * embed_dim + d];
                }
                logits_per_image[i * batch_size + j] = dot / temp;
                logits_per_text[j * batch_size + i] = dot / temp;
            }
        }

        // Compute cross-entropy loss (labels are diagonal - each image matches its text)
        var loss: f32 = 0.0;

        // Image-to-text loss
        for (0..batch_size) |i| {
            // Softmax denominator
            var max_logit: f32 = logits_per_image[i * batch_size];
            for (0..batch_size) |j| {
                const logit = logits_per_image[i * batch_size + j];
                if (logit > max_logit) max_logit = logit;
            }

            var sum_exp: f32 = 0.0;
            for (0..batch_size) |j| {
                sum_exp += @exp(logits_per_image[i * batch_size + j] - max_logit);
            }

            // Cross-entropy: -log(softmax[correct_index])
            const correct_logit = logits_per_image[i * batch_size + i];
            loss -= (correct_logit - max_logit) - @log(sum_exp);
        }

        // Text-to-image loss
        for (0..batch_size) |i| {
            var max_logit: f32 = logits_per_text[i * batch_size];
            for (0..batch_size) |j| {
                const logit = logits_per_text[i * batch_size + j];
                if (logit > max_logit) max_logit = logit;
            }

            var sum_exp: f32 = 0.0;
            for (0..batch_size) |j| {
                sum_exp += @exp(logits_per_text[i * batch_size + j] - max_logit);
            }

            const correct_logit = logits_per_text[i * batch_size + i];
            loss -= (correct_logit - max_logit) - @log(sum_exp);
        }

        // Average over both directions
        return loss / @as(f32, @floatFromInt(2 * batch_size));
    }

    /// Compute cosine similarity between two embeddings (SIMD-optimized via shared module)
    pub fn cosineSimilarity(embed1: []const f32, embed2: []const f32) f32 {
        return simd.cosineSimilarity(embed1, embed2);
    }
};

// ============================================================================
// Cross-Modal Attention
// ============================================================================

/// Cross-attention layer for fusing vision and language
pub const CrossAttention = struct {
    allocator: std.mem.Allocator,
    hidden_size: u32,
    num_heads: u32,
    head_dim: u32,

    /// Query projection (from target modality)
    wq: []f32,
    bq: []f32,

    /// Key projection (from source modality)
    wk: []f32,
    bk: []f32,

    /// Value projection (from source modality)
    wv: []f32,
    bv: []f32,

    /// Output projection
    wo: []f32,
    bo: []f32,

    pub fn init(allocator: std.mem.Allocator, hidden_size: u32, num_heads: u32) !CrossAttention {
        const head_dim = hidden_size / num_heads;
        const size = hidden_size * hidden_size;

        const self = CrossAttention{
            .allocator = allocator,
            .hidden_size = hidden_size,
            .num_heads = num_heads,
            .head_dim = head_dim,
            .wq = try allocator.alloc(f32, size),
            .bq = try allocator.alloc(f32, hidden_size),
            .wk = try allocator.alloc(f32, size),
            .bk = try allocator.alloc(f32, hidden_size),
            .wv = try allocator.alloc(f32, size),
            .bv = try allocator.alloc(f32, hidden_size),
            .wo = try allocator.alloc(f32, size),
            .bo = try allocator.alloc(f32, hidden_size),
        };

        // Xavier initialization
        const scale = @sqrt(2.0 / @as(f32, @floatFromInt(hidden_size * 2)));
        var prng = std.Random.DefaultPrng.init(555);

        for (self.wq) |*w| w.* = (prng.random().float(f32) * 2.0 - 1.0) * scale;
        for (self.wk) |*w| w.* = (prng.random().float(f32) * 2.0 - 1.0) * scale;
        for (self.wv) |*w| w.* = (prng.random().float(f32) * 2.0 - 1.0) * scale;
        for (self.wo) |*w| w.* = (prng.random().float(f32) * 2.0 - 1.0) * scale;

        @memset(self.bq, 0.0);
        @memset(self.bk, 0.0);
        @memset(self.bv, 0.0);
        @memset(self.bo, 0.0);

        return self;
    }

    pub fn deinit(self: *CrossAttention) void {
        self.allocator.free(self.wq);
        self.allocator.free(self.bq);
        self.allocator.free(self.wk);
        self.allocator.free(self.bk);
        self.allocator.free(self.wv);
        self.allocator.free(self.bv);
        self.allocator.free(self.wo);
        self.allocator.free(self.bo);
    }

    /// Forward pass
    /// query: [query_len, hidden] - target modality
    /// key_value: [kv_len, hidden] - source modality
    /// Returns: [query_len, hidden]
    pub fn forward(
        self: *const CrossAttention,
        query: []const f32,
        key_value: []const f32,
        query_len: usize,
        kv_len: usize,
    ) ![]f32 {
        const hidden = self.hidden_size;
        const num_heads = self.num_heads;
        const head_dim = self.head_dim;

        // Project query from target modality
        const q = try self.allocator.alloc(f32, query_len * hidden);
        defer self.allocator.free(q);

        // Project key and value from source modality
        const k = try self.allocator.alloc(f32, kv_len * hidden);
        defer self.allocator.free(k);
        const v = try self.allocator.alloc(f32, kv_len * hidden);
        defer self.allocator.free(v);

        // Q projection
        for (0..query_len) |s| {
            for (0..hidden) |h| {
                var sum: f32 = self.bq[h];
                for (0..hidden) |i| {
                    sum += query[s * hidden + i] * self.wq[h * hidden + i];
                }
                q[s * hidden + h] = sum;
            }
        }

        // K, V projections
        for (0..kv_len) |s| {
            for (0..hidden) |h| {
                var k_sum: f32 = self.bk[h];
                var v_sum: f32 = self.bv[h];
                for (0..hidden) |i| {
                    k_sum += key_value[s * hidden + i] * self.wk[h * hidden + i];
                    v_sum += key_value[s * hidden + i] * self.wv[h * hidden + i];
                }
                k[s * hidden + h] = k_sum;
                v[s * hidden + h] = v_sum;
            }
        }

        // Attention computation
        const attn_out = try self.allocator.alloc(f32, query_len * hidden);
        errdefer self.allocator.free(attn_out);
        @memset(attn_out, 0.0);

        const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
        const scores = try self.allocator.alloc(f32, kv_len);
        defer self.allocator.free(scores);

        for (0..num_heads) |head| {
            const offset = head * head_dim;

            for (0..query_len) |i| {
                // Compute attention scores
                for (0..kv_len) |j| {
                    var dot: f32 = 0.0;
                    for (0..head_dim) |d| {
                        dot += q[i * hidden + offset + d] * k[j * hidden + offset + d];
                    }
                    scores[j] = dot * attn_scale;
                }

                // Softmax
                vit.softmax(scores);

                // Weighted sum
                for (0..head_dim) |d| {
                    var sum: f32 = 0.0;
                    for (0..kv_len) |j| {
                        sum += scores[j] * v[j * hidden + offset + d];
                    }
                    attn_out[i * hidden + offset + d] = sum;
                }
            }
        }

        // Output projection
        const output = try self.allocator.alloc(f32, query_len * hidden);
        for (0..query_len) |s| {
            for (0..hidden) |h| {
                var sum: f32 = self.bo[h];
                for (0..hidden) |i| {
                    sum += attn_out[s * hidden + i] * self.wo[h * hidden + i];
                }
                output[s * hidden + h] = sum;
            }
        }

        self.allocator.free(attn_out);
        return output;
    }
};

// ============================================================================
// Multi-Modal Fusion Block
// ============================================================================

/// Fusion block combining vision and language through cross-attention
pub const FusionBlock = struct {
    allocator: std.mem.Allocator,
    hidden_size: u32,

    /// Vision attends to text
    vision_cross_attn: CrossAttention,
    vision_norm: vit.LayerNorm,
    vision_mlp: vit.MLP,
    vision_mlp_norm: vit.LayerNorm,

    /// Text attends to vision
    text_cross_attn: CrossAttention,
    text_norm: vit.LayerNorm,
    text_mlp: vit.MLP,
    text_mlp_norm: vit.LayerNorm,

    pub fn init(allocator: std.mem.Allocator, hidden_size: u32, num_heads: u32) !FusionBlock {
        return .{
            .allocator = allocator,
            .hidden_size = hidden_size,
            .vision_cross_attn = try CrossAttention.init(allocator, hidden_size, num_heads),
            .vision_norm = try vit.LayerNorm.init(allocator, hidden_size, 1e-5),
            .vision_mlp = try vit.MLP.init(allocator, hidden_size, hidden_size * 4, true),
            .vision_mlp_norm = try vit.LayerNorm.init(allocator, hidden_size, 1e-5),
            .text_cross_attn = try CrossAttention.init(allocator, hidden_size, num_heads),
            .text_norm = try vit.LayerNorm.init(allocator, hidden_size, 1e-5),
            .text_mlp = try vit.MLP.init(allocator, hidden_size, hidden_size * 4, true),
            .text_mlp_norm = try vit.LayerNorm.init(allocator, hidden_size, 1e-5),
        };
    }

    pub fn deinit(self: *FusionBlock) void {
        self.vision_cross_attn.deinit();
        self.vision_norm.deinit();
        self.vision_mlp.deinit();
        self.vision_mlp_norm.deinit();
        self.text_cross_attn.deinit();
        self.text_norm.deinit();
        self.text_mlp.deinit();
        self.text_mlp_norm.deinit();
    }

    /// Forward pass with bidirectional cross-attention
    pub fn forward(
        self: *const FusionBlock,
        vision_features: []const f32,
        text_features: []const f32,
        vision_len: usize,
        text_len: usize,
    ) !struct { vision: []f32, text: []f32 } {
        const hidden = self.hidden_size;

        // Vision attends to text
        const vision_normed = try self.vision_norm.forward(vision_features, vision_len);
        defer self.allocator.free(vision_normed);

        const vision_attn = try self.vision_cross_attn.forward(vision_normed, text_features, vision_len, text_len);
        defer self.allocator.free(vision_attn);

        // Residual connection
        const vision_residual = try self.allocator.alloc(f32, vision_len * hidden);
        for (0..vision_len * hidden) |i| {
            vision_residual[i] = vision_features[i] + vision_attn[i];
        }

        // MLP
        const vision_mlp_normed = try self.vision_mlp_norm.forward(vision_residual, vision_len);
        defer self.allocator.free(vision_mlp_normed);

        const vision_mlp_out = try self.vision_mlp.forward(vision_mlp_normed, vision_len);
        defer self.allocator.free(vision_mlp_out);

        for (0..vision_len * hidden) |i| {
            vision_residual[i] += vision_mlp_out[i];
        }

        // Text attends to vision
        const text_normed = try self.text_norm.forward(text_features, text_len);
        defer self.allocator.free(text_normed);

        const text_attn = try self.text_cross_attn.forward(text_normed, vision_features, text_len, vision_len);
        defer self.allocator.free(text_attn);

        const text_residual = try self.allocator.alloc(f32, text_len * hidden);
        for (0..text_len * hidden) |i| {
            text_residual[i] = text_features[i] + text_attn[i];
        }

        const text_mlp_normed = try self.text_mlp_norm.forward(text_residual, text_len);
        defer self.allocator.free(text_mlp_normed);

        const text_mlp_out = try self.text_mlp.forward(text_mlp_normed, text_len);
        defer self.allocator.free(text_mlp_out);

        for (0..text_len * hidden) |i| {
            text_residual[i] += text_mlp_out[i];
        }

        return .{ .vision = vision_residual, .text = text_residual };
    }
};

// ============================================================================
// CLIP-Style Model
// ============================================================================

/// Complete CLIP-style contrastive learning model
pub const CLIPModel = struct {
    allocator: std.mem.Allocator,
    config: MultiModalConfig,

    /// Vision encoder (ViT)
    vision_encoder: vit.VisionTransformer,

    /// Vision projection to shared space
    vision_projection: []f32,
    vision_projection_bias: []f32,

    /// Text encoder
    text_encoder: TextEncoder,

    /// Contrastive loss
    contrastive_loss: ContrastiveLoss,

    /// Optional fusion blocks
    fusion_blocks: ?[]FusionBlock,

    pub fn init(allocator: std.mem.Allocator, config: MultiModalConfig) !CLIPModel {
        // Vision encoder
        var vision_config = config.vision_config;
        vision_config.num_classes = 0; // No classification head
        const vision_encoder = try vit.VisionTransformer.init(allocator, vision_config);
        errdefer vision_encoder.deinit();

        // Vision projection
        const vision_projection = try allocator.alloc(f32, config.projection_dim * vision_config.hidden_size);
        const vision_projection_bias = try allocator.alloc(f32, config.projection_dim);

        const scale = @sqrt(2.0 / @as(f32, @floatFromInt(vision_config.hidden_size + config.projection_dim)));
        var prng = std.Random.DefaultPrng.init(111);
        for (vision_projection) |*w| w.* = (prng.random().float(f32) * 2.0 - 1.0) * scale;
        @memset(vision_projection_bias, 0.0);

        // Text encoder
        const text_encoder = try TextEncoder.init(allocator, config);

        // Contrastive loss
        const contrastive_loss = ContrastiveLoss.init(config.temperature, config.learnable_temperature);

        // Optional fusion blocks
        var fusion_blocks: ?[]FusionBlock = null;
        if (config.use_cross_attention and config.cross_attention_layers > 0) {
            const blocks = try allocator.alloc(FusionBlock, config.cross_attention_layers);
            var initialized: usize = 0;
            errdefer {
                for (blocks[0..initialized]) |*b| b.deinit();
                allocator.free(blocks);
            }

            for (blocks) |*block| {
                // Use smaller hidden size for fusion
                const fusion_hidden = @min(vision_config.hidden_size, config.text_hidden_size);
                block.* = try FusionBlock.init(allocator, fusion_hidden, 8);
                initialized += 1;
            }
            fusion_blocks = blocks;
        }

        return .{
            .allocator = allocator,
            .config = config,
            .vision_encoder = vision_encoder,
            .vision_projection = vision_projection,
            .vision_projection_bias = vision_projection_bias,
            .text_encoder = text_encoder,
            .contrastive_loss = contrastive_loss,
            .fusion_blocks = fusion_blocks,
        };
    }

    pub fn deinit(self: *CLIPModel) void {
        self.vision_encoder.deinit();
        self.allocator.free(self.vision_projection);
        self.allocator.free(self.vision_projection_bias);
        self.text_encoder.deinit();
        if (self.fusion_blocks) |blocks| {
            for (blocks) |*block| block.deinit();
            self.allocator.free(blocks);
        }
    }

    /// Encode an image to the shared embedding space
    pub fn encodeImage(self: *const CLIPModel, image: []const f32) ![]f32 {
        const hidden = self.config.vision_config.hidden_size;
        const proj_dim = self.config.projection_dim;

        // Get vision features
        const features = try self.vision_encoder.forward(image);
        defer self.allocator.free(features);

        // Project to shared space
        const output = try self.allocator.alloc(f32, proj_dim);
        for (0..proj_dim) |p| {
            var sum: f32 = self.vision_projection_bias[p];
            for (0..hidden) |h| {
                sum += features[h] * self.vision_projection[p * hidden + h];
            }
            output[p] = sum;
        }

        return output;
    }

    /// Encode text to the shared embedding space
    pub fn encodeText(self: *const CLIPModel, token_ids: []const u32) ![]f32 {
        return self.text_encoder.forward(token_ids);
    }

    /// Compute similarity between image and text
    pub fn computeSimilarity(_: *const CLIPModel, image_embed: []const f32, text_embed: []const f32) f32 {
        return ContrastiveLoss.cosineSimilarity(image_embed, text_embed);
    }

    /// Compute contrastive loss for a batch
    pub fn computeLoss(
        self: *const CLIPModel,
        image_embeds: []const f32,
        text_embeds: []const f32,
        batch_size: usize,
    ) !f32 {
        return self.contrastive_loss.forward(
            self.allocator,
            image_embeds,
            text_embeds,
            batch_size,
            self.config.projection_dim,
        );
    }

    /// Get total parameter count
    pub fn numParams(self: *const CLIPModel) usize {
        var total: usize = 0;

        // Vision encoder
        total += self.vision_encoder.numParams();

        // Vision projection
        total += self.config.projection_dim * self.config.vision_config.hidden_size;
        total += self.config.projection_dim;

        // Text encoder (approximate)
        total += self.config.vocab_size * self.config.text_hidden_size; // Token embeddings
        total += self.config.max_text_length * self.config.text_hidden_size; // Pos embeddings
        total += self.config.text_num_layers * (
            // Per block
            4 * self.config.text_hidden_size * self.config.text_hidden_size + // QKV + O
                4 * self.config.text_hidden_size + // Biases
                self.config.text_hidden_size * 4 * self.config.text_hidden_size * 2 + // MLP
                4 * self.config.text_hidden_size // Layer norms
        );
        total += self.config.projection_dim * self.config.text_hidden_size; // Projection
        total += self.config.projection_dim; // Projection bias

        return total;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "ContrastiveLoss cosine similarity" {
    const a = [_]f32{ 1.0, 0.0, 0.0 };
    const b = [_]f32{ 1.0, 0.0, 0.0 };
    const c = [_]f32{ 0.0, 1.0, 0.0 };

    const sim_ab = ContrastiveLoss.cosineSimilarity(&a, &b);
    const sim_ac = ContrastiveLoss.cosineSimilarity(&a, &c);

    try std.testing.expect(@abs(sim_ab - 1.0) < 0.001);
    try std.testing.expect(@abs(sim_ac) < 0.001);
}

test "ContrastiveLoss forward" {
    const allocator = std.testing.allocator;
    const loss = ContrastiveLoss.init(0.07, false);

    // Simple batch of 2
    const image_embeds = [_]f32{ 1.0, 0.0, 0.0, 1.0, 0.0, 1.0 };
    const text_embeds = [_]f32{ 1.0, 0.0, 0.0, 0.0, 0.0, 1.0 };

    const loss_val = try loss.forward(allocator, &image_embeds, &text_embeds, 2, 3);

    // Loss should be positive
    try std.testing.expect(loss_val >= 0.0);
}

test "CrossAttention forward" {
    const allocator = std.testing.allocator;

    var cross_attn = try CrossAttention.init(allocator, 64, 4);
    defer cross_attn.deinit();

    const query = try allocator.alloc(f32, 4 * 64);
    defer allocator.free(query);
    for (query) |*v| v.* = 0.1;

    const kv = try allocator.alloc(f32, 8 * 64);
    defer allocator.free(kv);
    for (kv) |*v| v.* = 0.2;

    const output = try cross_attn.forward(query, kv, 4, 8);
    defer allocator.free(output);

    try std.testing.expectEqual(@as(usize, 4 * 64), output.len);
}

test {
    std.testing.refAllDecls(@This());
}
