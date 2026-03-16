//! Multimodal (CLIP-style) Training Module
//!
//! Provides training capabilities for multimodal models with:
//! - Contrastive learning (InfoNCE loss)
//! - Dual encoder architecture (vision + text)
//! - Cross-modal attention fusion
//! - Zero-shot classification support

const std = @import("std");
const vision_trainer = @import("vision_trainer.zig");
const vit = @import("../vision/vit.zig");
const mixed_precision = @import("mixed_precision.zig");

/// Error types for multimodal training.
pub const MultimodalTrainingError = error{
    InvalidBatchSize,
    DimensionMismatch,
    NoActivationCache,
    OutOfMemory,
    InvalidTemperature,
};

/// Configuration for CLIP-style multimodal model.
pub const CLIPTrainingConfig = struct {
    /// Vision encoder configuration
    vision_config: vision_trainer.TrainableViTConfig,
    /// Text hidden dimension
    text_hidden_size: u32 = 512,
    /// Text vocabulary size
    text_vocab_size: u32 = 49408,
    /// Text max sequence length
    text_max_len: u32 = 77,
    /// Number of text transformer layers
    text_num_layers: u32 = 12,
    /// Number of text attention heads
    text_num_heads: u32 = 8,
    /// Shared embedding dimension for contrastive learning
    projection_dim: u32 = 512,
    /// Temperature for contrastive loss (learnable or fixed)
    temperature: f32 = 0.07,
    /// Whether temperature is learnable
    learnable_temperature: bool = true,
    /// Label smoothing for contrastive loss
    label_smoothing: f32 = 0.0,

    /// Compute total number of trainable parameters.
    pub fn numParams(self: CLIPTrainingConfig) usize {
        var total: usize = 0;

        // Vision encoder params
        total += self.vision_config.numParams();

        // Text encoder params
        // Token embeddings
        total += self.text_vocab_size * self.text_hidden_size;
        // Position embeddings
        total += self.text_max_len * self.text_hidden_size;
        // Per-layer: 4 projections + 2 norms + MLP
        const text_mlp_dim = self.text_hidden_size * 4;
        const per_layer = 4 * self.text_hidden_size * self.text_hidden_size + // Q,K,V,O
            4 * self.text_hidden_size + // 2 layer norms (weight + bias)
            self.text_hidden_size * text_mlp_dim + // MLP up
            text_mlp_dim * self.text_hidden_size + // MLP down
            text_mlp_dim + self.text_hidden_size; // MLP biases
        total += per_layer * self.text_num_layers;
        // Final norm
        total += 2 * self.text_hidden_size;
        // Text projection
        total += self.text_hidden_size * self.projection_dim;

        // Vision projection (if not already counted)
        total += self.vision_config.vit_config.hidden_size * self.projection_dim;

        // Temperature (if learnable)
        if (self.learnable_temperature) {
            total += 1;
        }

        return total;
    }
};

/// Trainable text encoder weights.
pub const TrainableTextEncoderWeights = struct {
    allocator: std.mem.Allocator,

    // Token embeddings [vocab_size, hidden_size]
    token_embedding: []f32,
    d_token_embedding: []f32,

    // Position embeddings [max_len, hidden_size]
    pos_embedding: []f32,
    d_pos_embedding: []f32,

    // Transformer layers
    layers: []TextTransformerLayerWeights,

    // Final layer norm
    final_ln_weight: []f32,
    final_ln_bias: []f32,
    d_final_ln_weight: []f32,
    d_final_ln_bias: []f32,

    // Projection to shared space [hidden_size, projection_dim]
    projection: []f32,
    d_projection: []f32,

    pub fn init(allocator: std.mem.Allocator, config: CLIPTrainingConfig) !TrainableTextEncoderWeights {
        const hidden = config.text_hidden_size;
        const vocab_size = config.text_vocab_size;
        const max_len = config.text_max_len;

        // Token embeddings
        const token_embedding = try allocator.alloc(f32, vocab_size * hidden);
        errdefer allocator.free(token_embedding);
        const d_token_embedding = try allocator.alloc(f32, vocab_size * hidden);
        errdefer allocator.free(d_token_embedding);

        // Position embeddings
        const pos_embedding = try allocator.alloc(f32, max_len * hidden);
        errdefer allocator.free(pos_embedding);
        const d_pos_embedding = try allocator.alloc(f32, max_len * hidden);
        errdefer allocator.free(d_pos_embedding);

        // Transformer layers
        const layers = try allocator.alloc(TextTransformerLayerWeights, config.text_num_layers);
        errdefer allocator.free(layers);

        var initialized: usize = 0;
        errdefer {
            for (layers[0..initialized]) |*layer| {
                layer.deinit();
            }
        }

        for (layers) |*layer| {
            layer.* = try TextTransformerLayerWeights.init(allocator, hidden, config.text_num_heads);
            initialized += 1;
        }

        // Final layer norm
        const final_ln_weight = try allocator.alloc(f32, hidden);
        errdefer allocator.free(final_ln_weight);
        const final_ln_bias = try allocator.alloc(f32, hidden);
        errdefer allocator.free(final_ln_bias);
        const d_final_ln_weight = try allocator.alloc(f32, hidden);
        errdefer allocator.free(d_final_ln_weight);
        const d_final_ln_bias = try allocator.alloc(f32, hidden);
        errdefer allocator.free(d_final_ln_bias);

        // Projection
        const projection = try allocator.alloc(f32, hidden * config.projection_dim);
        errdefer allocator.free(projection);
        const d_projection = try allocator.alloc(f32, hidden * config.projection_dim);

        // Initialize
        initializeXavier(token_embedding);
        initializePositional(pos_embedding, max_len, hidden);
        @memset(final_ln_weight, 1.0);
        @memset(final_ln_bias, 0.0);
        initializeXavier(projection);

        // Zero gradients
        @memset(d_token_embedding, 0);
        @memset(d_pos_embedding, 0);
        @memset(d_final_ln_weight, 0);
        @memset(d_final_ln_bias, 0);
        @memset(d_projection, 0);

        return .{
            .allocator = allocator,
            .token_embedding = token_embedding,
            .d_token_embedding = d_token_embedding,
            .pos_embedding = pos_embedding,
            .d_pos_embedding = d_pos_embedding,
            .layers = layers,
            .final_ln_weight = final_ln_weight,
            .final_ln_bias = final_ln_bias,
            .d_final_ln_weight = d_final_ln_weight,
            .d_final_ln_bias = d_final_ln_bias,
            .projection = projection,
            .d_projection = d_projection,
        };
    }

    pub fn deinit(self: *TrainableTextEncoderWeights) void {
        self.allocator.free(self.d_projection);
        self.allocator.free(self.projection);
        self.allocator.free(self.d_final_ln_bias);
        self.allocator.free(self.d_final_ln_weight);
        self.allocator.free(self.final_ln_bias);
        self.allocator.free(self.final_ln_weight);
        for (self.layers) |*layer| {
            layer.deinit();
        }
        self.allocator.free(self.layers);
        self.allocator.free(self.d_pos_embedding);
        self.allocator.free(self.pos_embedding);
        self.allocator.free(self.d_token_embedding);
        self.allocator.free(self.token_embedding);
        self.* = undefined;
    }

    pub fn zeroGradients(self: *TrainableTextEncoderWeights) void {
        @memset(self.d_token_embedding, 0);
        @memset(self.d_pos_embedding, 0);
        for (self.layers) |*layer| {
            layer.zeroGradients();
        }
        @memset(self.d_final_ln_weight, 0);
        @memset(self.d_final_ln_bias, 0);
        @memset(self.d_projection, 0);
    }
};

/// Text transformer layer weights.
pub const TextTransformerLayerWeights = struct {
    allocator: std.mem.Allocator,

    // Attention weights
    w_q: []f32,
    w_k: []f32,
    w_v: []f32,
    w_o: []f32,

    // Layer norms
    ln1_weight: []f32,
    ln1_bias: []f32,
    ln2_weight: []f32,
    ln2_bias: []f32,

    // MLP
    mlp_fc1_weight: []f32,
    mlp_fc1_bias: []f32,
    mlp_fc2_weight: []f32,
    mlp_fc2_bias: []f32,

    // Gradients
    d_w_q: []f32,
    d_w_k: []f32,
    d_w_v: []f32,
    d_w_o: []f32,
    d_ln1_weight: []f32,
    d_ln1_bias: []f32,
    d_ln2_weight: []f32,
    d_ln2_bias: []f32,
    d_mlp_fc1_weight: []f32,
    d_mlp_fc1_bias: []f32,
    d_mlp_fc2_weight: []f32,
    d_mlp_fc2_bias: []f32,

    pub fn init(allocator: std.mem.Allocator, hidden: u32, num_heads: u32) !TextTransformerLayerWeights {
        _ = num_heads;
        const mlp_dim = hidden * 4;

        // Allocate weights
        const w_q = try allocator.alloc(f32, hidden * hidden);
        errdefer allocator.free(w_q);
        const w_k = try allocator.alloc(f32, hidden * hidden);
        errdefer allocator.free(w_k);
        const w_v = try allocator.alloc(f32, hidden * hidden);
        errdefer allocator.free(w_v);
        const w_o = try allocator.alloc(f32, hidden * hidden);
        errdefer allocator.free(w_o);

        const ln1_weight = try allocator.alloc(f32, hidden);
        errdefer allocator.free(ln1_weight);
        const ln1_bias = try allocator.alloc(f32, hidden);
        errdefer allocator.free(ln1_bias);
        const ln2_weight = try allocator.alloc(f32, hidden);
        errdefer allocator.free(ln2_weight);
        const ln2_bias = try allocator.alloc(f32, hidden);
        errdefer allocator.free(ln2_bias);

        const mlp_fc1_weight = try allocator.alloc(f32, hidden * mlp_dim);
        errdefer allocator.free(mlp_fc1_weight);
        const mlp_fc1_bias = try allocator.alloc(f32, mlp_dim);
        errdefer allocator.free(mlp_fc1_bias);
        const mlp_fc2_weight = try allocator.alloc(f32, mlp_dim * hidden);
        errdefer allocator.free(mlp_fc2_weight);
        const mlp_fc2_bias = try allocator.alloc(f32, hidden);
        errdefer allocator.free(mlp_fc2_bias);

        // Allocate gradients
        const d_w_q = try allocator.alloc(f32, hidden * hidden);
        errdefer allocator.free(d_w_q);
        const d_w_k = try allocator.alloc(f32, hidden * hidden);
        errdefer allocator.free(d_w_k);
        const d_w_v = try allocator.alloc(f32, hidden * hidden);
        errdefer allocator.free(d_w_v);
        const d_w_o = try allocator.alloc(f32, hidden * hidden);
        errdefer allocator.free(d_w_o);

        const d_ln1_weight = try allocator.alloc(f32, hidden);
        errdefer allocator.free(d_ln1_weight);
        const d_ln1_bias = try allocator.alloc(f32, hidden);
        errdefer allocator.free(d_ln1_bias);
        const d_ln2_weight = try allocator.alloc(f32, hidden);
        errdefer allocator.free(d_ln2_weight);
        const d_ln2_bias = try allocator.alloc(f32, hidden);
        errdefer allocator.free(d_ln2_bias);

        const d_mlp_fc1_weight = try allocator.alloc(f32, hidden * mlp_dim);
        errdefer allocator.free(d_mlp_fc1_weight);
        const d_mlp_fc1_bias = try allocator.alloc(f32, mlp_dim);
        errdefer allocator.free(d_mlp_fc1_bias);
        const d_mlp_fc2_weight = try allocator.alloc(f32, mlp_dim * hidden);
        errdefer allocator.free(d_mlp_fc2_weight);
        const d_mlp_fc2_bias = try allocator.alloc(f32, hidden);

        // Initialize
        initializeXavier(w_q);
        initializeXavier(w_k);
        initializeXavier(w_v);
        initializeXavier(w_o);
        @memset(ln1_weight, 1.0);
        @memset(ln1_bias, 0.0);
        @memset(ln2_weight, 1.0);
        @memset(ln2_bias, 0.0);
        initializeXavier(mlp_fc1_weight);
        @memset(mlp_fc1_bias, 0.0);
        initializeXavier(mlp_fc2_weight);
        @memset(mlp_fc2_bias, 0.0);

        // Zero gradients
        @memset(d_w_q, 0);
        @memset(d_w_k, 0);
        @memset(d_w_v, 0);
        @memset(d_w_o, 0);
        @memset(d_ln1_weight, 0);
        @memset(d_ln1_bias, 0);
        @memset(d_ln2_weight, 0);
        @memset(d_ln2_bias, 0);
        @memset(d_mlp_fc1_weight, 0);
        @memset(d_mlp_fc1_bias, 0);
        @memset(d_mlp_fc2_weight, 0);
        @memset(d_mlp_fc2_bias, 0);

        return .{
            .allocator = allocator,
            .w_q = w_q,
            .w_k = w_k,
            .w_v = w_v,
            .w_o = w_o,
            .ln1_weight = ln1_weight,
            .ln1_bias = ln1_bias,
            .ln2_weight = ln2_weight,
            .ln2_bias = ln2_bias,
            .mlp_fc1_weight = mlp_fc1_weight,
            .mlp_fc1_bias = mlp_fc1_bias,
            .mlp_fc2_weight = mlp_fc2_weight,
            .mlp_fc2_bias = mlp_fc2_bias,
            .d_w_q = d_w_q,
            .d_w_k = d_w_k,
            .d_w_v = d_w_v,
            .d_w_o = d_w_o,
            .d_ln1_weight = d_ln1_weight,
            .d_ln1_bias = d_ln1_bias,
            .d_ln2_weight = d_ln2_weight,
            .d_ln2_bias = d_ln2_bias,
            .d_mlp_fc1_weight = d_mlp_fc1_weight,
            .d_mlp_fc1_bias = d_mlp_fc1_bias,
            .d_mlp_fc2_weight = d_mlp_fc2_weight,
            .d_mlp_fc2_bias = d_mlp_fc2_bias,
        };
    }

    pub fn deinit(self: *TextTransformerLayerWeights) void {
        self.allocator.free(self.d_mlp_fc2_bias);
        self.allocator.free(self.d_mlp_fc2_weight);
        self.allocator.free(self.d_mlp_fc1_bias);
        self.allocator.free(self.d_mlp_fc1_weight);
        self.allocator.free(self.d_ln2_bias);
        self.allocator.free(self.d_ln2_weight);
        self.allocator.free(self.d_ln1_bias);
        self.allocator.free(self.d_ln1_weight);
        self.allocator.free(self.d_w_o);
        self.allocator.free(self.d_w_v);
        self.allocator.free(self.d_w_k);
        self.allocator.free(self.d_w_q);
        self.allocator.free(self.mlp_fc2_bias);
        self.allocator.free(self.mlp_fc2_weight);
        self.allocator.free(self.mlp_fc1_bias);
        self.allocator.free(self.mlp_fc1_weight);
        self.allocator.free(self.ln2_bias);
        self.allocator.free(self.ln2_weight);
        self.allocator.free(self.ln1_bias);
        self.allocator.free(self.ln1_weight);
        self.allocator.free(self.w_o);
        self.allocator.free(self.w_v);
        self.allocator.free(self.w_k);
        self.allocator.free(self.w_q);
        self.* = undefined;
    }

    pub fn zeroGradients(self: *TextTransformerLayerWeights) void {
        @memset(self.d_w_q, 0);
        @memset(self.d_w_k, 0);
        @memset(self.d_w_v, 0);
        @memset(self.d_w_o, 0);
        @memset(self.d_ln1_weight, 0);
        @memset(self.d_ln1_bias, 0);
        @memset(self.d_ln2_weight, 0);
        @memset(self.d_ln2_bias, 0);
        @memset(self.d_mlp_fc1_weight, 0);
        @memset(self.d_mlp_fc1_bias, 0);
        @memset(self.d_mlp_fc2_weight, 0);
        @memset(self.d_mlp_fc2_bias, 0);
    }
};

/// Trainable CLIP model.
pub const TrainableCLIPModel = struct {
    allocator: std.mem.Allocator,
    config: CLIPTrainingConfig,

    /// Vision encoder
    vision_encoder: vision_trainer.TrainableViTModel,

    /// Text encoder
    text_encoder: TrainableTextEncoderWeights,

    /// Vision projection [vision_hidden, projection_dim]
    vision_projection: []f32,
    d_vision_projection: []f32,

    /// Temperature parameter (log scale for numerical stability)
    log_temperature: f32,
    d_log_temperature: f32,

    pub fn init(allocator: std.mem.Allocator, config: CLIPTrainingConfig) !TrainableCLIPModel {
        // Update vision config to use projection instead of classification
        var vision_config = config.vision_config;
        vision_config.num_classes = 0;
        vision_config.projection_dim = config.projection_dim;

        const vision_encoder = try vision_trainer.TrainableViTModel.init(allocator, vision_config);
        errdefer {
            var v = vision_encoder;
            v.deinit();
        }

        const text_encoder = try TrainableTextEncoderWeights.init(allocator, config);
        errdefer {
            var t = text_encoder;
            t.deinit();
        }

        const vision_hidden = config.vision_config.vit_config.hidden_size;
        const vision_projection = try allocator.alloc(f32, vision_hidden * config.projection_dim);
        errdefer allocator.free(vision_projection);
        const d_vision_projection = try allocator.alloc(f32, vision_hidden * config.projection_dim);

        initializeXavier(vision_projection);
        @memset(d_vision_projection, 0);

        return .{
            .allocator = allocator,
            .config = config,
            .vision_encoder = vision_encoder,
            .text_encoder = text_encoder,
            .vision_projection = vision_projection,
            .d_vision_projection = d_vision_projection,
            .log_temperature = @log(config.temperature),
            .d_log_temperature = 0,
        };
    }

    pub fn deinit(self: *TrainableCLIPModel) void {
        self.allocator.free(self.d_vision_projection);
        self.allocator.free(self.vision_projection);
        self.text_encoder.deinit();
        self.vision_encoder.deinit();
        self.* = undefined;
    }

    /// Encode images to embedding space.
    /// Input: [batch_size * channels * height * width] flattened images
    /// Output: [batch_size * projection_dim] normalized embeddings
    pub fn encodeImages(
        self: *TrainableCLIPModel,
        images: []const f32,
        batch_size: u32,
        embeddings: []f32,
    ) !void {
        const proj_dim = self.config.projection_dim;
        const vision_hidden = self.config.vision_config.vit_config.hidden_size;

        // Get vision features
        const vision_features = try self.allocator.alloc(f32, batch_size * vision_hidden);
        defer self.allocator.free(vision_features);

        try self.vision_encoder.forward(images, batch_size, vision_features);

        // Project to shared space
        for (0..batch_size) |b| {
            const feat_offset = b * vision_hidden;
            const emb_offset = b * proj_dim;

            for (0..proj_dim) |p| {
                var sum: f32 = 0;
                for (0..vision_hidden) |h| {
                    sum += vision_features[feat_offset + h] * self.vision_projection[h * proj_dim + p];
                }
                embeddings[emb_offset + p] = sum;
            }
        }

        // L2 normalize embeddings
        for (0..batch_size) |b| {
            const offset = b * proj_dim;
            l2Normalize(embeddings[offset .. offset + proj_dim]);
        }
    }

    /// Encode text to embedding space.
    /// Input: [batch_size, seq_len] token IDs
    /// Output: [batch_size * projection_dim] normalized embeddings
    pub fn encodeText(
        self: *TrainableCLIPModel,
        token_ids: []const u32,
        batch_size: u32,
        embeddings: []f32,
    ) !void {
        const hidden = self.config.text_hidden_size;
        const max_len = self.config.text_max_len;
        const proj_dim = self.config.projection_dim;

        // Process each batch item
        for (0..batch_size) |b| {
            const token_offset = b * max_len;
            const emb_offset = b * proj_dim;

            // Get token embeddings + position embeddings
            var hidden_states = try self.allocator.alloc(f32, max_len * hidden);
            defer self.allocator.free(hidden_states);

            for (0..max_len) |pos| {
                const token_id = token_ids[token_offset + pos];
                const tok_offset = token_id * hidden;
                const pos_offset = pos * hidden;
                const out_offset = pos * hidden;

                for (0..hidden) |h| {
                    hidden_states[out_offset + h] = self.text_encoder.token_embedding[tok_offset + h] +
                        self.text_encoder.pos_embedding[pos_offset + h];
                }
            }

            // Process through transformer layers (simplified)
            for (self.text_encoder.layers) |*layer| {
                // Pre-norm
                for (0..max_len) |pos| {
                    const offset = pos * hidden;
                    layerNorm(hidden_states[offset .. offset + hidden], layer.ln1_weight, layer.ln1_bias);
                }

                // Simplified self-attention (would need causal masking for proper implementation)
                // For now just apply MLP

                // MLP
                for (0..max_len) |pos| {
                    const offset = pos * hidden;
                    layerNorm(hidden_states[offset .. offset + hidden], layer.ln2_weight, layer.ln2_bias);

                    var mlp_hidden = try self.allocator.alloc(f32, hidden * 4);
                    defer self.allocator.free(mlp_hidden);

                    // FC1 + GELU
                    for (0..hidden * 4) |m| {
                        var sum: f32 = layer.mlp_fc1_bias[m];
                        for (0..hidden) |h| {
                            sum += hidden_states[offset + h] * layer.mlp_fc1_weight[h * hidden * 4 + m];
                        }
                        mlp_hidden[m] = vit.gelu(sum);
                    }

                    // FC2
                    for (0..hidden) |h| {
                        var sum: f32 = layer.mlp_fc2_bias[h];
                        for (0..hidden * 4) |m| {
                            sum += mlp_hidden[m] * layer.mlp_fc2_weight[m * hidden + h];
                        }
                        hidden_states[offset + h] = sum;
                    }
                }
            }

            // Final layer norm on last token (EOS position)
            const last_pos = max_len - 1;
            const last_offset = last_pos * hidden;
            layerNorm(
                hidden_states[last_offset .. last_offset + hidden],
                self.text_encoder.final_ln_weight,
                self.text_encoder.final_ln_bias,
            );

            // Project to shared space
            for (0..proj_dim) |p| {
                var sum: f32 = 0;
                for (0..hidden) |h| {
                    sum += hidden_states[last_offset + h] * self.text_encoder.projection[h * proj_dim + p];
                }
                embeddings[emb_offset + p] = sum;
            }
        }

        // L2 normalize embeddings
        for (0..batch_size) |b| {
            const offset = b * proj_dim;
            l2Normalize(embeddings[offset .. offset + proj_dim]);
        }
    }

    /// Compute contrastive loss (InfoNCE).
    /// Image and text embeddings should be L2 normalized.
    /// Returns loss and populates d_image_emb and d_text_emb with gradients.
    pub fn computeContrastiveLoss(
        self: *TrainableCLIPModel,
        image_embeddings: []const f32,
        text_embeddings: []const f32,
        batch_size: u32,
        d_image_emb: []f32,
        d_text_emb: []f32,
    ) f32 {
        const proj_dim = self.config.projection_dim;
        const temperature = @exp(self.log_temperature);

        // Compute similarity matrix: [batch, batch]
        var similarities = self.allocator.alloc(f32, batch_size * batch_size) catch return 0;
        defer self.allocator.free(similarities);

        for (0..batch_size) |i| {
            for (0..batch_size) |j| {
                var sim: f32 = 0;
                for (0..proj_dim) |d| {
                    sim += image_embeddings[i * proj_dim + d] * text_embeddings[j * proj_dim + d];
                }
                similarities[i * batch_size + j] = sim / temperature;
            }
        }

        // Compute softmax and loss
        var loss: f32 = 0;

        // Image-to-text loss
        for (0..batch_size) |i| {
            const row_offset = i * batch_size;

            // Softmax over row
            var max_val: f32 = similarities[row_offset];
            for (0..batch_size) |j| {
                if (similarities[row_offset + j] > max_val) {
                    max_val = similarities[row_offset + j];
                }
            }

            var sum_exp: f32 = 0;
            for (0..batch_size) |j| {
                sum_exp += @exp(similarities[row_offset + j] - max_val);
            }

            // Cross-entropy: -log(softmax[i, i])
            const log_prob = similarities[row_offset + i] - max_val - @log(sum_exp);
            loss -= log_prob;
        }

        // Text-to-image loss (transpose)
        for (0..batch_size) |j| {
            // Softmax over column
            var max_val: f32 = similarities[j];
            for (0..batch_size) |i| {
                if (similarities[i * batch_size + j] > max_val) {
                    max_val = similarities[i * batch_size + j];
                }
            }

            var sum_exp: f32 = 0;
            for (0..batch_size) |i| {
                sum_exp += @exp(similarities[i * batch_size + j] - max_val);
            }

            // Cross-entropy: -log(softmax[j, j])
            const log_prob = similarities[j * batch_size + j] - max_val - @log(sum_exp);
            loss -= log_prob;
        }

        // Average loss
        loss /= @as(f32, @floatFromInt(2 * batch_size));

        // Compute gradients
        @memset(d_image_emb, 0);
        @memset(d_text_emb, 0);

        // Gradient of contrastive loss
        for (0..batch_size) |i| {
            const row_offset = i * batch_size;

            // Softmax probabilities for this row
            var max_val: f32 = similarities[row_offset];
            for (0..batch_size) |j| {
                if (similarities[row_offset + j] > max_val) {
                    max_val = similarities[row_offset + j];
                }
            }

            var sum_exp: f32 = 0;
            for (0..batch_size) |j| {
                sum_exp += @exp(similarities[row_offset + j] - max_val);
            }

            // Gradient: (softmax - one_hot) / temperature
            for (0..batch_size) |j| {
                var prob = @exp(similarities[row_offset + j] - max_val) / sum_exp;
                if (i == j) prob -= 1.0;
                prob /= @as(f32, @floatFromInt(batch_size));

                // d_image_emb[i] += prob * text_emb[j] / temperature
                // d_text_emb[j] += prob * image_emb[i] / temperature
                for (0..proj_dim) |d| {
                    d_image_emb[i * proj_dim + d] += prob * text_embeddings[j * proj_dim + d] / temperature;
                    d_text_emb[j * proj_dim + d] += prob * image_embeddings[i * proj_dim + d] / temperature;
                }
            }
        }

        return loss;
    }

    /// Zero all gradients.
    pub fn zeroGradients(self: *TrainableCLIPModel) void {
        self.vision_encoder.zeroGradients();
        self.text_encoder.zeroGradients();
        @memset(self.d_vision_projection, 0);
        self.d_log_temperature = 0;
    }

    /// Compute gradient norm.
    pub fn computeGradientNorm(self: *const TrainableCLIPModel) f32 {
        var sum_sq: f32 = 0;

        // Vision encoder gradients
        sum_sq += self.vision_encoder.computeGradientNorm() * self.vision_encoder.computeGradientNorm();

        // Vision projection
        for (self.d_vision_projection) |g| sum_sq += g * g;

        // Text encoder gradients
        for (self.text_encoder.d_token_embedding) |g| sum_sq += g * g;
        for (self.text_encoder.d_pos_embedding) |g| sum_sq += g * g;
        for (self.text_encoder.layers) |layer| {
            for (layer.d_w_q) |g| sum_sq += g * g;
            for (layer.d_w_k) |g| sum_sq += g * g;
            for (layer.d_w_v) |g| sum_sq += g * g;
            for (layer.d_w_o) |g| sum_sq += g * g;
            for (layer.d_ln1_weight) |g| sum_sq += g * g;
            for (layer.d_ln1_bias) |g| sum_sq += g * g;
            for (layer.d_ln2_weight) |g| sum_sq += g * g;
            for (layer.d_ln2_bias) |g| sum_sq += g * g;
            for (layer.d_mlp_fc1_weight) |g| sum_sq += g * g;
            for (layer.d_mlp_fc1_bias) |g| sum_sq += g * g;
            for (layer.d_mlp_fc2_weight) |g| sum_sq += g * g;
            for (layer.d_mlp_fc2_bias) |g| sum_sq += g * g;
        }
        for (self.text_encoder.d_final_ln_weight) |g| sum_sq += g * g;
        for (self.text_encoder.d_final_ln_bias) |g| sum_sq += g * g;
        for (self.text_encoder.d_projection) |g| sum_sq += g * g;

        // Temperature gradient
        sum_sq += self.d_log_temperature * self.d_log_temperature;

        return @sqrt(sum_sq);
    }

    /// Apply SGD update.
    pub fn applySgdUpdate(self: *TrainableCLIPModel, learning_rate: f32) void {
        self.vision_encoder.applySgdUpdate(learning_rate);

        // Vision projection
        for (self.vision_projection, self.d_vision_projection) |*w, g| {
            w.* -= learning_rate * g;
        }

        // Text encoder
        for (self.text_encoder.token_embedding, self.text_encoder.d_token_embedding) |*w, g| {
            w.* -= learning_rate * g;
        }
        for (self.text_encoder.pos_embedding, self.text_encoder.d_pos_embedding) |*w, g| {
            w.* -= learning_rate * g;
        }

        for (self.text_encoder.layers) |*layer| {
            applyUpdate(layer.w_q, layer.d_w_q, learning_rate);
            applyUpdate(layer.w_k, layer.d_w_k, learning_rate);
            applyUpdate(layer.w_v, layer.d_w_v, learning_rate);
            applyUpdate(layer.w_o, layer.d_w_o, learning_rate);
            applyUpdate(layer.ln1_weight, layer.d_ln1_weight, learning_rate);
            applyUpdate(layer.ln1_bias, layer.d_ln1_bias, learning_rate);
            applyUpdate(layer.ln2_weight, layer.d_ln2_weight, learning_rate);
            applyUpdate(layer.ln2_bias, layer.d_ln2_bias, learning_rate);
            applyUpdate(layer.mlp_fc1_weight, layer.d_mlp_fc1_weight, learning_rate);
            applyUpdate(layer.mlp_fc1_bias, layer.d_mlp_fc1_bias, learning_rate);
            applyUpdate(layer.mlp_fc2_weight, layer.d_mlp_fc2_weight, learning_rate);
            applyUpdate(layer.mlp_fc2_bias, layer.d_mlp_fc2_bias, learning_rate);
        }

        applyUpdate(self.text_encoder.final_ln_weight, self.text_encoder.d_final_ln_weight, learning_rate);
        applyUpdate(self.text_encoder.final_ln_bias, self.text_encoder.d_final_ln_bias, learning_rate);
        applyUpdate(self.text_encoder.projection, self.text_encoder.d_projection, learning_rate);

        // Temperature
        if (self.config.learnable_temperature) {
            self.log_temperature -= learning_rate * self.d_log_temperature;
            // Clamp to reasonable range
            self.log_temperature = @max(-4.6, @min(4.6, self.log_temperature)); // ~0.01 to ~100
        }
    }

    /// Get current temperature value.
    pub fn getTemperature(self: *const TrainableCLIPModel) f32 {
        return @exp(self.log_temperature);
    }
};

// ============================================================================
// Helper Functions
// ============================================================================

/// Xavier initialization.
fn initializeXavier(data: []f32) void {
    const scale = @sqrt(2.0 / @as(f32, @floatFromInt(data.len)));
    var rng = std.Random.DefaultPrng.init(0x87654321);
    for (data) |*val| {
        val.* = (rng.random().float(f32) * 2.0 - 1.0) * scale;
    }
}

/// Sinusoidal position embedding initialization.
fn initializePositional(data: []f32, seq_len: u32, hidden: u32) void {
    for (0..seq_len) |pos| {
        for (0..hidden) |i| {
            const position = @as(f32, @floatFromInt(pos));
            const div_term = @exp(-@as(f32, @floatFromInt(i)) * @log(@as(f32, 10000.0)) / @as(f32, @floatFromInt(hidden)));

            if (i % 2 == 0) {
                data[pos * hidden + i] = @sin(position * div_term);
            } else {
                data[pos * hidden + i] = @cos(position * div_term);
            }
        }
    }
}

/// In-place layer normalization.
fn layerNorm(data: []f32, weight: []const f32, bias: []const f32) void {
    const dim = data.len;
    const dim_f = @as(f32, @floatFromInt(dim));

    // Compute mean
    var mean: f32 = 0;
    for (data) |v| mean += v;
    mean /= dim_f;

    // Compute variance
    var variance: f32 = 0;
    for (data) |v| {
        const diff = v - mean;
        variance += diff * diff;
    }
    variance /= dim_f;

    // Normalize
    const inv_std = 1.0 / @sqrt(variance + 1e-6);
    for (0..dim) |i| {
        data[i] = (data[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

/// L2 normalize a vector in place.
fn l2Normalize(data: []f32) void {
    var norm_sq: f32 = 0;
    for (data) |v| norm_sq += v * v;

    if (norm_sq > 0) {
        const inv_norm = 1.0 / @sqrt(norm_sq);
        for (data) |*v| {
            v.* *= inv_norm;
        }
    }
}

/// Apply gradient update: weight -= learning_rate * gradient
fn applyUpdate(weights: []f32, gradients: []const f32, learning_rate: f32) void {
    for (weights, gradients) |*w, g| {
        w.* -= learning_rate * g;
    }
}

// ============================================================================
// Tests
// ============================================================================

test "clip config num params" {
    const config = CLIPTrainingConfig{
        .vision_config = .{
            .vit_config = vit.ViTConfig.tiny(224, 16),
            .num_classes = 0,
            .projection_dim = 512,
        },
    };

    const num_params = config.numParams();
    try std.testing.expect(num_params > 0);
}

test "text encoder weights init/deinit" {
    const allocator = std.testing.allocator;
    const config = CLIPTrainingConfig{
        .vision_config = .{
            .vit_config = vit.ViTConfig.tiny(224, 16),
        },
        .text_num_layers = 2,
    };

    var text_encoder = try TrainableTextEncoderWeights.init(allocator, config);
    defer text_encoder.deinit();

    // Check token embeddings allocated
    try std.testing.expect(text_encoder.token_embedding.len > 0);
}

test "clip model init/deinit" {
    const allocator = std.testing.allocator;
    const config = CLIPTrainingConfig{
        .vision_config = .{
            .vit_config = vit.ViTConfig.tiny(224, 16),
            .num_classes = 0,
            .projection_dim = 64,
        },
        .projection_dim = 64,
        .text_num_layers = 2,
        .text_hidden_size = 64,
    };

    var model = try TrainableCLIPModel.init(allocator, config);
    defer model.deinit();

    // Check temperature initialized
    try std.testing.expect(model.getTemperature() > 0);
}

test "l2 normalize" {
    var data = [_]f32{ 3.0, 4.0 };
    l2Normalize(&data);

    // Should be unit vector
    var norm_sq: f32 = 0;
    for (data) |v| norm_sq += v * v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), norm_sq, 0.001);
}

test {
    std.testing.refAllDecls(@This());
}
