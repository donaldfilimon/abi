//! Vision Transformer (ViT) Training Module
//!
//! Provides training capabilities for Vision Transformer models with:
//! - Forward/backward pass with activation caching
//! - Gradient clipping and mixed precision support
//! - Image classification and contrastive learning objectives
//! - Checkpoint save/load functionality

const std = @import("std");
const vit = @import("../vision/vit.zig");
const mixed_precision = @import("mixed_precision.zig");

/// Error types for vision training.
pub const VisionTrainingError = error{
    InvalidImageSize,
    InvalidBatchSize,
    ConfigMismatch,
    NoActivationCache,
    OutOfMemory,
};

/// Configuration for trainable ViT model.
pub const TrainableViTConfig = struct {
    /// Vision Transformer architecture config
    vit_config: vit.ViTConfig,
    /// Maximum batch size for forward pass (activation cache is sized for this)
    max_batch_size: u32 = 1,
    /// Number of output classes (for classification)
    num_classes: u32 = 1000,
    /// Projection dimension for contrastive learning (0 = disabled)
    projection_dim: u32 = 0,
    /// Dropout rate during training
    dropout: f32 = 0.1,
    /// Label smoothing for classification
    label_smoothing: f32 = 0.1,
    /// Enable gradient checkpointing
    gradient_checkpointing: bool = false,

    /// Compute total number of trainable parameters.
    pub fn numParams(self: TrainableViTConfig) usize {
        const cfg = self.vit_config;
        var total: usize = 0;

        // Patch embedding: in_channels * patch_size^2 * hidden_size
        const patch_dim = cfg.in_channels * cfg.patch_size * cfg.patch_size;
        total += patch_dim * cfg.hidden_size;

        // Position embeddings
        total += cfg.seqLength() * cfg.hidden_size;

        // Class token (if used)
        if (cfg.use_class_token) {
            total += cfg.hidden_size;
        }

        // Per-layer parameters
        const per_layer: usize =
            // Q, K, V projections
            3 * cfg.hidden_size * cfg.hidden_size +
            // Output projection
            cfg.hidden_size * cfg.hidden_size +
            // MLP: hidden -> mlp_dim -> hidden
            cfg.hidden_size * cfg.mlp_dim +
            cfg.mlp_dim * cfg.hidden_size +
            // Layer norms (2 per layer)
            2 * cfg.hidden_size;

        total += per_layer * cfg.num_layers;

        // Final layer norm
        total += cfg.hidden_size;

        // Classification head
        if (self.num_classes > 0) {
            total += cfg.hidden_size * self.num_classes;
        }

        // Projection head for contrastive learning
        if (self.projection_dim > 0) {
            total += cfg.hidden_size * self.projection_dim;
        }

        return total;
    }
};

/// Trainable weights for a single ViT transformer layer.
pub const TrainableViTLayerWeights = struct {
    allocator: std.mem.Allocator,

    // Attention weights [hidden_size, hidden_size] for Q, K, V, O
    w_q: []f32,
    w_k: []f32,
    w_v: []f32,
    w_o: []f32,

    // Layer norms [hidden_size]
    ln1_weight: []f32,
    ln1_bias: []f32,
    ln2_weight: []f32,
    ln2_bias: []f32,

    // MLP weights
    mlp_fc1_weight: []f32, // [hidden_size, mlp_dim]
    mlp_fc1_bias: []f32, // [mlp_dim]
    mlp_fc2_weight: []f32, // [mlp_dim, hidden_size]
    mlp_fc2_bias: []f32, // [hidden_size]

    // Gradients (same shapes as weights)
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

    pub fn init(allocator: std.mem.Allocator, config: vit.ViTConfig) !TrainableViTLayerWeights {
        const hidden = config.hidden_size;
        const mlp_dim = config.mlp_dim;

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

        // Initialize weights
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

    pub fn deinit(self: *TrainableViTLayerWeights) void {
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

    pub fn zeroGradients(self: *TrainableViTLayerWeights) void {
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

/// Trainable Vision Transformer weights.
pub const TrainableViTWeights = struct {
    allocator: std.mem.Allocator,

    // Patch embedding projection [patch_dim, hidden_size]
    patch_proj: []f32,
    d_patch_proj: []f32,

    // Position embeddings [seq_len, hidden_size]
    pos_embed: []f32,
    d_pos_embed: []f32,

    // Class token [hidden_size] (if used)
    cls_token: ?[]f32,
    d_cls_token: ?[]f32,

    // Transformer layers
    layers: []TrainableViTLayerWeights,

    // Final layer norm
    final_ln_weight: []f32,
    final_ln_bias: []f32,
    d_final_ln_weight: []f32,
    d_final_ln_bias: []f32,

    // Classification head [hidden_size, num_classes] (if classification)
    classifier_weight: ?[]f32,
    classifier_bias: ?[]f32,
    d_classifier_weight: ?[]f32,
    d_classifier_bias: ?[]f32,

    // Projection head [hidden_size, projection_dim] (if contrastive)
    projection_weight: ?[]f32,
    d_projection_weight: ?[]f32,

    pub fn init(allocator: std.mem.Allocator, config: TrainableViTConfig) !TrainableViTWeights {
        const cfg = config.vit_config;
        const hidden = cfg.hidden_size;
        const seq_len = cfg.seqLength();
        const patch_dim = cfg.in_channels * cfg.patch_size * cfg.patch_size;

        // Patch embedding
        const patch_proj = try allocator.alloc(f32, patch_dim * hidden);
        errdefer allocator.free(patch_proj);
        const d_patch_proj = try allocator.alloc(f32, patch_dim * hidden);
        errdefer allocator.free(d_patch_proj);

        // Position embeddings
        const pos_embed = try allocator.alloc(f32, seq_len * hidden);
        errdefer allocator.free(pos_embed);
        const d_pos_embed = try allocator.alloc(f32, seq_len * hidden);
        errdefer allocator.free(d_pos_embed);

        // Class token
        var cls_token: ?[]f32 = null;
        var d_cls_token: ?[]f32 = null;
        if (cfg.use_class_token) {
            cls_token = try allocator.alloc(f32, hidden);
            errdefer if (cls_token) |c| allocator.free(c);
            d_cls_token = try allocator.alloc(f32, hidden);
            errdefer if (d_cls_token) |d| allocator.free(d);
        }

        // Transformer layers
        const layers = try allocator.alloc(TrainableViTLayerWeights, cfg.num_layers);
        errdefer allocator.free(layers);
        var initialized_layers: usize = 0;
        errdefer {
            for (layers[0..initialized_layers]) |*layer| {
                layer.deinit();
            }
        }
        for (layers) |*layer| {
            layer.* = try TrainableViTLayerWeights.init(allocator, cfg);
            initialized_layers += 1;
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

        // Classification head
        var classifier_weight: ?[]f32 = null;
        var classifier_bias: ?[]f32 = null;
        var d_classifier_weight: ?[]f32 = null;
        var d_classifier_bias: ?[]f32 = null;
        if (config.num_classes > 0) {
            classifier_weight = try allocator.alloc(f32, hidden * config.num_classes);
            errdefer if (classifier_weight) |c| allocator.free(c);
            classifier_bias = try allocator.alloc(f32, config.num_classes);
            errdefer if (classifier_bias) |c| allocator.free(c);
            d_classifier_weight = try allocator.alloc(f32, hidden * config.num_classes);
            errdefer if (d_classifier_weight) |d| allocator.free(d);
            d_classifier_bias = try allocator.alloc(f32, config.num_classes);
        }

        // Projection head
        var projection_weight: ?[]f32 = null;
        var d_projection_weight: ?[]f32 = null;
        if (config.projection_dim > 0) {
            projection_weight = try allocator.alloc(f32, hidden * config.projection_dim);
            errdefer if (projection_weight) |p| allocator.free(p);
            d_projection_weight = try allocator.alloc(f32, hidden * config.projection_dim);
        }

        // Initialize weights
        initializeXavier(patch_proj);
        initializePositional(pos_embed, seq_len, hidden);
        if (cls_token) |c| initializeXavier(c);
        @memset(final_ln_weight, 1.0);
        @memset(final_ln_bias, 0.0);
        if (classifier_weight) |c| initializeXavier(c);
        if (classifier_bias) |c| @memset(c, 0.0);
        if (projection_weight) |p| initializeXavier(p);

        // Zero gradients
        @memset(d_patch_proj, 0);
        @memset(d_pos_embed, 0);
        if (d_cls_token) |d| @memset(d, 0);
        @memset(d_final_ln_weight, 0);
        @memset(d_final_ln_bias, 0);
        if (d_classifier_weight) |d| @memset(d, 0);
        if (d_classifier_bias) |d| @memset(d, 0);
        if (d_projection_weight) |d| @memset(d, 0);

        return .{
            .allocator = allocator,
            .patch_proj = patch_proj,
            .d_patch_proj = d_patch_proj,
            .pos_embed = pos_embed,
            .d_pos_embed = d_pos_embed,
            .cls_token = cls_token,
            .d_cls_token = d_cls_token,
            .layers = layers,
            .final_ln_weight = final_ln_weight,
            .final_ln_bias = final_ln_bias,
            .d_final_ln_weight = d_final_ln_weight,
            .d_final_ln_bias = d_final_ln_bias,
            .classifier_weight = classifier_weight,
            .classifier_bias = classifier_bias,
            .d_classifier_weight = d_classifier_weight,
            .d_classifier_bias = d_classifier_bias,
            .projection_weight = projection_weight,
            .d_projection_weight = d_projection_weight,
        };
    }

    pub fn deinit(self: *TrainableViTWeights) void {
        if (self.d_projection_weight) |d| self.allocator.free(d);
        if (self.projection_weight) |p| self.allocator.free(p);
        if (self.d_classifier_bias) |d| self.allocator.free(d);
        if (self.d_classifier_weight) |d| self.allocator.free(d);
        if (self.classifier_bias) |c| self.allocator.free(c);
        if (self.classifier_weight) |c| self.allocator.free(c);
        self.allocator.free(self.d_final_ln_bias);
        self.allocator.free(self.d_final_ln_weight);
        self.allocator.free(self.final_ln_bias);
        self.allocator.free(self.final_ln_weight);
        for (self.layers) |*layer| {
            layer.deinit();
        }
        self.allocator.free(self.layers);
        if (self.d_cls_token) |d| self.allocator.free(d);
        if (self.cls_token) |c| self.allocator.free(c);
        self.allocator.free(self.d_pos_embed);
        self.allocator.free(self.pos_embed);
        self.allocator.free(self.d_patch_proj);
        self.allocator.free(self.patch_proj);
        self.* = undefined;
    }

    pub fn zeroGradients(self: *TrainableViTWeights) void {
        @memset(self.d_patch_proj, 0);
        @memset(self.d_pos_embed, 0);
        if (self.d_cls_token) |d| @memset(d, 0);
        for (self.layers) |*layer| {
            layer.zeroGradients();
        }
        @memset(self.d_final_ln_weight, 0);
        @memset(self.d_final_ln_bias, 0);
        if (self.d_classifier_weight) |d| @memset(d, 0);
        if (self.d_classifier_bias) |d| @memset(d, 0);
        if (self.d_projection_weight) |d| @memset(d, 0);
    }
};

/// Trainable Vision Transformer model.
pub const TrainableViTModel = struct {
    allocator: std.mem.Allocator,
    config: TrainableViTConfig,
    weights: TrainableViTWeights,

    /// Activation cache for backward pass
    activations: ?ViTActivationCache,

    pub fn init(allocator: std.mem.Allocator, config: TrainableViTConfig) !TrainableViTModel {
        const weights = try TrainableViTWeights.init(allocator, config);
        errdefer {
            var w = weights;
            w.deinit();
        }

        const activations = try ViTActivationCache.init(allocator, config);

        return .{
            .allocator = allocator,
            .config = config,
            .weights = weights,
            .activations = activations,
        };
    }

    pub fn deinit(self: *TrainableViTModel) void {
        if (self.activations) |*a| {
            a.deinit();
        }
        self.weights.deinit();
        self.* = undefined;
    }

    /// Forward pass through the ViT model.
    /// Input: [batch_size, channels, height, width] flattened image tensor
    /// Output: [batch_size, num_classes] class logits OR [batch_size, hidden_size] embeddings
    pub fn forward(
        self: *TrainableViTModel,
        images: []const f32,
        batch_size: u32,
        output: []f32,
    ) !void {
        if (batch_size > self.config.max_batch_size) return error.InvalidBatchSize;
        const cfg = self.config.vit_config;
        const hidden = cfg.hidden_size;
        const seq_len = cfg.seqLength();
        const patch_dim = cfg.in_channels * cfg.patch_size * cfg.patch_size;

        var cache = &(self.activations orelse return error.NoActivationCache);

        // Step 1: Patch embedding
        // For each image, extract patches and project to hidden dimension
        for (0..batch_size) |b| {
            const img_offset = b * cfg.in_channels * cfg.image_size * cfg.image_size;

            // Extract and project patches
            var patch_idx: usize = 0;
            var y: u32 = 0;
            while (y < cfg.image_size) : (y += cfg.patch_size) {
                var x: u32 = 0;
                while (x < cfg.image_size) : (x += cfg.patch_size) {
                    // Extract patch pixels into temporary buffer
                    var patch_sum: f32 = 0;
                    for (0..patch_dim) |pd| {
                        // Simplified: just sum the patch (real implementation would extract properly)
                        const pixel_idx = img_offset + pd % (cfg.image_size * cfg.image_size);
                        if (pixel_idx < images.len) {
                            patch_sum += images[pixel_idx];
                        }
                    }

                    // Project patch to hidden dimension
                    const emb_offset = (b * seq_len + patch_idx + @intFromBool(cfg.use_class_token)) * hidden;
                    for (0..hidden) |h| {
                        var proj_sum: f32 = 0;
                        for (0..patch_dim) |pd| {
                            proj_sum += patch_sum / @as(f32, @floatFromInt(patch_dim)) * self.weights.patch_proj[pd * hidden + h];
                        }
                        cache.embeddings[emb_offset + h] = proj_sum;
                    }

                    patch_idx += 1;
                }
            }

            // Add class token (if used)
            if (cfg.use_class_token) {
                if (self.weights.cls_token) |cls| {
                    const cls_offset = b * seq_len * hidden;
                    @memcpy(cache.embeddings[cls_offset .. cls_offset + hidden], cls);
                }
            }

            // Add position embeddings
            for (0..seq_len) |pos| {
                const offset = (b * seq_len + pos) * hidden;
                const pos_offset = pos * hidden;
                for (0..hidden) |h| {
                    cache.embeddings[offset + h] += self.weights.pos_embed[pos_offset + h];
                }
            }
        }

        // Step 2: Process through transformer layers
        var hidden_states = try self.allocator.alloc(f32, batch_size * seq_len * hidden);
        defer self.allocator.free(hidden_states);
        @memcpy(hidden_states, cache.embeddings[0 .. batch_size * seq_len * hidden]);

        for (self.weights.layers, 0..) |*layer, layer_idx| {
            try self.processTransformerLayer(
                hidden_states,
                layer,
                batch_size,
                seq_len,
                &cache.layer_caches[layer_idx],
            );
        }

        // Step 3: Final layer norm and output projection
        for (0..batch_size) |b| {
            // Get class token output (or mean pool)
            const cls_offset = b * seq_len * hidden;
            const cls_features = hidden_states[cls_offset .. cls_offset + hidden];

            // Apply final layer norm
            layerNorm(cls_features, self.weights.final_ln_weight, self.weights.final_ln_bias);

            // Classification head or projection head
            if (self.config.num_classes > 0) {
                // Classification: [hidden] @ [hidden, num_classes] -> [num_classes]
                if (self.weights.classifier_weight) |cw| {
                    const out_offset = b * self.config.num_classes;
                    for (0..self.config.num_classes) |c| {
                        var sum: f32 = 0;
                        for (0..hidden) |h| {
                            sum += cls_features[h] * cw[h * self.config.num_classes + c];
                        }
                        if (self.weights.classifier_bias) |cb| {
                            sum += cb[c];
                        }
                        output[out_offset + c] = sum;
                    }
                }
            } else if (self.config.projection_dim > 0) {
                // Projection: [hidden] @ [hidden, proj_dim] -> [proj_dim]
                if (self.weights.projection_weight) |pw| {
                    const out_offset = b * self.config.projection_dim;
                    for (0..self.config.projection_dim) |p| {
                        var sum: f32 = 0;
                        for (0..hidden) |h| {
                            sum += cls_features[h] * pw[h * self.config.projection_dim + p];
                        }
                        output[out_offset + p] = sum;
                    }
                }
            } else {
                // Raw embeddings
                const out_offset = b * hidden;
                @memcpy(output[out_offset .. out_offset + hidden], cls_features);
            }
        }
    }

    /// Process a single transformer layer.
    fn processTransformerLayer(
        self: *TrainableViTModel,
        hidden: []f32,
        layer: *TrainableViTLayerWeights,
        batch_size: u32,
        seq_len: u32,
        cache: *ViTLayerCache,
    ) !void {
        const hidden_dim = self.config.vit_config.hidden_size;
        const mlp_dim = self.config.vit_config.mlp_dim;
        const num_heads = self.config.vit_config.num_heads;
        const head_dim = hidden_dim / num_heads;

        // Allocate temporaries
        const residual = try self.allocator.alloc(f32, batch_size * seq_len * hidden_dim);
        defer self.allocator.free(residual);

        // Save pre-norm for backward
        @memcpy(cache.pre_ln1, hidden[0 .. batch_size * seq_len * hidden_dim]);

        // Pre-norm for attention
        @memcpy(residual, hidden);
        for (0..batch_size * seq_len) |i| {
            const offset = i * hidden_dim;
            layerNorm(hidden[offset .. offset + hidden_dim], layer.ln1_weight, layer.ln1_bias);
        }

        // Multi-head self-attention (simplified)
        var attn_out = try self.allocator.alloc(f32, batch_size * seq_len * hidden_dim);
        defer self.allocator.free(attn_out);
        @memset(attn_out, 0);

        // For each batch and head
        for (0..batch_size) |b| {
            for (0..num_heads) |h| {
                // Compute Q, K, V for this head
                for (0..seq_len) |s| {
                    const in_offset = (b * seq_len + s) * hidden_dim;
                    const head_offset = h * head_dim;

                    // Simplified attention computation
                    for (0..head_dim) |d| {
                        var q_val: f32 = 0;
                        for (0..hidden_dim) |hd| {
                            q_val += hidden[in_offset + hd] * layer.w_q[hd * hidden_dim + head_offset + d];
                        }
                        attn_out[in_offset + head_offset + d] += q_val * 0.1; // Simplified
                    }
                }
            }

            // Output projection
            for (0..seq_len) |s| {
                const offset = (b * seq_len + s) * hidden_dim;
                var proj_out = try self.allocator.alloc(f32, hidden_dim);
                defer self.allocator.free(proj_out);

                for (0..hidden_dim) |h| {
                    var sum: f32 = 0;
                    for (0..hidden_dim) |hd| {
                        sum += attn_out[offset + hd] * layer.w_o[hd * hidden_dim + h];
                    }
                    proj_out[h] = sum;
                }

                @memcpy(attn_out[offset .. offset + hidden_dim], proj_out);
            }
        }

        // Residual connection
        for (0..batch_size * seq_len * hidden_dim) |i| {
            hidden[i] = residual[i] + attn_out[i];
        }

        // Save for backward
        @memcpy(cache.post_attn, hidden[0 .. batch_size * seq_len * hidden_dim]);
        @memcpy(cache.pre_ln2, hidden[0 .. batch_size * seq_len * hidden_dim]);

        // Pre-norm for MLP
        @memcpy(residual, hidden);
        for (0..batch_size * seq_len) |i| {
            const offset = i * hidden_dim;
            layerNorm(hidden[offset .. offset + hidden_dim], layer.ln2_weight, layer.ln2_bias);
        }

        // MLP
        for (0..batch_size * seq_len) |i| {
            const offset = i * hidden_dim;
            const input = hidden[offset .. offset + hidden_dim];

            // FC1: [hidden] -> [mlp_dim] with GELU
            var mlp_hidden = try self.allocator.alloc(f32, mlp_dim);
            defer self.allocator.free(mlp_hidden);

            for (0..mlp_dim) |m| {
                var sum: f32 = layer.mlp_fc1_bias[m];
                for (0..hidden_dim) |h| {
                    sum += input[h] * layer.mlp_fc1_weight[h * mlp_dim + m];
                }
                mlp_hidden[m] = vit.gelu(sum);
            }

            // FC2: [mlp_dim] -> [hidden]
            for (0..hidden_dim) |h| {
                var sum: f32 = layer.mlp_fc2_bias[h];
                for (0..mlp_dim) |m| {
                    sum += mlp_hidden[m] * layer.mlp_fc2_weight[m * hidden_dim + h];
                }
                hidden[offset + h] = sum;
            }
        }

        // Residual connection
        for (0..batch_size * seq_len * hidden_dim) |i| {
            hidden[i] = residual[i] + hidden[i];
        }

        // Save post-FFN
        @memcpy(cache.post_ffn, hidden[0 .. batch_size * seq_len * hidden_dim]);
    }

    /// Zero all gradients.
    pub fn zeroGradients(self: *TrainableViTModel) void {
        self.weights.zeroGradients();
    }

    /// Compute gradient norm.
    pub fn computeGradientNorm(self: *const TrainableViTModel) f32 {
        var sum_sq: f32 = 0;

        for (self.weights.d_patch_proj) |g| sum_sq += g * g;
        for (self.weights.d_pos_embed) |g| sum_sq += g * g;
        if (self.weights.d_cls_token) |d| {
            for (d) |g| sum_sq += g * g;
        }

        for (self.weights.layers) |layer| {
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

        for (self.weights.d_final_ln_weight) |g| sum_sq += g * g;
        for (self.weights.d_final_ln_bias) |g| sum_sq += g * g;

        if (self.weights.d_classifier_weight) |d| {
            for (d) |g| sum_sq += g * g;
        }
        if (self.weights.d_classifier_bias) |d| {
            for (d) |g| sum_sq += g * g;
        }
        if (self.weights.d_projection_weight) |d| {
            for (d) |g| sum_sq += g * g;
        }

        return @sqrt(sum_sq);
    }

    /// Clip gradients by norm.
    pub fn clipGradients(self: *TrainableViTModel, max_norm: f32) f32 {
        const grad_norm = self.computeGradientNorm();

        if (grad_norm > max_norm and grad_norm > 0) {
            const scale = max_norm / grad_norm;
            scaleSlice(self.weights.d_patch_proj, scale);
            scaleSlice(self.weights.d_pos_embed, scale);
            if (self.weights.d_cls_token) |d| scaleSlice(d, scale);

            for (self.weights.layers) |*layer| {
                scaleSlice(layer.d_w_q, scale);
                scaleSlice(layer.d_w_k, scale);
                scaleSlice(layer.d_w_v, scale);
                scaleSlice(layer.d_w_o, scale);
                scaleSlice(layer.d_ln1_weight, scale);
                scaleSlice(layer.d_ln1_bias, scale);
                scaleSlice(layer.d_ln2_weight, scale);
                scaleSlice(layer.d_ln2_bias, scale);
                scaleSlice(layer.d_mlp_fc1_weight, scale);
                scaleSlice(layer.d_mlp_fc1_bias, scale);
                scaleSlice(layer.d_mlp_fc2_weight, scale);
                scaleSlice(layer.d_mlp_fc2_bias, scale);
            }

            scaleSlice(self.weights.d_final_ln_weight, scale);
            scaleSlice(self.weights.d_final_ln_bias, scale);

            if (self.weights.d_classifier_weight) |d| scaleSlice(d, scale);
            if (self.weights.d_classifier_bias) |d| scaleSlice(d, scale);
            if (self.weights.d_projection_weight) |d| scaleSlice(d, scale);
        }

        return grad_norm;
    }

    /// Apply SGD update.
    pub fn applySgdUpdate(self: *TrainableViTModel, learning_rate: f32) void {
        applyUpdate(self.weights.patch_proj, self.weights.d_patch_proj, learning_rate);
        applyUpdate(self.weights.pos_embed, self.weights.d_pos_embed, learning_rate);
        if (self.weights.cls_token) |w| {
            if (self.weights.d_cls_token) |d| {
                applyUpdate(w, d, learning_rate);
            }
        }

        for (self.weights.layers) |*layer| {
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

        applyUpdate(self.weights.final_ln_weight, self.weights.d_final_ln_weight, learning_rate);
        applyUpdate(self.weights.final_ln_bias, self.weights.d_final_ln_bias, learning_rate);

        if (self.weights.classifier_weight) |w| {
            if (self.weights.d_classifier_weight) |d| {
                applyUpdate(w, d, learning_rate);
            }
        }
        if (self.weights.classifier_bias) |w| {
            if (self.weights.d_classifier_bias) |d| {
                applyUpdate(w, d, learning_rate);
            }
        }
        if (self.weights.projection_weight) |w| {
            if (self.weights.d_projection_weight) |d| {
                applyUpdate(w, d, learning_rate);
            }
        }
    }
};

/// Activation cache for ViT backward pass.
pub const ViTActivationCache = struct {
    allocator: std.mem.Allocator,
    embeddings: []f32,
    layer_caches: []ViTLayerCache,

    pub fn init(allocator: std.mem.Allocator, config: TrainableViTConfig) !ViTActivationCache {
        const cfg = config.vit_config;
        const max_batch = config.max_batch_size;
        const batch_seq_hidden = max_batch * cfg.seqLength() * cfg.hidden_size;

        const embeddings = try allocator.alloc(f32, batch_seq_hidden);
        errdefer allocator.free(embeddings);
        @memset(embeddings, 0);

        const layer_caches = try allocator.alloc(ViTLayerCache, cfg.num_layers);
        errdefer allocator.free(layer_caches);

        var initialized: usize = 0;
        errdefer {
            for (layer_caches[0..initialized]) |*cache| {
                cache.deinit();
            }
        }

        for (layer_caches) |*cache| {
            cache.* = try ViTLayerCache.init(allocator, cfg, max_batch);
            initialized += 1;
        }

        return .{
            .allocator = allocator,
            .embeddings = embeddings,
            .layer_caches = layer_caches,
        };
    }

    pub fn deinit(self: *ViTActivationCache) void {
        for (self.layer_caches) |*cache| {
            cache.deinit();
        }
        self.allocator.free(self.layer_caches);
        self.allocator.free(self.embeddings);
        self.* = undefined;
    }
};

/// Layer-wise activation cache for ViT backward pass.
pub const ViTLayerCache = struct {
    allocator: std.mem.Allocator,
    pre_ln1: []f32,
    post_attn: []f32,
    pre_ln2: []f32,
    post_ffn: []f32,

    pub fn init(allocator: std.mem.Allocator, config: vit.ViTConfig, max_batch_size: u32) !ViTLayerCache {
        const size = max_batch_size * config.seqLength() * config.hidden_size;

        const pre_ln1 = try allocator.alloc(f32, size);
        errdefer allocator.free(pre_ln1);
        const post_attn = try allocator.alloc(f32, size);
        errdefer allocator.free(post_attn);
        const pre_ln2 = try allocator.alloc(f32, size);
        errdefer allocator.free(pre_ln2);
        const post_ffn = try allocator.alloc(f32, size);

        @memset(pre_ln1, 0);
        @memset(post_attn, 0);
        @memset(pre_ln2, 0);
        @memset(post_ffn, 0);

        return .{
            .allocator = allocator,
            .pre_ln1 = pre_ln1,
            .post_attn = post_attn,
            .pre_ln2 = pre_ln2,
            .post_ffn = post_ffn,
        };
    }

    pub fn deinit(self: *ViTLayerCache) void {
        self.allocator.free(self.post_ffn);
        self.allocator.free(self.pre_ln2);
        self.allocator.free(self.post_attn);
        self.allocator.free(self.pre_ln1);
        self.* = undefined;
    }
};

// ============================================================================
// Helper Functions
// ============================================================================

/// Xavier initialization.
fn initializeXavier(data: []f32) void {
    const scale = @sqrt(2.0 / @as(f32, @floatFromInt(data.len)));
    var rng = std.Random.DefaultPrng.init(0x12345678);
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

/// Scale a slice by a factor.
fn scaleSlice(data: []f32, scale: f32) void {
    for (data) |*v| {
        v.* *= scale;
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

test "trainable vit config num params" {
    const config = TrainableViTConfig{
        .vit_config = vit.ViTConfig.tiny(224, 16),
        .num_classes = 1000,
    };

    const num_params = config.numParams();
    try std.testing.expect(num_params > 0);
}

test "trainable vit layer weights init/deinit" {
    const allocator = std.testing.allocator;
    const vit_config = vit.ViTConfig.tiny(224, 16);

    var layer = try TrainableViTLayerWeights.init(allocator, vit_config);
    defer layer.deinit();

    // Check gradients are zeroed
    for (layer.d_w_q) |g| {
        try std.testing.expectEqual(@as(f32, 0), g);
    }
}

test "trainable vit weights init/deinit" {
    const allocator = std.testing.allocator;
    const config = TrainableViTConfig{
        .vit_config = vit.ViTConfig.tiny(224, 16),
        .num_classes = 10,
    };

    var weights = try TrainableViTWeights.init(allocator, config);
    defer weights.deinit();

    // Check classifier weight exists
    try std.testing.expect(weights.classifier_weight != null);
}

test "trainable vit model init/deinit" {
    const allocator = std.testing.allocator;
    const config = TrainableViTConfig{
        .vit_config = vit.ViTConfig.tiny(224, 16),
        .num_classes = 10,
    };

    var model = try TrainableViTModel.init(allocator, config);
    defer model.deinit();

    // Check activations initialized
    try std.testing.expect(model.activations != null);
}

test "layer norm" {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const weight = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const bias = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    layerNorm(&data, &weight, &bias);

    // After normalization, mean should be ~0 and variance ~1
    var mean: f32 = 0;
    for (data) |v| mean += v;
    mean /= 4.0;

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mean, 0.001);
}

test {
    std.testing.refAllDecls(@This());
}
