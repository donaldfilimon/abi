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
const math = std.math;

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
// Activation Functions
// ============================================================================

/// GELU activation function (Gaussian Error Linear Unit)
pub fn gelu(x: f32) f32 {
    // Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const sqrt_2_over_pi: f32 = 0.7978845608;
    const coeff: f32 = 0.044715;
    const inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    return 0.5 * x * (1.0 + math.tanh(inner));
}

/// Apply GELU activation to a slice
pub fn geluSlice(data: []f32) void {
    for (data) |*v| {
        v.* = gelu(v.*);
    }
}

/// Softmax over a slice (in-place)
pub fn softmax(data: []f32) void {
    if (data.len == 0) return;

    // Find max for numerical stability
    var max_val: f32 = data[0];
    for (data[1..]) |v| {
        if (v > max_val) max_val = v;
    }

    // Compute exp and sum
    var sum: f32 = 0.0;
    for (data) |*v| {
        v.* = @exp(v.* - max_val);
        sum += v.*;
    }

    // Normalize
    if (sum > 0.0) {
        for (data) |*v| {
            v.* /= sum;
        }
    }
}

// ============================================================================
// Patch Embedding Layer
// ============================================================================

/// Patch embedding layer that converts image patches to embeddings
pub const PatchEmbedding = struct {
    allocator: std.mem.Allocator,
    config: ViTConfig,

    /// Projection weights [hidden_size, patch_size * patch_size * in_channels]
    proj_weights: []f32,

    /// Projection bias [hidden_size]
    proj_bias: []f32,

    /// Class token embedding [hidden_size]
    cls_token: ?[]f32,

    /// Position embeddings [seq_length, hidden_size]
    pos_embed: []f32,

    pub fn init(allocator: std.mem.Allocator, config: ViTConfig) !PatchEmbedding {
        const patch_dim = config.patch_size * config.patch_size * config.in_channels;
        const hidden = config.hidden_size;
        const seq_len = config.seqLength();

        // Allocate projection weights and bias
        const proj_weights = try allocator.alloc(f32, hidden * patch_dim);
        const proj_bias = try allocator.alloc(f32, hidden);

        // Initialize with Xavier/Glorot
        const scale = @sqrt(2.0 / @as(f32, @floatFromInt(patch_dim + hidden)));
        var prng = std.Random.DefaultPrng.init(42);
        for (proj_weights) |*w| {
            w.* = (prng.random().float(f32) * 2.0 - 1.0) * scale;
        }
        @memset(proj_bias, 0.0);

        // Allocate class token if needed
        const cls_token = if (config.use_class_token) blk: {
            const token = try allocator.alloc(f32, hidden);
            for (token) |*t| {
                t.* = (prng.random().float(f32) * 2.0 - 1.0) * 0.02;
            }
            break :blk token;
        } else null;

        // Allocate position embeddings
        const pos_embed = try allocator.alloc(f32, seq_len * hidden);
        for (pos_embed) |*p| {
            p.* = (prng.random().float(f32) * 2.0 - 1.0) * 0.02;
        }

        return .{
            .allocator = allocator,
            .config = config,
            .proj_weights = proj_weights,
            .proj_bias = proj_bias,
            .cls_token = cls_token,
            .pos_embed = pos_embed,
        };
    }

    pub fn deinit(self: *PatchEmbedding) void {
        self.allocator.free(self.proj_weights);
        self.allocator.free(self.proj_bias);
        if (self.cls_token) |tok| self.allocator.free(tok);
        self.allocator.free(self.pos_embed);
    }

    /// Forward pass: [batch, channels, height, width] -> [batch, seq_len, hidden]
    /// For simplicity, we process one image at a time here
    pub fn forward(self: *const PatchEmbedding, image: []const f32) ![]f32 {
        const cfg = self.config;
        const seq_len = cfg.seqLength();
        const hidden = cfg.hidden_size;
        const patch_dim = cfg.patch_size * cfg.patch_size * cfg.in_channels;
        const patches_per_side = cfg.image_size / cfg.patch_size;

        // Allocate output
        const output = try self.allocator.alloc(f32, seq_len * hidden);
        errdefer self.allocator.free(output);

        // Start index (after class token if present)
        var out_idx: usize = 0;

        // Add class token if present
        if (self.cls_token) |cls| {
            @memcpy(output[0..hidden], cls);
            out_idx = hidden;
        }

        // Extract and project each patch
        for (0..patches_per_side) |py| {
            for (0..patches_per_side) |px| {
                // Extract patch
                var patch: [16 * 16 * 3]f32 = undefined; // Max patch size
                var patch_idx: usize = 0;

                for (0..cfg.in_channels) |c| {
                    for (0..cfg.patch_size) |dy| {
                        for (0..cfg.patch_size) |dx| {
                            const y = py * cfg.patch_size + dy;
                            const x = px * cfg.patch_size + dx;
                            const img_idx = c * cfg.image_size * cfg.image_size + y * cfg.image_size + x;
                            if (img_idx < image.len and patch_idx < patch_dim) {
                                patch[patch_idx] = image[img_idx];
                            }
                            patch_idx += 1;
                        }
                    }
                }

                // Project patch to hidden dimension
                for (0..hidden) |h| {
                    var sum: f32 = self.proj_bias[h];
                    for (0..patch_dim) |p| {
                        sum += patch[p] * self.proj_weights[h * patch_dim + p];
                    }
                    output[out_idx + h] = sum;
                }
                out_idx += hidden;
            }
        }

        // Add position embeddings
        for (0..seq_len * hidden) |i| {
            output[i] += self.pos_embed[i];
        }

        return output;
    }
};

// ============================================================================
// Multi-Head Self-Attention
// ============================================================================

/// Multi-head self-attention layer
pub const MultiHeadAttention = struct {
    allocator: std.mem.Allocator,
    hidden_size: u32,
    num_heads: u32,
    head_dim: u32,

    /// Query projection [hidden_size, hidden_size]
    wq: []f32,
    /// Key projection [hidden_size, hidden_size]
    wk: []f32,
    /// Value projection [hidden_size, hidden_size]
    wv: []f32,
    /// Output projection [hidden_size, hidden_size]
    wo: []f32,

    /// Biases
    bq: []f32,
    bk: []f32,
    bv: []f32,
    bo: []f32,

    pub fn init(allocator: std.mem.Allocator, hidden_size: u32, num_heads: u32) !MultiHeadAttention {
        const head_dim = hidden_size / num_heads;
        const size = hidden_size * hidden_size;

        const wq = try allocator.alloc(f32, size);
        const wk = try allocator.alloc(f32, size);
        const wv = try allocator.alloc(f32, size);
        const wo = try allocator.alloc(f32, size);

        const bq = try allocator.alloc(f32, hidden_size);
        const bk = try allocator.alloc(f32, hidden_size);
        const bv = try allocator.alloc(f32, hidden_size);
        const bo = try allocator.alloc(f32, hidden_size);

        // Xavier initialization
        const scale = @sqrt(2.0 / @as(f32, @floatFromInt(hidden_size * 2)));
        var prng = std.Random.DefaultPrng.init(123);

        for (wq) |*w| w.* = (prng.random().float(f32) * 2.0 - 1.0) * scale;
        for (wk) |*w| w.* = (prng.random().float(f32) * 2.0 - 1.0) * scale;
        for (wv) |*w| w.* = (prng.random().float(f32) * 2.0 - 1.0) * scale;
        for (wo) |*w| w.* = (prng.random().float(f32) * 2.0 - 1.0) * scale;

        @memset(bq, 0.0);
        @memset(bk, 0.0);
        @memset(bv, 0.0);
        @memset(bo, 0.0);

        return .{
            .allocator = allocator,
            .hidden_size = hidden_size,
            .num_heads = num_heads,
            .head_dim = head_dim,
            .wq = wq,
            .wk = wk,
            .wv = wv,
            .wo = wo,
            .bq = bq,
            .bk = bk,
            .bv = bv,
            .bo = bo,
        };
    }

    pub fn deinit(self: *MultiHeadAttention) void {
        self.allocator.free(self.wq);
        self.allocator.free(self.wk);
        self.allocator.free(self.wv);
        self.allocator.free(self.wo);
        self.allocator.free(self.bq);
        self.allocator.free(self.bk);
        self.allocator.free(self.bv);
        self.allocator.free(self.bo);
    }

    /// Forward pass: [seq_len, hidden] -> [seq_len, hidden]
    pub fn forward(self: *const MultiHeadAttention, x: []const f32, seq_len: usize) ![]f32 {
        const hidden = self.hidden_size;
        const num_heads = self.num_heads;
        const head_dim = self.head_dim;

        // Allocate Q, K, V
        const q = try self.allocator.alloc(f32, seq_len * hidden);
        defer self.allocator.free(q);
        const k = try self.allocator.alloc(f32, seq_len * hidden);
        defer self.allocator.free(k);
        const v = try self.allocator.alloc(f32, seq_len * hidden);
        defer self.allocator.free(v);

        // Linear projections
        for (0..seq_len) |s| {
            for (0..hidden) |h| {
                var q_sum: f32 = self.bq[h];
                var k_sum: f32 = self.bk[h];
                var v_sum: f32 = self.bv[h];

                for (0..hidden) |i| {
                    const x_val = x[s * hidden + i];
                    q_sum += x_val * self.wq[h * hidden + i];
                    k_sum += x_val * self.wk[h * hidden + i];
                    v_sum += x_val * self.wv[h * hidden + i];
                }

                q[s * hidden + h] = q_sum;
                k[s * hidden + h] = k_sum;
                v[s * hidden + h] = v_sum;
            }
        }

        // Allocate attention output
        const attn_out = try self.allocator.alloc(f32, seq_len * hidden);
        errdefer self.allocator.free(attn_out);
        @memset(attn_out, 0.0);

        // Compute attention for each head
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        // Temporary attention scores
        const scores = try self.allocator.alloc(f32, seq_len);
        defer self.allocator.free(scores);

        for (0..num_heads) |head| {
            const offset = head * head_dim;

            for (0..seq_len) |i| {
                // Compute attention scores for position i
                for (0..seq_len) |j| {
                    var dot: f32 = 0.0;
                    for (0..head_dim) |d| {
                        dot += q[i * hidden + offset + d] * k[j * hidden + offset + d];
                    }
                    scores[j] = dot * scale;
                }

                // Softmax
                softmax(scores);

                // Weighted sum of values
                for (0..head_dim) |d| {
                    var sum: f32 = 0.0;
                    for (0..seq_len) |j| {
                        sum += scores[j] * v[j * hidden + offset + d];
                    }
                    attn_out[i * hidden + offset + d] = sum;
                }
            }
        }

        // Output projection
        const output = try self.allocator.alloc(f32, seq_len * hidden);
        for (0..seq_len) |s| {
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
// Feed-Forward Network (MLP)
// ============================================================================

/// MLP block used in transformer
pub const MLP = struct {
    allocator: std.mem.Allocator,
    hidden_size: u32,
    mlp_dim: u32,
    use_gelu: bool,

    /// First linear [mlp_dim, hidden_size]
    w1: []f32,
    b1: []f32,

    /// Second linear [hidden_size, mlp_dim]
    w2: []f32,
    b2: []f32,

    pub fn init(allocator: std.mem.Allocator, hidden_size: u32, mlp_dim: u32, use_gelu_act: bool) !MLP {
        const w1 = try allocator.alloc(f32, mlp_dim * hidden_size);
        const b1 = try allocator.alloc(f32, mlp_dim);
        const w2 = try allocator.alloc(f32, hidden_size * mlp_dim);
        const b2 = try allocator.alloc(f32, hidden_size);

        // Xavier initialization
        const scale1 = @sqrt(2.0 / @as(f32, @floatFromInt(hidden_size + mlp_dim)));
        const scale2 = @sqrt(2.0 / @as(f32, @floatFromInt(mlp_dim + hidden_size)));
        var prng = std.Random.DefaultPrng.init(456);

        for (w1) |*w| w.* = (prng.random().float(f32) * 2.0 - 1.0) * scale1;
        for (w2) |*w| w.* = (prng.random().float(f32) * 2.0 - 1.0) * scale2;

        @memset(b1, 0.0);
        @memset(b2, 0.0);

        return .{
            .allocator = allocator,
            .hidden_size = hidden_size,
            .mlp_dim = mlp_dim,
            .use_gelu = use_gelu_act,
            .w1 = w1,
            .b1 = b1,
            .w2 = w2,
            .b2 = b2,
        };
    }

    pub fn deinit(self: *MLP) void {
        self.allocator.free(self.w1);
        self.allocator.free(self.b1);
        self.allocator.free(self.w2);
        self.allocator.free(self.b2);
    }

    /// Forward pass: [seq_len, hidden] -> [seq_len, hidden]
    pub fn forward(self: *const MLP, x: []const f32, seq_len: usize) ![]f32 {
        const hidden = self.hidden_size;
        const mlp_dim = self.mlp_dim;

        // First linear + activation
        const intermediate = try self.allocator.alloc(f32, seq_len * mlp_dim);
        defer self.allocator.free(intermediate);

        for (0..seq_len) |s| {
            for (0..mlp_dim) |m| {
                var sum: f32 = self.b1[m];
                for (0..hidden) |h| {
                    sum += x[s * hidden + h] * self.w1[m * hidden + h];
                }
                intermediate[s * mlp_dim + m] = if (self.use_gelu) gelu(sum) else @max(sum, 0.0);
            }
        }

        // Second linear
        const output = try self.allocator.alloc(f32, seq_len * hidden);
        for (0..seq_len) |s| {
            for (0..hidden) |h| {
                var sum: f32 = self.b2[h];
                for (0..mlp_dim) |m| {
                    sum += intermediate[s * mlp_dim + m] * self.w2[h * mlp_dim + m];
                }
                output[s * hidden + h] = sum;
            }
        }

        return output;
    }
};

// ============================================================================
// Layer Normalization
// ============================================================================

/// Layer normalization
pub const LayerNorm = struct {
    allocator: std.mem.Allocator,
    hidden_size: u32,
    eps: f32,

    /// Scale parameter (gamma)
    gamma: []f32,
    /// Shift parameter (beta)
    beta: []f32,

    pub fn init(allocator: std.mem.Allocator, hidden_size: u32, eps: f32) !LayerNorm {
        const gamma = try allocator.alloc(f32, hidden_size);
        const beta = try allocator.alloc(f32, hidden_size);

        // Initialize gamma to 1, beta to 0
        for (gamma) |*g| g.* = 1.0;
        @memset(beta, 0.0);

        return .{
            .allocator = allocator,
            .hidden_size = hidden_size,
            .eps = eps,
            .gamma = gamma,
            .beta = beta,
        };
    }

    pub fn deinit(self: *LayerNorm) void {
        self.allocator.free(self.gamma);
        self.allocator.free(self.beta);
    }

    /// Forward pass: normalizes the last dimension
    pub fn forward(self: *const LayerNorm, x: []const f32, seq_len: usize) ![]f32 {
        const hidden = self.hidden_size;
        const output = try self.allocator.alloc(f32, seq_len * hidden);

        for (0..seq_len) |s| {
            // Compute mean
            var mean: f32 = 0.0;
            for (0..hidden) |h| {
                mean += x[s * hidden + h];
            }
            mean /= @floatFromInt(hidden);

            // Compute variance
            var variance: f32 = 0.0;
            for (0..hidden) |h| {
                const diff = x[s * hidden + h] - mean;
                variance += diff * diff;
            }
            variance /= @floatFromInt(hidden);

            // Normalize
            const std_dev = @sqrt(variance + self.eps);
            for (0..hidden) |h| {
                const normalized = (x[s * hidden + h] - mean) / std_dev;
                output[s * hidden + h] = normalized * self.gamma[h] + self.beta[h];
            }
        }

        return output;
    }
};

// ============================================================================
// Transformer Encoder Block
// ============================================================================

/// Single transformer encoder block
pub const TransformerBlock = struct {
    allocator: std.mem.Allocator,
    hidden_size: u32,
    pre_norm: bool,

    attention: MultiHeadAttention,
    mlp: MLP,
    norm1: LayerNorm,
    norm2: LayerNorm,

    pub fn init(allocator: std.mem.Allocator, config: ViTConfig) !TransformerBlock {
        return .{
            .allocator = allocator,
            .hidden_size = config.hidden_size,
            .pre_norm = config.pre_norm,
            .attention = try MultiHeadAttention.init(allocator, config.hidden_size, config.num_heads),
            .mlp = try MLP.init(allocator, config.hidden_size, config.mlp_dim, config.use_gelu),
            .norm1 = try LayerNorm.init(allocator, config.hidden_size, config.layer_norm_eps),
            .norm2 = try LayerNorm.init(allocator, config.hidden_size, config.layer_norm_eps),
        };
    }

    pub fn deinit(self: *TransformerBlock) void {
        self.attention.deinit();
        self.mlp.deinit();
        self.norm1.deinit();
        self.norm2.deinit();
    }

    /// Forward pass with residual connections
    pub fn forward(self: *const TransformerBlock, x: []const f32, seq_len: usize) ![]f32 {
        const hidden = self.hidden_size;

        if (self.pre_norm) {
            // Pre-norm: norm -> attention -> residual -> norm -> mlp -> residual
            const normed1 = try self.norm1.forward(x, seq_len);
            defer self.allocator.free(normed1);

            const attn_out = try self.attention.forward(normed1, seq_len);
            defer self.allocator.free(attn_out);

            // Add residual
            const residual1 = try self.allocator.alloc(f32, seq_len * hidden);
            for (0..seq_len * hidden) |i| {
                residual1[i] = x[i] + attn_out[i];
            }

            const normed2 = try self.norm2.forward(residual1, seq_len);
            defer self.allocator.free(normed2);

            const mlp_out = try self.mlp.forward(normed2, seq_len);
            defer self.allocator.free(mlp_out);

            // Add residual
            for (0..seq_len * hidden) |i| {
                residual1[i] = residual1[i] + mlp_out[i];
            }

            return residual1;
        } else {
            // Post-norm: attention -> residual -> norm -> mlp -> residual -> norm
            const attn_out = try self.attention.forward(x, seq_len);
            defer self.allocator.free(attn_out);

            // Add residual and norm
            const residual1 = try self.allocator.alloc(f32, seq_len * hidden);
            for (0..seq_len * hidden) |i| {
                residual1[i] = x[i] + attn_out[i];
            }

            const normed1 = try self.norm1.forward(residual1, seq_len);
            self.allocator.free(residual1);
            defer self.allocator.free(normed1);

            const mlp_out = try self.mlp.forward(normed1, seq_len);
            defer self.allocator.free(mlp_out);

            // Add residual and norm
            const residual2 = try self.allocator.alloc(f32, seq_len * hidden);
            for (0..seq_len * hidden) |i| {
                residual2[i] = normed1[i] + mlp_out[i];
            }

            const output = try self.norm2.forward(residual2, seq_len);
            self.allocator.free(residual2);

            return output;
        }
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
        const patch_embed = try PatchEmbedding.init(allocator, config);
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
        defer self.allocator.free(x);

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
        const embedding = try self.allocator.alloc(f32, hidden);

        if (self.config.use_class_token) {
            // Use class token (first position)
            @memcpy(embedding, normed[0..hidden]);
        } else {
            // Global average pooling
            @memset(embedding, 0.0);
            for (0..seq_len) |s| {
                for (0..hidden) |h| {
                    embedding[h] += normed[s * hidden + h];
                }
            }
            for (embedding) |*e| {
                e.* /= @floatFromInt(seq_len);
            }
        }

        self.allocator.free(normed);
        return embedding;
    }

    /// Forward pass with classification
    /// Returns logits if num_classes > 0, otherwise returns embedding
    pub fn classify(self: *const VisionTransformer, image: []const f32) ![]f32 {
        const embedding = try self.forward(image);

        if (self.cls_head) |head| {
            defer self.allocator.free(embedding);

            const num_classes = self.config.num_classes;
            const hidden = self.config.hidden_size;
            const logits = try self.allocator.alloc(f32, num_classes);

            for (0..num_classes) |c| {
                var sum: f32 = self.cls_bias.?[c];
                for (0..hidden) |h| {
                    sum += embedding[h] * head[c * hidden + h];
                }
                logits[c] = sum;
            }

            return logits;
        }

        return embedding;
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
    const base = ViTConfig.base(224, 16);
    try std.testing.expectEqual(@as(u32, 768), base.hidden_size);
    try std.testing.expectEqual(@as(u32, 12), base.num_layers);
    try std.testing.expectEqual(@as(u32, 196), base.numPatches());
    try std.testing.expectEqual(@as(u32, 197), base.seqLength());

    const large = ViTConfig.large(224, 16);
    try std.testing.expectEqual(@as(u32, 1024), large.hidden_size);
    try std.testing.expectEqual(@as(u32, 24), large.num_layers);
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

    var vit = try VisionTransformer.init(allocator, config);
    defer vit.deinit();

    // Create dummy image
    const img_size = config.in_channels * config.image_size * config.image_size;
    const image = try allocator.alloc(f32, img_size);
    defer allocator.free(image);
    for (image) |*p| p.* = 0.5;

    // Forward pass
    const embedding = try vit.forward(image);
    defer allocator.free(embedding);

    try std.testing.expectEqual(@as(usize, config.hidden_size), embedding.len);
}

test "VisionTransformer parameter count" {
    const config = ViTConfig.base(224, 16);
    const allocator = std.testing.allocator;

    var vit = try VisionTransformer.init(allocator, config);
    defer vit.deinit();

    const params = vit.numParams();
    // ViT-Base should have ~86M parameters
    try std.testing.expect(params > 80_000_000 and params < 90_000_000);
}
