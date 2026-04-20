//! Patch Embedding Layer
//!
//! Converts image patches to embeddings with learnable position encoding.
//! Implements the patch embedding and position encoding components of the
//! Vision Transformer (ViT) architecture.

const std = @import("std");
const ViTConfig = @import("../vit.zig").ViTConfig;

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
