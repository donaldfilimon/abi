//! Trainable weights container.

const std = @import("std");
const config_mod = @import("config.zig");
const layer = @import("layer.zig");
const utils = @import("utils.zig");

/// Trainable weights for the full model.
pub const TrainableWeights = struct {
    allocator: std.mem.Allocator,
    config: config_mod.TrainableModelConfig,

    // Embedding
    token_embedding: []f32,
    d_token_embedding: []f32,

    // Layers
    layers: []layer.TrainableLayerWeights,

    // Final norm
    final_norm: []f32,
    d_final_norm: []f32,

    // Output projection (if not tied)
    output_proj: ?[]f32,
    d_output_proj: ?[]f32,

    pub fn init(allocator: std.mem.Allocator, config: config_mod.TrainableModelConfig) !TrainableWeights {
        const token_embedding = try allocator.alloc(f32, config.vocab_size * config.hidden_dim);
        errdefer allocator.free(token_embedding);
        const d_token_embedding = try allocator.alloc(f32, config.vocab_size * config.hidden_dim);
        errdefer allocator.free(d_token_embedding);

        const layers = try allocator.alloc(layer.TrainableLayerWeights, config.num_layers);
        errdefer allocator.free(layers);

        var initialized_layers: usize = 0;
        errdefer {
            for (0..initialized_layers) |i| {
                layers[i].deinit();
            }
        }

        for (layers) |*l| {
            l.* = try layer.TrainableLayerWeights.init(allocator, config);
            initialized_layers += 1;
        }

        const final_norm = try allocator.alloc(f32, config.hidden_dim);
        errdefer allocator.free(final_norm);
        const d_final_norm = try allocator.alloc(f32, config.hidden_dim);
        errdefer allocator.free(d_final_norm);

        var output_proj: ?[]f32 = null;
        var d_output_proj: ?[]f32 = null;
        if (!config.tie_embeddings) {
            output_proj = try allocator.alloc(f32, config.hidden_dim * config.vocab_size);
            d_output_proj = try allocator.alloc(f32, config.hidden_dim * config.vocab_size);
        }

        // Initialize
        utils.initializeXavier(token_embedding);
        @memset(d_token_embedding, 0);
        @memset(final_norm, 1.0);
        @memset(d_final_norm, 0);
        if (output_proj) |op| {
            utils.initializeXavier(op);
        }
        if (d_output_proj) |dop| {
            @memset(dop, 0);
        }

        return .{
            .allocator = allocator,
            .config = config,
            .token_embedding = token_embedding,
            .d_token_embedding = d_token_embedding,
            .layers = layers,
            .final_norm = final_norm,
            .d_final_norm = d_final_norm,
            .output_proj = output_proj,
            .d_output_proj = d_output_proj,
        };
    }

    pub fn deinit(self: *TrainableWeights) void {
        if (self.d_output_proj) |dop| {
            self.allocator.free(dop);
        }
        if (self.output_proj) |op| {
            self.allocator.free(op);
        }
        self.allocator.free(self.d_final_norm);
        self.allocator.free(self.final_norm);
        for (self.layers) |*l| {
            l.deinit();
        }
        self.allocator.free(self.layers);
        self.allocator.free(self.d_token_embedding);
        self.allocator.free(self.token_embedding);
        self.* = undefined;
    }

    /// Zero all gradients.
    pub fn zeroGradients(self: *TrainableWeights) void {
        @memset(self.d_token_embedding, 0);
        for (self.layers) |*l| {
            l.zeroGradients();
        }
        @memset(self.d_final_norm, 0);
        if (self.d_output_proj) |dop| {
            @memset(dop, 0);
        }
    }

    /// Get output projection weights (tied or separate).
    pub fn getOutputProj(self: *const TrainableWeights) []const f32 {
        return self.output_proj orelse self.token_embedding;
    }

    /// Get output projection gradient.
    pub fn getOutputProjGrad(self: *TrainableWeights) []f32 {
        return self.d_output_proj orelse self.d_token_embedding;
    }

    /// Number of parameters.
    pub fn numParams(self: *const TrainableWeights) usize {
        return self.config.numParams();
    }
};

test {
    std.testing.refAllDecls(@This());
}
