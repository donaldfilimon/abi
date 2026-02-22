//! Core `TrainableModel` type with behavior delegated to focused modules.
//!
//! This file keeps API-shape stability while `forward`, `backward`, `io`, and
//! checkpoint persistence are implemented in dedicated modules.

const std = @import("std");
const ops = @import("../../llm/ops/mod.zig");
const model_config = @import("../model/config.zig");
const trainable_ckpt = @import("../trainable_checkpoint.zig");

// Re-exports from related modules.
const weights_mod = @import("../weights.zig");
const forward_mod = @import("forward.zig");
const backward_mod = @import("backward.zig");
const io_mod = @import("io.zig");
const checkpoint_mod = @import("checkpoint.zig");
const training_bridge = @import("../../../gpu/training_bridge.zig");

pub const TrainableLayerWeights = weights_mod.TrainableLayerWeights;
pub const TrainableWeights = weights_mod.TrainableWeights;
pub const ActivationCache = weights_mod.ActivationCache;

pub const GradientCheckpointer = trainable_ckpt.GradientCheckpointer;
pub const ModelCheckpoint = trainable_ckpt.ModelCheckpoint;
pub const LoadError = trainable_ckpt.LoadError;

pub const CheckpointingStrategy = model_config.CheckpointingStrategy;
pub const TrainableModelConfig = model_config.TrainableModelConfig;

pub const TrainableModel = struct {
    allocator: std.mem.Allocator,
    config: TrainableModelConfig,
    weights: TrainableWeights,
    activations: ?*ActivationCache,
    rope_cache: ?*ops.rope.RopeCache,

    pub fn init(allocator: std.mem.Allocator, config: TrainableModelConfig) !TrainableModel {
        var weights = try TrainableWeights.init(allocator, config);
        errdefer weights.deinit();

        const rope_cache = try allocator.create(ops.rope.RopeCache);
        errdefer allocator.destroy(rope_cache);
        rope_cache.* = try ops.rope.RopeCache.init(allocator, .{
            .head_dim = config.headDim(),
            .theta_base = config.rope_theta,
            .max_seq_len = config.max_seq_len,
        });

        return .{
            .allocator = allocator,
            .config = config,
            .weights = weights,
            .activations = null,
            .rope_cache = rope_cache,
        };
    }

    pub fn deinit(self: *TrainableModel) void {
        if (self.activations) |act| {
            act.deinit();
            self.allocator.destroy(act);
        }
        if (self.rope_cache) |rc| {
            rc.deinit();
            self.allocator.destroy(rc);
        }
        self.weights.deinit();
        self.* = undefined;
    }

    /// Zero all gradients.
    pub fn zeroGradients(self: *TrainableModel) void {
        self.weights.zeroGradients();
    }

    /// Get number of parameters.
    pub fn numParams(self: *const TrainableModel) usize {
        return self.config.numParams();
    }

    /// Prepare activation cache for training.
    pub fn prepareForTraining(self: *TrainableModel, max_seq_len: u32) !void {
        if (self.activations) |act| {
            act.deinit();
            self.allocator.destroy(act);
        }
        const act = try self.allocator.create(ActivationCache);
        act.* = try ActivationCache.init(self.allocator, self.config, max_seq_len);
        self.activations = act;
    }

    /// Forward pass through the model.
    /// Returns logits: [seq_len, vocab_size]
    pub fn forward(self: *TrainableModel, input_ids: []const u32, logits_out: []f32) !void {
        return forward_mod.run(self, input_ids, logits_out);
    }

    /// Forward pass with GPU bridge acceleration.
    pub fn forwardGpu(self: *TrainableModel, input_ids: []const u32, logits_out: []f32, bridge: *training_bridge.GpuTrainingBridge) !void {
        return forward_mod.runWithBridge(self, input_ids, logits_out, bridge);
    }

    /// Backward pass through the model.
    pub fn backward(self: *TrainableModel, d_logits: []const f32, input_ids: []const u32) !void {
        return backward_mod.backward(self, d_logits, input_ids);
    }

    /// Backward pass with GPU bridge acceleration.
    pub fn backwardGpu(self: *TrainableModel, d_logits: []const f32, input_ids: []const u32, bridge: *training_bridge.GpuTrainingBridge) !void {
        return backward_mod.backwardWithBridge(self, d_logits, input_ids, bridge);
    }

    /// Compute cross-entropy loss and its gradient.
    pub fn computeCrossEntropyLoss(logits: []const f32, targets: []const u32, d_logits: []f32, vocab_size: u32) f32 {
        return backward_mod.computeCrossEntropyLoss(logits, targets, d_logits, vocab_size);
    }

    /// Complete training step: forward, loss, backward.
    /// Returns the loss value.
    pub fn trainStep(self: *TrainableModel, input_ids: []const u32, target_ids: []const u32) !f32 {
        return backward_mod.trainStep(self, input_ids, target_ids);
    }

    /// Load weights from a GGUF file.
    /// Dequantizes quantized tensors to f32 for training.
    pub fn loadFromGguf(self: *TrainableModel, path: []const u8) !void {
        return io_mod.loadFromGguf(self, path);
    }

    /// Create a TrainableModel from a GGUF file.
    /// Extracts config from metadata and loads weights.
    pub fn fromGguf(allocator: std.mem.Allocator, path: []const u8) !TrainableModel {
        const config = try io_mod.readConfigFromGguf(allocator, path);
        var model = try TrainableModel.init(allocator, config);
        errdefer model.deinit();
        try io_mod.loadFromGguf(&model, path);
        return model;
    }

    /// Save model checkpoint to a file.
    /// This flattens all weights and saves them with config metadata.
    pub fn saveCheckpoint(self: *const TrainableModel, path: []const u8, step: u64) !void {
        return checkpoint_mod.saveCheckpoint(self, path, step);
    }

    /// Load model checkpoint from a file.
    /// The model config must match the checkpoint.
    pub fn loadCheckpointFile(self: *TrainableModel, path: []const u8) !u64 {
        return checkpoint_mod.loadCheckpointFile(self, path);
    }

    /// Collect all weights into a flat array for checkpointing.
    pub fn collectWeights(self: *const TrainableModel) ![]f32 {
        return checkpoint_mod.collectWeights(self);
    }

    /// Distribute a flat weight array back to model weights.
    pub fn distributeWeights(self: *TrainableModel, weights: []const f32) !void {
        return checkpoint_mod.distributeWeights(self, weights);
    }

    /// Create a checkpoint from current model state.
    pub fn createCheckpoint(self: *const TrainableModel, step: u64) !ModelCheckpoint {
        return checkpoint_mod.createCheckpoint(self, step);
    }

    /// Load model state from a checkpoint.
    pub fn loadFromCheckpoint(self: *TrainableModel, ckpt: *const ModelCheckpoint) !void {
        return checkpoint_mod.loadFromCheckpoint(self, ckpt);
    }

    /// Compute global L2 norm of all gradients.
    pub fn computeGradientNorm(self: *const TrainableModel) f32 {
        var sum_sq: f32 = 0;

        for (self.weights.d_token_embedding) |g| sum_sq += g * g;

        for (self.weights.layers) |layer| {
            for (layer.d_w_q) |g| sum_sq += g * g;
            for (layer.d_w_k) |g| sum_sq += g * g;
            for (layer.d_w_v) |g| sum_sq += g * g;
            for (layer.d_w_o) |g| sum_sq += g * g;
            for (layer.d_attn_norm) |g| sum_sq += g * g;
            for (layer.d_w_gate) |g| sum_sq += g * g;
            for (layer.d_w_up) |g| sum_sq += g * g;
            for (layer.d_w_down) |g| sum_sq += g * g;
            for (layer.d_ffn_norm) |g| sum_sq += g * g;
        }

        for (self.weights.d_final_norm) |g| sum_sq += g * g;

        if (self.weights.d_output_proj) |d_op| {
            for (d_op) |g| sum_sq += g * g;
        }

        return @sqrt(sum_sq);
    }

    /// Clip gradients by global norm.
    /// If the global norm exceeds max_norm, gradients are scaled down.
    /// Returns the original gradient norm before clipping.
    pub fn clipGradients(self: *TrainableModel, max_norm: f32) f32 {
        const grad_norm = self.computeGradientNorm();

        if (grad_norm > max_norm and grad_norm > 0) {
            const scale = max_norm / grad_norm;
            for (self.weights.d_token_embedding) |*g| g.* *= scale;

            for (self.weights.layers) |*layer| {
                for (layer.d_w_q) |*g| g.* *= scale;
                for (layer.d_w_k) |*g| g.* *= scale;
                for (layer.d_w_v) |*g| g.* *= scale;
                for (layer.d_w_o) |*g| g.* *= scale;
                for (layer.d_attn_norm) |*g| g.* *= scale;
                for (layer.d_w_gate) |*g| g.* *= scale;
                for (layer.d_w_up) |*g| g.* *= scale;
                for (layer.d_w_down) |*g| g.* *= scale;
                for (layer.d_ffn_norm) |*g| g.* *= scale;
            }

            for (self.weights.d_final_norm) |*g| {
                g.* *= scale;
            }

            if (self.weights.d_output_proj) |d_op| {
                for (d_op) |*g| g.* *= scale;
            }
        }

        return grad_norm;
    }

    /// Check if gradients contain any non-finite values (NaN or Inf).
    pub fn hasNonFiniteGradients(self: *const TrainableModel) bool {
        for (self.weights.d_token_embedding) |g| {
            if (!std.math.isFinite(g)) return true;
        }

        for (self.weights.layers) |layer| {
            for (layer.d_w_q) |g| if (!std.math.isFinite(g)) return true;
            for (layer.d_w_k) |g| if (!std.math.isFinite(g)) return true;
            for (layer.d_w_v) |g| if (!std.math.isFinite(g)) return true;
            for (layer.d_w_o) |g| if (!std.math.isFinite(g)) return true;
            for (layer.d_attn_norm) |g| if (!std.math.isFinite(g)) return true;
            for (layer.d_w_gate) |g| if (!std.math.isFinite(g)) return true;
            for (layer.d_w_up) |g| if (!std.math.isFinite(g)) return true;
            for (layer.d_w_down) |g| if (!std.math.isFinite(g)) return true;
            for (layer.d_ffn_norm) |g| if (!std.math.isFinite(g)) return true;
        }

        for (self.weights.d_final_norm) |g| {
            if (!std.math.isFinite(g)) return true;
        }

        if (self.weights.d_output_proj) |d_op| {
            for (d_op) |g| {
                if (!std.math.isFinite(g)) return true;
            }
        }

        return false;
    }

    /// Apply SGD update to all weights.
    /// weights = weights - learning_rate * gradients
    pub fn applySgdUpdate(self: *TrainableModel, learning_rate: f32) void {
        for (self.weights.d_token_embedding, self.weights.token_embedding) |*g, *w| {
            w.* -= learning_rate * g.*;
        }

        for (self.weights.layers) |*layer| {
            for (layer.w_q, layer.d_w_q) |*w, g| {
                w.* -= learning_rate * g;
            }
            for (layer.w_k, layer.d_w_k) |*w, g| {
                w.* -= learning_rate * g;
            }
            for (layer.w_v, layer.d_w_v) |*w, g| {
                w.* -= learning_rate * g;
            }
            for (layer.w_o, layer.d_w_o) |*w, g| {
                w.* -= learning_rate * g;
            }
            for (layer.attn_norm, layer.d_attn_norm) |*w, g| {
                w.* -= learning_rate * g;
            }
            for (layer.w_gate, layer.d_w_gate) |*w, g| {
                w.* -= learning_rate * g;
            }
            for (layer.w_up, layer.d_w_up) |*w, g| {
                w.* -= learning_rate * g;
            }
            for (layer.w_down, layer.d_w_down) |*w, g| {
                w.* -= learning_rate * g;
            }
            for (layer.ffn_norm, layer.d_ffn_norm) |*w, g| {
                w.* -= learning_rate * g;
            }
        }

        for (self.weights.d_final_norm, self.weights.final_norm) |*g, *w| {
            w.* -= learning_rate * g.*;
        }

        if (self.weights.output_proj) |op| {
            if (self.weights.d_output_proj) |d_op| {
                for (op, d_op) |*w, g| {
                    w.* -= learning_rate * g;
                }
            }
        }
    }

    /// Scale all gradients by a factor (for mixed precision unscaling).
    pub fn scaleGradients(self: *TrainableModel, scale: f32) void {
        for (self.weights.d_token_embedding) |*g| g.* *= scale;

        for (self.weights.layers) |*layer| {
            for (layer.d_w_q) |*g| g.* *= scale;
            for (layer.d_w_k) |*g| g.* *= scale;
            for (layer.d_w_v) |*g| g.* *= scale;
            for (layer.d_w_o) |*g| g.* *= scale;
            for (layer.d_attn_norm) |*g| g.* *= scale;
            for (layer.d_w_gate) |*g| g.* *= scale;
            for (layer.d_w_up) |*g| g.* *= scale;
            for (layer.d_w_down) |*g| g.* *= scale;
            for (layer.d_ffn_norm) |*g| g.* *= scale;
        }

        for (self.weights.d_final_norm) |*g| g.* *= scale;

        if (self.weights.d_output_proj) |d_op| {
            for (d_op) |*g| g.* *= scale;
        }
    }

    /// Training step with gradient clipping and optional mixed precision.
    /// Returns the loss value and gradient norm before clipping.
    pub fn trainStepWithClipping(
        self: *TrainableModel,
        input_ids: []const u32,
        target_ids: []const u32,
        learning_rate: f32,
        max_grad_norm: f32,
        loss_scale: ?f32,
    ) !TrainStepResult {
        const seq_len: u32 = @intCast(input_ids.len);
        const vocab_size = self.config.vocab_size;

        const logits = try self.allocator.alloc(f32, seq_len * vocab_size);
        defer self.allocator.free(logits);
        const d_logits = try self.allocator.alloc(f32, seq_len * vocab_size);
        defer self.allocator.free(d_logits);

        try self.forward(input_ids, logits);
        const loss = TrainableModel.computeCrossEntropyLoss(logits, target_ids, d_logits, vocab_size);

        if (loss_scale) |scale| {
            for (d_logits) |*g| {
                g.* *= scale;
            }
        }

        try self.backward(d_logits, input_ids);

        if (loss_scale) |scale| {
            self.scaleGradients(1.0 / scale);
        }

        const has_nan = self.hasNonFiniteGradients();
        if (has_nan) {
            self.zeroGradients();
            return .{
                .loss = loss,
                .grad_norm = 0,
                .grad_norm_clipped = 0,
                .skipped = true,
            };
        }

        const grad_norm = self.clipGradients(max_grad_norm);
        const grad_norm_clipped = self.computeGradientNorm();

        self.applySgdUpdate(learning_rate);
        self.zeroGradients();

        return .{
            .loss = loss,
            .grad_norm = grad_norm,
            .grad_norm_clipped = grad_norm_clipped,
            .skipped = false,
        };
    }

    pub const TrainStepResult = struct {
        loss: f32,
        grad_norm: f32,
        grad_norm_clipped: f32,
        skipped: bool,
    };

    /// Export trainable weights to GGUF format (weights only).
    pub fn exportToGguf(
        self: *const TrainableModel,
        allocator: std.mem.Allocator,
        path: []const u8,
        config: checkpoint_mod.GgufExportConfig,
    ) !void {
        return checkpoint_mod.exportToGguf(self, allocator, path, config);
    }
};
