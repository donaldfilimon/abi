//! Checkpoint-related types and helpers for trainable models.
//!
//! Contains GradientCheckpointer, ModelCheckpoint, LoadError,
//! and dequantization helpers.

const std = @import("std");
const gguf = @import("../llm/io/gguf.zig");
const model_config = @import("model/config.zig");

/// Configuration for a trainable model.
pub const TrainableModelConfig = model_config.TrainableModelConfig;

/// Gradient checkpointing strategy.
pub const CheckpointingStrategy = model_config.CheckpointingStrategy;

pub const LoadError = error{
    MissingMetadata,
    ConfigMismatch,
    TensorNotFound,
    BufferTooSmall,
    UnsupportedTensorType,
    NameTooLong,
    OutOfMemory,
} || gguf.GgufError;

/// Gradient checkpointing manager.
/// Manages selective storage/recomputation of activations to reduce memory.
pub const GradientCheckpointer = struct {
    allocator: std.mem.Allocator,
    strategy: CheckpointingStrategy,
    checkpoint_interval: u32,
    num_layers: u32,
    /// Which layers to checkpoint (store activations)
    checkpointed_layers: []bool,
    /// Checkpointed inputs for recomputation
    layer_inputs: []?[]f32,

    pub fn init(
        allocator: std.mem.Allocator,
        config: TrainableModelConfig,
    ) !GradientCheckpointer {
        const checkpointed_layers = try allocator.alloc(bool, config.num_layers);
        @memset(checkpointed_layers, false);

        const layer_inputs = try allocator.alloc(?[]f32, config.num_layers);
        @memset(layer_inputs, null);

        var self = GradientCheckpointer{
            .allocator = allocator,
            .strategy = config.checkpointing,
            .checkpoint_interval = config.checkpoint_interval,
            .num_layers = config.num_layers,
            .checkpointed_layers = checkpointed_layers,
            .layer_inputs = layer_inputs,
        };

        // Mark which layers should be checkpointed based on strategy
        self.computeCheckpointMask();

        return self;
    }

    pub fn deinit(self: *GradientCheckpointer) void {
        for (self.layer_inputs) |maybe_input| {
            if (maybe_input) |input| {
                self.allocator.free(input);
            }
        }
        self.allocator.free(self.layer_inputs);
        self.allocator.free(self.checkpointed_layers);
        self.* = undefined;
    }

    fn computeCheckpointMask(self: *GradientCheckpointer) void {
        switch (self.strategy) {
            .none => {
                // Checkpoint all layers (store all activations)
                @memset(self.checkpointed_layers, true);
            },
            .every_n_layers => {
                // Checkpoint every N layers
                for (0..self.num_layers) |i| {
                    self.checkpointed_layers[i] = (i % self.checkpoint_interval == 0);
                }
                // Always checkpoint first and last
                self.checkpointed_layers[0] = true;
                if (self.num_layers > 0) {
                    self.checkpointed_layers[self.num_layers - 1] = true;
                }
            },
            .attention_only => {
                // Only checkpoint attention layers (assuming alternating)
                @memset(self.checkpointed_layers, false);
                // We still need layer boundaries
                for (0..self.num_layers) |i| {
                    if (i % 2 == 0) { // Mark every other layer
                        self.checkpointed_layers[i] = true;
                    }
                }
            },
            .full => {
                // Don't checkpoint any layers (recompute everything)
                @memset(self.checkpointed_layers, false);
                // Still need to checkpoint first layer input
                if (self.num_layers > 0) {
                    self.checkpointed_layers[0] = true;
                }
            },
        }
    }

    /// Check if layer should store activations.
    pub fn shouldStoreActivations(self: *const GradientCheckpointer, layer_idx: u32) bool {
        if (layer_idx >= self.num_layers) return false;
        return self.checkpointed_layers[layer_idx];
    }

    /// Store layer input for potential recomputation.
    pub fn storeLayerInput(self: *GradientCheckpointer, layer_idx: u32, input: []const f32) !void {
        if (layer_idx >= self.num_layers) return;
        if (!self.checkpointed_layers[layer_idx]) return;

        // Free existing if any
        if (self.layer_inputs[layer_idx]) |existing| {
            self.allocator.free(existing);
        }

        // Copy input
        const copy = try self.allocator.alloc(f32, input.len);
        @memcpy(copy, input);
        self.layer_inputs[layer_idx] = copy;
    }

    /// Get stored layer input for recomputation.
    pub fn getLayerInput(self: *const GradientCheckpointer, layer_idx: u32) ?[]const f32 {
        if (layer_idx >= self.num_layers) return null;
        return self.layer_inputs[layer_idx];
    }

    /// Find the nearest checkpoint before this layer.
    pub fn findNearestCheckpoint(self: *const GradientCheckpointer, layer_idx: u32) ?u32 {
        if (layer_idx == 0) return 0;

        var i: u32 = layer_idx;
        while (i > 0) {
            i -= 1;
            if (self.checkpointed_layers[i]) return i;
        }
        return if (self.checkpointed_layers[0]) @as(u32, 0) else null;
    }

    /// Clear all stored inputs (call after backward pass).
    pub fn clearStoredInputs(self: *GradientCheckpointer) void {
        for (self.layer_inputs) |*maybe_input| {
            if (maybe_input.*) |input| {
                self.allocator.free(input);
                maybe_input.* = null;
            }
        }
    }

    /// Estimate memory savings compared to full activation storage.
    pub fn estimateMemorySavings(self: *const GradientCheckpointer) f32 {
        var stored_count: u32 = 0;
        for (self.checkpointed_layers) |is_stored| {
            if (is_stored) stored_count += 1;
        }
        const full = @as(f32, @floatFromInt(self.num_layers));
        const actual = @as(f32, @floatFromInt(stored_count));
        return 1.0 - (actual / full);
    }
};

/// Model checkpoint for saving/loading.
pub const ModelCheckpoint = struct {
    allocator: std.mem.Allocator,
    config: TrainableModelConfig,
    weights: []f32,
    step: u64,
    timestamp: u64,

    pub fn deinit(self: *ModelCheckpoint) void {
        self.allocator.free(self.weights);
        self.* = undefined;
    }
};

/// Dequantize Q4_0 data to f32.
pub fn dequantizeQ4_0(data: []const u8, dest: []f32, allocator: std.mem.Allocator) !void {
    _ = allocator;
    const block_size: usize = 32;
    const bytes_per_block: usize = 18; // 2 byte scale + 16 bytes data

    var dest_idx: usize = 0;
    var data_idx: usize = 0;

    while (data_idx + bytes_per_block <= data.len and dest_idx + block_size <= dest.len) {
        // Read scale (f16)
        const scale_bytes = data[data_idx..][0..2];
        const scale: f32 = @floatCast(@as(f16, @bitCast(scale_bytes.*)));
        data_idx += 2;

        // Read quantized values (4 bits each, packed)
        for (0..16) |i| {
            const byte = data[data_idx + i];
            const lo: i8 = @as(i8, @intCast(byte & 0x0F)) - 8;
            const hi: i8 = @as(i8, @intCast(byte >> 4)) - 8;

            dest[dest_idx] = @as(f32, @floatFromInt(lo)) * scale;
            dest[dest_idx + 1] = @as(f32, @floatFromInt(hi)) * scale;
            dest_idx += 2;
        }
        data_idx += 16;
    }
}

/// Dequantize Q8_0 data to f32.
pub fn dequantizeQ8_0(data: []const u8, dest: []f32, allocator: std.mem.Allocator) !void {
    _ = allocator;
    const block_size: usize = 32;
    const bytes_per_block: usize = 34; // 2 byte scale + 32 bytes data

    var dest_idx: usize = 0;
    var data_idx: usize = 0;

    while (data_idx + bytes_per_block <= data.len and dest_idx + block_size <= dest.len) {
        // Read scale (f16)
        const scale_bytes = data[data_idx..][0..2];
        const scale: f32 = @floatCast(@as(f16, @bitCast(scale_bytes.*)));
        data_idx += 2;

        // Read quantized values (8 bits each)
        for (0..32) |i| {
            const qval: i8 = @bitCast(data[data_idx + i]);
            dest[dest_idx + i] = @as(f32, @floatFromInt(qval)) * scale;
        }
        dest_idx += 32;
        data_idx += 32;
    }
}
