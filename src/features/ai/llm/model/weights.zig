//! Weight management for LLaMA models.
//!
//! Handles loading, storing, and accessing model weights.

const std = @import("std");
const config_mod = @import("config.zig");
const gguf = @import("../io/gguf.zig");
const tensor_loader = @import("../io/tensor_loader.zig");

/// Layer weights for a single transformer layer.
pub const LayerWeights = struct {
    /// Attention weights
    q_proj: []const f32,
    k_proj: []const f32,
    v_proj: []const f32,
    o_proj: []const f32,

    /// FFN weights (SwiGLU)
    gate_proj: []const f32, // gate
    up_proj: []const f32, // up
    down_proj: []const f32,

    /// Normalization weights
    input_norm: []const f32, // attention input norm
    post_attn_norm: []const f32, // FFN input norm

    /// Whether weights are quantized
    quantized: bool,
    quant_type: ?gguf.GgufTensorType,
};

/// All weights for a LLaMA model.
pub const LlamaWeights = struct {
    allocator: std.mem.Allocator,
    config: config_mod.LlamaConfig,

    /// Token embeddings: [vocab_size, dim]
    token_embedding: []f32,

    /// Per-layer weights
    layers: []LayerWeights,

    /// Output projection (if not tied)
    output_proj: ?[]f32,

    /// Final normalization
    final_norm: []f32,

    /// Original GGUF file (for quantized weights)
    gguf_file: ?*gguf.GgufFile,

    pub fn init(allocator: std.mem.Allocator, llama_config: config_mod.LlamaConfig) !LlamaWeights {
        const layers = try allocator.alloc(LayerWeights, llama_config.n_layers);
        @memset(layers, std.mem.zeroes(LayerWeights));

        return .{
            .allocator = allocator,
            .config = llama_config,
            .token_embedding = &[_]f32{},
            .layers = layers,
            .output_proj = null,
            .final_norm = &[_]f32{},
            .gguf_file = null,
        };
    }

    pub fn deinit(self: *LlamaWeights) void {
        // Free allocated weight buffers
        if (self.token_embedding.len > 0) {
            self.allocator.free(self.token_embedding);
        }

        for (self.layers) |layer| {
            if (layer.q_proj.len > 0) self.allocator.free(@constCast(layer.q_proj));
            if (layer.k_proj.len > 0) self.allocator.free(@constCast(layer.k_proj));
            if (layer.v_proj.len > 0) self.allocator.free(@constCast(layer.v_proj));
            if (layer.o_proj.len > 0) self.allocator.free(@constCast(layer.o_proj));
            if (layer.gate_proj.len > 0) self.allocator.free(@constCast(layer.gate_proj));
            if (layer.up_proj.len > 0) self.allocator.free(@constCast(layer.up_proj));
            if (layer.down_proj.len > 0) self.allocator.free(@constCast(layer.down_proj));
        }

        self.allocator.free(self.layers);

        if (self.output_proj) |proj| {
            self.allocator.free(proj);
        }

        if (self.final_norm.len > 0) {
            self.allocator.free(self.final_norm);
        }

        if (self.gguf_file) |file| {
            @constCast(file).deinit();
            self.allocator.destroy(file);
        }

        self.* = undefined;
    }

    /// Check if weights are memory-mapped (not copied).
    pub fn isMemoryMapped(self: *const LlamaWeights) bool {
        return self.gguf_file != null;
    }

    /// Load weights from a GGUF file.
    pub fn loadFromGguf(self: *LlamaWeights, path: []const u8) !void {
        var file = try gguf.GgufFile.open(self.allocator, path);
        errdefer file.deinit();

        // Update config from GGUF
        self.config = config_mod.LlamaConfig.fromGguf(&file);

        // Load token embeddings
        if (file.getTensorData("token_embd.weight")) |data| {
            const info = file.getTensor("token_embd.weight").?;
            self.token_embedding = try self.dequantizeTensor(data, info);
        }

        // Load layer weights
        for (0..self.config.n_layers) |i| {
            try self.loadLayerWeights(&file, @intCast(i));
        }

        // Load final norm
        if (file.getTensorData("output_norm.weight")) |data| {
            self.final_norm = try self.allocator.dupe(f32, @alignCast(std.mem.bytesAsSlice(f32, data)));
        }

        // Load output projection
        if (file.getTensorData("output.weight")) |data| {
            const info = file.getTensor("output.weight").?;
            self.output_proj = self.dequantizeTensor(data, info) catch |err| switch (err) {
                error.UnsupportedQuantization => null,
                else => return err,
            };
        }

        // Keep file reference for potential future access
        const file_ptr = try self.allocator.create(gguf.GgufFile);
        file_ptr.* = file;
        self.gguf_file = file_ptr;
    }

    fn loadLayerWeights(self: *LlamaWeights, file: *gguf.GgufFile, layer_idx: u32) !void {
        var buf: [64]u8 = undefined;

        // Attention weights
        const q_name = try std.fmt.bufPrint(&buf, "blk.{d}.attn_q.weight", .{layer_idx});
        if (file.getTensorData(q_name)) |data| {
            const info = file.getTensor(q_name).?;
            self.layers[layer_idx].q_proj = try self.dequantizeTensor(data, info);
        }

        const k_name = try std.fmt.bufPrint(&buf, "blk.{d}.attn_k.weight", .{layer_idx});
        if (file.getTensorData(k_name)) |data| {
            const info = file.getTensor(k_name).?;
            self.layers[layer_idx].k_proj = try self.dequantizeTensor(data, info);
        }

        const v_name = try std.fmt.bufPrint(&buf, "blk.{d}.attn_v.weight", .{layer_idx});
        if (file.getTensorData(v_name)) |data| {
            const info = file.getTensor(v_name).?;
            self.layers[layer_idx].v_proj = try self.dequantizeTensor(data, info);
        }

        const o_name = try std.fmt.bufPrint(&buf, "blk.{d}.attn_output.weight", .{layer_idx});
        if (file.getTensorData(o_name)) |data| {
            const info = file.getTensor(o_name).?;
            self.layers[layer_idx].o_proj = try self.dequantizeTensor(data, info);
        }

        // FFN weights
        const gate_name = try std.fmt.bufPrint(&buf, "blk.{d}.ffn_gate.weight", .{layer_idx});
        if (file.getTensorData(gate_name)) |data| {
            const info = file.getTensor(gate_name).?;
            self.layers[layer_idx].gate_proj = try self.dequantizeTensor(data, info);
        }

        const up_name = try std.fmt.bufPrint(&buf, "blk.{d}.ffn_up.weight", .{layer_idx});
        if (file.getTensorData(up_name)) |data| {
            const info = file.getTensor(up_name).?;
            self.layers[layer_idx].up_proj = try self.dequantizeTensor(data, info);
        }

        const down_name = try std.fmt.bufPrint(&buf, "blk.{d}.ffn_down.weight", .{layer_idx});
        if (file.getTensorData(down_name)) |data| {
            const info = file.getTensor(down_name).?;
            self.layers[layer_idx].down_proj = try self.dequantizeTensor(data, info);
        }

        // Normalization weights
        const input_norm_name = try std.fmt.bufPrint(&buf, "blk.{d}.attn_norm.weight", .{layer_idx});
        if (file.getTensorData(input_norm_name)) |data| {
            self.layers[layer_idx].input_norm = @alignCast(std.mem.bytesAsSlice(f32, data));
        }

        const post_norm_name = try std.fmt.bufPrint(&buf, "blk.{d}.ffn_norm.weight", .{layer_idx});
        if (file.getTensorData(post_norm_name)) |data| {
            self.layers[layer_idx].post_attn_norm = @alignCast(std.mem.bytesAsSlice(f32, data));
        }
    }

    fn dequantizeTensor(self: *LlamaWeights, data: []const u8, info: gguf.TensorInfo) ![]f32 {
        const element_count = info.elementCount();
        const result = try self.allocator.alloc(f32, @intCast(element_count));
        errdefer self.allocator.free(result);

        switch (info.tensor_type) {
            .f32 => {
                const f32_slice: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, data));
                @memcpy(result, f32_slice);
            },
            .f16 => {
                const f16_data: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, data));
                for (f16_data, 0..) |v, i| {
                    result[i] = @floatCast(v);
                }
            },
            .q4_0 => {
                try tensor_loader.dequantizeQ4_0(data, result);
            },
            .q8_0 => {
                try tensor_loader.dequantizeQ8_0(data, result);
            },
            .q6_k => {
                try tensor_loader.dequantizeQ6_K(data, result);
            },
            else => {
                return error.UnsupportedQuantization;
            },
        }

        return result;
    }

    /// Get layer weights.
    pub fn getLayer(self: *const LlamaWeights, idx: u32) *const LayerWeights {
        return &self.layers[idx];
    }

    /// Compute memory usage.
    pub fn memoryBytes(self: *const LlamaWeights) u64 {
        var total: u64 = 0;

        total += self.token_embedding.len * @sizeOf(f32);
        total += self.final_norm.len * @sizeOf(f32);

        if (self.output_proj) |proj| {
            total += proj.len * @sizeOf(f32);
        }

        for (self.layers) |layer| {
            total += layer.q_proj.len * @sizeOf(f32);
            total += layer.k_proj.len * @sizeOf(f32);
            total += layer.v_proj.len * @sizeOf(f32);
            total += layer.o_proj.len * @sizeOf(f32);
            total += layer.gate_proj.len * @sizeOf(f32);
            total += layer.up_proj.len * @sizeOf(f32);
            total += layer.down_proj.len * @sizeOf(f32);
            total += layer.input_norm.len * @sizeOf(f32);
            total += layer.post_attn_norm.len * @sizeOf(f32);
        }

        return total;
    }
};

test "llama weights init" {
    const allocator = std.testing.allocator;
    const llama_config = config_mod.LlamaConfig.llama7B();

    var w = try LlamaWeights.init(allocator, llama_config);
    defer w.deinit();

    try std.testing.expectEqual(@as(usize, 32), w.layers.len);
}
