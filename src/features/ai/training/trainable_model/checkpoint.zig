const std = @import("std");
const gguf_writer = @import("../../llm/io/gguf_writer.zig");
const checkpoint = @import("../checkpoint.zig");

pub const GgufExportConfig = struct {
    name: []const u8 = "abi-llama",
    tokenizer: ?gguf_writer.TokenizerConfig = null,
};

pub fn saveCheckpoint(model: anytype, path: []const u8, step: u64) !void {
    // Collect all weights into a flat array
    const weights = try collectWeights(model);
    defer model.allocator.free(weights);

    const view = checkpoint.CheckpointView{
        .step = step,
        .timestamp = 0, // Will be set by checkpoint.saveCheckpoint
        .weights = weights,
    };

    try checkpoint.saveCheckpoint(model.allocator, path, view);
    std.log.info("Saved checkpoint to {s} (step {d}, {d} params)", .{ path, step, weights.len });
}

pub fn loadCheckpointFile(model: anytype, path: []const u8) !u64 {
    var ckpt = try checkpoint.loadCheckpoint(model.allocator, path);
    defer ckpt.deinit(model.allocator);

    // Verify weight count matches
    const expected = model.config.numParams();
    if (ckpt.weights.len != expected) {
        std.log.err("Checkpoint weight count mismatch: expected {d}, got {d}", .{ expected, ckpt.weights.len });
        return error.ConfigMismatch;
    }

    // Distribute weights back to model
    try distributeWeights(model, ckpt.weights);

    std.log.info("Loaded checkpoint from {s} (step {d})", .{ path, ckpt.step });
    return ckpt.step;
}

pub fn collectWeights(model: anytype) ![]f32 {
    const total_params = model.config.numParams();
    const weights = try model.allocator.alloc(f32, total_params);
    errdefer model.allocator.free(weights);

    var offset: usize = 0;

    // Token embedding
    @memcpy(weights[offset..][0..model.weights.token_embedding.len], model.weights.token_embedding);
    offset += model.weights.token_embedding.len;

    // Per-layer weights
    for (model.weights.layers) |layer| {
        @memcpy(weights[offset..][0..layer.w_q.len], layer.w_q);
        offset += layer.w_q.len;
        @memcpy(weights[offset..][0..layer.w_k.len], layer.w_k);
        offset += layer.w_k.len;
        @memcpy(weights[offset..][0..layer.w_v.len], layer.w_v);
        offset += layer.w_v.len;
        @memcpy(weights[offset..][0..layer.w_o.len], layer.w_o);
        offset += layer.w_o.len;
        @memcpy(weights[offset..][0..layer.attn_norm.len], layer.attn_norm);
        offset += layer.attn_norm.len;
        @memcpy(weights[offset..][0..layer.w_gate.len], layer.w_gate);
        offset += layer.w_gate.len;
        @memcpy(weights[offset..][0..layer.w_up.len], layer.w_up);
        offset += layer.w_up.len;
        @memcpy(weights[offset..][0..layer.w_down.len], layer.w_down);
        offset += layer.w_down.len;
        @memcpy(weights[offset..][0..layer.ffn_norm.len], layer.ffn_norm);
        offset += layer.ffn_norm.len;
    }

    // Final norm
    @memcpy(weights[offset..][0..model.weights.final_norm.len], model.weights.final_norm);
    offset += model.weights.final_norm.len;

    // Output projection (if not tied)
    if (model.weights.output_proj) |op| {
        @memcpy(weights[offset..][0..op.len], op);
        offset += op.len;
    }

    return weights;
}

pub fn distributeWeights(model: anytype, weights: []const f32) !void {
    var offset: usize = 0;

    // Token embedding
    @memcpy(model.weights.token_embedding, weights[offset..][0..model.weights.token_embedding.len]);
    offset += model.weights.token_embedding.len;

    // Per-layer weights
    for (model.weights.layers) |*layer| {
        @memcpy(layer.w_q, weights[offset..][0..layer.w_q.len]);
        offset += layer.w_q.len;
        @memcpy(layer.w_k, weights[offset..][0..layer.w_k.len]);
        offset += layer.w_k.len;
        @memcpy(layer.w_v, weights[offset..][0..layer.w_v.len]);
        offset += layer.w_v.len;
        @memcpy(layer.w_o, weights[offset..][0..layer.w_o.len]);
        offset += layer.w_o.len;
        @memcpy(layer.attn_norm, weights[offset..][0..layer.attn_norm.len]);
        offset += layer.attn_norm.len;
        @memcpy(layer.w_gate, weights[offset..][0..layer.w_gate.len]);
        offset += layer.w_gate.len;
        @memcpy(layer.w_up, weights[offset..][0..layer.w_up.len]);
        offset += layer.w_up.len;
        @memcpy(layer.w_down, weights[offset..][0..layer.w_down.len]);
        offset += layer.w_down.len;
        @memcpy(layer.ffn_norm, weights[offset..][0..layer.ffn_norm.len]);
        offset += layer.ffn_norm.len;
    }

    // Final norm
    @memcpy(model.weights.final_norm, weights[offset..][0..model.weights.final_norm.len]);
    offset += model.weights.final_norm.len;

    // Output projection (if not tied)
    if (model.weights.output_proj) |op| {
        @memcpy(op, weights[offset..][0..op.len]);
    }
}

pub fn createCheckpoint(model: anytype, step: u64) !trainable_checkpoint.ModelCheckpoint {
    const weights = try collectWeights(model);
    return .{
        .allocator = model.allocator,
        .config = model.config,
        .weights = weights,
        .step = step,
        .timestamp = 0,
    };
}

pub fn loadFromCheckpoint(model: anytype, ckpt: *const trainable_checkpoint.ModelCheckpoint) !void {
    // Verify config matches
    if (ckpt.config.hidden_dim != model.config.hidden_dim or
        ckpt.config.num_layers != model.config.num_layers or
        ckpt.config.num_heads != model.config.num_heads)
    {
        return error.ConfigMismatch;
    }
    try distributeWeights(model, ckpt.weights);
}

pub fn exportToGguf(
    model: anytype,
    allocator: std.mem.Allocator,
    path: []const u8,
    config: GgufExportConfig,
) !void {
    const layer_count: usize = @intCast(model.config.num_layers);
    const layers = try allocator.alloc(gguf_writer.LayerWeights, layer_count);
    defer allocator.free(layers);

    for (model.weights.layers, 0..) |layer, i| {
        layers[i] = .{
            .attn_norm = layer.attn_norm,
            .ffn_norm = layer.ffn_norm,
            .wq = layer.w_q,
            .wk = layer.w_k,
            .wv = layer.w_v,
            .wo = layer.w_o,
            .w_gate = layer.w_gate,
            .w_up = layer.w_up,
            .w_down = layer.w_down,
        };
    }

    const export_config = gguf_writer.ExportConfig{
        .name = config.name,
        .vocab_size = model.config.vocab_size,
        .context_length = model.config.max_seq_len,
        .embedding_length = model.config.hidden_dim,
        .block_count = model.config.num_layers,
        .head_count = model.config.num_heads,
        .head_count_kv = model.config.num_kv_heads,
        .ffn_hidden_dim = model.config.intermediate_dim,
        .rope_freq_base = model.config.rope_theta,
        .layer_norm_rms_epsilon = model.config.norm_eps,
        .tokenizer = config.tokenizer,
    };

    const export_weights = gguf_writer.ExportWeights{
        .token_embedding = model.weights.token_embedding,
        .output_weight = model.weights.output_proj,
        .output_norm = model.weights.final_norm,
        .layers = layers,
    };

    try gguf_writer.exportToGguf(allocator, path, export_config, export_weights);
}

const trainable_checkpoint = @import("../trainable_checkpoint.zig");
