const std = @import("std");
const gguf = @import("../../llm/io/gguf.zig");
const tensor_loader = @import("../../llm/io/tensor_loader.zig");
const model_config = @import("../model/config.zig");

pub fn loadFromGguf(model: anytype, path: []const u8) !void {
    var gguf_file = try gguf.GgufFile.open(model.allocator, path);
    defer gguf_file.deinit();

    // Verify config matches
    const gguf_hidden = gguf_file.getEmbeddingLength() orelse return error.MissingMetadata;
    const gguf_layers = gguf_file.getBlockCount() orelse return error.MissingMetadata;
    const gguf_heads = gguf_file.getHeadCount() orelse return error.MissingMetadata;

    if (gguf_hidden != model.config.hidden_dim) return error.ConfigMismatch;
    if (gguf_layers != model.config.num_layers) return error.ConfigMismatch;
    if (gguf_heads != model.config.num_heads) return error.ConfigMismatch;

    // Load token embedding
    try loadTensor(model, &gguf_file, "token_embd.weight", model.weights.token_embedding);

    // Load layer weights
    for (model.weights.layers, 0..) |*layer, i| {
        try loadLayerWeights(model, &gguf_file, layer, @intCast(i));
    }

    // Load final norm
    try loadTensor(model, &gguf_file, "output_norm.weight", model.weights.final_norm);

    // Load output projection if not tied
    if (model.weights.output_proj) |out_proj| {
        try loadTensor(model, &gguf_file, "output.weight", out_proj);
    }

    std.log.info("Loaded weights from GGUF: {s}", .{path});
}

pub fn loadTensor(
    model: anytype,
    gguf_file: *gguf.GgufFile,
    name: []const u8,
    dest: []f32,
) !void {
    _ = model;
    const info = gguf_file.getTensor(name) orelse {
        std.log.warn("Tensor not found: {s}", .{name});
        return error.TensorNotFound;
    };

    const data = gguf_file.getTensorData(name) orelse return error.TensorNotFound;

    // Check destination size
    const elem_count = info.elementCount();
    if (elem_count > dest.len) return error.BufferTooSmall;

    // Dequantize based on tensor type
    switch (info.tensor_type) {
        .f32 => {
            const src = std.mem.bytesAsSlice(f32, data);
            @memcpy(dest[0..src.len], src);
        },
        .f16 => {
            const src = std.mem.bytesAsSlice(f16, data);
            for (dest[0..src.len], src) |*d, s| {
                d.* = @floatCast(s);
            }
        },
        .bf16 => {
            const src = std.mem.bytesAsSlice(u16, data);
            for (dest[0..src.len], src) |*d, s| {
                d.* = tensor_loader.bf16ToF32(s);
            }
        },
        .q4_0 => {
            try tensor_loader.dequantizeQ4_0(data, dest[0..@intCast(elem_count)]);
        },
        .q8_0 => {
            try tensor_loader.dequantizeQ8_0(data, dest[0..@intCast(elem_count)]);
        },
        .mxfp4 => {
            try tensor_loader.dequantizeMXFP4(data, dest[0..@intCast(elem_count)]);
        },
        else => {
            std.log.warn("Unsupported tensor type for training: {t}", .{info.tensor_type});
            return error.UnsupportedTensorType;
        },
    }
}

pub fn loadLayerWeights(model: anytype, gguf_file: *gguf.GgufFile, layer: anytype, layer_idx: u32) !void {
    var name_buf: [128]u8 = undefined;

    // Attention weights
    const attn_q = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_q.weight", .{layer_idx}) catch return error.NameTooLong;
    loadTensor(model, gguf_file, attn_q, layer.w_q) catch |err| {
        if (err != error.TensorNotFound) return err;
    };

    const attn_k = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_k.weight", .{layer_idx}) catch return error.NameTooLong;
    loadTensor(model, gguf_file, attn_k, layer.w_k) catch |err| {
        if (err != error.TensorNotFound) return err;
    };

    const attn_v = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_v.weight", .{layer_idx}) catch return error.NameTooLong;
    loadTensor(model, gguf_file, attn_v, layer.w_v) catch |err| {
        if (err != error.TensorNotFound) return err;
    };

    const attn_output = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_output.weight", .{layer_idx}) catch return error.NameTooLong;
    loadTensor(model, gguf_file, attn_output, layer.w_o) catch |err| {
        if (err != error.TensorNotFound) return err;
    };

    const attn_norm = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_norm.weight", .{layer_idx}) catch return error.NameTooLong;
    loadTensor(model, gguf_file, attn_norm, layer.attn_norm) catch |err| {
        if (err != error.TensorNotFound) return err;
    };

    // FFN weights
    const ffn_gate = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_gate.weight", .{layer_idx}) catch return error.NameTooLong;
    loadTensor(model, gguf_file, ffn_gate, layer.w_gate) catch |err| {
        if (err != error.TensorNotFound) return err;
    };

    const ffn_up = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_up.weight", .{layer_idx}) catch return error.NameTooLong;
    loadTensor(model, gguf_file, ffn_up, layer.w_up) catch |err| {
        if (err != error.TensorNotFound) return err;
    };

    const ffn_down = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_down.weight", .{layer_idx}) catch return error.NameTooLong;
    loadTensor(model, gguf_file, ffn_down, layer.w_down) catch |err| {
        if (err != error.TensorNotFound) return err;
    };

    const ffn_norm = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_norm.weight", .{layer_idx}) catch return error.NameTooLong;
    loadTensor(model, gguf_file, ffn_norm, layer.ffn_norm) catch |err| {
        if (err != error.TensorNotFound) return err;
    };
}

pub fn loadFromGgufInternal(model: anytype, gguf_file: *gguf.GgufFile) !void {
    // Load token embedding
    loadTensor(model, gguf_file, "token_embd.weight", model.weights.token_embedding) catch |err| {
        std.log.warn("Failed to load token embedding: {t}", .{err});
    };

    // Load layer weights
    for (model.weights.layers, 0..) |*layer, i| {
        loadLayerWeights(model, gguf_file, layer, @intCast(i)) catch |err| {
            std.log.warn("Failed to load layer {d} weights: {t}", .{ i, err });
        };
    }

    // Load final norm
    loadTensor(model, gguf_file, "output_norm.weight", model.weights.final_norm) catch |err| {
        std.log.warn("Failed to load final norm: {t}", .{err});
    };

    // Load output projection if not tied
    if (model.weights.output_proj) |out_proj| {
        loadTensor(model, gguf_file, "output.weight", out_proj) catch |err| {
            std.log.warn("Failed to load output projection: {t}", .{err});
        };
    }
}

pub fn readConfigFromGguf(allocator: std.mem.Allocator, path: []const u8) !model_config.TrainableModelConfig {
    var gguf_file = try gguf.GgufFile.open(allocator, path);
    defer gguf_file.deinit();

    return .{
        .hidden_dim = gguf_file.getEmbeddingLength() orelse return error.MissingMetadata,
        .num_layers = gguf_file.getBlockCount() orelse return error.MissingMetadata,
        .num_heads = gguf_file.getHeadCount() orelse return error.MissingMetadata,
        .num_kv_heads = gguf_file.getHeadCountKV() orelse gguf_file.getHeadCount() orelse return error.MissingMetadata,
        .intermediate_dim = getIntermediateDim(&gguf_file) orelse return error.MissingMetadata,
        .vocab_size = gguf_file.getVocabSize() orelse return error.MissingMetadata,
        .max_seq_len = gguf_file.getContextLength() orelse 2048,
        .rope_theta = getRopeTheta(&gguf_file) orelse 10000.0,
        .norm_eps = getNormEps(&gguf_file) orelse 1e-5,
    };
}

pub fn getIntermediateDim(gguf_file: *gguf.GgufFile) ?u32 {
    const arch = gguf_file.getArchitecture() orelse "llama";
    var buf: [64]u8 = undefined;
    const key = std.fmt.bufPrint(&buf, "{s}.feed_forward_length", .{arch}) catch return null;
    const val = gguf_file.getMetadata(key) orelse return null;
    return val.asU32();
}

pub fn getRopeTheta(gguf_file: *gguf.GgufFile) ?f32 {
    const arch = gguf_file.getArchitecture() orelse "llama";
    var buf: [64]u8 = undefined;
    const key = std.fmt.bufPrint(&buf, "{s}.rope.freq_base", .{arch}) catch return null;
    const val = gguf_file.getMetadata(key) orelse return null;
    return val.asF32();
}

pub fn getNormEps(gguf_file: *gguf.GgufFile) ?f32 {
    const arch = gguf_file.getArchitecture() orelse "llama";
    var buf: [64]u8 = undefined;
    const key = std.fmt.bufPrint(&buf, "{s}.attention.layer_norm_rms_epsilon", .{arch}) catch return null;
    const val = gguf_file.getMetadata(key) orelse return null;
    return val.asF32();
}
