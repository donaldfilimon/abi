//! GGUF file format writer for exporting trained models.
//!
//! Creates llama.cpp-compatible GGUF files from trained model weights.
//! Supports:
//! - GGUF v3 format
//! - Model metadata (architecture, hyperparameters)
//! - Tokenizer vocabulary
//! - Quantized tensor export (F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0)

const std = @import("std");
const gguf = @import("gguf.zig");

pub const GgufWriterError = error{
    InvalidAlignment,
    WriteError,
    OutOfMemory,
    InvalidTensorType,
    DuplicateKey,
    DuplicateTensor,
};

pub const TokenizerConfig = struct {
    model: []const u8 = "llama",
    tokens: []const []const u8,
    scores: ?[]const f32 = null,
    token_types: ?[]const i32 = null,
    merges: ?[]const []const u8 = null,
    bos_token_id: u32,
    eos_token_id: u32,
    unknown_token_id: ?u32 = null,
    padding_token_id: ?u32 = null,
    add_bos_token: bool = true,
    add_eos_token: bool = false,
};

/// GGUF file writer for creating model export files.
pub const GgufWriter = struct {
    allocator: std.mem.Allocator,
    io_backend: std.Io.Threaded,
    io: std.Io,
    file: std.Io.File,
    metadata: std.ArrayListUnmanaged(MetadataEntry),
    tensors: std.ArrayListUnmanaged(TensorEntry),
    tensor_data: std.ArrayListUnmanaged(u8),
    alignment: usize,

    const MetadataEntry = struct {
        key: []const u8,
        value_type: gguf.GgufMetadataValueType,
        data: []const u8, // Serialized value
    };

    const TensorEntry = struct {
        name: []const u8,
        n_dims: u32,
        dims: [4]u64,
        tensor_type: gguf.GgufTensorType,
        offset: u64,
        size: u64,
    };

    /// Create a new GGUF writer.
    pub fn init(allocator: std.mem.Allocator, path: []const u8) !GgufWriter {
        var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
        errdefer io_backend.deinit();
        const io = io_backend.io();

        const file = std.Io.Dir.cwd().createFile(io, path, .{}) catch |err| {
            std.log.err("Failed to create GGUF file: {s}: {t}", .{ path, err });
            return error.WriteError;
        };
        errdefer file.close(io);

        return .{
            .allocator = allocator,
            .io_backend = io_backend,
            .io = io,
            .file = file,
            .metadata = std.ArrayListUnmanaged(MetadataEntry).empty,
            .tensors = std.ArrayListUnmanaged(TensorEntry).empty,
            .tensor_data = std.ArrayListUnmanaged(u8).empty,
            .alignment = gguf.GGUF_DEFAULT_ALIGNMENT,
        };
    }

    pub fn deinit(self: *GgufWriter) void {
        for (self.metadata.items) |entry| {
            self.allocator.free(entry.key);
            self.allocator.free(entry.data);
        }
        self.metadata.deinit(self.allocator);

        for (self.tensors.items) |entry| {
            self.allocator.free(entry.name);
        }
        self.tensors.deinit(self.allocator);
        self.tensor_data.deinit(self.allocator);
        self.file.close(self.io);
        self.io_backend.deinit();
        self.* = undefined;
    }

    //==========================================================================
    // Metadata API
    //==========================================================================

    /// Add a string metadata entry.
    pub fn addMetadataString(self: *GgufWriter, key: []const u8, value: []const u8) !void {
        // Serialize: length (u64) + string bytes
        const data = try self.allocator.alloc(u8, 8 + value.len);
        std.mem.writeInt(u64, data[0..8], value.len, .little);
        @memcpy(data[8..], value);

        try self.addMetadataRaw(key, .string, data);
    }

    /// Add a u32 metadata entry.
    pub fn addMetadataU32(self: *GgufWriter, key: []const u8, value: u32) !void {
        const data = try self.allocator.alloc(u8, 4);
        std.mem.writeInt(u32, data[0..4], value, .little);
        try self.addMetadataRaw(key, .uint32, data);
    }

    /// Add a u64 metadata entry.
    pub fn addMetadataU64(self: *GgufWriter, key: []const u8, value: u64) !void {
        const data = try self.allocator.alloc(u8, 8);
        std.mem.writeInt(u64, data[0..8], value, .little);
        try self.addMetadataRaw(key, .uint64, data);
    }

    /// Add an i32 metadata entry.
    pub fn addMetadataI32(self: *GgufWriter, key: []const u8, value: i32) !void {
        const data = try self.allocator.alloc(u8, 4);
        std.mem.writeInt(i32, data[0..4], value, .little);
        try self.addMetadataRaw(key, .int32, data);
    }

    /// Add an f32 metadata entry.
    pub fn addMetadataF32(self: *GgufWriter, key: []const u8, value: f32) !void {
        const data = try self.allocator.alloc(u8, 4);
        @memcpy(data[0..4], std.mem.asBytes(&value));
        try self.addMetadataRaw(key, .float32, data);
    }

    /// Add a bool metadata entry.
    pub fn addMetadataBool(self: *GgufWriter, key: []const u8, value: bool) !void {
        const data = try self.allocator.alloc(u8, 1);
        data[0] = if (value) 1 else 0;
        try self.addMetadataRaw(key, .bool_, data);
    }

    /// Add an array of strings metadata entry.
    pub fn addMetadataStringArray(self: *GgufWriter, key: []const u8, values: []const []const u8) !void {
        // Calculate total size: element_type (4) + count (8) + string data
        var total_size: usize = 12; // type + count
        for (values) |v| {
            total_size += 8 + v.len; // length + bytes for each string
        }

        const data = try self.allocator.alloc(u8, total_size);
        var offset: usize = 0;

        // Element type
        std.mem.writeInt(u32, data[offset..][0..4], @intFromEnum(gguf.GgufMetadataValueType.string), .little);
        offset += 4;

        // Count
        std.mem.writeInt(u64, data[offset..][0..8], values.len, .little);
        offset += 8;

        // String data
        for (values) |v| {
            std.mem.writeInt(u64, data[offset..][0..8], v.len, .little);
            offset += 8;
            @memcpy(data[offset..][0..v.len], v);
            offset += v.len;
        }

        try self.addMetadataRaw(key, .array, data);
    }

    /// Add an array of f32 metadata entry.
    pub fn addMetadataF32Array(self: *GgufWriter, key: []const u8, values: []const f32) !void {
        // Size: element_type (4) + count (8) + f32 data
        const data = try self.allocator.alloc(u8, 12 + values.len * 4);
        var offset: usize = 0;

        // Element type
        std.mem.writeInt(u32, data[offset..][0..4], @intFromEnum(gguf.GgufMetadataValueType.float32), .little);
        offset += 4;

        // Count
        std.mem.writeInt(u64, data[offset..][0..8], values.len, .little);
        offset += 8;

        // Values
        for (values) |v| {
            @memcpy(data[offset..][0..4], std.mem.asBytes(&v));
            offset += 4;
        }

        try self.addMetadataRaw(key, .array, data);
    }

    /// Add an array of i32 metadata entry.
    pub fn addMetadataI32Array(self: *GgufWriter, key: []const u8, values: []const i32) !void {
        const data = try self.allocator.alloc(u8, 12 + values.len * 4);
        var offset: usize = 0;

        std.mem.writeInt(u32, data[offset..][0..4], @intFromEnum(gguf.GgufMetadataValueType.int32), .little);
        offset += 4;

        std.mem.writeInt(u64, data[offset..][0..8], values.len, .little);
        offset += 8;

        for (values) |v| {
            std.mem.writeInt(i32, data[offset..][0..4], v, .little);
            offset += 4;
        }

        try self.addMetadataRaw(key, .array, data);
    }

    fn addMetadataRaw(self: *GgufWriter, key: []const u8, value_type: gguf.GgufMetadataValueType, data: []const u8) !void {
        // Check for duplicates
        for (self.metadata.items) |entry| {
            if (std.mem.eql(u8, entry.key, key)) {
                self.allocator.free(data);
                return error.DuplicateKey;
            }
        }

        const key_copy = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_copy);

        try self.metadata.append(self.allocator, .{
            .key = key_copy,
            .value_type = value_type,
            .data = data,
        });
    }

    //==========================================================================
    // Tensor API
    //==========================================================================

    /// Add a tensor with F32 data.
    pub fn addTensorF32(self: *GgufWriter, name: []const u8, data: []const f32, shape: []const u64) !void {
        try self.addTensorRaw(name, std.mem.sliceAsBytes(data), shape, .f32);
    }

    /// Add a tensor with F16 data.
    pub fn addTensorF16(self: *GgufWriter, name: []const u8, data: []const f16, shape: []const u64) !void {
        try self.addTensorRaw(name, std.mem.sliceAsBytes(data), shape, .f16);
    }

    /// Add a tensor with quantized data.
    pub fn addTensorQuantized(
        self: *GgufWriter,
        name: []const u8,
        data: []const u8,
        shape: []const u64,
        tensor_type: gguf.GgufTensorType,
    ) !void {
        try self.addTensorRaw(name, data, shape, tensor_type);
    }

    fn addTensorRaw(
        self: *GgufWriter,
        name: []const u8,
        data: []const u8,
        shape: []const u64,
        tensor_type: gguf.GgufTensorType,
    ) !void {
        if (shape.len == 0 or shape.len > 4) return error.InvalidTensorType;

        // Check for duplicates
        for (self.tensors.items) |entry| {
            if (std.mem.eql(u8, entry.name, name)) {
                return error.DuplicateTensor;
            }
        }

        // Align tensor data
        const current_offset = self.tensor_data.items.len;
        const aligned_offset = (current_offset + self.alignment - 1) & ~(self.alignment - 1);
        const padding = aligned_offset - current_offset;

        // Add padding
        try self.tensor_data.appendNTimes(self.allocator, 0, padding);

        const tensor_offset = self.tensor_data.items.len;

        // Add tensor data
        try self.tensor_data.appendSlice(self.allocator, data);

        // Create tensor entry
        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);

        var dims: [4]u64 = .{ 1, 1, 1, 1 };
        for (shape, 0..) |dim, i| {
            dims[i] = dim;
        }

        try self.tensors.append(self.allocator, .{
            .name = name_copy,
            .n_dims = @intCast(shape.len),
            .dims = dims,
            .tensor_type = tensor_type,
            .offset = tensor_offset,
            .size = data.len,
        });
    }

    //==========================================================================
    // Convenience Methods
    //==========================================================================

    /// Add standard LLaMA model metadata.
    pub fn addLlamaMetadata(
        self: *GgufWriter,
        config: struct {
            name: []const u8 = "abi-llama",
            vocab_size: u32,
            context_length: u32,
            embedding_length: u32,
            block_count: u32,
            head_count: u32,
            head_count_kv: ?u32 = null,
            ffn_hidden_dim: u32,
            rope_dimension_count: ?u32 = null,
            rope_freq_base: f32 = 10000.0,
            layer_norm_rms_epsilon: f32 = 1e-5,
        },
    ) !void {
        try self.addMetadataString("general.architecture", "llama");
        try self.addMetadataString("general.name", config.name);
        try self.addMetadataU32("general.quantization_version", 2);

        try self.addMetadataU32("llama.vocab_size", config.vocab_size);
        try self.addMetadataU32("llama.context_length", config.context_length);
        try self.addMetadataU32("llama.embedding_length", config.embedding_length);
        try self.addMetadataU32("llama.block_count", config.block_count);
        try self.addMetadataU32("llama.attention.head_count", config.head_count);
        try self.addMetadataU32("llama.attention.head_count_kv", config.head_count_kv orelse config.head_count);
        try self.addMetadataU32("llama.feed_forward_length", config.ffn_hidden_dim);

        const head_dim = config.embedding_length / config.head_count;
        try self.addMetadataU32("llama.rope.dimension_count", config.rope_dimension_count orelse head_dim);
        try self.addMetadataF32("llama.rope.freq_base", config.rope_freq_base);
        try self.addMetadataF32("llama.attention.layer_norm_rms_epsilon", config.layer_norm_rms_epsilon);
    }

    /// Add tokenizer metadata.
    pub fn addTokenizerMetadata(
        self: *GgufWriter,
        config: TokenizerConfig,
    ) !void {
        try self.addMetadataString("tokenizer.ggml.model", config.model);
        try self.addMetadataStringArray("tokenizer.ggml.tokens", config.tokens);

        if (config.scores) |scores| {
            try self.addMetadataF32Array("tokenizer.ggml.scores", scores);
        }
        if (config.token_types) |types| {
            try self.addMetadataI32Array("tokenizer.ggml.token_type", types);
        }
        if (config.merges) |merges| {
            try self.addMetadataStringArray("tokenizer.ggml.merges", merges);
        }

        try self.addMetadataU32("tokenizer.ggml.bos_token_id", config.bos_token_id);
        try self.addMetadataU32("tokenizer.ggml.eos_token_id", config.eos_token_id);

        if (config.unknown_token_id) |unk| {
            try self.addMetadataU32("tokenizer.ggml.unknown_token_id", unk);
        }
        if (config.padding_token_id) |pad| {
            try self.addMetadataU32("tokenizer.ggml.padding_token_id", pad);
        }

        try self.addMetadataBool("tokenizer.ggml.add_bos_token", config.add_bos_token);
        try self.addMetadataBool("tokenizer.ggml.add_eos_token", config.add_eos_token);
    }

    //==========================================================================
    // File Writing
    //==========================================================================

    /// Finalize and write the GGUF file.
    pub fn finalize(self: *GgufWriter) !void {
        // Build entire file in memory for Zig 0.16 compatibility
        var buffer = std.ArrayListUnmanaged(u8).empty;
        defer buffer.deinit(self.allocator);

        // Write header
        try appendIntLe(u32, &buffer, self.allocator, gguf.GGUF_MAGIC);
        try appendIntLe(u32, &buffer, self.allocator, gguf.GGUF_VERSION_3);
        try appendIntLe(u64, &buffer, self.allocator, self.tensors.items.len);
        try appendIntLe(u64, &buffer, self.allocator, self.metadata.items.len);

        // Write metadata
        for (self.metadata.items) |entry| {
            // Key: length + bytes
            try appendIntLe(u64, &buffer, self.allocator, entry.key.len);
            try buffer.appendSlice(self.allocator, entry.key);
            // Type
            try appendIntLe(u32, &buffer, self.allocator, @intFromEnum(entry.value_type));
            // Value data
            try buffer.appendSlice(self.allocator, entry.data);
        }

        // Write tensor info
        for (self.tensors.items) |entry| {
            // Name: length + bytes
            try appendIntLe(u64, &buffer, self.allocator, entry.name.len);
            try buffer.appendSlice(self.allocator, entry.name);
            // Dimensions
            try appendIntLe(u32, &buffer, self.allocator, entry.n_dims);
            for (0..entry.n_dims) |i| {
                try appendIntLe(u64, &buffer, self.allocator, entry.dims[i]);
            }
            // Type
            try appendIntLe(u32, &buffer, self.allocator, @intFromEnum(entry.tensor_type));
            // Offset (within tensor data section)
            try appendIntLe(u64, &buffer, self.allocator, entry.offset);
        }

        // Align to GGUF_DEFAULT_ALIGNMENT before tensor data
        const current_pos = buffer.items.len;
        const aligned_pos = (current_pos + self.alignment - 1) & ~(self.alignment - 1);
        const padding_needed = aligned_pos - current_pos;

        if (padding_needed > 0) {
            try buffer.appendNTimes(self.allocator, 0, padding_needed);
        }

        // Write tensor data
        try buffer.appendSlice(self.allocator, self.tensor_data.items);

        // Write all at once using writeStreamingAll for Zig 0.16 compatibility
        try self.file.writeStreamingAll(self.io, buffer.items);

        std.log.info("GGUF file written: {d} tensors, {d} metadata entries", .{
            self.tensors.items.len,
            self.metadata.items.len,
        });
    }

    fn appendIntLe(comptime T: type, buffer: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, value: T) !void {
        var bytes: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &bytes, value, .little);
        try buffer.appendSlice(allocator, &bytes);
    }
};

/// Export trained model weights to GGUF format.
/// This is a high-level function that handles the entire export process.
pub fn exportToGguf(
    allocator: std.mem.Allocator,
    path: []const u8,
    config: ExportConfig,
    weights: ExportWeights,
) !void {
    var writer = try GgufWriter.init(allocator, path);
    defer writer.deinit();

    // Add model metadata
    try writer.addLlamaMetadata(.{
        .name = config.name,
        .vocab_size = config.vocab_size,
        .context_length = config.context_length,
        .embedding_length = config.embedding_length,
        .block_count = config.block_count,
        .head_count = config.head_count,
        .head_count_kv = config.head_count_kv,
        .ffn_hidden_dim = config.ffn_hidden_dim,
        .rope_freq_base = config.rope_freq_base,
        .layer_norm_rms_epsilon = config.layer_norm_rms_epsilon,
    });

    // Add tokenizer metadata if provided
    if (config.tokenizer) |tok| {
        try writer.addTokenizerMetadata(tok);
    }

    // Add tensors
    // Token embedding
    try writer.addTensorF32("token_embd.weight", weights.token_embedding, &.{ config.vocab_size, config.embedding_length });

    // Output projection
    if (weights.output_weight) |output| {
        try writer.addTensorF32("output.weight", output, &.{ config.vocab_size, config.embedding_length });
    }

    // Final norm
    try writer.addTensorF32("output_norm.weight", weights.output_norm, &.{config.embedding_length});

    // Layer weights
    // SAFETY: name_buf is 128 bytes. Longest format is "blk.{d}.attn_output.weight" = ~30 chars.
    // Even with max u32 layer index (10 digits), output is ~40 chars, well under 128.
    for (weights.layers, 0..) |layer, i| {
        var name_buf: [128]u8 = undefined;

        // Attention norms
        const attn_norm_name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_norm.weight", .{i}) catch unreachable;
        try writer.addTensorF32(attn_norm_name, layer.attn_norm, &.{config.embedding_length});

        const ffn_norm_name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_norm.weight", .{i}) catch unreachable;
        try writer.addTensorF32(ffn_norm_name, layer.ffn_norm, &.{config.embedding_length});

        // Attention weights
        const head_dim = config.embedding_length / config.head_count;
        const kv_dim = head_dim * (config.head_count_kv orelse config.head_count);

        const attn_q_name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_q.weight", .{i}) catch unreachable;
        try writer.addTensorF32(attn_q_name, layer.wq, &.{ config.embedding_length, config.embedding_length });

        const attn_k_name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_k.weight", .{i}) catch unreachable;
        try writer.addTensorF32(attn_k_name, layer.wk, &.{ kv_dim, config.embedding_length });

        const attn_v_name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_v.weight", .{i}) catch unreachable;
        try writer.addTensorF32(attn_v_name, layer.wv, &.{ kv_dim, config.embedding_length });

        const attn_out_name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_output.weight", .{i}) catch unreachable;
        try writer.addTensorF32(attn_out_name, layer.wo, &.{ config.embedding_length, config.embedding_length });

        // FFN weights
        const ffn_gate_name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_gate.weight", .{i}) catch unreachable;
        try writer.addTensorF32(ffn_gate_name, layer.w_gate, &.{ config.ffn_hidden_dim, config.embedding_length });

        const ffn_up_name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_up.weight", .{i}) catch unreachable;
        try writer.addTensorF32(ffn_up_name, layer.w_up, &.{ config.ffn_hidden_dim, config.embedding_length });

        const ffn_down_name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_down.weight", .{i}) catch unreachable;
        try writer.addTensorF32(ffn_down_name, layer.w_down, &.{ config.embedding_length, config.ffn_hidden_dim });
    }

    try writer.finalize();
}

/// Configuration for GGUF export.
pub const ExportConfig = struct {
    name: []const u8 = "abi-llama",
    vocab_size: u32,
    context_length: u32,
    embedding_length: u32,
    block_count: u32,
    head_count: u32,
    head_count_kv: ?u32 = null,
    ffn_hidden_dim: u32,
    rope_freq_base: f32 = 10000.0,
    layer_norm_rms_epsilon: f32 = 1e-5,
    tokenizer: ?TokenizerConfig = null,
};

/// Layer weights for export.
pub const LayerWeights = struct {
    attn_norm: []const f32,
    ffn_norm: []const f32,
    wq: []const f32,
    wk: []const f32,
    wv: []const f32,
    wo: []const f32,
    w_gate: []const f32,
    w_up: []const f32,
    w_down: []const f32,
};

/// Collected model weights for export.
pub const ExportWeights = struct {
    token_embedding: []const f32,
    output_weight: ?[]const f32 = null,
    output_norm: []const f32,
    layers: []const LayerWeights,
};

test "gguf writer basic" {
    const allocator = std.testing.allocator;

    // Create a temp file
    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const path = tmp_dir.dir.realPathFileAlloc(std.testing.io, ".", allocator) catch return;
    defer allocator.free(path);

    const full_path = std.fmt.allocPrint(allocator, "{s}/test.gguf", .{path}) catch return;
    defer allocator.free(full_path);

    var writer = GgufWriter.init(allocator, full_path) catch return;
    defer writer.deinit();

    // Add some metadata
    try writer.addMetadataString("general.architecture", "llama");
    try writer.addMetadataU32("llama.vocab_size", 32000);
    try writer.addMetadataF32("llama.rope.freq_base", 10000.0);
    try writer.addMetadataBool("tokenizer.ggml.add_bos_token", true);

    // Add a small tensor
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    try writer.addTensorF32("test_tensor", &data, &.{ 2, 2 });

    // Finalize
    try writer.finalize();

    // Verify file was created
    const stat = tmp_dir.dir.statFile("test.gguf") catch return;
    try std.testing.expect(stat.size > 0);
}

test "gguf writer duplicate key" {
    const allocator = std.testing.allocator;

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const path = tmp_dir.dir.realPathFileAlloc(std.testing.io, ".", allocator) catch return;
    defer allocator.free(path);

    const full_path = std.fmt.allocPrint(allocator, "{s}/dup.gguf", .{path}) catch return;
    defer allocator.free(full_path);

    var writer = GgufWriter.init(allocator, full_path) catch return;
    defer writer.deinit();

    try writer.addMetadataString("test.key", "value1");

    // Second add should fail
    const result = writer.addMetadataString("test.key", "value2");
    try std.testing.expectError(error.DuplicateKey, result);
}
