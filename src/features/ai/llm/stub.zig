//! Stub for LLM feature when disabled
const std = @import("std");

pub const LlmError = error{
    LlmDisabled,
    InvalidModelFormat,
    OutOfMemory,
};

pub const DType = enum {
    f32,
    f16,
    q4_0,
    q8_0,
};

pub const TensorInfo = struct {
    name: []const u8 = "",
    dtype: DType = .f32,
    shape: []const usize = &.{},
    offset: usize = 0,
};

pub const GgufHeader = struct {
    magic: u32 = 0,
    version: u32 = 0,
    tensor_count: u64 = 0,
    metadata_kv_count: u64 = 0,
};

pub const GgufMetadata = struct {};

pub const GgufTensorInfo = struct {
    name: []const u8 = "",
    tensor_type: DType = .f32,
    n_dims: usize = 0,
    dims: [4]u64 = .{ 0, 0, 0, 0 },
};

pub const TensorMap = struct {
    pub const Iterator = struct {
        done: bool = false,

        pub fn next(self: *Iterator) ?struct { key_ptr: *[]const u8, value_ptr: *GgufTensorInfo } {
            if (self.done) return null;
            self.done = true;
            return null;
        }
    };

    pub fn iterator(_: *const TensorMap) Iterator {
        return .{};
    }
};

pub const GgufFile = struct {
    tensors: TensorMap = .{},

    pub fn open(_: std.mem.Allocator, _: []const u8) LlmError!@This() {
        return LlmError.LlmDisabled;
    }

    pub fn deinit(_: *@This()) void {}

    pub fn getMetadata(_: *const @This(), _: []const u8) ?[]const u8 {
        return null;
    }

    pub fn getMetadataInt(_: *const @This(), _: []const u8) ?i64 {
        return null;
    }

    pub fn getTensorInfo(_: *const @This(), _: []const u8) ?TensorInfo {
        return null;
    }

    pub fn printSummaryDebug(_: *const @This()) void {}
};

pub const MappedFile = struct {
    pub fn open(_: []const u8) LlmError!@This() {
        return LlmError.LlmDisabled;
    }

    pub fn deinit(_: *@This()) void {}
};

pub const Tensor = struct {
    shape: []const usize = &.{},
    dtype: DType = .f32,
    data: []const u8 = &.{},

    pub fn deinit(_: *@This()) void {}
};

pub const Q4_0Block = struct {};
pub const Q8_0Block = struct {};

pub const BpeTokenizer = struct {
    pub fn init(_: std.mem.Allocator) @This() {
        return .{};
    }

    pub fn deinit(_: *@This()) void {}

    pub fn encode(_: *@This(), _: std.mem.Allocator, _: []const u8) LlmError![]u32 {
        return LlmError.LlmDisabled;
    }

    pub fn decode(_: *@This(), _: std.mem.Allocator, _: []const u32) LlmError![]u8 {
        return LlmError.LlmDisabled;
    }
};

pub const Vocab = struct {};

pub const Model = struct {
    pub fn load(_: std.mem.Allocator, _: []const u8) LlmError!@This() {
        return LlmError.LlmDisabled;
    }

    pub fn deinit(_: *@This()) void {}
};

pub const ModelConfig = struct {
    dim: u32 = 0,
    hidden_dim: u32 = 0,
    n_layers: u32 = 0,
    n_heads: u32 = 0,
    n_kv_heads: u32 = 0,
    vocab_size: u32 = 0,
    seq_len: u32 = 0,

    pub fn fromGguf(_: *const GgufFile) ModelConfig {
        return .{};
    }

    pub fn estimateMemory(_: ModelConfig) u64 {
        return 0;
    }

    pub fn estimateParameters(_: ModelConfig) u64 {
        return 0;
    }
};

pub const Generator = struct {};
pub const Sampler = struct {};
pub const SamplerConfig = struct {};
pub const KvCache = struct {};

// Submodule namespaces (for code that accesses llm.io.GgufFile, llm.model.LlamaModel, etc.)
pub const io = struct {
    pub const GgufFile = @import("stub.zig").GgufFile;
    pub const MappedFile = @import("stub.zig").MappedFile;
    pub const TensorInfo = @import("stub.zig").TensorInfo;
    pub const GgufHeader = @import("stub.zig").GgufHeader;
    pub const GgufMetadata = @import("stub.zig").GgufMetadata;
};

pub const tensor = struct {
    pub const Tensor = @import("stub.zig").Tensor;
    pub const DType = @import("stub.zig").DType;
    pub const Q4_0Block = @import("stub.zig").Q4_0Block;
    pub const Q8_0Block = @import("stub.zig").Q8_0Block;
};

pub const tokenizer = struct {
    pub const BpeTokenizer = @import("stub.zig").BpeTokenizer;
    pub const Vocab = @import("stub.zig").Vocab;
};

pub const model = struct {
    pub const LlamaModel = @import("stub.zig").Model;
    pub const LlamaConfig = @import("stub.zig").ModelConfig;
};

pub const generation = struct {
    pub const Generator = @import("stub.zig").Generator;
    pub const Sampler = @import("stub.zig").Sampler;
    pub const SamplerConfig = @import("stub.zig").SamplerConfig;
};

pub const cache = struct {
    pub const KvCache = @import("stub.zig").KvCache;
};

pub const ops = struct {};

pub const InferenceConfig = struct {
    max_context_length: u32 = 2048,
    max_new_tokens: u32 = 256,
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    top_k: u32 = 40,
    repetition_penalty: f32 = 1.1,
    use_gpu: bool = true,
    num_threads: u32 = 0,
    streaming: bool = true,
    batch_size: u32 = 512,
};

pub const InferenceStats = struct {
    load_time_ns: u64 = 0,
    prefill_time_ns: u64 = 0,
    decode_time_ns: u64 = 0,
    prompt_tokens: u32 = 0,
    generated_tokens: u32 = 0,
    peak_memory_bytes: u64 = 0,
    used_gpu: bool = false,

    pub fn prefillTokensPerSecond(_: InferenceStats) f64 {
        return 0;
    }

    pub fn decodeTokensPerSecond(_: InferenceStats) f64 {
        return 0;
    }
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    config: InferenceConfig,
    stats: InferenceStats,

    pub fn init(allocator: std.mem.Allocator, config: InferenceConfig) Engine {
        return .{
            .allocator = allocator,
            .config = config,
            .stats = .{},
        };
    }

    pub fn deinit(_: *Engine) void {}

    pub fn loadModel(_: *Engine, _: []const u8) LlmError!void {
        return LlmError.LlmDisabled;
    }

    pub fn generate(_: *Engine, _: []const u8) LlmError![]u8 {
        return LlmError.LlmDisabled;
    }

    pub fn getStats(self: *Engine) InferenceStats {
        return self.stats;
    }
};

pub fn isEnabled() bool {
    return false;
}

pub fn infer(_: std.mem.Allocator, _: []const u8, _: []const u8) LlmError![]u8 {
    return LlmError.LlmDisabled;
}
