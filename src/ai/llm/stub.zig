//! LLM Stub Module
//!
//! Stub implementation when LLM is disabled at compile time.
//! This mirrors the API of src/features/ai/llm/stub.zig for consistent
//! feature gating across the codebase.

const std = @import("std");
const config_module = @import("../../config.zig");
const stub_root = @This();

pub const Error = error{
    LlmDisabled,
    ModelNotFound,
    ModelLoadFailed,
    InferenceFailed,
    TokenizationFailed,
    InvalidConfig,
    InvalidModelFormat,
    OutOfMemory,
};

pub const LlmError = Error;

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

const GgufTensorInfo = struct {
    name: []const u8 = "",
    tensor_type: DType = .f32,
    n_dims: usize = 0,
    dims: [4]u64 = .{ 0, 0, 0, 0 },
};

const TensorMap = struct {
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

    pub fn open(_: std.mem.Allocator, _: []const u8) Error!@This() {
        return error.LlmDisabled;
    }

    pub fn deinit(_: *@This()) void {}

    pub fn close(_: *@This()) void {}

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
    pub fn open(_: []const u8) Error!@This() {
        return error.LlmDisabled;
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

    pub fn encode(_: *@This(), _: std.mem.Allocator, _: []const u8) Error![]u32 {
        return error.LlmDisabled;
    }

    pub fn decode(_: *@This(), _: std.mem.Allocator, _: []const u32) Error![]u8 {
        return error.LlmDisabled;
    }
};

pub const Vocab = struct {};

pub const Model = struct {
    pub fn load(_: std.mem.Allocator, _: []const u8) Error!@This() {
        return error.LlmDisabled;
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

    pub fn fromGguf(_: anytype) ModelConfig {
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

pub const ParallelStrategy = enum {
    tensor,
    pipeline,
    hybrid,
};

pub const ParallelMode = enum {
    none,
    tensor,
    pipeline,
    hybrid,
};

pub const TensorParallelConfig = struct {
    shard_count: u32 = 1,
    shard_axis: u32 = 0,
};

pub const PipelineParallelConfig = struct {
    stage_count: u32 = 1,
    micro_batch_count: u32 = 1,
};

pub const ParallelConfig = struct {
    mode: ParallelMode = .none,
    strategy: ParallelStrategy = .tensor,
    thread_count: ?usize = null,
    tensor_config: TensorParallelConfig = .{},
    pipeline_config: PipelineParallelConfig = .{},
};

pub const ParallelCoordinator = struct {
    allocator: std.mem.Allocator,
    config: ParallelConfig,

    pub fn init(allocator: std.mem.Allocator, config: ParallelConfig) ParallelCoordinator {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }

    pub fn deinit(_: *ParallelCoordinator) void {}
};

// Submodule namespaces (for code that accesses llm.io.GgufFile, llm.model.LlamaModel, etc.)
pub const io = struct {
    pub const GgufFile = stub_root.GgufFile;
    pub const MappedFile = stub_root.MappedFile;
    pub const TensorInfo = stub_root.TensorInfo;
    pub const GgufHeader = stub_root.GgufHeader;
    pub const GgufMetadata = stub_root.GgufMetadata;
};

pub const tensor = struct {
    pub const Tensor = stub_root.Tensor;
    pub const DType = stub_root.DType;
    pub const Q4_0Block = stub_root.Q4_0Block;
    pub const Q8_0Block = stub_root.Q8_0Block;
};

pub const tokenizer = struct {
    pub const BpeTokenizer = stub_root.BpeTokenizer;
    pub const Vocab = stub_root.Vocab;
};

pub const model = struct {
    pub const LlamaModel = stub_root.Model;
    pub const LlamaConfig = stub_root.ModelConfig;
};

pub const generation = struct {
    pub const Generator = stub_root.Generator;
    pub const Sampler = stub_root.Sampler;
    pub const SamplerConfig = stub_root.SamplerConfig;
};

pub const cache = struct {
    pub const KvCache = stub_root.KvCache;
};

pub const ops = struct {
    pub fn matrixMultiply(
        _: []const f32,
        _: []const f32,
        _: []f32,
        _: usize,
        _: usize,
        _: usize,
    ) void {}
};

pub const parallel = struct {
    pub const ParallelStrategy = stub_root.ParallelStrategy;
    pub const ParallelConfig = stub_root.ParallelConfig;
    pub const ParallelMode = stub_root.ParallelMode;
    pub const ParallelCoordinator = stub_root.ParallelCoordinator;
    pub const TensorParallelConfig = stub_root.TensorParallelConfig;
    pub const PipelineParallelConfig = stub_root.PipelineParallelConfig;
};

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

    pub fn loadModel(_: *Engine, _: []const u8) Error!void {
        return error.LlmDisabled;
    }

    pub fn generate(_: *Engine, _: []const u8) Error![]u8 {
        return error.LlmDisabled;
    }

    pub fn getStats(self: *Engine) InferenceStats {
        return self.stats;
    }
};

pub const Context = struct {
    pub fn init(_: std.mem.Allocator, _: config_module.LlmConfig) error{LlmDisabled}!*Context {
        return error.LlmDisabled;
    }

    pub fn deinit(_: *Context) void {}

    pub fn getEngine(_: *Context) Error!*Engine {
        return error.LlmDisabled;
    }

    pub fn generate(_: *Context, _: []const u8) Error![]u8 {
        return error.LlmDisabled;
    }

    pub fn tokenize(_: *Context, _: []const u8) Error![]u32 {
        return error.LlmDisabled;
    }

    pub fn detokenize(_: *Context, _: []const u32) Error![]u8 {
        return error.LlmDisabled;
    }
};

pub fn isEnabled() bool {
    return false;
}

pub fn infer(_: std.mem.Allocator, _: []const u8, _: []const u8) Error![]u8 {
    return error.LlmDisabled;
}
