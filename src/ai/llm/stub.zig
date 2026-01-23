//! LLM Stub Module
//!
//! Stub implementation when LLM is disabled at compile time.

const std = @import("std");
const config_module = @import("../../config.zig");

pub const LlmError = error{
    LlmDisabled,
    InvalidModelFormat,
    UnsupportedQuantization,
    ModelTooLarge,
    ContextLengthExceeded,
    TokenizationFailed,
    InferenceError,
    OutOfMemory,
    GpuUnavailable,
    InvalidGgufMagic,
    UnsupportedGgufVersion,
    MissingRequiredMetadata,
    TensorNotFound,
    ShapeMismatch,
};

pub const Error = LlmError;

// Stub types
pub const DType = enum { f32, f16, q4_0, q8_0 };
pub const TensorInfo = struct {};
pub const GgufHeader = struct {};
pub const GgufMetadata = struct {};
pub const GgufFile = struct {};
pub const MappedFile = struct {};
pub const Tensor = struct {};
pub const Q4_0Block = struct {};
pub const Q8_0Block = struct {};
pub const BpeTokenizer = struct {};
pub const Vocab = struct {};
pub const Model = struct {};
pub const ModelConfig = struct {};
pub const Generator = struct {};
pub const Sampler = struct {};
pub const SamplerConfig = struct {};
pub const KvCache = struct {};
pub const ParallelStrategy = enum { none, tensor, pipeline, hybrid };
pub const ParallelMode = enum { single, distributed };
pub const TensorParallelConfig = struct {};
pub const PipelineParallelConfig = struct {};
pub const ParallelConfig = struct {};
pub const ParallelCoordinator = struct {};

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
    context_size: u32 = 2048,
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
    pub fn init(_: std.mem.Allocator, _: InferenceConfig) Engine {
        return .{};
    }

    pub fn deinit(_: *Engine) void {}

    pub fn loadModel(_: *Engine, _: []const u8) LlmError!void {
        return error.LlmDisabled;
    }

    pub fn generate(_: *Engine, _: std.mem.Allocator, _: []const u8) LlmError![]u8 {
        return error.LlmDisabled;
    }

    pub fn generateStreaming(_: *Engine, _: []const u8, _: *const fn ([]const u8) void) LlmError!void {
        return error.LlmDisabled;
    }

    pub fn tokenize(_: *Engine, _: std.mem.Allocator, _: []const u8) LlmError![]u32 {
        return error.LlmDisabled;
    }

    pub fn detokenize(_: *Engine, _: std.mem.Allocator, _: []const u32) LlmError![]u8 {
        return error.LlmDisabled;
    }

    pub fn getStats(_: *Engine) InferenceStats {
        return .{};
    }
};

// Submodule namespace stubs
pub const io = struct {
    pub const MappedFile = stub_root.MappedFile;
    pub const GgufFile = stub_root.GgufFile;
    pub const GgufHeader = stub_root.GgufHeader;
    pub const GgufMetadata = stub_root.GgufMetadata;
    pub const TensorInfo = stub_root.TensorInfo;
};
const stub_root = @This();

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

pub const ops = struct {};

pub const parallel = struct {
    pub const ParallelStrategy = stub_root.ParallelStrategy;
    pub const ParallelMode = stub_root.ParallelMode;
    pub const ParallelConfig = stub_root.ParallelConfig;
    pub const ParallelCoordinator = stub_root.ParallelCoordinator;
    pub const TensorParallelConfig = stub_root.TensorParallelConfig;
    pub const PipelineParallelConfig = stub_root.PipelineParallelConfig;
};

/// Public API Context wrapper
pub const Context = struct {
    pub fn init(_: std.mem.Allocator, _: config_module.LlmConfig) error{LlmDisabled}!*Context {
        return error.LlmDisabled;
    }

    pub fn deinit(_: *Context) void {}

    pub fn getEngine(_: *Context) LlmError!*Engine {
        return error.LlmDisabled;
    }

    pub fn generate(_: *Context, _: []const u8) LlmError![]u8 {
        return error.LlmDisabled;
    }

    pub fn tokenize(_: *Context, _: []const u8) LlmError![]u32 {
        return error.LlmDisabled;
    }

    pub fn detokenize(_: *Context, _: []const u32) LlmError![]u8 {
        return error.LlmDisabled;
    }
};

pub fn isEnabled() bool {
    return false;
}

pub fn infer(_: std.mem.Allocator, _: []const u8, _: []const u8) LlmError![]u8 {
    return error.LlmDisabled;
}
