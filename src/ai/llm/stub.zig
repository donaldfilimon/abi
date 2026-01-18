//! LLM Stub Module
//!
//! Stub implementation when LLM is disabled at compile time.
//! Re-exports from implementation stub and adds Context wrapper.

const std = @import("std");
const config_module = @import("../../config.zig");

// Re-export all types from implementation stub
const impl_stub = @import("../implementation/llm/stub.zig");

pub const LlmError = impl_stub.LlmError;
pub const Error = LlmError;

pub const DType = impl_stub.DType;
pub const TensorInfo = impl_stub.TensorInfo;
pub const GgufHeader = impl_stub.GgufHeader;
pub const GgufMetadata = impl_stub.GgufMetadata;
pub const GgufFile = impl_stub.GgufFile;
pub const MappedFile = impl_stub.MappedFile;
pub const Tensor = impl_stub.Tensor;
pub const Q4_0Block = impl_stub.Q4_0Block;
pub const Q8_0Block = impl_stub.Q8_0Block;
pub const BpeTokenizer = impl_stub.BpeTokenizer;
pub const Vocab = impl_stub.Vocab;
pub const Model = impl_stub.Model;
pub const ModelConfig = impl_stub.ModelConfig;
pub const Generator = impl_stub.Generator;
pub const Sampler = impl_stub.Sampler;
pub const SamplerConfig = impl_stub.SamplerConfig;
pub const KvCache = impl_stub.KvCache;
pub const ParallelStrategy = impl_stub.ParallelStrategy;
pub const ParallelMode = impl_stub.ParallelMode;
pub const TensorParallelConfig = impl_stub.TensorParallelConfig;
pub const PipelineParallelConfig = impl_stub.PipelineParallelConfig;
pub const ParallelConfig = impl_stub.ParallelConfig;
pub const ParallelCoordinator = impl_stub.ParallelCoordinator;
pub const InferenceConfig = impl_stub.InferenceConfig;
pub const InferenceStats = impl_stub.InferenceStats;
pub const Engine = impl_stub.Engine;

// Submodule namespaces
pub const io = impl_stub.io;
pub const tensor = impl_stub.tensor;
pub const tokenizer = impl_stub.tokenizer;
pub const model = impl_stub.model;
pub const generation = impl_stub.generation;
pub const cache = impl_stub.cache;
pub const ops = impl_stub.ops;
pub const parallel = impl_stub.parallel;

/// Public API Context wrapper (specific to this stub)
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
