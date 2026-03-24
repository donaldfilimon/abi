//! LLM Stub Module — disabled at compile time.

const std = @import("std");
const config_module = @import("../../../core/config/mod.zig");
const types = @import("types.zig");

// ── Re-exported types ──────────────────────────────────────────────────────

pub const LlmError = types.LlmError;
pub const Error = types.Error;
pub const DType = types.DType;
pub const EngineBackend = types.EngineBackend;
pub const TensorInfo = types.TensorInfo;
pub const GgufHeader = types.GgufHeader;
pub const GgufMetadata = types.GgufMetadata;
pub const GgufFile = types.GgufFile;
pub const MappedFile = types.MappedFile;
pub const Tensor = types.Tensor;
pub const Q4_0Block = types.Q4_0Block;
pub const Q8_0Block = types.Q8_0Block;
pub const BpeTokenizer = types.BpeTokenizer;
pub const Vocab = types.Vocab;
pub const Model = types.Model;
pub const ModelConfig = types.ModelConfig;
pub const Generator = types.Generator;
pub const Sampler = types.Sampler;
pub const SamplerConfig = types.SamplerConfig;
pub const KvCache = types.KvCache;
pub const ParallelExecutor = types.ParallelExecutor;
pub const TokenEvent = types.TokenEvent;
pub const StreamingStats = types.StreamingStats;
pub const StreamingCallbacks = types.StreamingCallbacks;
pub const StreamingConfig = types.StreamingConfig;
pub const StreamingGenerator = types.StreamingGenerator;
pub const StreamingResponse = types.StreamingResponse;
pub const StreamingError = types.StreamingError;
pub const StreamingState = types.StreamingState;
pub const SSEFormatter = types.SSEFormatter;
pub const InferenceConfig = types.InferenceConfig;
pub const InferenceStats = types.InferenceStats;
pub const Engine = types.Engine;
pub const LlmError = types.LlmError;
pub const FusionConfig = types.FusionConfig;
pub const WdbxFusion = types.WdbxFusion;
pub const Context = types.Context;
pub const providers = types.providers;

pub const unified_orchestrator = @import("unified_orchestrator/stub.zig");

// ── Submodule Namespace Stubs ──────────────────────────────────────────────

pub const io = struct {
    pub const MappedFile = types.MappedFile;
    pub const GgufFile = types.GgufFile;
    pub const GgufHeader = types.GgufHeader;
    pub const GgufMetadata = types.GgufMetadata;
    pub const TensorInfo = types.TensorInfo;
    pub const gguf = struct {
        pub const GgufFile = types.GgufFile;
    };
    pub const gguf_writer = struct {
        pub const TokenizerConfig = @import("io/gguf_writer.zig").TokenizerConfig;
    };
};

pub const tensor = struct {
    pub const Tensor = types.Tensor;
    pub const DType = types.DType;
    pub const Q4_0Block = types.Q4_0Block;
    pub const Q4_1Block = types.Q4_1Block;
    pub const Q8_0Block = types.Q8_0Block;
    pub const quantized = struct {};
};

pub const tokenizer = struct {
    pub const BpeTokenizer = types.BpeTokenizer;
    pub const Tokenizer = types.Tokenizer;
    pub const Vocab = types.Vocab;
    pub const TokenizerError = error{
        InvalidUtf8,
        VocabNotLoaded,
        UnknownToken,
        EncodingError,
        DecodingError,
        OutOfMemory,
        FeatureDisabled,
    };
    pub fn loadFromGguf(_: std.mem.Allocator, _: *const types.GgufFile) LlmError!types.Tokenizer {
        return error.FeatureDisabled;
    }
};

pub const model = struct {
    pub const LlamaModel = types.Model;
    pub const LlamaConfig = types.ModelConfig;
};

pub const generation = struct {
    pub const Generator = types.Generator;
    pub const GeneratorConfig = types.GeneratorConfig;
    pub const sampler = struct {
        pub const Sampler = types.Sampler;
        pub const TopKTopP = struct {};
    };
    pub const Sampler = types.Sampler;
    pub const SamplerConfig = types.SamplerConfig;
    pub const StreamingGenerator = types.StreamingGenerator;
    pub const StreamingResponse = types.StreamingResponse;
    pub const StreamingConfig = types.StreamingConfig;
    pub const StreamingState = types.StreamingState;
    pub const StreamingStats = types.StreamingStats;
    pub const StreamingCallbacks = types.StreamingCallbacks;
    pub const StreamingError = types.StreamingError;
    pub const TokenEvent = types.TokenEvent;
    pub const SSEFormatter = types.SSEFormatter;
    pub const collectStreamingResponse = @import("stub.zig").collectStreamingResponse;
};

pub const cache = struct {
    pub const KvCache = types.KvCache;
};

pub const ops = struct {
    pub const attention = struct {
        pub fn selfAttention(
            _: std.mem.Allocator,
            _: []const f32,
            _: []const f32,
            _: []const f32,
            output: []f32,
            _: u32,
            _: u32,
            _: u32,
            _: bool,
        ) !void {
            @memset(output, 0);
        }
        pub fn scaledDotProductAttention(
            _: std.mem.Allocator,
            _: []const f32,
            _: []const f32,
            _: []const f32,
            output: []f32,
            _: u32,
            _: u32,
            _: u32,
            _: bool,
        ) !void {
            @memset(output, 0);
        }
    };
    pub const activations = struct {
        pub fn softmax(input: []const f32, output: []f32) void {
            @memcpy(output, input);
            activations.softmaxInPlace(output);
        }
        pub fn softmaxInPlace(x: []f32) void {
            if (x.len == 0) return;
            const value = 1.0 / @as(f32, @floatFromInt(x.len));
            @memset(x, value);
        }
        pub fn silu(x: f32) f32 {
            return x / (1.0 + @exp(-x));
        }
        pub fn gelu(x: f32) f32 {
            return 0.5 * x * (1.0 + std.math.tanh(0.7978846 * (x + 0.044715 * x * x * x)));
        }
    };
    pub const rmsnorm = struct {
        pub fn rmsNorm(x: []const f32, weight: []const f32, output: []f32, eps: f32) void {
            _ = eps;
            if (x.len == 0) return;
            for (x, weight, 0..) |value, scale, idx| {
                output[idx] = value * scale;
            }
        }
        pub fn rmsNormInPlace(x: []f32, weight: []const f32, eps: f32) void {
            rmsnorm.rmsNorm(x, weight, x, eps);
        }
    };
    pub const matmul = struct {
        pub fn matrixMultiply(_: anytype, _: anytype, c: []f32, _: usize, _: usize, _: usize) void {
            @memset(c, 0);
        }
    };
    pub const matrixMultiply = matmul.matrixMultiply;
    pub const selfAttention = attention.selfAttention;
    pub const scaledDotProductAttention = attention.scaledDotProductAttention;
    pub const rmsNorm = rmsnorm.rmsNorm;
    pub const rmsNormInPlace = rmsnorm.rmsNormInPlace;
    pub const softmax = activations.softmax;
    pub const softmaxInPlace = activations.softmaxInPlace;
    pub const silu = activations.silu;
    pub const gelu = activations.gelu;
};

pub const parallel = struct {
    pub const ParallelStrategy = types.ParallelStrategy;
    pub const ParallelMode = types.ParallelMode;
    pub const ParallelConfig = types.ParallelConfig;
    pub const ParallelCoordinator = types.ParallelCoordinator;
    pub const TensorParallelConfig = types.TensorParallelConfig;
    pub const PipelineParallelConfig = types.PipelineParallelConfig;
    pub const ParallelExecutor = types.ParallelExecutor;
};

pub const wdbx_fusion = struct {
    pub const WdbxFusion = types.WdbxFusion;
    pub const FusionConfig = types.FusionConfig;
    pub const ContextChunk = types.ContextChunk;
    pub const CacheEntry = types.CacheEntry;
};

// ── Free functions ─────────────────────────────────────────────────────────

pub fn collectStreamingResponse(_: std.mem.Allocator, _: *types.StreamingResponse) LlmError!struct { text: ?[]u8, stats: types.StreamingStats } {
    return error.FeatureDisabled;
}

pub fn isEnabled() bool {
    return false;
}

pub fn infer(_: std.mem.Allocator, _: []const u8, _: []const u8) LlmError![]u8 {
    return error.FeatureDisabled;
}

test {
    std.testing.refAllDecls(@This());
}
