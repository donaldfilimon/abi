//! LLM Stub Module — disabled at compile time.

const std = @import("std");
const config_module = @import("../../core/config/mod.zig");
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
pub const FusionConfig = types.FusionConfig;
pub const WdbxFusion = types.WdbxFusion;
pub const Context = types.Context;
pub const providers = types.providers;

pub const unified_orchestrator = @import("unified_orchestrator/stub.zig");

// ── Submodule Namespace Stubs ──────────────────────────────────────────────

pub const io = struct {
    pub const MappedFile = types.MappedFile;
    pub const MmapError = types.MmapError;
    pub const GgufFile = types.GgufFile;
    pub const GgufHeader = types.GgufHeader;
    pub const GgufMetadata = types.GgufMetadata;
    pub const GgufMetadataValue = types.GgufMetadataValue;
    pub const GgufMetadataValueType = types.GgufMetadataValueType;
    pub const GgufTensorType = types.GgufTensorType;
    pub const TensorInfo = types.TensorInfo;
    pub const GgufError = types.GgufError;
    pub const TensorLoader = types.TensorLoader;
    pub const GgufWriter = struct {};
    pub const GgufWriterError = types.GgufWriterError;
    pub const ExportConfig = types.ExportConfig;
    pub const ExportWeights = types.ExportWeights;
    pub const LayerWeights = types.LayerWeights;
    pub fn exportToGguf(_: std.mem.Allocator, _: []const u8, _: ExportConfig, _: ExportWeights) !void {
        return error.FeatureDisabled;
    }
    pub const mmap = struct {
        pub const MappedFile = types.MappedFile;
    };
    pub const gguf = struct {
        pub const GgufFile = types.GgufFile;
        pub const GgufHeader = types.GgufHeader;
        pub const GgufMetadata = types.GgufMetadata;
        pub const GgufMetadataValue = types.GgufMetadataValue;
        pub const GgufMetadataValueType = types.GgufMetadataValueType;
        pub const GgufTensorType = types.GgufTensorType;
        pub const TensorInfo = types.TensorInfo;
        pub const GgufError = types.GgufError;
    };
    pub const gguf_writer = struct {
        pub const GgufWriter = struct {};
        pub const GgufWriterError = types.GgufWriterError;
        pub const ExportConfig = types.ExportConfig;
        pub const ExportWeights = types.ExportWeights;
        pub const LayerWeights = types.LayerWeights;
        pub const exportToGguf = @import("stub.zig").io.exportToGguf;
        pub const TokenizerConfig = @import("io/gguf_writer.zig").TokenizerConfig;
    };
    pub const tensor_loader = struct {
        pub const TensorLoader = types.TensorLoader;
    };
};

pub const tensor = struct {
    pub const Tensor = types.Tensor;
    pub const DType = types.DType;
    pub const Shape = types.Shape;
    pub const TensorError = types.TensorError;
    pub const Q4_0Block = types.Q4_0Block;
    pub const Q4_1Block = types.Q4_1Block;
    pub const Q8_0Block = types.Q8_0Block;
    pub const QuantType = types.QuantType;
    pub fn dequantizeQ4_0(_: []const u8, _: []f32) void {}
    pub fn dequantizeQ4_1(_: []const u8, _: []f32) void {}
    pub fn dequantizeQ8_0(_: []const u8, _: []f32) void {}
    pub fn quantizeToQ4_1(_: []const f32, _: []u8) void {}
    pub fn quantizeToQ8_0(_: []const f32, _: []u8) void {}
    pub fn quantizedSize(_: QuantType, _: usize) usize {
        return 0;
    }
    pub fn dequantizedSize(_: QuantType, _: usize) usize {
        return 0;
    }
    pub fn dotQ4_0F32(_: []const u8, _: []const f32, _: usize) f32 {
        return 0;
    }
    pub fn dotQ4_1F32(_: []const u8, _: []const f32, _: usize) f32 {
        return 0;
    }
    pub fn dotQ8_0F32(_: []const u8, _: []const f32, _: usize) f32 {
        return 0;
    }
    pub const TensorView = types.TensorView;
    pub fn zeros(_: std.mem.Allocator, _: types.Shape, _: types.DType) !types.Tensor {
        return error.FeatureDisabled;
    }
    pub fn ones(_: std.mem.Allocator, _: types.Shape, _: types.DType) !types.Tensor {
        return error.FeatureDisabled;
    }
    pub fn fromSlice(_: std.mem.Allocator, _: []const f32, _: types.Shape) !types.Tensor {
        return error.FeatureDisabled;
    }
    pub fn vector(_: std.mem.Allocator, _: []const f32) !types.Tensor {
        return error.FeatureDisabled;
    }
    pub fn matrix(_: std.mem.Allocator, _: []const f32, _: u32, _: u32) !types.Tensor {
        return error.FeatureDisabled;
    }
    pub const quantized = struct {
        pub const Q4_0Block = types.Q4_0Block;
        pub const Q4_1Block = types.Q4_1Block;
        pub const Q8_0Block = types.Q8_0Block;
        pub const QuantType = types.QuantType;
        pub const dequantizeQ4_0 = @import("stub.zig").tensor.dequantizeQ4_0;
        pub const dequantizeQ4_1 = @import("stub.zig").tensor.dequantizeQ4_1;
        pub const dequantizeQ8_0 = @import("stub.zig").tensor.dequantizeQ8_0;
        pub const quantizeToQ4_1 = @import("stub.zig").tensor.quantizeToQ4_1;
        pub const quantizeToQ8_0 = @import("stub.zig").tensor.quantizeToQ8_0;
        pub const quantizedSize = @import("stub.zig").tensor.quantizedSize;
        pub const dequantizedSize = @import("stub.zig").tensor.dequantizedSize;
        pub const dotQ4_0F32 = @import("stub.zig").tensor.dotQ4_0F32;
        pub const dotQ4_1F32 = @import("stub.zig").tensor.dotQ4_1F32;
        pub const dotQ8_0F32 = @import("stub.zig").tensor.dotQ8_0F32;
    };
    pub const tensor_mod = struct {
        pub const Tensor = types.Tensor;
        pub const DType = types.DType;
        pub const Shape = types.Shape;
        pub const TensorError = types.TensorError;
    };
    pub const view = struct {
        pub const TensorView = types.TensorView;
    };
};

pub const tokenizer = struct {
    pub const BpeTokenizer = types.BpeTokenizer;
    pub const CharTokenizer = types.CharTokenizer;
    pub const SentencePieceTokenizer = types.SentencePieceTokenizer;
    pub const Vocab = types.Vocab;
    pub const SpecialTokens = types.SpecialTokens;
    pub const TokenizerError = types.TokenizerError;
    pub const SentencePieceError = types.SentencePieceError;
    pub const TokenType = types.TokenType;
    pub const TokenizerLoadError = types.TokenizerLoadError;
    pub const TokenizerKind = types.TokenizerKind;
    pub const Tokenizer = types.Tokenizer;
    pub fn loadFromGguf(_: std.mem.Allocator, _: *const types.GgufFile) LlmError!types.Tokenizer {
        return error.FeatureDisabled;
    }
    pub fn encode(_: std.mem.Allocator, _: *types.BpeTokenizer, _: []const u8) ![]u32 {
        return error.FeatureDisabled;
    }
    pub fn decode(_: std.mem.Allocator, _: *types.BpeTokenizer, _: []const u32) ![]u8 {
        return error.FeatureDisabled;
    }
    pub const bpe = struct {
        pub const BpeTokenizer = types.BpeTokenizer;
        pub const CharTokenizer = types.CharTokenizer;
        pub const TokenizerError = types.TokenizerError;
    };
    pub const sentencepiece = struct {
        pub const SentencePieceTokenizer = types.SentencePieceTokenizer;
        pub const SentencePieceError = types.SentencePieceError;
        pub const TokenType = types.TokenType;
    };
    pub const vocab = struct {
        pub const Vocab = types.Vocab;
    };
    pub const special_tokens = struct {
        pub const SpecialTokens = types.SpecialTokens;
    };
};

pub const model = struct {
    pub const LlamaModel = types.Model;
    pub const LlamaConfig = types.ModelConfig;
    pub const Model = types.Model;
    pub const ModelConfig = types.ModelConfig;
};

pub const generation = struct {
    pub const Generator = types.Generator;
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
    pub const sampler = struct {
        pub const Sampler = types.Sampler;
        pub const SamplerConfig = types.SamplerConfig;
    };
};

pub const cache = struct {
    pub const KvCache = types.KvCache;
};

pub const ops = struct {
    pub const matmul = struct {
        pub fn matrixMultiply(_: anytype, _: anytype, c: []f32, _: usize, _: usize, _: usize) void {
            @memset(c, 0);
        }
        pub fn matrixMultiplyTransposed(_: anytype, _: anytype, _: []f32, _: usize, _: usize, _: usize) void {}
        pub fn matrixVectorMultiplyTransposed(_: anytype, _: anytype, _: []f32, _: usize, _: usize) void {}
    };
    pub const matmul_quant = struct {
        pub fn quantizedMatmulQ4(_: anytype, _: anytype, _: []f32, _: usize, _: usize, _: usize) void {}
        pub fn quantizedMatmulQ8(_: anytype, _: anytype, _: []f32, _: usize, _: usize, _: usize) void {}
    };
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
    pub const rope = struct {
        pub fn applyRope(_: []f32, _: u32, _: u32, _: u32, _: f32, _: ?*types.RopeCache) void {}
        pub const RopeCache = types.RopeCache;
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
    pub const ffn = struct {
        pub fn feedForward(_: []const f32, _: []const f32, _: []const f32, _: []f32, _: usize, _: usize) void {}
        pub fn swiglu(_: []const f32, _: []const f32, _: []f32, _: usize, _: usize) void {}
    };
    pub const gpu = struct {
        pub const GpuOpsContext = types.GpuOpsContext;
        pub const GpuStats = types.GpuStats;
        pub fn createContext(_: std.mem.Allocator, _: anytype) !types.GpuOpsContext {
            return error.FeatureDisabled;
        }
    };
    pub const gpu_memory_pool = struct {
        pub const LlmMemoryPool = types.LlmMemoryPool;
        pub const PooledBuffer = types.PooledBuffer;
        pub const PoolConfig = types.PoolConfig;
        pub const PoolStats = types.PoolStats;
    };
    pub const backward = struct {
        pub fn matmulBackward(_: anytype, _: anytype, _: anytype, _: anytype, _: usize, _: usize, _: usize) void {}
        pub fn matrixVectorBackward(_: anytype, _: anytype, _: anytype, _: anytype, _: usize, _: usize) void {}
        pub fn rmsNormBackward(_: anytype, _: anytype, _: anytype, _: anytype, _: f32) void {}
        pub fn softmaxBackward(_: anytype, _: anytype, _: anytype) void {}
        pub fn ropeBackward(_: anytype, _: anytype, _: u32, _: u32, _: u32, _: f32, _: anytype) void {}
        pub fn attentionBackward(_: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) void {}
        pub fn swigluBackward(_: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: usize, _: usize) void {}
        pub const AttentionCache = types.AttentionCache;
        pub const SwigluCache = types.SwigluCache;
    };

    pub const matrixMultiply = matmul.matrixMultiply;
    pub const matrixMultiplyTransposed = matmul.matrixMultiplyTransposed;
    pub const matrixVectorMultiplyTransposed = matmul.matrixVectorMultiplyTransposed;
    pub const quantizedMatmulQ4 = matmul_quant.quantizedMatmulQ4;
    pub const quantizedMatmulQ8 = matmul_quant.quantizedMatmulQ8;
    pub const selfAttention = attention.selfAttention;
    pub const scaledDotProductAttention = attention.scaledDotProductAttention;
    pub const applyRope = rope.applyRope;
    pub const RopeCache = rope.RopeCache;
    pub const rmsNorm = rmsnorm.rmsNorm;
    pub const rmsNormInPlace = rmsnorm.rmsNormInPlace;
    pub const silu = activations.silu;
    pub const gelu = activations.gelu;
    pub const softmax = activations.softmax;
    pub const softmaxInPlace = activations.softmaxInPlace;
    pub const feedForward = ffn.feedForward;
    pub const swiglu = ffn.swiglu;
    pub const GpuOpsContext = gpu.GpuOpsContext;
    pub const GpuStats = gpu.GpuStats;
    pub const createGpuContext = gpu.createContext;
    pub const LlmMemoryPool = gpu_memory_pool.LlmMemoryPool;
    pub const PooledBuffer = gpu_memory_pool.PooledBuffer;
    pub const PoolConfig = gpu_memory_pool.PoolConfig;
    pub const PoolStats = gpu_memory_pool.PoolStats;
    pub const matmulBackward = backward.matmulBackward;
    pub const matrixVectorBackward = backward.matrixVectorBackward;
    pub const rmsNormBackward = backward.rmsNormBackward;
    pub const softmaxBackward = backward.softmaxBackward;
    pub const ropeBackward = backward.ropeBackward;
    pub const attentionBackward = backward.attentionBackward;
    pub const swigluBackward = backward.swigluBackward;
    pub const AttentionCache = backward.AttentionCache;
    pub const SwigluCache = backward.SwigluCache;
};

pub const parallel = struct {
    pub const ParallelExecutor = types.ParallelExecutor;
};

pub const wdbx_fusion = struct {
    pub const FusionConfig = types.FusionConfig;
    pub const ContextChunk = types.ContextChunk;
    pub const WdbxFusion = types.WdbxFusion;
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
