//! LLM Stub Module — disabled at compile time.

const std = @import("std");
const config_module = @import("../../../core/config/mod.zig");

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

// --- Core Types ---

pub const DType = enum { f32, f16, q4_0, q8_0 };
pub const TensorInfo = struct {
    name: []const u8 = "",
    dims: [4]u32 = .{ 0, 0, 0, 0 },
    n_dims: usize = 0,
    tensor_type: DType = .f32,
};
pub const GgufHeader = struct {};
pub const GgufMetadata = struct {};
pub const TensorEntry = struct { value_ptr: *const TensorInfo };
pub const TensorIterator = struct {
    pub fn next(_: *TensorIterator) ?TensorEntry {
        return null;
    }
};
pub const TensorMap = struct {
    pub fn iterator(_: *const TensorMap) TensorIterator {
        return .{};
    }
};
pub const GgufFile = struct {
    tensors: TensorMap = .{},
    pub fn open(_: std.mem.Allocator, _: []const u8) LlmError!GgufFile {
        return error.LlmDisabled;
    }
    pub fn close(_: *GgufFile) void {}
    pub fn deinit(_: *GgufFile) void {}
    pub fn printSummaryDebug(_: *const GgufFile) void {}
};
pub const MappedFile = struct {};
pub const Tensor = struct {};
pub const Q4_0Block = struct {};
pub const Q8_0Block = struct {};

pub const BpeTokenizer = struct {
    pub fn init(_: std.mem.Allocator) BpeTokenizer {
        return .{};
    }
    pub fn deinit(_: *BpeTokenizer) void {}
    pub fn encode(_: *const BpeTokenizer, _: std.mem.Allocator, _: []const u8) LlmError![]u32 {
        return error.LlmDisabled;
    }
    pub fn decode(_: *const BpeTokenizer, _: std.mem.Allocator, _: []const u32) LlmError![]u8 {
        return error.LlmDisabled;
    }
};
pub const Tokenizer = BpeTokenizer;
pub const Vocab = struct {};

pub const ModelInfo = struct {
    model_name: []const u8 = "",
    architecture: []const u8 = "",
    vocab_size: u32 = 0,
    context_length: u32 = 0,
    num_layers: u32 = 0,
    n_layers: u32 = 0,
    hidden_size: u32 = 0,
    num_heads: u32 = 0,
    n_heads: u32 = 0,
    n_kv_heads: u32 = 0,
    dim: u32 = 0,
    max_seq_len: u32 = 0,
    current_pos: u32 = 0,
    kv_cache_memory: u64 = 0,
    weights_memory: u64 = 0,
};

pub const Model = struct {
    pub fn load(_: std.mem.Allocator, _: []const u8) LlmError!Model {
        return error.LlmDisabled;
    }
    pub fn deinit(_: *Model) void {}
    pub fn generate(_: *Model, _: []const u32, _: GeneratorConfig) LlmError![]u32 {
        return error.LlmDisabled;
    }
    pub fn generateText(_: *Model, _: std.mem.Allocator, _: []const u8) LlmError![]u8 {
        return error.LlmDisabled;
    }
    pub fn info(_: *const Model) ModelInfo {
        return .{};
    }
    pub fn encode(_: *Model, _: []const u8) LlmError![]u32 {
        return error.LlmDisabled;
    }
    pub fn decode(_: *Model, _: []const u32) LlmError![]u8 {
        return error.LlmDisabled;
    }
};

pub const ModelConfig = struct {
    dim: u32 = 0,
    n_layers: u32 = 0,
    n_heads: u32 = 0,
    n_kv_heads: u32 = 0,
    vocab_size: u32 = 0,
    max_seq_len: u32 = 0,
    ffn_dim: u32 = 0,
    norm_eps: f32 = 1e-6,
    rope_theta: f32 = 10000.0,
    tie_embeddings: bool = false,
    arch: []const u8 = "",
    attention_key_length: u32 = 0,
    attention_value_length: u32 = 0,

    pub fn fromGguf(_: *const GgufFile) ModelConfig {
        return .{};
    }
    pub fn estimateMemory(_: ModelConfig) u64 {
        return 0;
    }
    pub fn estimateParameters(_: ModelConfig) u64 {
        return 0;
    }
    pub fn queryHeadDim(_: ModelConfig) u32 {
        return 0;
    }
    pub fn keyHeadDim(_: ModelConfig) u32 {
        return 0;
    }
    pub fn valueHeadDim(_: ModelConfig) u32 {
        return 0;
    }
    pub fn queryDim(_: ModelConfig) u32 {
        return 0;
    }
    pub fn kvDim(_: ModelConfig) u32 {
        return 0;
    }
    pub fn valueDim(_: ModelConfig) u32 {
        return 0;
    }
    pub fn supportsLlamaAttentionLayout(_: ModelConfig) bool {
        return false;
    }
};

pub const Generator = struct {};
pub const GeneratorConfig = struct {
    max_tokens: u32 = 256,
    stop_tokens: []const u32 = &[_]u32{2},
    temperature: f32 = 0.7,
    top_k: u32 = 40,
    top_p: f32 = 0.9,
    repetition_penalty: f32 = 1.1,
    seed: u64 = 0,
};
pub const Sampler = struct {};
pub const SamplerConfig = struct {};
pub const KvCache = struct {};
pub const ParallelStrategy = enum { none, tensor, pipeline, hybrid };
pub const ParallelMode = enum { single, distributed };
pub const TensorParallelConfig = struct {};
pub const PipelineParallelConfig = struct {};
pub const ParallelConfig = struct {};
pub const ParallelCoordinator = struct {};

// --- Streaming Types ---

pub const StreamingError = error{
    OutOfMemory,
    WeightsNotLoaded,
    TokenizerNotLoaded,
    InvalidToken,
    ContextOverflow,
    InvalidState,
    Cancelled,
    BufferOverflow,
    AlreadyStreaming,
    TimerFailed,
    LlmDisabled,
};

pub const StreamingState = enum { idle, prefilling, generating, completed, cancelled, errored };

pub const TokenEvent = struct {
    token_id: u32 = 0,
    text: ?[]const u8 = null,
    position: u32 = 0,
    is_final: bool = false,
    timestamp_ns: u64 = 0,
};

pub const StreamingStats = struct {
    tokens_generated: u32 = 0,
    prefill_time_ns: u64 = 0,
    generation_time_ns: u64 = 0,
    time_to_first_token_ns: u64 = 0,
    prompt_tokens: u32 = 0,
    pub fn tokensPerSecond(_: StreamingStats) f64 {
        return 0;
    }
    pub fn timeToFirstTokenMs(_: StreamingStats) f64 {
        return 0;
    }
};

pub const StreamingCallbacks = struct {
    on_token: ?*const fn (TokenEvent) void = null,
    on_complete: ?*const fn (StreamingStats) void = null,
    on_error: ?*const fn (StreamingError) void = null,
    user_data: ?*anyopaque = null,
};

pub const StreamingConfig = struct {
    max_tokens: u32 = 256,
    temperature: f32 = 0.7,
    top_k: u32 = 40,
    top_p: f32 = 0.9,
    repetition_penalty: f32 = 1.1,
    seed: u64 = 0,
    stop_tokens: []const u32 = &[_]u32{2},
    initial_buffer_capacity: u32 = 256,
    max_buffer_size: u32 = 0,
    decode_tokens: bool = true,
    min_token_delay_ns: u64 = 0,
    generation_timeout_ns: u64 = 0,
    on_token: ?*const fn (TokenEvent) void = null,
    on_complete: ?*const fn (StreamingStats) void = null,
    on_error: ?*const fn (StreamingError) void = null,
};

pub const StreamingGenerator = struct {
    pub fn init(_: std.mem.Allocator, _: anytype) StreamingGenerator {
        return .{};
    }
    pub fn deinit(_: *StreamingGenerator) void {}
    pub fn setCallbacks(_: *StreamingGenerator, _: StreamingCallbacks) void {}
    pub fn cancel(_: *StreamingGenerator) void {}
    pub fn isCancelled(_: *StreamingGenerator) bool {
        return false;
    }
    pub fn getTokens(_: *const StreamingGenerator) []const u32 {
        return &[_]u32{};
    }
    pub fn getState(_: *const StreamingGenerator) StreamingState {
        return .idle;
    }
    pub fn getStats(_: *const StreamingGenerator) StreamingStats {
        return .{};
    }
    pub fn reset(_: *StreamingGenerator) void {}
};

pub const StreamingResponse = struct {
    pub fn init(_: std.mem.Allocator, _: anytype, _: []const u32, _: StreamingConfig, _: anytype) StreamingError!StreamingResponse {
        return error.LlmDisabled;
    }
    pub fn deinit(_: *StreamingResponse) void {}
    pub fn next(_: *StreamingResponse) StreamingError!?TokenEvent {
        return error.LlmDisabled;
    }
    pub fn cancel(_: *StreamingResponse) void {}
    pub fn isCancelled(_: *StreamingResponse) bool {
        return false;
    }
    pub fn getState(_: *const StreamingResponse) StreamingState {
        return .idle;
    }
    pub fn getStats(_: *const StreamingResponse) StreamingStats {
        return .{};
    }
    pub fn getTokens(_: *const StreamingResponse) []const u32 {
        return &[_]u32{};
    }
    pub fn getText(_: *StreamingResponse) StreamingError!?[]u8 {
        return error.LlmDisabled;
    }
    pub fn reset(_: *StreamingResponse, _: []const u32) void {}
};

pub const SSEFormatter = struct {
    pub fn formatTokenEvent(_: std.mem.Allocator, _: TokenEvent) LlmError![]u8 {
        return error.LlmDisabled;
    }
    pub fn formatCompletionEvent(_: std.mem.Allocator, _: StreamingStats) LlmError![]u8 {
        return error.LlmDisabled;
    }
    pub fn formatErrorEvent(_: std.mem.Allocator, _: StreamingError) LlmError![]u8 {
        return error.LlmDisabled;
    }
};

pub fn collectStreamingResponse(_: std.mem.Allocator, _: *StreamingResponse) LlmError!struct { text: ?[]u8, stats: StreamingStats } {
    return error.LlmDisabled;
}

// --- Inference Engine ---

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
    allow_ollama_fallback: bool = true,
    ollama_model: ?[]const u8 = null,
};

pub const EngineBackend = enum {
    none,
    local_gguf,
    ollama,
    pub fn label(self: EngineBackend) []const u8 {
        return switch (self) {
            .none => "none",
            .local_gguf => "local-gguf",
            .ollama => "ollama",
        };
    }
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
    pub fn createStreamingResponse(_: *Engine, _: []const u8, _: StreamingConfig) LlmError!StreamingResponse {
        return error.LlmDisabled;
    }
    pub fn generateStreamingWithConfig(_: *Engine, _: []const u8, _: StreamingConfig) LlmError!StreamingStats {
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
    pub fn getBackend(_: *const Engine) EngineBackend {
        return .none;
    }
    pub fn supportsStreaming(_: *const Engine) bool {
        return false;
    }
    pub fn getBackendModelName(_: *const Engine) ?[]const u8 {
        return null;
    }
};

// --- Submodule Namespace Stubs ---

const stub_root = @This();

pub const io = struct {
    pub const MappedFile = stub_root.MappedFile;
    pub const GgufFile = stub_root.GgufFile;
    pub const GgufHeader = stub_root.GgufHeader;
    pub const GgufMetadata = stub_root.GgufMetadata;
    pub const TensorInfo = stub_root.TensorInfo;
    pub const gguf = struct {
        pub const GgufFile = stub_root.GgufFile;
    };
    pub const gguf_writer = struct {
        pub const TokenizerConfig = @import("io/gguf_writer.zig").TokenizerConfig;
    };
};

pub const tensor = struct {
    pub const Tensor = stub_root.Tensor;
    pub const DType = stub_root.DType;
    pub const Q4_0Block = stub_root.Q4_0Block;
    pub const Q8_0Block = stub_root.Q8_0Block;
};

pub const tokenizer = struct {
    pub const BpeTokenizer = stub_root.BpeTokenizer;
    pub const Tokenizer = stub_root.Tokenizer;
    pub const Vocab = stub_root.Vocab;
    pub fn loadFromGguf(_: std.mem.Allocator, _: *const stub_root.GgufFile) LlmError!stub_root.Tokenizer {
        return error.LlmDisabled;
    }
};

pub const model = struct {
    pub const LlamaModel = stub_root.Model;
    pub const LlamaConfig = stub_root.ModelConfig;
};

pub const generation = struct {
    pub const Generator = stub_root.Generator;
    pub const GeneratorConfig = stub_root.GeneratorConfig;
    pub const Sampler = stub_root.Sampler;
    pub const SamplerConfig = stub_root.SamplerConfig;
    pub const StreamingGenerator = stub_root.StreamingGenerator;
    pub const StreamingResponse = stub_root.StreamingResponse;
    pub const StreamingConfig = stub_root.StreamingConfig;
    pub const StreamingState = stub_root.StreamingState;
    pub const StreamingStats = stub_root.StreamingStats;
    pub const StreamingCallbacks = stub_root.StreamingCallbacks;
    pub const StreamingError = stub_root.StreamingError;
    pub const TokenEvent = stub_root.TokenEvent;
    pub const SSEFormatter = stub_root.SSEFormatter;
    pub const collectStreamingResponse = stub_root.collectStreamingResponse;
};

pub const cache = struct {
    pub const KvCache = stub_root.KvCache;
};

pub const ops = struct {
    pub fn matrixMultiply(_: anytype, _: anytype, _: anytype, _: usize, _: usize, _: usize) void {}
    pub fn softmax(_: anytype) void {}
    pub fn rmsnorm(_: anytype, _: anytype, _: f32) void {}
};

pub const parallel = struct {
    pub const ParallelStrategy = stub_root.ParallelStrategy;
    pub const ParallelMode = stub_root.ParallelMode;
    pub const ParallelConfig = stub_root.ParallelConfig;
    pub const ParallelCoordinator = stub_root.ParallelCoordinator;
    pub const TensorParallelConfig = stub_root.TensorParallelConfig;
    pub const PipelineParallelConfig = stub_root.PipelineParallelConfig;
};

pub const providers = struct {
    pub const ProviderError = error{
        LlmDisabled,
        InvalidProvider,
        InvalidBackend,
        ModelRequired,
        PromptRequired,
        NotAvailable,
        NoProviderAvailable,
        PluginNotFound,
        PluginDisabled,
        InvalidPlugin,
        AbiVersionMismatch,
        SymbolMissing,
        GenerationFailed,
    };

    pub const ProviderId = enum {
        local_gguf,
        llama_cpp,
        mlx,
        ollama,
        lm_studio,
        vllm,
        plugin_http,
        plugin_native,

        pub fn label(self: ProviderId) []const u8 {
            return @tagName(self);
        }
    };

    pub const GenerateConfig = struct {
        model: []const u8,
        prompt: []const u8,
        backend: ?ProviderId = null,
        fallback: []const ProviderId = &.{},
        strict_backend: bool = false,
        plugin_id: ?[]const u8 = null,
        max_tokens: u32 = 256,
        temperature: f32 = 0.7,
        top_p: f32 = 0.9,
        top_k: u32 = 40,
        repetition_penalty: f32 = 1.1,
    };

    pub const GenerateResult = struct {
        provider: ProviderId,
        model_used: []u8,
        content: []u8,

        pub fn deinit(self: *GenerateResult, allocator: std.mem.Allocator) void {
            allocator.free(self.model_used);
            allocator.free(self.content);
            self.* = undefined;
        }
    };

    pub fn generate(_: std.mem.Allocator, _: GenerateConfig) ProviderError!GenerateResult {
        return error.LlmDisabled;
    }
};

// --- Context ---

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
