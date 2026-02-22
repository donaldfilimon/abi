//! LLM Sub-module
//!
//! Local LLM inference supporting GGUF models and transformer architectures.
//!
//! This module provides a pure Zig implementation for loading and running
//! large language models locally without external dependencies. Supports:
//! - GGUF model format (llama.cpp compatible)
//! - BPE tokenization
//! - Quantized inference (Q4_0, Q8_0)
//! - KV caching for efficient autoregressive generation
//! - GPU acceleration with CPU fallback
//!
//! Usage:
//! ```zig
//! const llm = @import("llm");
//!
//! var model = try llm.Model.load(allocator, "model.gguf");
//! defer model.deinit();
//!
//! var generator = model.generator(.{});
//! const output = try generator.generate(allocator, "Hello, world!");
//! ```

const std = @import("std");
const time = @import("../../../services/shared/time.zig");
const connectors = @import("../../../services/connectors/mod.zig");
const build_options = @import("build_options");
const config_module = @import("../../../core/config/mod.zig");

// Core modules (local imports)
pub const io = @import("io/mod.zig");
pub const tensor = @import("tensor/mod.zig");
pub const tokenizer = @import("tokenizer/mod.zig");
pub const ops = @import("ops/mod.zig");
pub const cache = @import("cache/mod.zig");
pub const model = @import("model/mod.zig");
pub const generation = @import("generation/mod.zig");
pub const parallel = @import("parallel.zig");
pub const providers = @import("providers/mod.zig");
pub const wdbx_fusion = @import("wdbx_fusion.zig");

// Re-exports for convenience
pub const MappedFile = io.MappedFile;
pub const GgufFile = io.GgufFile;
pub const GgufHeader = io.GgufHeader;
pub const GgufMetadata = io.GgufMetadata;
pub const TensorInfo = io.TensorInfo;

pub const Tensor = tensor.Tensor;
pub const DType = tensor.DType;
pub const Q4_0Block = tensor.Q4_0Block;
pub const Q8_0Block = tensor.Q8_0Block;

pub const BpeTokenizer = tokenizer.BpeTokenizer;
pub const Vocab = tokenizer.Vocab;

pub const KvCache = cache.KvCache;
pub const Model = model.LlamaModel;
pub const ModelConfig = model.LlamaConfig;
pub const Generator = generation.Generator;
pub const Sampler = generation.Sampler;
pub const SamplerConfig = generation.SamplerConfig;

// Streaming exports
pub const StreamingGenerator = generation.StreamingGenerator;
pub const StreamingResponse = generation.StreamingResponse;
pub const StreamingConfig = generation.StreamingConfig;
pub const StreamingState = generation.StreamingState;
pub const StreamingStats = generation.StreamingStats;
pub const StreamingCallbacks = generation.StreamingCallbacks;
pub const StreamingError = generation.StreamingError;
pub const TokenEvent = generation.TokenEvent;
pub const SSEFormatter = generation.SSEFormatter;
pub const collectStreamingResponse = generation.collectStreamingResponse;

// WDBX Fusion exports
pub const WdbxFusion = wdbx_fusion.WdbxFusion;
pub const FusionConfig = wdbx_fusion.FusionConfig;

// Parallel inference exports
pub const ParallelStrategy = parallel.ParallelStrategy;
pub const ParallelConfig = parallel.ParallelConfig;
pub const ParallelMode = parallel.ParallelMode;
pub const ParallelCoordinator = parallel.ParallelCoordinator;
pub const TensorParallelConfig = parallel.TensorParallelConfig;
pub const PipelineParallelConfig = parallel.PipelineParallelConfig;

/// LLM-specific errors
pub const LlmError = error{
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

pub const Error = error{
    LlmDisabled,
    ModelNotFound,
    ModelLoadFailed,
    InferenceFailed,
    TokenizationFailed,
    InvalidConfig,
};

/// Configuration for LLM inference
pub const InferenceConfig = struct {
    /// Maximum context length (tokens)
    max_context_length: u32 = 2048,
    /// Maximum tokens to generate
    max_new_tokens: u32 = 256,
    /// Temperature for sampling (0.0 = greedy, 1.0 = default)
    temperature: f32 = 0.7,
    /// Top-p nucleus sampling threshold
    top_p: f32 = 0.9,
    /// Top-k sampling (0 = disabled)
    top_k: u32 = 40,
    /// Repetition penalty (1.0 = disabled)
    repetition_penalty: f32 = 1.1,
    /// Use GPU acceleration if available
    use_gpu: bool = true,
    /// Number of threads for CPU inference (0 = auto). When use_gpu is false, batch
    /// parallelism can use this with abi.runtime.ThreadPool.parallelFor for multi-threaded CPU.
    num_threads: u32 = 0, // 0 = auto-detect
    /// Enable streaming output
    streaming: bool = true,
    /// Batch size for prefill
    batch_size: u32 = 512,
    /// Context size (for framework compatibility)
    context_size: u32 = 2048,
    /// Fall back to Ollama when a local model cannot be executed by ABI.
    allow_ollama_fallback: bool = true,
    /// Optional Ollama model override (otherwise connector/env default is used).
    ollama_model: ?[]const u8 = null,
};

/// Statistics from inference
pub const InferenceStats = struct {
    /// Time to load model (nanoseconds)
    load_time_ns: u64 = 0,
    /// Time for prompt processing (nanoseconds)
    prefill_time_ns: u64 = 0,
    /// Time for token generation (nanoseconds)
    decode_time_ns: u64 = 0,
    /// Number of prompt tokens processed
    prompt_tokens: u32 = 0,
    /// Number of tokens generated
    generated_tokens: u32 = 0,
    /// Peak memory usage (bytes)
    peak_memory_bytes: u64 = 0,
    /// Whether GPU was used
    used_gpu: bool = false,

    pub fn prefillTokensPerSecond(self: InferenceStats) f64 {
        if (self.prefill_time_ns == 0) return 0;
        return @as(f64, @floatFromInt(self.prompt_tokens)) /
            (@as(f64, @floatFromInt(self.prefill_time_ns)) / 1_000_000_000.0);
    }

    pub fn decodeTokensPerSecond(self: InferenceStats) f64 {
        if (self.decode_time_ns == 0) return 0;
        return @as(f64, @floatFromInt(self.generated_tokens)) /
            (@as(f64, @floatFromInt(self.decode_time_ns)) / 1_000_000_000.0);
    }

    pub fn format(
        self: InferenceStats,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        try writer.print(
            "InferenceStats{{ prefill: {d:.1} tok/s, decode: {d:.1} tok/s, prompt: {d}, generated: {d}, gpu: {} }}",
            .{
                self.prefillTokensPerSecond(),
                self.decodeTokensPerSecond(),
                self.prompt_tokens,
                self.generated_tokens,
                self.used_gpu,
            },
        );
    }
};

/// Active execution backend used by Engine.
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

/// High-level interface for loading and running models
pub const Engine = struct {
    allocator: std.mem.Allocator,
    loaded_model: ?*Model = null,
    ollama_client: ?*connectors.ollama.Client = null,
    backend: EngineBackend = .none,
    config: InferenceConfig,
    stats: InferenceStats,
    fusion: ?*WdbxFusion = null,

    pub fn init(allocator: std.mem.Allocator, config: InferenceConfig) Engine {
        return .{
            .allocator = allocator,
            .config = config,
            .stats = .{},
        };
    }

    /// Attach a WDBX fusion engine for RAG-augmented generation.
    pub fn setFusion(self: *Engine, fus: ?*WdbxFusion) void {
        self.fusion = fus;
    }

    pub fn deinit(self: *Engine) void {
        self.unloadLocalModel();
        self.unloadOllama();
        self.* = undefined;
    }

    /// Load a model from a GGUF file
    pub fn loadModel(self: *Engine, path: []const u8) !void {
        var timer = time.Timer.start() catch null;

        try self.loadModelImpl(path);

        self.stats.load_time_ns = if (timer) |*t| t.read() else 0;
    }

    fn loadModelImpl(self: *Engine, path: []const u8) !void {
        self.unloadLocalModel();
        self.unloadOllama();
        self.backend = .none;

        if (looksLikeExternalModelName(path)) {
            try self.loadOllama(resolveOllamaModelHint(self.config.ollama_model, path));
            return;
        }

        const m = try self.allocator.create(Model);
        errdefer self.allocator.destroy(m);

        m.* = Model.load(self.allocator, path) catch |err| {
            if (err == error.UnsupportedArchitecture and self.config.allow_ollama_fallback) {
                self.allocator.destroy(m);
                try self.loadOllama(resolveOllamaModelHint(self.config.ollama_model, path));
                return;
            }
            return err;
        };

        self.loaded_model = m;
        self.backend = .local_gguf;
    }

    /// Generate text from a prompt
    pub fn generate(self: *Engine, allocator: std.mem.Allocator, prompt: []const u8) ![]u8 {
        var timer = time.Timer.start() catch null;
        self.stats.prefill_time_ns = 0;
        self.stats.decode_time_ns = 0;
        self.stats.prompt_tokens = 0;
        self.stats.generated_tokens = 0;

        if (self.loaded_model) |m| {
            // Encode prompt to tokens
            const prompt_tokens = try m.encode(prompt);
            defer self.allocator.free(prompt_tokens);

            // Generate output tokens
            const output_tokens = try m.generate(prompt_tokens, .{
                .max_tokens = self.config.max_new_tokens,
                .temperature = self.config.temperature,
                .top_p = self.config.top_p,
                .top_k = self.config.top_k,
            });
            defer self.allocator.free(output_tokens);

            // Decode tokens to text
            const decoded = try m.decode(output_tokens);
            defer self.allocator.free(decoded);

            self.stats.prompt_tokens = @intCast(prompt_tokens.len);
            self.stats.generated_tokens = @intCast(output_tokens.len);
            self.stats.decode_time_ns = if (timer) |*t| t.read() else 0;
            self.stats.used_gpu = self.config.use_gpu;

            return allocator.dupe(u8, decoded);
        }

        if (self.ollama_client) |client| {
            var res = try client.generate(.{
                .model = client.config.model,
                .prompt = prompt,
                .stream = false,
                .options = .{
                    .temperature = self.config.temperature,
                    .num_predict = self.config.max_new_tokens,
                    .top_p = self.config.top_p,
                    .top_k = self.config.top_k,
                },
            });
            defer res.deinit(self.allocator);

            self.stats.prompt_tokens = res.prompt_eval_count orelse estimateTokenCount(prompt);
            self.stats.generated_tokens = res.eval_count orelse estimateTokenCount(res.response);
            self.stats.prefill_time_ns = res.prompt_eval_duration_ns orelse 0;
            self.stats.decode_time_ns = res.eval_duration_ns orelse if (timer) |*t| t.read() else 0;
            self.stats.load_time_ns = res.load_duration_ns orelse self.stats.load_time_ns;
            self.stats.used_gpu = false;

            return allocator.dupe(u8, res.response);
        }

        return LlmError.InvalidModelFormat;
    }

    /// Generate with streaming callback (per-token)
    ///
    /// This is the simple callback-based streaming API. For more control,
    /// use `createStreamingResponse()` which returns an iterator.
    pub fn generateStreaming(
        self: *Engine,
        prompt: []const u8,
        callback: *const fn ([]const u8) void,
    ) !void {
        if (self.ollama_client != null) {
            const output = try self.generate(self.allocator, prompt);
            defer self.allocator.free(output);
            callback(output);
            return;
        }

        const m = self.loaded_model orelse return LlmError.InvalidModelFormat;

        // Encode prompt to tokens
        const tok = if (m.tokenizer) |*t| t else return LlmError.TokenizationFailed;
        const prompt_tokens = try tok.encode(self.allocator, prompt);
        defer self.allocator.free(prompt_tokens);

        // Create generator with streaming enabled
        var gen = m.generator(.{
            .max_tokens = self.config.max_new_tokens,
            .temperature = self.config.temperature,
            .top_k = self.config.top_k,
            .top_p = self.config.top_p,
        });
        defer gen.deinit();

        // Generate with per-token streaming callback
        const output_tokens = try gen.generateTokensStreaming(prompt_tokens, tok, callback);
        defer self.allocator.free(output_tokens);

        // Update stats
        self.stats.generated_tokens += @intCast(output_tokens.len);
    }

    /// Generate with streaming using advanced configuration.
    ///
    /// This function provides more control over streaming behavior through
    /// the `StreamingConfig` struct. It returns a `StreamingResponse` iterator
    /// that can be used for pull-based streaming.
    ///
    /// ## Example
    ///
    /// ```zig
    /// var response = try engine.createStreamingResponse(prompt, .{
    ///     .max_tokens = 100,
    ///     .temperature = 0.8,
    ///     .on_token = myCallback,
    /// });
    /// defer response.deinit();
    ///
    /// while (try response.next()) |event| {
    ///     if (event.text) |text| {
    ///         try stdout.writeAll(text);
    ///     }
    ///     if (event.is_final) break;
    /// }
    /// ```
    pub fn createStreamingResponse(
        self: *Engine,
        prompt: []const u8,
        stream_config: StreamingConfig,
    ) !StreamingResponse {
        const m = self.loaded_model orelse return LlmError.InvalidModelFormat;

        // Get tokenizer
        const tok = if (m.tokenizer) |*t| t else return LlmError.TokenizationFailed;

        // Encode prompt to tokens
        const prompt_tokens = try tok.encode(self.allocator, prompt);
        // Note: prompt_tokens ownership transfers to caller, must be freed

        // Create streaming response with model and config
        return StreamingResponse.init(
            self.allocator,
            m,
            prompt_tokens,
            stream_config,
            tok,
        ) catch |e| switch (e) {
            StreamingError.OutOfMemory => return LlmError.OutOfMemory,
            StreamingError.TimerFailed => return LlmError.InferenceError,
            else => return LlmError.InferenceError,
        };
    }

    /// Generate with streaming callbacks using advanced configuration.
    ///
    /// This is a convenience function that sets up streaming with callbacks
    /// and iterates through all tokens automatically.
    pub fn generateStreamingWithConfig(
        self: *Engine,
        prompt: []const u8,
        stream_config: StreamingConfig,
    ) !StreamingStats {
        var response = try self.createStreamingResponse(prompt, stream_config);
        defer response.deinit();

        // Iterate through all tokens
        while (try response.next()) |event| {
            if (event.is_final) break;
        }

        // Update engine stats
        const stats = response.getStats();
        self.stats.generated_tokens += stats.tokens_generated;
        self.stats.prefill_time_ns += stats.prefill_time_ns;
        self.stats.decode_time_ns += stats.generation_time_ns;

        return stats;
    }

    /// Tokenize text
    pub fn tokenize(self: *Engine, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        var m = self.loaded_model orelse return LlmError.InvalidModelFormat;
        const tokens = try m.encode(text);
        defer self.allocator.free(tokens);

        const out = try allocator.alloc(u32, tokens.len);
        @memcpy(out, tokens);
        return out;
    }

    /// Detokenize tokens
    pub fn detokenize(self: *Engine, allocator: std.mem.Allocator, tokens: []const u32) ![]u8 {
        var m = self.loaded_model orelse return LlmError.InvalidModelFormat;
        const text = try m.decode(tokens);
        defer self.allocator.free(text);
        return allocator.dupe(u8, text);
    }

    /// Get active backend.
    pub fn getBackend(self: *const Engine) EngineBackend {
        return self.backend;
    }

    /// Return true when token-level local streaming is available.
    pub fn supportsStreaming(self: *const Engine) bool {
        return self.loaded_model != null;
    }

    /// Return active model name for connector-backed engines.
    pub fn getBackendModelName(self: *const Engine) ?[]const u8 {
        if (self.ollama_client) |client| return client.config.model;
        return null;
    }

    /// Get current statistics
    pub fn getStats(self: *Engine) InferenceStats {
        return self.stats;
    }

    fn unloadLocalModel(self: *Engine) void {
        if (self.loaded_model) |m| {
            m.deinit();
            self.allocator.destroy(m);
            self.loaded_model = null;
        }
    }

    fn unloadOllama(self: *Engine) void {
        if (self.ollama_client) |client| {
            client.deinit();
            self.allocator.destroy(client);
            self.ollama_client = null;
        }
    }

    fn loadOllama(self: *Engine, model_override: ?[]const u8) !void {
        const client_ptr = try self.allocator.create(connectors.ollama.Client);
        errdefer self.allocator.destroy(client_ptr);

        client_ptr.* = try connectors.ollama.createClient(self.allocator);
        errdefer client_ptr.deinit();

        if (model_override) |model_name| {
            if (model_name.len > 0) {
                if (client_ptr.config.model_owned) {
                    self.allocator.free(@constCast(client_ptr.config.model));
                }
                client_ptr.config.model = try self.allocator.dupe(u8, model_name);
                client_ptr.config.model_owned = true;
            }
        }

        self.ollama_client = client_ptr;
        self.backend = .ollama;
    }
};

fn estimateTokenCount(text: []const u8) u32 {
    if (text.len == 0) return 0;

    var count: u32 = 0;
    var in_word = false;
    for (text) |c| {
        const is_space = std.ascii.isWhitespace(c);
        if (is_space) {
            in_word = false;
            continue;
        }
        if (!in_word) {
            count += 1;
            in_word = true;
        }
    }
    return count;
}

fn looksLikeExternalModelName(path: []const u8) bool {
    if (path.len == 0) return false;
    if (std.mem.indexOfScalar(u8, path, '/')) |_| return false;
    if (std.mem.indexOfScalar(u8, path, '\\')) |_| return false;
    if (std.mem.endsWith(u8, path, ".gguf")) return false;
    if (std.mem.endsWith(u8, path, ".bin")) return false;
    if (std.mem.endsWith(u8, path, ".safetensors")) return false;
    if (std.mem.endsWith(u8, path, ".mlx")) return false;
    return true;
}

fn resolveOllamaModelHint(config_model: ?[]const u8, path: []const u8) ?[]const u8 {
    if (config_model) |name| {
        if (name.len > 0) return name;
    }

    if (looksLikeExternalModelName(path)) {
        return path;
    }

    const base = if (std.mem.lastIndexOfScalar(u8, path, '/')) |pos| path[pos + 1 ..] else path;
    if (std.mem.indexOf(u8, base, "gpt-oss") != null or std.mem.indexOf(u8, base, "GPT-OSS") != null) {
        return "gpt-oss";
    }
    return null;
}

/// LLM context for framework integration.
pub const Context = struct {
    allocator: std.mem.Allocator,
    config: config_module.LlmConfig,
    engine: ?*Engine = null,

    pub fn init(allocator: std.mem.Allocator, cfg: config_module.LlmConfig) !*Context {
        if (!isEnabled()) return error.LlmDisabled;

        const ctx = try allocator.create(Context);
        ctx.* = .{
            .allocator = allocator,
            .config = cfg,
        };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        if (self.engine) |e| {
            e.deinit();
            self.allocator.destroy(e);
        }
        self.allocator.destroy(self);
    }

    /// Get or initialize the LLM engine.
    pub fn getEngine(self: *Context) !*Engine {
        if (self.engine) |e| return e;

        const engine_ptr = try self.allocator.create(Engine);
        engine_ptr.* = Engine.init(self.allocator, .{
            .context_size = self.config.context_size,
            .batch_size = self.config.batch_size,
        });
        self.engine = engine_ptr;
        return engine_ptr;
    }

    /// Generate text from prompt.
    pub fn generate(self: *Context, prompt: []const u8) ![]u8 {
        const engine = try self.getEngine();
        return engine.generate(self.allocator, prompt);
    }

    /// Tokenize text.
    pub fn tokenize(self: *Context, text: []const u8) ![]u32 {
        const engine = try self.getEngine();
        return engine.tokenize(self.allocator, text);
    }

    /// Detokenize tokens.
    pub fn detokenize(self: *Context, tokens: []const u32) ![]u8 {
        const engine = try self.getEngine();
        return engine.detokenize(self.allocator, tokens);
    }
};

/// Check if LLM features are enabled
pub fn isEnabled() bool {
    return build_options.enable_ai and build_options.enable_llm;
}

/// Quick inference helper
pub fn infer(allocator: std.mem.Allocator, model_path: []const u8, prompt: []const u8) ![]u8 {
    var engine = Engine.init(allocator, .{});
    defer engine.deinit();

    try engine.loadModel(model_path);
    return engine.generate(allocator, prompt);
}

test "llm module compilation" {
    // Basic compilation test
    const config = InferenceConfig{};
    try std.testing.expect(config.max_context_length == 2048);
    try std.testing.expect(config.temperature == 0.7);
}

test "inference stats formatting" {
    const stats = InferenceStats{
        .prompt_tokens = 100,
        .generated_tokens = 50,
        .prefill_time_ns = 1_000_000_000, // 1 second
        .decode_time_ns = 2_000_000_000, // 2 seconds
    };

    try std.testing.expectEqual(@as(f64, 100.0), stats.prefillTokensPerSecond());
    try std.testing.expectEqual(@as(f64, 25.0), stats.decodeTokensPerSecond());
}
