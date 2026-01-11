//! LLM inference engine for local model execution.
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
const build_options = @import("build_options");

// Core modules
pub const io = @import("io/mod.zig");
pub const tensor = @import("tensor/mod.zig");
pub const tokenizer = @import("tokenizer/mod.zig");
pub const ops = @import("ops/mod.zig");
pub const cache = @import("cache/mod.zig");
pub const model = @import("model/mod.zig");
pub const generation = @import("generation/mod.zig");

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
    /// Number of threads for CPU inference
    num_threads: u32 = 0, // 0 = auto-detect
    /// Enable streaming output
    streaming: bool = true,
    /// Batch size for prefill
    batch_size: u32 = 512,
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

/// High-level interface for loading and running models
pub const Engine = struct {
    allocator: std.mem.Allocator,
    loaded_model: ?*Model = null,
    config: InferenceConfig,
    stats: InferenceStats,

    pub fn init(allocator: std.mem.Allocator, config: InferenceConfig) Engine {
        return .{
            .allocator = allocator,
            .config = config,
            .stats = .{},
        };
    }

    pub fn deinit(self: *Engine) void {
        if (self.loaded_model) |m| {
            m.deinit();
            self.allocator.destroy(m);
        }
        self.* = undefined;
    }

    /// Load a model from a GGUF file
    pub fn loadModel(self: *Engine, path: []const u8) !void {
        var timer = std.time.Timer.start() catch {
            self.stats.load_time_ns = 0;
            return self.loadModelImpl(path);
        };

        try self.loadModelImpl(path);

        self.stats.load_time_ns = timer.read();
    }

    fn loadModelImpl(self: *Engine, path: []const u8) !void {
        // Unload existing model if any
        if (self.loaded_model) |m| {
            m.deinit();
            self.allocator.destroy(m);
            self.loaded_model = null;
        }

        const m = try self.allocator.create(Model);
        errdefer self.allocator.destroy(m);

        m.* = try Model.load(self.allocator, path);
        self.loaded_model = m;
    }

    /// Generate text from a prompt
    pub fn generate(self: *Engine, prompt: []const u8) ![]u8 {
        const m = self.loaded_model orelse return LlmError.InvalidModelFormat;

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
        return m.decode(output_tokens);
    }

    /// Generate with streaming callback (per-token)
    pub fn generateStreaming(
        self: *Engine,
        prompt: []const u8,
        callback: *const fn ([]const u8) void,
    ) !void {
        const m = self.loaded_model orelse return LlmError.InvalidModelFormat;

        // Encode prompt to tokens
        const prompt_tokens = try m.tokenizer.encode(self.allocator, prompt);
        defer self.allocator.free(prompt_tokens);

        // Create generator with streaming enabled
        var gen = m.createGenerator(.{
            .max_tokens = self.config.max_new_tokens,
            .temperature = self.config.temperature,
            .top_k = self.config.top_k,
            .top_p = self.config.top_p,
            .stop_tokens = &[_]u32{m.tokenizer.eos_token_id},
        });
        defer gen.deinit();

        // Generate with per-token streaming callback
        const output_tokens = try gen.generateTokensStreaming(prompt_tokens, &m.tokenizer, callback);
        defer self.allocator.free(output_tokens);

        // Update stats
        self.stats.generated_tokens += @intCast(output_tokens.len);
    }

    /// Get current statistics
    pub fn getStats(self: *Engine) InferenceStats {
        return self.stats;
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
    return engine.generate(prompt);
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
