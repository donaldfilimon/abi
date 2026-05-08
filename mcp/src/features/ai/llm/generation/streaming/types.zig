const std = @import("std");

/// Error set for streaming generation callbacks.
pub const StreamingError = error{
    /// Memory allocation failed
    OutOfMemory,
    /// Model weights not loaded
    WeightsNotLoaded,
    /// Tokenizer not available
    TokenizerNotLoaded,
    /// Invalid token ID
    InvalidToken,
    /// Position exceeds context length
    ContextOverflow,
    /// Model is in invalid state
    InvalidState,
    /// Streaming was cancelled
    Cancelled,
    /// Buffer overflow
    BufferOverflow,
    /// Generation already in progress
    AlreadyStreaming,
    /// Timer failed to start
    TimerFailed,
};

/// Streaming generation state.
pub const StreamingState = enum {
    idle,
    prefilling,
    generating,
    completed,
    cancelled,
    errored,
};

/// Token event emitted during streaming.
pub const TokenEvent = struct {
    /// Token ID
    token_id: u32,
    /// Decoded text (if available)
    text: ?[]const u8,
    /// Position in sequence
    position: u32,
    /// Is this the final token?
    is_final: bool,
    /// Generation timestamp (ns since generation start)
    timestamp_ns: u64,
};

/// Streaming statistics.
pub const StreamingStats = struct {
    /// Total tokens generated
    tokens_generated: u32,
    /// Total time for prefill (ns)
    prefill_time_ns: u64,
    /// Total time for generation (ns)
    generation_time_ns: u64,
    /// Time of first token (ns from start)
    time_to_first_token_ns: u64,
    /// Number of prompt tokens
    prompt_tokens: u32,

    /// Tokens per second.
    pub fn tokensPerSecond(self: StreamingStats) f64 {
        if (self.generation_time_ns == 0) return 0;
        return @as(f64, @floatFromInt(self.tokens_generated)) /
            (@as(f64, @floatFromInt(self.generation_time_ns)) / 1e9);
    }

    /// Time to first token in milliseconds.
    pub fn timeToFirstTokenMs(self: StreamingStats) f64 {
        return @as(f64, @floatFromInt(self.time_to_first_token_ns)) / 1e6;
    }
};

/// Callback types for streaming events.
pub const StreamingCallbacks = struct {
    /// Called for each generated token.
    on_token: ?*const fn (TokenEvent) void = null,
    /// Called when streaming completes.
    on_complete: ?*const fn (StreamingStats) void = null,
    /// Called on error.
    on_error: ?*const fn (StreamingError) void = null,
    /// User context.
    user_data: ?*anyopaque = null,
};

const generator_mod = @import("../generator.zig");

/// Configuration for streaming inference.
pub const StreamingConfig = struct {
    // Generation parameters
    /// Maximum tokens to generate
    max_tokens: u32 = 256,
    /// Temperature for sampling (0.0 = greedy, 1.0 = default)
    temperature: f32 = 0.7,
    /// Top-k sampling (0 = disabled)
    top_k: u32 = 40,
    /// Top-p nucleus sampling threshold
    top_p: f32 = 0.9,
    /// Repetition penalty (1.0 = disabled)
    repetition_penalty: f32 = 1.1,
    /// Random seed (0 = use system time)
    seed: u64 = 0,
    /// Stop token IDs
    stop_tokens: []const u32 = &[_]u32{2},

    // Buffer configuration
    /// Initial capacity for token buffer
    initial_buffer_capacity: u32 = 256,
    /// Maximum buffer size before flush (0 = unlimited)
    max_buffer_size: u32 = 0,
    /// Decode tokens to text during streaming
    decode_tokens: bool = true,

    // Timing configuration
    /// Minimum delay between tokens (ns, 0 = no delay)
    min_token_delay_ns: u64 = 0,
    /// Timeout for entire generation (ns, 0 = no timeout)
    generation_timeout_ns: u64 = 0,

    // Callbacks
    /// Callback for each token
    on_token: ?*const fn (TokenEvent) void = null,
    /// Callback when generation completes
    on_complete: ?*const fn (StreamingStats) void = null,
    /// Callback on error
    on_error: ?*const fn (StreamingError) void = null,

    /// Create config from GeneratorConfig.
    pub fn fromGeneratorConfig(gen_config: generator_mod.GeneratorConfig) StreamingConfig {
        return .{
            .max_tokens = gen_config.max_tokens,
            .temperature = gen_config.temperature,
            .top_k = gen_config.top_k,
            .top_p = gen_config.top_p,
            .repetition_penalty = gen_config.repetition_penalty,
            .seed = gen_config.seed,
            .stop_tokens = gen_config.stop_tokens,
        };
    }

    /// Convert to GeneratorConfig.
    pub fn toGeneratorConfig(self: StreamingConfig) generator_mod.GeneratorConfig {
        return .{
            .max_tokens = self.max_tokens,
            .temperature = self.temperature,
            .top_k = self.top_k,
            .top_p = self.top_p,
            .repetition_penalty = self.repetition_penalty,
            .seed = self.seed,
            .stop_tokens = self.stop_tokens,
        };
    }
};
