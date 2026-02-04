//! Async streaming generation support.
//!
//! Provides asynchronous token generation with callback-based streaming,
//! compatible with web servers and interactive applications.
//!
//! ## Features
//!
//! - **Callback-based streaming**: Register callbacks for tokens, completion, and errors
//! - **Iterator-based streaming**: Pull tokens one at a time with `StreamingResponse`
//! - **Cancellation support**: Cancel generation mid-stream with thread-safe flag
//! - **SSE formatting**: Built-in Server-Sent Events formatting for web APIs
//! - **Configurable buffering**: Control memory usage with buffer size options
//!
//! ## Example (Callback-based)
//!
//! ```zig
//! var gen = StreamingGenerator.init(allocator, config);
//! defer gen.deinit();
//!
//! gen.setCallbacks(.{
//!     .on_token = onToken,
//!     .on_complete = onComplete,
//! });
//!
//! try gen.startStreaming(getLogits, prompt_tokens, tokenizer);
//! ```
//!
//! ## Example (Iterator-based)
//!
//! ```zig
//! var response = try StreamingResponse.init(allocator, model, prompt_tokens, streaming_config);
//! defer response.deinit();
//!
//! while (try response.next()) |event| {
//!     if (event.text) |text| {
//!         std.debug.print("{s}", .{text});
//!     }
//!     if (event.is_final) break;
//! }
//! ```

const std = @import("std");
const sampler_mod = @import("sampler.zig");
const tokenizer = @import("../tokenizer/mod.zig");
const generator_mod = @import("generator.zig");

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

/// Configuration for streaming inference.
///
/// This struct provides fine-grained control over streaming behavior including
/// buffer sizes, timing, and generation parameters.
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

/// Async streaming generator.
pub const StreamingGenerator = struct {
    allocator: std.mem.Allocator,
    config: generator_mod.GeneratorConfig,
    sampler: sampler_mod.Sampler,
    state: StreamingState,
    callbacks: StreamingCallbacks,

    /// Buffer for accumulating tokens
    token_buffer: std.ArrayListUnmanaged(u32),

    /// Statistics
    stats: StreamingStats,

    /// Start time for timing
    start_time: ?std.time.Timer,

    /// Cancellation flag
    cancel_requested: std.atomic.Value(bool),

    pub fn init(allocator: std.mem.Allocator, config: generator_mod.GeneratorConfig) StreamingGenerator {
        return .{
            .allocator = allocator,
            .config = config,
            .sampler = sampler_mod.Sampler.init(allocator, .{
                .temperature = config.temperature,
                .top_k = config.top_k,
                .top_p = config.top_p,
                .repetition_penalty = config.repetition_penalty,
                .seed = config.seed,
            }),
            .state = .idle,
            .callbacks = .{},
            .token_buffer = std.ArrayListUnmanaged(u32).empty,
            .stats = std.mem.zeroes(StreamingStats),
            .start_time = null,
            .cancel_requested = std.atomic.Value(bool).init(false),
        };
    }

    pub fn deinit(self: *StreamingGenerator) void {
        self.token_buffer.deinit(self.allocator);
        self.sampler.deinit();
        self.* = undefined;
    }

    /// Set streaming callbacks.
    pub fn setCallbacks(self: *StreamingGenerator, callbacks: StreamingCallbacks) void {
        self.callbacks = callbacks;
    }

    /// Request cancellation of ongoing generation.
    pub fn cancel(self: *StreamingGenerator) void {
        self.cancel_requested.store(true, .seq_cst);
    }

    /// Check if cancellation was requested.
    pub fn isCancelled(self: *StreamingGenerator) bool {
        return self.cancel_requested.load(.seq_cst);
    }

    /// Start streaming generation from logits provider.
    pub fn startStreaming(
        self: *StreamingGenerator,
        getLogits: *const fn (u32, u32) StreamingError![]f32,
        prompt_tokens: []const u32,
        tok: ?*tokenizer.BpeTokenizer,
    ) StreamingError!void {
        self.state = .prefilling;
        self.cancel_requested.store(false, .seq_cst);
        self.token_buffer.clearRetainingCapacity();
        self.stats = std.mem.zeroes(StreamingStats);
        self.stats.prompt_tokens = @intCast(prompt_tokens.len);

        self.start_time = std.time.Timer.start() catch null;

        // Prefill phase
        var pos: u32 = 0;
        for (prompt_tokens) |token| {
            if (self.isCancelled()) {
                self.state = .cancelled;
                return;
            }
            _ = try getLogits(token, pos);
            pos += 1;
        }

        if (self.start_time) |*timer| {
            self.stats.prefill_time_ns = timer.read();
        }

        self.state = .generating;

        // Generation phase
        var last_token = prompt_tokens[prompt_tokens.len - 1];
        var first_token = true;

        while (self.stats.tokens_generated < self.config.max_tokens) {
            if (self.isCancelled()) {
                self.state = .cancelled;
                break;
            }

            // Get logits
            const logits = getLogits(last_token, pos) catch |err| {
                self.state = .errored;
                if (self.callbacks.on_error) |on_error| {
                    on_error(err);
                }
                return err;
            };

            // Sample next token
            const next_token = self.sampler.sample(logits);

            // Check for stop token
            var should_stop = false;
            for (self.config.stop_tokens) |stop| {
                if (next_token == stop) {
                    should_stop = true;
                    break;
                }
            }

            // Record first token time
            if (first_token) {
                if (self.start_time) |*timer| {
                    self.stats.time_to_first_token_ns = timer.read();
                }
                first_token = false;
            }

            // Emit token event
            self.token_buffer.append(self.allocator, next_token) catch |err| {
                std.log.warn("Token buffer append failed: {t}", .{err});
            };
            self.stats.tokens_generated += 1;

            if (self.callbacks.on_token) |on_token| {
                var text: ?[]const u8 = null;
                if (tok) |t| {
                    const token_slice = [_]u32{next_token};
                    text = t.decode(self.allocator, &token_slice) catch null;
                }
                defer if (text) |txt| self.allocator.free(txt);

                var current_time: u64 = 0;
                if (self.start_time) |*timer| {
                    current_time = timer.read();
                }

                on_token(.{
                    .token_id = next_token,
                    .text = text,
                    .position = pos,
                    .is_final = should_stop,
                    .timestamp_ns = current_time,
                });
            }

            if (should_stop) break;

            last_token = next_token;
            pos += 1;
        }

        // Finalize
        if (self.start_time) |*timer| {
            self.stats.generation_time_ns = timer.read() - self.stats.prefill_time_ns;
        }

        self.state = .completed;

        if (self.callbacks.on_complete) |on_complete| {
            on_complete(self.stats);
        }
    }

    /// Get collected tokens.
    pub fn getTokens(self: *const StreamingGenerator) []const u32 {
        return self.token_buffer.items;
    }

    /// Get current state.
    pub fn getState(self: *const StreamingGenerator) StreamingState {
        return self.state;
    }

    /// Get current statistics.
    pub fn getStats(self: *const StreamingGenerator) StreamingStats {
        return self.stats;
    }

    /// Reset generator for new generation.
    pub fn reset(self: *StreamingGenerator) void {
        self.state = .idle;
        self.cancel_requested.store(false, .seq_cst);
        self.token_buffer.clearRetainingCapacity();
        self.sampler.reset();
        self.stats = std.mem.zeroes(StreamingStats);
        self.start_time = null;
    }
};

/// Server-Sent Events (SSE) formatter for streaming.
pub const SSEFormatter = struct {
    /// Format a token event as SSE data.
    pub fn formatTokenEvent(allocator: std.mem.Allocator, event: TokenEvent) ![]u8 {
        var buffer = std.ArrayListUnmanaged(u8).empty;
        errdefer buffer.deinit(allocator);

        // SSE format: data: {json}\n\n
        try buffer.appendSlice(allocator, "data: {\"token_id\":");
        const id_str = try std.fmt.allocPrint(allocator, "{d}", .{event.token_id});
        defer allocator.free(id_str);
        try buffer.appendSlice(allocator, id_str);

        if (event.text) |text| {
            try buffer.appendSlice(allocator, ",\"text\":\"");
            // Escape JSON string
            for (text) |c| {
                switch (c) {
                    '"' => try buffer.appendSlice(allocator, "\\\""),
                    '\\' => try buffer.appendSlice(allocator, "\\\\"),
                    '\n' => try buffer.appendSlice(allocator, "\\n"),
                    '\r' => try buffer.appendSlice(allocator, "\\r"),
                    '\t' => try buffer.appendSlice(allocator, "\\t"),
                    else => try buffer.append(allocator, c),
                }
            }
            try buffer.appendSlice(allocator, "\"");
        }

        try buffer.appendSlice(allocator, ",\"position\":");
        const pos_str = try std.fmt.allocPrint(allocator, "{d}", .{event.position});
        defer allocator.free(pos_str);
        try buffer.appendSlice(allocator, pos_str);

        try buffer.appendSlice(allocator, ",\"is_final\":");
        try buffer.appendSlice(allocator, if (event.is_final) "true" else "false");

        try buffer.appendSlice(allocator, "}\n\n");

        return buffer.toOwnedSlice(allocator);
    }

    /// Format completion event as SSE.
    pub fn formatCompletionEvent(allocator: std.mem.Allocator, stats: StreamingStats) ![]u8 {
        return std.fmt.allocPrint(allocator,
            \\data: {{"event":"complete","tokens_generated":{d},"tokens_per_second":{d:.1},"time_to_first_token_ms":{d:.1}}}
            \\
            \\
        , .{
            stats.tokens_generated,
            stats.tokensPerSecond(),
            stats.timeToFirstTokenMs(),
        });
    }

    /// Format error event as SSE.
    pub fn formatErrorEvent(allocator: std.mem.Allocator, err: StreamingError) ![]u8 {
        return std.fmt.allocPrint(allocator,
            \\data: {{"event":"error","error":"{t}"}}
            \\
            \\
        , .{err});
    }
};

/// Simple stdout streaming helper.
pub fn streamToStdout(event: TokenEvent) void {
    if (event.text) |text| {
        std.debug.print("{s}", .{text});
    }
}

/// Completion callback that prints stats.
pub fn printCompletionStats(stats: StreamingStats) void {
    std.debug.print("\n\n--- Generation Complete ---\n", .{});
    std.debug.print("Tokens: {d}\n", .{stats.tokens_generated});
    std.debug.print("Speed: {d:.1} tok/s\n", .{stats.tokensPerSecond()});
    std.debug.print("Time to first token: {d:.1}ms\n", .{stats.timeToFirstTokenMs()});
}

/// Iterator-based streaming response.
///
/// Provides a pull-based API for streaming token generation. Use this when you
/// want more control over the generation loop, such as in async contexts or
/// when you need to process tokens between generation steps.
///
/// ## Example
///
/// ```zig
/// var response = try StreamingResponse.init(allocator, &model, prompt_tokens, .{});
/// defer response.deinit();
///
/// while (try response.next()) |event| {
///     if (event.text) |text| {
///         try writer.writeAll(text);
///     }
///     if (event.is_final) break;
/// }
///
/// const stats = response.getStats();
/// ```
pub const StreamingResponse = struct {
    allocator: std.mem.Allocator,
    config: StreamingConfig,
    sampler: sampler_mod.Sampler,
    state: StreamingState,
    tok: ?*tokenizer.BpeTokenizer,

    // Model forward function
    model: *anyopaque,
    forward_fn: *const fn (*anyopaque, u32, u32) StreamingError![]f32,

    // Generation state
    prompt_tokens: []const u32,
    current_position: u32,
    last_token: u32,
    tokens_generated: u32,

    // Buffer for generated tokens
    token_buffer: std.ArrayListUnmanaged(u32),

    // Timing
    start_time: ?std.time.Timer,
    first_token_time: ?u64,
    stats: StreamingStats,

    // Cancellation
    cancel_requested: std.atomic.Value(bool),

    // Last decoded text (owned, must be freed)
    last_text: ?[]u8,

    /// Initialize streaming response.
    ///
    /// The model must have a `forward(token: u32, pos: u32) ![]f32` method.
    pub fn init(
        allocator: std.mem.Allocator,
        model: anytype,
        prompt_tokens: []const u32,
        config: StreamingConfig,
        tok: ?*tokenizer.BpeTokenizer,
    ) StreamingError!StreamingResponse {
        const Model = @TypeOf(model);
        const Ptr = if (@typeInfo(Model) == .pointer) Model else *Model;

        var response = StreamingResponse{
            .allocator = allocator,
            .config = config,
            .sampler = sampler_mod.Sampler.init(allocator, .{
                .temperature = config.temperature,
                .top_k = config.top_k,
                .top_p = config.top_p,
                .repetition_penalty = config.repetition_penalty,
                .seed = config.seed,
            }),
            .state = .idle,
            .tok = tok,
            .model = @ptrCast(@constCast(model)),
            .forward_fn = &struct {
                fn forward(m: *anyopaque, token: u32, pos: u32) StreamingError![]f32 {
                    const model_ptr: Ptr = @ptrCast(@alignCast(m));
                    return model_ptr.forward(token, pos) catch |e| {
                        // Map allocation errors, treat other errors as invalid state
                        return if (e == error.OutOfMemory) StreamingError.OutOfMemory else StreamingError.InvalidState;
                    };
                }
            }.forward,
            .prompt_tokens = prompt_tokens,
            .current_position = 0,
            .last_token = 0,
            .tokens_generated = 0,
            .token_buffer = std.ArrayListUnmanaged(u32).empty,
            .start_time = null,
            .first_token_time = null,
            .stats = std.mem.zeroes(StreamingStats),
            .cancel_requested = std.atomic.Value(bool).init(false),
            .last_text = null,
        };

        // Pre-allocate buffer
        if (config.initial_buffer_capacity > 0) {
            response.token_buffer.ensureTotalCapacity(allocator, config.initial_buffer_capacity) catch |err| {
                std.log.debug("Failed to pre-allocate streaming buffer: {t}", .{err});
            };
        }

        return response;
    }

    pub fn deinit(self: *StreamingResponse) void {
        if (self.last_text) |text| {
            self.allocator.free(text);
        }
        self.token_buffer.deinit(self.allocator);
        self.sampler.deinit();
        self.* = undefined;
    }

    /// Get the next token event from the stream.
    ///
    /// Returns null when generation is complete or cancelled.
    /// The returned TokenEvent's text field is valid until the next call to next().
    pub fn next(self: *StreamingResponse) StreamingError!?TokenEvent {
        // Free previous text
        if (self.last_text) |text| {
            self.allocator.free(text);
            self.last_text = null;
        }

        // Check cancellation
        if (self.cancel_requested.load(.seq_cst)) {
            self.state = .cancelled;
            return null;
        }

        // Handle different states
        switch (self.state) {
            .idle => {
                // Start prefill
                self.state = .prefilling;
                self.start_time = std.time.Timer.start() catch return StreamingError.TimerFailed;
                self.stats.prompt_tokens = @intCast(self.prompt_tokens.len);

                // Process all prompt tokens
                for (self.prompt_tokens) |token| {
                    if (self.cancel_requested.load(.seq_cst)) {
                        self.state = .cancelled;
                        return null;
                    }
                    _ = try self.forward_fn(self.model, token, self.current_position);
                    self.current_position += 1;
                }

                // Record prefill time
                if (self.start_time) |*timer| {
                    self.stats.prefill_time_ns = timer.read();
                }

                // Switch to generating
                self.state = .generating;
                self.last_token = self.prompt_tokens[self.prompt_tokens.len - 1];

                // Return prefill complete event (position 0, no text)
                return TokenEvent{
                    .token_id = self.last_token,
                    .text = null,
                    .position = self.current_position - 1,
                    .is_final = false,
                    .timestamp_ns = self.stats.prefill_time_ns,
                };
            },

            .prefilling => {
                // Should not happen - prefill is done in one shot
                self.state = .generating;
                return self.next();
            },

            .generating => {
                // Check if we've hit max tokens
                if (self.tokens_generated >= self.config.max_tokens) {
                    return self.finalize();
                }

                // Check timeout
                if (self.config.generation_timeout_ns > 0) {
                    if (self.start_time) |*timer| {
                        if (timer.read() > self.config.generation_timeout_ns) {
                            return self.finalize();
                        }
                    }
                }

                // Forward pass
                const logits = try self.forward_fn(self.model, self.last_token, self.current_position);

                // Sample next token
                const next_token = self.sampler.sample(logits);

                // Check for stop token
                for (self.config.stop_tokens) |stop| {
                    if (next_token == stop) {
                        return self.finalize();
                    }
                }

                // Record first token time
                if (self.first_token_time == null) {
                    if (self.start_time) |*timer| {
                        self.first_token_time = timer.read();
                        self.stats.time_to_first_token_ns = self.first_token_time.?;
                    }
                }

                // Update state
                self.token_buffer.append(self.allocator, next_token) catch return StreamingError.OutOfMemory;
                self.tokens_generated += 1;
                self.last_token = next_token;
                self.current_position += 1;

                // Decode token if requested
                var text: ?[]const u8 = null;
                if (self.config.decode_tokens) {
                    if (self.tok) |t| {
                        const token_slice = [_]u32{next_token};
                        self.last_text = t.decode(self.allocator, &token_slice) catch null;
                        text = self.last_text;
                    }
                }

                // Get current timestamp
                var current_time: u64 = 0;
                if (self.start_time) |*timer| {
                    current_time = timer.read();
                }

                // Call callback if set
                const event = TokenEvent{
                    .token_id = next_token,
                    .text = text,
                    .position = self.current_position - 1,
                    .is_final = false,
                    .timestamp_ns = current_time,
                };

                if (self.config.on_token) |on_token| {
                    on_token(event);
                }

                return event;
            },

            .completed, .cancelled, .errored => {
                return null;
            },
        }
    }

    /// Finalize the stream and return the final event.
    fn finalize(self: *StreamingResponse) ?TokenEvent {
        self.state = .completed;

        // Calculate final stats
        if (self.start_time) |*timer| {
            const total_time = timer.read();
            self.stats.generation_time_ns = total_time - self.stats.prefill_time_ns;
        }
        self.stats.tokens_generated = self.tokens_generated;

        // Call completion callback
        if (self.config.on_complete) |on_complete| {
            on_complete(self.stats);
        }

        // Return final event
        var current_time: u64 = 0;
        if (self.start_time) |*timer| {
            current_time = timer.read();
        }

        return TokenEvent{
            .token_id = self.last_token,
            .text = null,
            .position = self.current_position - 1,
            .is_final = true,
            .timestamp_ns = current_time,
        };
    }

    /// Request cancellation of the stream.
    pub fn cancel(self: *StreamingResponse) void {
        self.cancel_requested.store(true, .seq_cst);
    }

    /// Check if the stream is cancelled.
    pub fn isCancelled(self: *StreamingResponse) bool {
        return self.cancel_requested.load(.seq_cst);
    }

    /// Get current state.
    pub fn getState(self: *const StreamingResponse) StreamingState {
        return self.state;
    }

    /// Get statistics.
    pub fn getStats(self: *const StreamingResponse) StreamingStats {
        return self.stats;
    }

    /// Get all generated tokens.
    pub fn getTokens(self: *const StreamingResponse) []const u32 {
        return self.token_buffer.items;
    }

    /// Get full decoded text.
    pub fn getText(self: *StreamingResponse) !?[]u8 {
        if (self.tok) |t| {
            return t.decode(self.allocator, self.token_buffer.items);
        }
        return null;
    }

    /// Reset for new generation with different prompt.
    pub fn reset(self: *StreamingResponse, new_prompt_tokens: []const u32) void {
        if (self.last_text) |text| {
            self.allocator.free(text);
            self.last_text = null;
        }
        self.state = .idle;
        self.cancel_requested.store(false, .seq_cst);
        self.token_buffer.clearRetainingCapacity();
        self.sampler.reset();
        self.prompt_tokens = new_prompt_tokens;
        self.current_position = 0;
        self.last_token = 0;
        self.tokens_generated = 0;
        self.start_time = null;
        self.first_token_time = null;
        self.stats = std.mem.zeroes(StreamingStats);
    }
};

/// Collect all tokens from a streaming response.
///
/// Convenience function that iterates through all tokens and returns
/// the decoded text and statistics.
pub fn collectStreamingResponse(
    allocator: std.mem.Allocator,
    response: *StreamingResponse,
) !struct { text: ?[]u8, stats: StreamingStats } {
    var text_buffer = std.ArrayListUnmanaged(u8).empty;
    errdefer text_buffer.deinit(allocator);

    while (try response.next()) |event| {
        if (event.text) |text| {
            try text_buffer.appendSlice(allocator, text);
        }
        if (event.is_final) break;
    }

    return .{
        .text = if (text_buffer.items.len > 0) text_buffer.toOwnedSlice(allocator) catch null else null,
        .stats = response.getStats(),
    };
}

test "streaming generator init" {
    const allocator = std.testing.allocator;

    var gen = StreamingGenerator.init(allocator, .{});
    defer gen.deinit();

    try std.testing.expectEqual(StreamingState.idle, gen.getState());
}

test "streaming stats calculation" {
    const stats = StreamingStats{
        .tokens_generated = 100,
        .prefill_time_ns = 50_000_000, // 50ms
        .generation_time_ns = 1_000_000_000, // 1s
        .time_to_first_token_ns = 100_000_000, // 100ms
        .prompt_tokens = 50,
    };

    // 100 tokens / 1 second = 100 tok/s
    try std.testing.expectEqual(@as(f64, 100.0), stats.tokensPerSecond());
    // 100ms
    try std.testing.expectEqual(@as(f64, 100.0), stats.timeToFirstTokenMs());
}

test "sse formatter" {
    const allocator = std.testing.allocator;

    const event = TokenEvent{
        .token_id = 42,
        .text = "hello",
        .position = 10,
        .is_final = false,
        .timestamp_ns = 1000,
    };

    const sse = try SSEFormatter.formatTokenEvent(allocator, event);
    defer allocator.free(sse);

    try std.testing.expect(std.mem.indexOf(u8, sse, "data:") != null);
    try std.testing.expect(std.mem.indexOf(u8, sse, "\"token_id\":42") != null);
    try std.testing.expect(std.mem.indexOf(u8, sse, "\"text\":\"hello\"") != null);
}

test "cancellation" {
    const allocator = std.testing.allocator;

    var gen = StreamingGenerator.init(allocator, .{});
    defer gen.deinit();

    try std.testing.expect(!gen.isCancelled());
    gen.cancel();
    try std.testing.expect(gen.isCancelled());
    gen.reset();
    try std.testing.expect(!gen.isCancelled());
}

test "streaming config defaults" {
    const config = StreamingConfig{};

    try std.testing.expectEqual(@as(u32, 256), config.max_tokens);
    try std.testing.expectEqual(@as(f32, 0.7), config.temperature);
    try std.testing.expectEqual(@as(u32, 40), config.top_k);
    try std.testing.expectEqual(@as(f32, 0.9), config.top_p);
    try std.testing.expectEqual(@as(u32, 256), config.initial_buffer_capacity);
    try std.testing.expect(config.decode_tokens);
}

test "streaming config from generator config" {
    const gen_config = generator_mod.GeneratorConfig{
        .max_tokens = 100,
        .temperature = 0.5,
        .top_k = 20,
        .top_p = 0.8,
    };

    const stream_config = StreamingConfig.fromGeneratorConfig(gen_config);

    try std.testing.expectEqual(@as(u32, 100), stream_config.max_tokens);
    try std.testing.expectEqual(@as(f32, 0.5), stream_config.temperature);
    try std.testing.expectEqual(@as(u32, 20), stream_config.top_k);
    try std.testing.expectEqual(@as(f32, 0.8), stream_config.top_p);
}

test "streaming config to generator config" {
    const stream_config = StreamingConfig{
        .max_tokens = 150,
        .temperature = 0.6,
        .top_k = 30,
        .top_p = 0.85,
    };

    const gen_config = stream_config.toGeneratorConfig();

    try std.testing.expectEqual(@as(u32, 150), gen_config.max_tokens);
    try std.testing.expectEqual(@as(f32, 0.6), gen_config.temperature);
    try std.testing.expectEqual(@as(u32, 30), gen_config.top_k);
    try std.testing.expectEqual(@as(f32, 0.85), gen_config.top_p);
}

test "token event creation" {
    const event = TokenEvent{
        .token_id = 123,
        .text = "test",
        .position = 5,
        .is_final = false,
        .timestamp_ns = 1000,
    };

    try std.testing.expectEqual(@as(u32, 123), event.token_id);
    try std.testing.expectEqualStrings("test", event.text.?);
    try std.testing.expectEqual(@as(u32, 5), event.position);
    try std.testing.expect(!event.is_final);
}

test "streaming state transitions" {
    // Test that states are properly defined
    const states = [_]StreamingState{
        .idle,
        .prefilling,
        .generating,
        .completed,
        .cancelled,
        .errored,
    };

    try std.testing.expectEqual(@as(usize, 6), states.len);
}

test "sse formatter with special characters" {
    const allocator = std.testing.allocator;

    const event = TokenEvent{
        .token_id = 1,
        .text = "hello\nworld\t\"test\"",
        .position = 0,
        .is_final = false,
        .timestamp_ns = 0,
    };

    const sse = try SSEFormatter.formatTokenEvent(allocator, event);
    defer allocator.free(sse);

    // Verify JSON escaping
    try std.testing.expect(std.mem.indexOf(u8, sse, "\\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, sse, "\\t") != null);
    try std.testing.expect(std.mem.indexOf(u8, sse, "\\\"") != null);
}

test "sse formatter completion event" {
    const allocator = std.testing.allocator;

    const stats = StreamingStats{
        .tokens_generated = 50,
        .prefill_time_ns = 10_000_000, // 10ms
        .generation_time_ns = 500_000_000, // 500ms
        .time_to_first_token_ns = 20_000_000, // 20ms
        .prompt_tokens = 10,
    };

    const sse = try SSEFormatter.formatCompletionEvent(allocator, stats);
    defer allocator.free(sse);

    try std.testing.expect(std.mem.indexOf(u8, sse, "\"event\":\"complete\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, sse, "\"tokens_generated\":50") != null);
}

test "streaming callbacks struct" {
    var called = false;

    const callbacks = StreamingCallbacks{
        .on_token = struct {
            fn cb(_: TokenEvent) void {}
        }.cb,
        .on_complete = null,
        .on_error = null,
    };

    // Verify callback can be called
    if (callbacks.on_token) |cb| {
        cb(.{ .token_id = 0, .text = null, .position = 0, .is_final = false, .timestamp_ns = 0 });
        called = true;
    }

    try std.testing.expect(called);
}
