//! Async streaming generation support.
//!
//! Provides asynchronous token generation with callback-based streaming,
//! compatible with web servers and interactive applications.

const std = @import("std");
const sampler_mod = @import("sampler.zig");
const tokenizer = @import("../tokenizer/mod.zig");
const generator_mod = @import("generator.zig");

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
    on_error: ?*const fn (anyerror) void = null,
    /// User context.
    user_data: ?*anyopaque = null,
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
        getLogits: *const fn (u32, u32) anyerror![]f32,
        prompt_tokens: []const u32,
        tok: ?*tokenizer.BpeTokenizer,
    ) !void {
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
            self.token_buffer.append(self.allocator, next_token) catch {};
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
    pub fn formatErrorEvent(allocator: std.mem.Allocator, err: anyerror) ![]u8 {
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
