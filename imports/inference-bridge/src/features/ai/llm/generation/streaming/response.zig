const std = @import("std");
const types = @import("types.zig");
const time = @import("../../../../../foundation/mod.zig").time;
const sampler_mod = @import("../sampler.zig");
const tokenizer = @import("../../tokenizer/mod.zig");

/// Iterator-based streaming response.
pub const StreamingResponse = struct {
    allocator: std.mem.Allocator,
    config: types.StreamingConfig,
    sampler: sampler_mod.Sampler,
    state: types.StreamingState,
    tok: ?*tokenizer.Tokenizer,

    // Model forward function
    model: *anyopaque,
    forward_fn: *const fn (*anyopaque, u32, u32) types.StreamingError![]f32,

    // Generation state
    prompt_tokens: []const u32,
    current_position: u32,
    last_token: u32,
    tokens_generated: u32,

    // Buffer for generated tokens
    token_buffer: std.ArrayListUnmanaged(u32),

    // Timing
    start_time: ?time.Timer,
    first_token_time: ?u64,
    stats: types.StreamingStats,

    // Cancellation
    cancel_requested: std.atomic.Value(bool),

    // Last decoded text (owned, must be freed)
    last_text: ?[]u8,

    /// Initialize streaming response.
    pub fn init(
        allocator: std.mem.Allocator,
        model: anytype,
        prompt_tokens: []const u32,
        config: types.StreamingConfig,
        tok: ?*tokenizer.Tokenizer,
    ) types.StreamingError!StreamingResponse {
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
                fn forward(m: *anyopaque, token: u32, pos: u32) types.StreamingError![]f32 {
                    const model_ptr: Ptr = @ptrCast(@alignCast(m));
                    return model_ptr.forward(token, pos) catch |e| {
                        return if (e == error.OutOfMemory) types.StreamingError.OutOfMemory else types.StreamingError.InvalidState;
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
            .stats = std.mem.zeroes(types.StreamingStats),
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
    pub fn next(self: *StreamingResponse) types.StreamingError!?types.TokenEvent {
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
                self.start_time = time.Timer.start() catch return types.StreamingError.TimerFailed;
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
                return types.TokenEvent{
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
                self.token_buffer.append(self.allocator, next_token) catch return types.StreamingError.OutOfMemory;
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
                const event = types.TokenEvent{
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
    fn finalize(self: *StreamingResponse) ?types.TokenEvent {
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

        return types.TokenEvent{
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
    pub fn getState(self: *const StreamingResponse) types.StreamingState {
        return self.state;
    }

    /// Get statistics.
    pub fn getStats(self: *const StreamingResponse) types.StreamingStats {
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
        self.stats = std.mem.zeroes(types.StreamingStats);
    }
};

/// Collect all tokens from a streaming response.
pub fn collectStreamingResponse(
    allocator: std.mem.Allocator,
    response: *StreamingResponse,
) !struct { text: ?[]u8, stats: types.StreamingStats } {
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
