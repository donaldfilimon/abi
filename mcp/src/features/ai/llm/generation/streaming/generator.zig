const std = @import("std");
const types = @import("types.zig");
const time = @import("../../../../../foundation/mod.zig").time;
const sampler_mod = @import("../sampler.zig");
const tokenizer = @import("../../tokenizer/mod.zig");
const generator_mod = @import("../generator.zig");

pub const StreamingGenerator = struct {
    allocator: std.mem.Allocator,
    config: generator_mod.GeneratorConfig,
    sampler: sampler_mod.Sampler,
    state: types.StreamingState,
    callbacks: types.StreamingCallbacks,

    /// Buffer for accumulating tokens
    token_buffer: std.ArrayListUnmanaged(u32),

    /// Statistics
    stats: types.StreamingStats,

    /// Start time for timing
    start_time: ?time.Timer,

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
            .stats = std.mem.zeroes(types.StreamingStats),
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
    pub fn setCallbacks(self: *StreamingGenerator, callbacks: types.StreamingCallbacks) void {
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
        getLogits: *const fn (u32, u32) types.StreamingError![]f32,
        prompt_tokens: []const u32,
        tok: ?*tokenizer.Tokenizer,
    ) types.StreamingError!void {
        self.state = .prefilling;
        self.cancel_requested.store(false, .seq_cst);
        self.token_buffer.clearRetainingCapacity();
        self.stats = std.mem.zeroes(types.StreamingStats);
        self.stats.prompt_tokens = @intCast(prompt_tokens.len);

        self.start_time = time.Timer.start() catch null;

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
    pub fn getState(self: *const StreamingGenerator) types.StreamingState {
        return self.state;
    }

    /// Get current statistics.
    pub fn getStats(self: *const StreamingGenerator) types.StreamingStats {
        return self.stats;
    }

    /// Reset generator for new generation.
    pub fn reset(self: *StreamingGenerator) void {
        self.state = .idle;
        self.cancel_requested.store(false, .seq_cst);
        self.token_buffer.clearRetainingCapacity();
        self.sampler.reset();
        self.stats = std.mem.zeroes(types.StreamingStats);
        self.start_time = null;
    }
};

/// Display function: logs streaming token text via std.log.
pub fn streamToStdout(event: types.TokenEvent) void {
    if (event.text) |text| {
        std.log.info("{s}", .{text});
    }
}

/// Write streaming token text to an arbitrary writer.
pub fn streamEventToWriter(event: types.TokenEvent, writer: anytype) !void {
    if (event.text) |text| {
        try writer.writeAll(text);
    }
}

/// Display function: logs generation stats via std.log.
pub fn printCompletionStats(stats: types.StreamingStats) void {
    std.log.info("--- Generation Complete ---", .{});
    std.log.info("Tokens: {d}", .{stats.tokens_generated});
    std.log.info("Speed: {d:.1} tok/s", .{stats.tokensPerSecond()});
    std.log.info("Time to first token: {d:.1}ms", .{stats.timeToFirstTokenMs()});
}

/// Write generation stats to an arbitrary writer.
pub fn writeCompletionStats(stats: types.StreamingStats, writer: anytype) !void {
    try writer.print("\n\n--- Generation Complete ---\n", .{});
    try writer.print("Tokens: {d}\n", .{stats.tokens_generated});
    try writer.print("Speed: {d:.1} tok/s\n", .{stats.tokensPerSecond()});
    try writer.print("Time to first token: {d:.1}ms\n", .{stats.timeToFirstTokenMs()});
}
