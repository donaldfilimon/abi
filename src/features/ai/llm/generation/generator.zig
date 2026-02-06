//! Text generator for autoregressive generation.

const std = @import("std");
const time = @import("../../../../services/shared/time.zig");
const sync = @import("../../../../services/shared/sync.zig");
const sampler_mod = @import("sampler.zig");
const tokenizer = @import("../tokenizer/mod.zig");

/// Error set for model forward pass operations.
pub const ForwardError = error{
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
};

/// Function pointer type for model forward pass.
pub const ForwardFn = *const fn (*anyopaque, u32, u32) ForwardError![]f32;

/// Generator configuration.
pub const GeneratorConfig = struct {
    /// Maximum tokens to generate
    max_tokens: u32 = 256,
    /// Stop generation on these token IDs
    stop_tokens: []const u32 = &[_]u32{2}, // Default EOS
    /// Sampling configuration
    temperature: f32 = 0.7,
    top_k: u32 = 40,
    top_p: f32 = 0.9,
    repetition_penalty: f32 = 1.1,
    /// Random seed
    seed: u64 = 0,
};

/// Generation result with statistics.
pub const GenerationResult = struct {
    /// Generated token IDs
    tokens: []u32,
    /// Generated text (if decoded)
    text: ?[]u8,
    /// Number of prompt tokens processed
    prompt_tokens: u32,
    /// Number of tokens generated
    generated_tokens: u32,
    /// Time for prefill (nanoseconds)
    prefill_time_ns: u64,
    /// Time for generation (nanoseconds)
    generation_time_ns: u64,

    pub fn deinit(self: *GenerationResult, allocator: std.mem.Allocator) void {
        allocator.free(self.tokens);
        if (self.text) |t| {
            allocator.free(t);
        }
        self.* = undefined;
    }

    /// Tokens per second during generation.
    pub fn tokensPerSecond(self: GenerationResult) f64 {
        if (self.generation_time_ns == 0) return 0;
        return @as(f64, @floatFromInt(self.generated_tokens)) /
            (@as(f64, @floatFromInt(self.generation_time_ns)) / 1e9);
    }

    /// Print summary.
    pub fn printSummary(self: GenerationResult, writer: anytype) !void {
        try writer.print("Generation Summary:\n", .{});
        try writer.print("  Prompt tokens: {d}\n", .{self.prompt_tokens});
        try writer.print("  Generated tokens: {d}\n", .{self.generated_tokens});
        try writer.print("  Prefill time: {d:.2}ms\n", .{@as(f64, @floatFromInt(self.prefill_time_ns)) / 1e6});
        try writer.print("  Generation time: {d:.2}ms\n", .{@as(f64, @floatFromInt(self.generation_time_ns)) / 1e6});
        try writer.print("  Speed: {d:.1} tok/s\n", .{self.tokensPerSecond()});
    }
};

/// Text generator wrapping a model.
pub const Generator = struct {
    allocator: std.mem.Allocator,
    model: *anyopaque, // LlamaModel pointer
    forward_fn: ForwardFn,
    config: GeneratorConfig,
    sampler: sampler_mod.Sampler,
    streaming_callback: ?*const fn ([]const u8) void,

    pub fn init(allocator: std.mem.Allocator, model: anytype, config: GeneratorConfig) Generator {
        const Model = @TypeOf(model);
        const Ptr = if (@typeInfo(Model) == .pointer) Model else *Model;

        return .{
            .allocator = allocator,
            .model = @ptrCast(@constCast(model)),
            .forward_fn = &struct {
                fn forward(m: *anyopaque, token: u32, pos: u32) ForwardError![]f32 {
                    const model_ptr: Ptr = @ptrCast(@alignCast(m));
                    return model_ptr.forward(token, pos);
                }
            }.forward,
            .config = config,
            .sampler = sampler_mod.Sampler.init(allocator, .{
                .temperature = config.temperature,
                .top_k = config.top_k,
                .top_p = config.top_p,
                .repetition_penalty = config.repetition_penalty,
                .seed = config.seed,
            }),
            .streaming_callback = null,
        };
    }

    pub fn deinit(self: *Generator) void {
        self.sampler.deinit();
        self.* = undefined;
    }

    /// Set streaming callback for token-by-token output.
    pub fn setStreamingCallback(self: *Generator, callback: *const fn ([]const u8) void) void {
        self.streaming_callback = callback;
    }

    /// Generate tokens from prompt tokens.
    pub fn generateTokens(self: *Generator, prompt_tokens: []const u32) ![]u32 {
        var output = std.ArrayListUnmanaged(u32).empty;
        errdefer output.deinit(self.allocator);

        // Prefill: process prompt tokens
        var pos: u32 = 0;
        for (prompt_tokens) |token| {
            _ = try self.forward_fn(self.model, token, pos);
            pos += 1;
        }

        // Generate new tokens
        var last_token = prompt_tokens[prompt_tokens.len - 1];
        var generated: u32 = 0;

        while (generated < self.config.max_tokens) {
            // Forward pass
            const logits = try self.forward_fn(self.model, last_token, pos);

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
            if (should_stop) break;

            try output.append(self.allocator, next_token);
            last_token = next_token;
            pos += 1;
            generated += 1;
        }

        return output.toOwnedSlice(self.allocator);
    }

    /// Generate tokens from prompt tokens with streaming callback.
    pub fn generateTokensStreaming(
        self: *Generator,
        prompt_tokens: []const u32,
        tok: *tokenizer.Tokenizer,
        callback: *const fn ([]const u8) void,
    ) ![]u32 {
        var output = std.ArrayListUnmanaged(u32).empty;
        errdefer output.deinit(self.allocator);

        // Prefill: process prompt tokens
        var pos: u32 = 0;
        for (prompt_tokens) |token| {
            _ = try self.forward_fn(self.model, token, pos);
            pos += 1;
        }

        // Generate new tokens
        var last_token = prompt_tokens[prompt_tokens.len - 1];
        var generated: u32 = 0;

        while (generated < self.config.max_tokens) {
            // Forward pass
            const logits = try self.forward_fn(self.model, last_token, pos);

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
            if (should_stop) break;

            try output.append(self.allocator, next_token);
            last_token = next_token;
            pos += 1;
            generated += 1;

            // Streaming callback - decode single token and call
            const token_slice = try self.allocator.alloc(u32, 1);
            defer self.allocator.free(token_slice);
            token_slice[0] = next_token;

            const token_text = try tok.decode(self.allocator, token_slice);
            defer self.allocator.free(token_text);

            callback(token_text);
        }

        return output.toOwnedSlice(self.allocator);
    }

    /// Generate text from a text prompt.
    pub fn generate(self: *Generator, prompt: []const u8, tok: *tokenizer.Tokenizer) !GenerationResult {
        var prefill_timer = time.Timer.start() catch return error.TimerFailed;

        // Encode prompt
        const prompt_tokens = try tok.encode(self.allocator, prompt);
        defer self.allocator.free(prompt_tokens);

        // Generate
        const prefill_time_ns = prefill_timer.read();
        var gen_timer = time.Timer.start() catch return error.TimerFailed;
        const output_tokens = try self.generateTokens(prompt_tokens);
        const generation_time_ns = gen_timer.read();

        // Decode output
        const text = try tok.decode(self.allocator, output_tokens);

        return .{
            .tokens = output_tokens,
            .text = text,
            .prompt_tokens = @intCast(prompt_tokens.len),
            .generated_tokens = @intCast(output_tokens.len),
            .prefill_time_ns = @intCast(prefill_time_ns),
            .generation_time_ns = @intCast(generation_time_ns),
        };
    }

    /// Generate with streaming output.
    pub fn generateStreaming(
        self: *Generator,
        prompt: []const u8,
        callback: *const fn ([]const u8) void,
    ) !void {
        self.streaming_callback = callback;
        defer self.streaming_callback = null;

        _ = prompt; // Requires tokenizer integration
    }

    /// Reset generator state.
    pub fn reset(self: *Generator) void {
        self.sampler.reset();
    }
};

/// Simple streaming helper.
pub fn streamToStdout(text: []const u8) void {
    std.debug.print("{s}", .{text});
}

test "generator config defaults" {
    const config = GeneratorConfig{};
    try std.testing.expectEqual(@as(u32, 256), config.max_tokens);
    try std.testing.expectEqual(@as(f32, 0.7), config.temperature);
}

test "generation result stats" {
    var result = GenerationResult{
        .tokens = &[_]u32{ 1, 2, 3 },
        .text = null,
        .prompt_tokens = 10,
        .generated_tokens = 50,
        .prefill_time_ns = 100_000_000, // 100ms
        .generation_time_ns = 1_000_000_000, // 1s
    };

    // 50 tokens / 1 second = 50 tok/s
    try std.testing.expectEqual(@as(f64, 50.0), result.tokensPerSecond());
}
