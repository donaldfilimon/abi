//! Streaming response API for AI models.
//!
//! Provides support for streaming token-by-token responses from AI models,
//! enabling real-time output display and reduced perceived latency.

const std = @import("std");
const transformer = @import("transformer/mod.zig");

pub const StreamingError = error{
    StreamClosed,
    InvalidState,
    GenerationFailed,
};

pub const StreamState = enum {
    idle,
    generating,
    paused,
    completed,
    failed,
};

pub const StreamToken = struct {
    id: u32,
    text: []const u8,
    log_prob: ?f32 = null,
    is_end: bool = false,
};

pub const GenerationConfig = struct {
    max_tokens: u32 = 256,
    temperature: f32 = 0.8,
    top_p: f32 = 0.9,
    top_k: u32 = 40,
    repeat_penalty: f32 = 1.1,
    presence_penalty: f32 = 0.0,
    frequency_penalty: f32 = 0.0,
    stop_tokens: []const []const u8 = &.{},
};

pub const StreamingGenerator = struct {
    allocator: std.mem.Allocator,
    model: *transformer.TransformerModel,
    config: GenerationConfig,
    state: StreamState,
    generated_tokens: std.ArrayListUnmanaged(u32),
    current_token_id: u32,

    pub fn init(allocator: std.mem.Allocator, model: *transformer.TransformerModel, config: GenerationConfig) StreamingGenerator {
        return .{
            .allocator = allocator,
            .model = model,
            .config = config,
            .state = .idle,
            .generated_tokens = .{},
            .current_token_id = 0,
        };
    }

    pub fn deinit(self: *StreamingGenerator) void {
        self.generated_tokens.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn start(self: *StreamingGenerator, prompt: []const u8) !void {
        if (self.state != .idle) {
            return StreamingError.InvalidState;
        }

        const tokens = try self.model.encode(self.allocator, prompt);
        defer self.allocator.free(tokens);

        try self.generated_tokens.appendSlice(self.allocator, tokens);
        self.state = .generating;
        self.current_token_id = 0;
    }

    pub fn next(self: *StreamingGenerator) !?StreamToken {
        if (self.state != .generating) {
            if (self.state == .completed) {
                return null;
            }
            return StreamingError.InvalidState;
        }

        if (self.current_token_id >= self.config.max_tokens) {
            self.state = .completed;
            return null;
        }

        const logits = try self.model.forward(self.allocator, self.generated_tokens.items);
        defer self.allocator.free(logits);

        const token = self.model.sampleToken(logits);
        const decoded = decodeToken(token);

        self.current_token_id += 1;
        try self.generated_tokens.append(self.allocator, token);

        const is_stop = self.checkStopToken(decoded);
        if (is_stop or token == 0) {
            self.state = .completed;
            return StreamToken{
                .id = self.current_token_id,
                .text = decoded,
                .is_end = true,
            };
        }

        return StreamToken{
            .id = self.current_token_id,
            .text = decoded,
        };
    }

    pub fn pause(self: *StreamingGenerator) void {
        if (self.state == .generating) {
            self.state = .paused;
        }
    }

    pub fn resumeGeneration(self: *StreamingGenerator) void {
        if (self.state == .paused) {
            self.state = .generating;
        }
    }

    pub fn cancel(self: *StreamingGenerator) void {
        self.state = .idle;
        self.generated_tokens.clearRetainingCapacity();
        self.current_token_id = 0;
    }

    pub fn reset(self: *StreamingGenerator, new_config: GenerationConfig) void {
        self.cancel();
        self.config = new_config;
    }

    pub fn getGeneratedText(self: *StreamingGenerator, allocator: std.mem.Allocator) ![]u8 {
        return self.model.decode(allocator, self.generated_tokens.items);
    }

    pub fn tokenCount(self: *const StreamingGenerator) usize {
        return self.generated_tokens.items.len;
    }

    pub fn isComplete(self: *const StreamingGenerator) bool {
        return self.state == .completed;
    }

    fn checkStopToken(self: *StreamingGenerator, token: []const u8) bool {
        for (self.config.stop_tokens) |stop| {
            if (std.mem.startsWith(u8, token, stop)) {
                return true;
            }
        }
        return false;
    }
};

pub fn streamInference(
    allocator: std.mem.Allocator,
    model: *transformer.TransformerModel,
    prompt: []const u8,
    config: GenerationConfig,
    callback: anytype,
) !void {
    var generator = StreamingGenerator.init(allocator, model, config);
    defer generator.deinit();

    try generator.start(prompt);

    while (try generator.next()) |token| {
        try callback(token);
        if (token.is_end) break;
    }
}

pub fn formatStreamOutput(tokens: []const StreamToken, allocator: std.mem.Allocator) ![]u8 {
    var output = std.ArrayListUnmanaged(u8).empty;
    errdefer output.deinit(allocator);

    for (tokens) |token| {
        try output.appendSlice(allocator, token.text);
    }

    return output.toOwnedSlice(allocator);
}

pub fn createChunkedStream(
    allocator: std.mem.Allocator,
    tokens: []const StreamToken,
    chunk_size: usize,
) ![]const []const u8 {
    if (tokens.len == 0) {
        const empty = try allocator.alloc([]const u8, 0);
        return empty;
    }

    const num_chunks = (tokens.len + chunk_size - 1) / chunk_size;
    const chunks = try allocator.alloc([]const u8, num_chunks);

    for (0..num_chunks) |i| {
        const start = i * chunk_size;
        const end = @min(start + chunk_size, tokens.len);
        var chunk_text = std.ArrayListUnmanaged(u8).empty;
        errdefer chunk_text.deinit(allocator);

        for (tokens[start..end]) |token| {
            try chunk_text.appendSlice(allocator, token.text);
        }

        chunks[i] = try chunk_text.toOwnedSlice(allocator);
    }

    return chunks;
}

/// Static buffer for single-byte token decoding to avoid dangling pointer
const single_byte_tokens: [256][1]u8 = blk: {
    var tokens: [256][1]u8 = undefined;
    for (0..256) |i| {
        tokens[i] = .{@as(u8, @intCast(i))};
    }
    break :blk tokens;
};

fn decodeToken(token: u32) []const u8 {
    if (token == 0) return "";
    if (token == 1) return "<unk>";
    if (token < 256) {
        return &single_byte_tokens[token];
    }
    return "<token>";
}

test "streaming generator initialization" {
    const allocator = std.testing.allocator;
    var model = transformer.TransformerModel.init(.{
        .layers = 2,
        .hidden_size = 64,
        .vocab_size = 512,
    });

    const config = GenerationConfig{ .max_tokens = 10 };
    var generator = StreamingGenerator.init(allocator, &model, config);
    defer generator.deinit();

    try std.testing.expectEqual(.idle, generator.state);
    try std.testing.expectEqual(@as(usize, 0), generator.tokenCount());
}

test "streaming generator text generation" {
    const allocator = std.testing.allocator;
    var model = try transformer.TransformerModel.init(allocator, .{
        .layers = 2,
        .hidden_size = 64,
        .num_heads = 4,
        .vocab_size = 256,
        .max_tokens = 32,
        .seed = 12345,
    });
    defer model.deinit();

    var generator = StreamingGenerator.init(allocator, &model, .{
        .max_tokens = 5,
    });
    defer generator.deinit();

    try generator.start("hello");

    var token_count: usize = 0;
    while (try generator.next()) |token| : (token_count += 1) {
        try std.testing.expect(token.id == token_count + 1);
        try std.testing.expect(token.text.len > 0);
    }

    try std.testing.expect(generator.isComplete());
    try std.testing.expect(token_count <= 5);
}

test "streaming generator pause and resume" {
    const allocator = std.testing.allocator;
    var model = try transformer.TransformerModel.init(allocator, .{
        .layers = 2,
        .hidden_size = 64,
        .num_heads = 4,
        .vocab_size = 256,
        .max_tokens = 32,
        .seed = 54321,
    });
    defer model.deinit();

    var generator = StreamingGenerator.init(allocator, &model, .{
        .max_tokens = 10,
    });
    defer generator.deinit();

    try generator.start("test");

    const first_token = try generator.next();
    try std.testing.expect(first_token != null);

    generator.pause();
    try std.testing.expectEqual(.paused, generator.state);

    generator.resumeGeneration();
    try std.testing.expectEqual(.generating, generator.state);
}

test "streaming generator cancel" {
    const allocator = std.testing.allocator;
    var model = try transformer.TransformerModel.init(allocator, .{
        .layers = 2,
        .hidden_size = 64,
        .num_heads = 4,
        .vocab_size = 256,
        .max_tokens = 32,
        .seed = 99999,
    });
    defer model.deinit();

    var generator = StreamingGenerator.init(allocator, &model, .{
        .max_tokens = 100,
    });
    defer generator.deinit();

    try generator.start("cancel test");

    const first_token = try generator.next();
    try std.testing.expect(first_token != null);

    generator.cancel();
    try std.testing.expectEqual(.idle, generator.state);
    try std.testing.expectEqual(@as(usize, 0), generator.tokenCount());
}

test "format stream output" {
    const allocator = std.testing.allocator;

    const tokens = [_]StreamToken{
        .{ .id = 1, .text = "hello" },
        .{ .id = 2, .text = " " },
        .{ .id = 3, .text = "world" },
        .{ .id = 4, .text = "!", .is_end = true },
    };

    const output = try formatStreamOutput(&tokens, allocator);
    defer allocator.free(output);

    try std.testing.expectEqualStrings("hello world!", output);
}

test "create chunked stream" {
    const allocator = std.testing.allocator;

    const tokens = [_]StreamToken{
        .{ .id = 1, .text = "a" },
        .{ .id = 2, .text = "b" },
        .{ .id = 3, .text = "c" },
        .{ .id = 4, .text = "d" },
        .{ .id = 5, .text = "e" },
    };

    const chunks = try createChunkedStream(allocator, &tokens, 2);
    defer {
        for (chunks) |chunk| allocator.free(chunk);
        allocator.free(chunks);
    }

    try std.testing.expectEqual(@as(usize, 3), chunks.len);
    try std.testing.expectEqualStrings("ab", chunks[0]);
    try std.testing.expectEqualStrings("cd", chunks[1]);
    try std.testing.expectEqualStrings("e", chunks[2]);
}
