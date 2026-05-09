const std = @import("std");
const shared = @import("../shared.zig");
const contract = @import("contract.zig");

pub const OpenAIError = error{
    MissingApiKey,
    ApiRequestFailed,
    InvalidResponse,
    RateLimitExceeded,
};

pub const Config = struct {
    api_key: []u8,
    base_url: []u8,
    model: []const u8 = "gpt-4",
    model_owned: bool = false,
    timeout_ms: u32 = 60_000,

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        shared.deinitConfig(allocator, self.api_key, self.base_url);
        if (self.model_owned) allocator.free(@constCast(self.model));
        self.* = undefined;
    }
};

pub const Message = struct {
    role: []const u8,
    content: []const u8,
};

pub const ChatCompletionRequest = struct {
    model: []const u8,
    messages: []Message,
    temperature: f32 = 0.7,
    max_tokens: ?u32 = null,
    stream: bool = false,
};

pub const ChatCompletionResponse = struct {
    id: []const u8,
    object: []const u8,
    created: u64,
    model: []const u8,
    choices: []Choice,
    usage: Usage,
};

pub const Choice = struct {
    index: u32,
    message: Message,
    finish_reason: []const u8,
};

pub const Usage = struct {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
};

pub const StreamingChunk = struct {
    id: []const u8,
    object: []const u8,
    created: u64,
    model: []const u8,
    choices: []StreamingChoice,
    delta: ?StreamingDelta = null,
};

pub const StreamingChoice = struct {
    index: u32,
    delta: ?StreamingDelta,
    finish_reason: ?[]const u8 = null,
};

pub const StreamingDelta = struct {
    role: ?[]const u8 = null,
    content: ?[]const u8 = null,
};

pub const Client = struct {
    allocator: std.mem.Allocator,

    pub fn init(_: std.mem.Allocator, _: Config) !Client {
        return contract.disabled(Client);
    }

    pub fn deinit(_: *Client) void {}

    pub fn chatCompletion(_: *Client, _: ChatCompletionRequest) !ChatCompletionResponse {
        return contract.disabled(ChatCompletionResponse);
    }

    pub fn chat(_: *Client, _: []Message) !ChatCompletionResponse {
        return contract.disabled(ChatCompletionResponse);
    }

    pub fn chatSimple(_: *Client, _: []const u8) !ChatCompletionResponse {
        return contract.disabled(ChatCompletionResponse);
    }
};

pub fn loadFromEnv(_: std.mem.Allocator) !Config {
    return contract.disabled(Config);
}

pub fn createClient(_: std.mem.Allocator) !Client {
    return contract.disabled(Client);
}

pub fn isAvailable() bool {
    return contract.unavailable();
}

test {
    std.testing.refAllDecls(@This());
}
