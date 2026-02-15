const std = @import("std");
const shared = @import("../shared.zig");
const contract = @import("contract.zig");

pub const MLXError = error{
    ApiRequestFailed,
    InvalidResponse,
    RateLimitExceeded,
};

pub const Config = struct {
    host: []u8,
    api_key: ?[]u8 = null,
    model: []const u8 = "default",
    model_owned: bool = false,
    timeout_ms: u32 = 120_000,

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        shared.secureFree(allocator, self.host);
        if (self.api_key) |key| shared.secureFree(allocator, key);
        if (self.model_owned) allocator.free(@constCast(self.model));
        self.* = undefined;
    }
};

pub const Message = shared.ChatMessage;

pub const ChatCompletionRequest = struct {
    model: []const u8,
    messages: []const Message,
    temperature: f32 = 0.7,
    max_tokens: ?u32 = null,
    top_p: f32 = 1.0,
    stream: bool = false,
};

pub const ChatCompletionResponse = struct {
    id: []const u8,
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

pub const Client = struct {
    allocator: std.mem.Allocator,

    pub fn init(_: std.mem.Allocator, _: Config) !Client {
        return contract.disabled(Client);
    }

    pub fn deinit(_: *Client) void {}

    pub fn chatCompletion(_: *Client, _: ChatCompletionRequest) !ChatCompletionResponse {
        return contract.disabled(ChatCompletionResponse);
    }

    pub fn chat(_: *Client, _: []const Message) !ChatCompletionResponse {
        return contract.disabled(ChatCompletionResponse);
    }

    pub fn chatSimple(_: *Client, _: []const u8) !ChatCompletionResponse {
        return contract.disabled(ChatCompletionResponse);
    }

    pub fn generate(_: *Client, _: []const u8, _: ?u32) ![]u8 {
        return contract.disabled([]u8);
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
