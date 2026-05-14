const std = @import("std");
const shared = @import("../shared.zig");

pub const GrokError = shared.ProviderError || error{
    MissingApiKey,
};

pub const Config = struct {
    api_key: []u8,
    base_url: []u8,
    model: []const u8 = "grok-2",
    model_owned: bool = false,
    timeout_ms: u32 = 60_000,

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        shared.deinitConfig(allocator, self.api_key, self.base_url);
        if (self.model_owned) allocator.free(@constCast(self.model));
        self.* = undefined;
    }
};

pub const Message = shared.ChatMessage;

pub const ChatCompletionRequest = struct {
    model: []const u8,
    messages: []const Message,
    temperature: f32 = 0.7,
    top_p: f32 = 0.95,
    max_tokens: ?u32 = null,
    stream: bool = false,
};

pub const Usage = struct {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
};

pub const MessageDelta = struct {
    role: ?[]const u8 = null,
    content: ?[]const u8 = null,
};

pub const Choice = struct {
    index: u32,
    message: Message,
    finish_reason: ?[]const u8 = null,
};

pub const StreamingChoice = struct {
    index: u32,
    delta: MessageDelta,
    finish_reason: ?[]const u8 = null,
};

pub const ChatCompletionResponse = struct {
    id: []const u8,
    object: []const u8,
    created: u64,
    model: []const u8,
    choices: []Choice,
    usage: Usage,
};

pub const StreamingChunk = struct {
    id: []const u8,
    object: []const u8,
    created: u64,
    model: []const u8,
    choices: []StreamingChoice,
    delta: ?MessageDelta = null,
};

pub const Client = struct {
    allocator: std.mem.Allocator,

    pub fn init(_: std.mem.Allocator, _: Config) !Client {
        return error.ConnectorsDisabled;
    }

    pub fn deinit(_: *Client) void {}

    pub fn chatSimple(_: *Client, _: []const u8) ![]const u8 {
        return error.ConnectorsDisabled;
    }

    pub fn chatComplete(_: *Client, _: ChatCompletionRequest) !ChatCompletionResponse {
        return error.ConnectorsDisabled;
    }
};

pub fn loadFromEnv(_: std.mem.Allocator) !Config {
    return error.ConnectorsDisabled;
}

pub fn createClient(_: std.mem.Allocator) !Client {
    return error.ConnectorsDisabled;
}

pub fn isAvailable() bool {
    return false;
}

test {
    std.testing.refAllDecls(@This());
}
