const std = @import("std");
const shared = @import("../shared.zig");

pub const MistralError = error{
    MissingApiKey,
    ApiRequestFailed,
    InvalidResponse,
    RateLimitExceeded,
};

pub const Config = struct {
    api_key: []u8,
    base_url: []u8,
    model: []const u8 = "mistral-large-latest",
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
    messages: []const Message,
    temperature: f32 = 0.7,
    max_tokens: ?u32 = null,
    top_p: f32 = 1.0,
    stream: bool = false,
    safe_prompt: bool = false,
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

pub const EmbeddingRequest = struct {
    model: []const u8 = "mistral-embed",
    input: []const []const u8,
    encoding_format: []const u8 = "float",
};

pub const EmbeddingResponse = struct {
    id: []const u8,
    object: []const u8,
    model: []const u8,
    data: []EmbeddingData,
    usage: EmbeddingUsage,
};

pub const EmbeddingData = struct {
    object: []const u8,
    index: u32,
    embedding: []f32,
};

pub const EmbeddingUsage = struct {
    prompt_tokens: u32,
    total_tokens: u32,
};

pub const Client = struct {
    allocator: std.mem.Allocator,

    pub fn init(_: std.mem.Allocator, _: Config) !Client {
        return error.ConnectorsDisabled;
    }

    pub fn deinit(_: *Client) void {}

    pub fn chatCompletion(_: *Client, _: ChatCompletionRequest) !ChatCompletionResponse {
        return error.ConnectorsDisabled;
    }

    pub fn chat(_: *Client, _: []const Message) !ChatCompletionResponse {
        return error.ConnectorsDisabled;
    }

    pub fn chatSimple(_: *Client, _: []const u8) !ChatCompletionResponse {
        return error.ConnectorsDisabled;
    }

    pub fn embeddings(_: *Client, _: EmbeddingRequest) !EmbeddingResponse {
        return error.ConnectorsDisabled;
    }

    pub fn embed(_: *Client, _: []const []const u8) !EmbeddingResponse {
        return error.ConnectorsDisabled;
    }

    pub fn embedSingle(_: *Client, _: []const u8) ![]f32 {
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
