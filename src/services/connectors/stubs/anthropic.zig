const std = @import("std");
const shared = @import("../shared.zig");

pub const AnthropicError = error{
    MissingApiKey,
    ApiRequestFailed,
    InvalidResponse,
    RateLimitExceeded,
    ContentFiltered,
};

pub const Config = struct {
    api_key: []u8,
    base_url: []u8,
    model: []const u8 = "claude-3-5-sonnet-20241022",
    model_owned: bool = false,
    max_tokens: u32 = 4096,
    timeout_ms: u32 = 120_000,

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

pub const MessagesRequest = struct {
    model: []const u8,
    messages: []const Message,
    max_tokens: u32 = 4096,
    temperature: f32 = 0.7,
    system: ?[]const u8 = null,
    stream: bool = false,
};

pub const ContentBlock = struct {
    type: []const u8,
    text: []const u8,
};

pub const MessagesResponse = struct {
    id: []const u8,
    type: []const u8,
    role: []const u8,
    content: []ContentBlock,
    model: []const u8,
    stop_reason: ?[]const u8,
    usage: Usage,
};

pub const Usage = struct {
    input_tokens: u32,
    output_tokens: u32,
};

pub const EmbeddingRequest = struct {
    model: []const u8 = "voyage-3",
    input: []const []const u8,
    input_type: []const u8 = "document",
};

pub const EmbeddingResponse = struct {
    object: []const u8,
    data: []EmbeddingData,
    model: []const u8,
    usage: EmbeddingUsage,
};

pub const EmbeddingData = struct {
    object: []const u8,
    index: u32,
    embedding: []f32,
};

pub const EmbeddingUsage = struct {
    total_tokens: u32,
};

pub const Client = struct {
    allocator: std.mem.Allocator,

    pub fn init(_: std.mem.Allocator, _: Config) !Client {
        return error.ConnectorsDisabled;
    }

    pub fn deinit(_: *Client) void {}

    pub fn messages(_: *Client, _: MessagesRequest) !MessagesResponse {
        return error.ConnectorsDisabled;
    }

    pub fn chat(_: *Client, _: []const Message) !MessagesResponse {
        return error.ConnectorsDisabled;
    }

    pub fn chatSimple(_: *Client, _: []const u8) !MessagesResponse {
        return error.ConnectorsDisabled;
    }

    pub fn chatWithSystem(_: *Client, _: []const u8, _: []const Message) !MessagesResponse {
        return error.ConnectorsDisabled;
    }

    pub fn getResponseText(_: *Client, _: MessagesResponse) ![]u8 {
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
