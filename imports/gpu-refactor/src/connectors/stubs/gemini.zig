const std = @import("std");
const shared = @import("../shared.zig");
const contract = @import("contract.zig");

pub const GeminiError = error{
    MissingApiKey,
    ApiRequestFailed,
    InvalidResponse,
    RateLimitExceeded,
};

pub const Config = struct {
    api_key: []u8,
    base_url: []u8,
    model: []const u8 = "gemini-1.5-pro",
    model_owned: bool = false,
    timeout_ms: u32 = 120_000,

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        shared.deinitConfig(allocator, self.api_key, self.base_url);
        if (self.model_owned) allocator.free(@constCast(self.model));
        self.* = undefined;
    }
};

pub const Message = shared.ChatMessage;

pub const GenerateRequest = struct {
    model: []const u8,
    messages: []const Message,
    temperature: f32 = 0.7,
    max_output_tokens: ?u32 = null,
    top_p: f32 = 0.95,
    system_prompt: ?[]const u8 = null,
};

pub const GenerateResponse = struct {
    model: []u8,
    text: []u8,

    pub fn deinit(self: *GenerateResponse, allocator: std.mem.Allocator) void {
        allocator.free(self.model);
        allocator.free(self.text);
        self.* = undefined;
    }
};

pub const Client = struct {
    allocator: std.mem.Allocator,

    pub fn init(_: std.mem.Allocator, _: Config) !Client {
        return contract.disabled(Client);
    }

    pub fn deinit(_: *Client) void {}

    pub fn generate(_: *Client, _: GenerateRequest) !GenerateResponse {
        return contract.disabled(GenerateResponse);
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
