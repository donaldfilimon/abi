const std = @import("std");
const shared = @import("../shared.zig");

pub const HuggingFaceError = error{
    MissingApiToken,
    ModelLoading,
    ApiRequestFailed,
    InvalidResponse,
    RateLimitExceeded,
};

pub const Config = struct {
    api_token: []u8,
    base_url: []u8,
    model: []const u8 = "gpt2",
    model_owned: bool = false,
    timeout_ms: u32 = 60_000,

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        shared.secureFree(allocator, self.api_token);
        allocator.free(self.base_url);
        if (self.model_owned) allocator.free(@constCast(self.model));
        self.* = undefined;
    }
};

pub const InferenceRequest = struct {
    inputs: ?[]const u8 = null,
    parameters: ?Parameters = null,
};

pub const Parameters = struct {
    top_k: ?u32 = null,
    top_p: ?f32 = null,
    temperature: ?f32 = null,
    max_new_tokens: ?u32 = null,
    return_full_text: ?bool = null,
};

pub const InferenceResponse = struct {
    generated_text: []const u8,
};

pub const TextGenerationRequest = struct {
    inputs: []const u8,
    parameters: ?Parameters = null,
};

pub const Client = struct {
    allocator: std.mem.Allocator,

    pub fn init(_: std.mem.Allocator, _: Config) !Client {
        return error.ConnectorsDisabled;
    }

    pub fn deinit(_: *Client) void {}

    pub fn inference(_: *Client, _: InferenceRequest) !InferenceResponse {
        return error.ConnectorsDisabled;
    }

    pub fn generateText(_: *Client, _: []const u8, _: ?Parameters) !InferenceResponse {
        return error.ConnectorsDisabled;
    }

    pub fn generateTextSimple(_: *Client, _: []const u8) !InferenceResponse {
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
