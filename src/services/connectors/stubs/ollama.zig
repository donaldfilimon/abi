const std = @import("std");

pub const Config = struct {
    host: []u8,
    model: []const u8 = "gpt-oss",
    model_owned: bool = false,
    timeout_ms: u32 = 120_000,

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        allocator.free(self.host);
        if (self.model_owned) {
            allocator.free(@constCast(self.model));
        }
        self.* = undefined;
    }
};

pub const Message = struct {
    role: []const u8,
    content: []const u8,
};

pub const GenerateRequest = struct {
    model: []const u8,
    prompt: []const u8,
    stream: bool = false,
    options: ?Options = null,
};

pub const ChatRequest = struct {
    model: []const u8,
    messages: []Message,
    stream: bool = false,
};

pub const Options = struct {
    temperature: f32 = 0.7,
    num_predict: u32 = 128,
    top_p: f32 = 0.9,
    top_k: u32 = 40,
};

pub const GenerateResponse = struct {
    model: []const u8,
    response: []const u8,
    done: bool,
    context: ?[]u64 = null,

    pub fn deinit(_: *GenerateResponse, _: std.mem.Allocator) void {}
};

pub const ChatResponse = struct {
    model: []const u8,
    message: Message,
    done: bool,

    pub fn deinit(_: *ChatResponse, _: std.mem.Allocator) void {}
};

pub const Client = struct {
    allocator: std.mem.Allocator,

    pub fn init(_: std.mem.Allocator, _: Config) !Client {
        return error.ConnectorsDisabled;
    }

    pub fn deinit(_: *Client) void {}

    pub fn generate(_: *Client, _: GenerateRequest) !GenerateResponse {
        return error.ConnectorsDisabled;
    }

    pub fn generateSimple(_: *Client, _: []const u8) !GenerateResponse {
        return error.ConnectorsDisabled;
    }

    pub fn chat(_: *Client, _: ChatRequest) !ChatResponse {
        return error.ConnectorsDisabled;
    }

    pub fn chatSimple(_: *Client, _: []const u8) !ChatResponse {
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
