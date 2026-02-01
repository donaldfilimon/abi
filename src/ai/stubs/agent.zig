const std = @import("std");

pub const MIN_TEMPERATURE: f32 = 0.0;
pub const MAX_TEMPERATURE: f32 = 2.0;
pub const MIN_TOP_P: f32 = 0.0;
pub const MAX_TOP_P: f32 = 1.0;
pub const MAX_TOKENS_LIMIT: u32 = 128000;
pub const DEFAULT_TEMPERATURE: f32 = 0.7;
pub const DEFAULT_TOP_P: f32 = 0.9;
pub const DEFAULT_MAX_TOKENS: u32 = 1024;

pub const AgentError = error{
    InvalidTemperature,
    InvalidTopP,
    InvalidMaxTokens,
    NoMessages,
    EmptyResponse,
    ConnectionFailed,
    RateLimited,
    AuthenticationFailed,
    ContextLengthExceeded,
    ModelNotAvailable,
    BackendError,
    AiDisabled,
};

pub const AgentBackend = enum {
    openai,
    ollama,
    huggingface,
    local,
};

pub const AgentConfig = struct {
    name: []const u8 = "",
    backend: AgentBackend = .openai,
    model: []const u8 = "gpt-4",
    temperature: f32 = DEFAULT_TEMPERATURE,
    top_p: f32 = DEFAULT_TOP_P,
    max_tokens: u32 = DEFAULT_MAX_TOKENS,
    system_prompt: ?[]const u8 = null,
    enable_history: bool = true,
};

pub const Message = struct {
    role: []const u8,
    content: []const u8,
};

pub const Agent = struct {
    const Self = @This();

    pub fn init(_: std.mem.Allocator, _: AgentConfig) AgentError!Self {
        return error.AiDisabled;
    }

    pub fn deinit(_: *Self) void {}

    pub fn chat(_: *Self, _: []const u8, _: std.mem.Allocator) AgentError![]const u8 {
        return error.AiDisabled;
    }

    pub fn process(_: *Self, _: []const u8, _: std.mem.Allocator) AgentError![]const u8 {
        return error.AiDisabled;
    }
};
