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
    InvalidConfiguration,
    OutOfMemory,
    ConnectorNotAvailable,
    GenerationFailed,
    ApiKeyMissing,
    HttpRequestFailed,
    InvalidApiResponse,
    RateLimitExceeded,
    Timeout,
    ConnectionRefused,
    ModelNotFound,
};

pub const AgentBackend = enum {
    echo,
    openai,
    ollama,
    huggingface,
    local,
};

pub const OperationContext = enum {
    initialization,
    configuration_validation,
    message_processing,
    response_generation,
    history_management,
    api_request,
    json_parsing,
    token_counting,

    pub fn toString(self: OperationContext) []const u8 {
        return switch (self) {
            .initialization => "initialization",
            .configuration_validation => "configuration validation",
            .message_processing => "message processing",
            .response_generation => "response generation",
            .history_management => "history management",
            .api_request => "API request",
            .json_parsing => "JSON parsing",
            .token_counting => "token counting",
        };
    }
};

pub const ErrorContext = struct {
    @"error": AgentError,
    backend: AgentBackend,
    operation: OperationContext,
    http_status: ?u16 = null,
    endpoint: ?[]const u8 = null,
    message: ?[]const u8 = null,
    retry_count: ?u32 = null,
    max_retries: ?u32 = null,

    pub fn format(self: ErrorContext, writer: anytype) !void {
        _ = self;
        _ = writer;
    }

    pub fn formatToString(self: ErrorContext, allocator: std.mem.Allocator) ![]u8 {
        _ = self;
        return allocator.dupe(u8, "disabled");
    }

    pub fn log(self: ErrorContext) void {
        _ = self;
    }
};

pub const AgentConfig = struct {
    name: []const u8 = "",
    enable_history: bool = true,
    temperature: f32 = DEFAULT_TEMPERATURE,
    top_p: f32 = DEFAULT_TOP_P,
    max_tokens: u32 = DEFAULT_MAX_TOKENS,
    backend: AgentBackend = .echo,
    model: []const u8 = "gpt-4",
    system_prompt: ?[]const u8 = null,

    pub fn validate(self: AgentConfig) AgentError!void {
        if (self.name.len == 0) return AgentError.InvalidConfiguration;
        if (self.temperature < MIN_TEMPERATURE or self.temperature > MAX_TEMPERATURE) {
            return AgentError.InvalidConfiguration;
        }
        if (self.top_p < MIN_TOP_P or self.top_p > MAX_TOP_P) {
            return AgentError.InvalidConfiguration;
        }
        if (self.max_tokens == 0 or self.max_tokens > MAX_TOKENS_LIMIT) {
            return AgentError.InvalidConfiguration;
        }
    }
};

pub const Message = struct {
    role: Role,
    content: []const u8,

    pub const Role = enum {
        system,
        user,
        assistant,
    };
};

pub const Agent = struct {
    allocator: std.mem.Allocator,
    config: AgentConfig,
    history: std.ArrayListUnmanaged(Message) = .{},
    total_tokens_used: u64 = 0,

    pub fn init(allocator: std.mem.Allocator, config: AgentConfig) AgentError!Agent {
        _ = config;
        _ = allocator;
        return error.ConnectorNotAvailable;
    }

    pub fn deinit(_: *Agent) void {}

    pub fn historyCount(self: *const Agent) usize {
        return self.history.items.len;
    }

    pub fn chat(_: *Agent, _: []const u8, _: std.mem.Allocator) AgentError![]const u8 {
        return error.ConnectorNotAvailable;
    }

    pub fn process(_: *Agent, _: []const u8, _: std.mem.Allocator) AgentError![]const u8 {
        return error.ConnectorNotAvailable;
    }
};
