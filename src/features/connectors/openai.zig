const std = @import("std");
const connectors = @import("mod.zig");
const async_http = @import("../../shared/utils/http/async_http.zig");

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
    timeout_ms: u32 = 60_000,

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        allocator.free(self.api_key);
        allocator.free(self.base_url);
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

pub const Client = struct {
    allocator: std.mem.Allocator,
    config: Config,
    http: async_http.AsyncHttpClient,

    pub fn init(allocator: std.mem.Allocator, config: Config) !Client {
        const http = try async_http.AsyncHttpClient.init(allocator);
        return .{
            .allocator = allocator,
            .config = config,
            .http = http,
        };
    }

    pub fn deinit(self: *Client) void {
        self.http.deinit();
        self.config.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn chatCompletion(self: *Client, request: ChatCompletionRequest) !ChatCompletionResponse {
        const url = try std.fmt.allocPrint(self.allocator, "{s}/chat/completions", .{self.config.base_url});
        defer self.allocator.free(url);

        const json = try self.encodeChatRequest(request);
        defer self.allocator.free(json);

        var http_req = try async_http.HttpRequest.init(self.allocator, .POST, url);
        defer http_req.deinit();

        try http_req.setBearerToken(self.config.api_key);
        try http_req.setJsonBody(json);

        const http_res = try self.http.fetchJson(&http_req);
        defer http_res.deinit();

        if (!http_res.isSuccess()) {
            return OpenAIError.ApiRequestFailed;
        }

        return try self.decodeChatResponse(http_res.body);
    }

    pub fn chat(self: *Client, messages: []Message) !ChatCompletionResponse {
        return try self.chatCompletion(.{
            .model = self.config.model,
            .messages = messages,
        });
    }

    pub fn chatSimple(self: *Client, prompt: []const u8) !ChatCompletionResponse {
        const messages = [_]Message{
            .{ .role = "user", .content = prompt },
        };
        return try self.chat(&messages);
    }

    pub fn encodeChatRequest(self: *Client, request: ChatCompletionRequest) ![]u8 {
        var json_str = std.ArrayList(u8).init(self.allocator);
        errdefer json_str.deinit();

        try json_str.writer().print("{{\"model\":\"{s\",\"messages\":[", .{request.model});

        for (request.messages, 0..) |msg, i| {
            if (i > 0) try json_str.append(',');
            try json_str.writer().print(
                "{{\"role\":\"{s\",\"content\":\"{s}\"}}",
                .{ msg.role, std.zig.fmtEscapes(msg.content) },
            );
        }

        try json_str.writer().print("],\"temperature\":{d:.2}", .{request.temperature});

        if (request.max_tokens) |max_tokens| {
            try json_str.writer().print(",\"max_tokens\":{d}", .{max_tokens});
        }

        try json_str.append('}');

        return json_str.toOwnedSlice();
    }

    pub fn decodeChatResponse(self: *Client, json: []const u8) !ChatCompletionResponse {
        _ = self;
        _ = json;
        return OpenAIError.InvalidResponse;
    }
};

pub fn loadFromEnv(allocator: std.mem.Allocator) !Config {
    const api_key = (try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_OPENAI_API_KEY",
        "OPENAI_API_KEY",
    })) orelse return OpenAIError.MissingApiKey;

    const base_url = (try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_OPENAI_BASE_URL",
        "OPENAI_BASE_URL",
    })) orelse try allocator.dupe(u8, "https://api.openai.com/v1");

    const model = (try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_OPENAI_MODEL",
        "OPENAI_MODEL",
    })) orelse try allocator.dupe(u8, "gpt-4");
    errdefer allocator.free(model);

    return .{
        .api_key = api_key,
        .base_url = base_url,
        .model = model,
        .timeout_ms = 60_000,
    };
}

pub fn createClient(allocator: std.mem.Allocator) !Client {
    const config = try loadFromEnv(allocator);
    return try Client.init(allocator, config);
}
