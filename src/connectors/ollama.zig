const std = @import("std");
const connectors = @import("mod.zig");
const shared = @import("shared.zig");
const async_http = @import("../shared/utils/http/async_http.zig");
const json_utils = @import("../shared/utils/json/mod.zig");

pub const Config = struct {
    host: []u8,
    model: []const u8 = "llama2",
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

pub const Message = shared.ChatMessage;

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

    pub fn deinit(self: *GenerateResponse, allocator: std.mem.Allocator) void {
        allocator.free(self.model);
        allocator.free(self.response);
        if (self.context) |ctx| {
            allocator.free(ctx);
        }
        self.* = undefined;
    }
};

pub const ChatResponse = struct {
    model: []const u8,
    message: Message,
    done: bool,

    pub fn deinit(self: *ChatResponse, allocator: std.mem.Allocator) void {
        allocator.free(self.model);
        allocator.free(self.message.role);
        allocator.free(self.message.content);
        self.* = undefined;
    }
};

pub const Client = struct {
    allocator: std.mem.Allocator,
    config: Config,
    http: async_http.AsyncHttpClient,

    pub fn init(allocator: std.mem.Allocator, config: Config) !Client {
        const http = try async_http.AsyncHttpClient.init(allocator);
        errdefer http.deinit();

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

    pub fn generate(self: *Client, request: GenerateRequest) !GenerateResponse {
        const url = try std.fmt.allocPrint(self.allocator, "{s}/api/generate", .{self.config.host});
        defer self.allocator.free(url);

        const json = try self.encodeGenerateRequest(request);
        defer self.allocator.free(json);

        var http_req = try async_http.HttpRequest.init(self.allocator, .POST, url);
        defer http_req.deinit();

        try http_req.setJsonBody(json);

        const http_res = try self.http.fetchJson(&http_req);
        defer http_res.deinit();

        if (!http_res.isSuccess()) {
            return error.ApiRequestFailed;
        }

        return try self.decodeGenerateResponse(http_res.body);
    }

    pub fn generateSimple(self: *Client, prompt: []const u8) !GenerateResponse {
        return try self.generate(.{
            .model = self.config.model,
            .prompt = prompt,
        });
    }

    pub fn chat(self: *Client, request: ChatRequest) !ChatResponse {
        const url = try std.fmt.allocPrint(self.allocator, "{s}/api/chat", .{self.config.host});
        defer self.allocator.free(url);

        const json = try self.encodeChatRequest(request);
        defer self.allocator.free(json);

        var http_req = try async_http.HttpRequest.init(self.allocator, .POST, url);
        defer http_req.deinit();

        try http_req.setJsonBody(json);

        const http_res = try self.http.fetchJson(&http_req);
        defer http_res.deinit();

        if (!http_res.isSuccess()) {
            return error.ApiRequestFailed;
        }

        return try self.decodeChatResponse(http_res.body);
    }

    pub fn chatSimple(self: *Client, prompt: []const u8) !ChatResponse {
        const messages = [_]Message{
            .{ .role = "user", .content = prompt },
        };
        return try self.chat(.{
            .model = self.config.model,
            .messages = &messages,
        });
    }

    pub fn encodeGenerateRequest(self: *Client, request: GenerateRequest) ![]u8 {
        var json_str = std.ArrayListUnmanaged(u8){};
        errdefer json_str.deinit(self.allocator);

        try json_str.print(
            self.allocator,
            "{{\"model\":\"{s}\",\"prompt\":\"{}\",\"stream\":{d}}}",
            .{ request.model, json_utils.jsonEscape(request.prompt), @intFromBool(request.stream) },
        );

        return json_str.toOwnedSlice(self.allocator);
    }

    pub fn encodeChatRequest(self: *Client, request: ChatRequest) ![]u8 {
        var json_str = std.ArrayListUnmanaged(u8){};
        errdefer json_str.deinit(self.allocator);

        try json_str.print(self.allocator, "{{\"model\":\"{s\",\"messages\":[", .{request.model});

        for (request.messages, 0..) |msg, i| {
            if (i > 0) try json_str.append(self.allocator, ',');
            try json_str.print(
                self.allocator,
                "{{\"role\":\"{s}\",\"content\":\"{}\"}}",
                .{ msg.role, json_utils.jsonEscape(msg.content) },
            );
        }

        try json_str.append(self.allocator, '}');

        return json_str.toOwnedSlice(self.allocator);
    }

    pub fn decodeGenerateResponse(self: *Client, json: []const u8) !GenerateResponse {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const object = try json_utils.getRequiredObject(parsed.value);

        const model = try json_utils.parseStringField(object, "model", self.allocator);
        const response = try json_utils.parseStringField(object, "response", self.allocator);
        const done = try json_utils.parseBoolField(object, "done");

        var context: ?[]u64 = null;
        if (object.get("context")) |context_val| {
            if (context_val == .array) {
                context = try self.allocator.alloc(u64, context_val.array.items.len);
                for (context_val.array.items, 0..) |item, i| {
                    const num = try json_utils.parseUint(item);
                    context.?[i] = num;
                }
            }
        }

        return GenerateResponse{
            .model = model,
            .response = response,
            .done = done,
            .context = context,
        };
    }

    pub fn decodeChatResponse(self: *Client, json: []const u8) !ChatResponse {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const object = try json_utils.getRequiredObject(parsed.value);

        const model = try json_utils.parseStringField(object, "model", self.allocator);
        const message_obj = try json_utils.parseObjectField(object, "message");
        const role = try json_utils.parseStringField(message_obj, "role", self.allocator);
        const content = try json_utils.parseStringField(message_obj, "content", self.allocator);
        const done = try json_utils.parseBoolField(object, "done");

        return ChatResponse{
            .model = model,
            .message = .{
                .role = role,
                .content = content,
            },
            .done = done,
        };
    }
};

pub fn loadFromEnv(allocator: std.mem.Allocator) !Config {
    const host = (try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_OLLAMA_HOST",
        "OLLAMA_HOST",
    })) orelse try allocator.dupe(u8, "http://127.0.0.1:11434");

    const model = (try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_OLLAMA_MODEL",
        "OLLAMA_MODEL",
    })) orelse try allocator.dupe(u8, "gpt-oss");
    errdefer allocator.free(model);

    return .{
        .host = host,
        .model = model,
        .model_owned = true,
        .timeout_ms = 120_000,
    };
}

pub fn createClient(allocator: std.mem.Allocator) !Client {
    const config = try loadFromEnv(allocator);
    return try Client.init(allocator, config);
}

test "ollama endpoint join" {
    var config = Config{ .host = try std.testing.allocator.dupe(u8, "http://localhost:11434") };
    defer config.deinit(std.testing.allocator);

    const url = try std.fmt.allocPrint(std.testing.allocator, "{s}/api/generate", .{config.host});
    defer std.testing.allocator.free(url);
    try std.testing.expectEqualStrings("http://localhost:11434/api/generate", url);
}
