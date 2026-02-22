//! Gemini API connector.
//!
//! Vendor-native connector using Google's Generative Language API
//! (`models/*:generateContent`).

const std = @import("std");
const connectors = @import("mod.zig");
const shared = @import("shared.zig");
const async_http = @import("../shared/utils.zig").async_http;
const json_utils = @import("../shared/utils.zig").json;

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
        const url = try std.fmt.allocPrint(
            self.allocator,
            "{s}/models/{s}:generateContent?key={s}",
            .{ self.config.base_url, request.model, self.config.api_key },
        );
        defer self.allocator.free(url);

        const json = try self.encodeGenerateRequest(request);
        defer self.allocator.free(json);

        var http_req = try async_http.HttpRequest.init(self.allocator, .post, url);
        defer http_req.deinit();

        try http_req.setJsonBody(json);

        var http_res = try self.http.fetchJsonWithRetry(&http_req, shared.DEFAULT_RETRY_OPTIONS);
        defer http_res.deinit();

        if (!http_res.isSuccess()) {
            if (http_res.status_code == 429) {
                return GeminiError.RateLimitExceeded;
            }
            return GeminiError.ApiRequestFailed;
        }

        return try self.decodeGenerateResponse(http_res.body, request.model);
    }

    fn encodeGenerateRequest(self: *Client, request: GenerateRequest) ![]u8 {
        var json = std.ArrayListUnmanaged(u8).empty;
        errdefer json.deinit(self.allocator);

        try json.appendSlice(self.allocator, "{");

        if (request.system_prompt) |system| {
            try json.appendSlice(self.allocator, "\"system_instruction\":{\"parts\":[{\"text\":\"");
            try json_utils.appendJsonEscaped(self.allocator, &json, system);
            try json.appendSlice(self.allocator, "\"}]},");
        }

        try json.appendSlice(self.allocator, "\"contents\":[");
        for (request.messages, 0..) |msg, i| {
            if (i > 0) try json.append(self.allocator, ',');

            const role = if (std.mem.eql(u8, msg.role, "assistant")) "model" else "user";
            try json.appendSlice(self.allocator, "{\"role\":\"");
            try json.appendSlice(self.allocator, role);
            try json.appendSlice(self.allocator, "\",\"parts\":[{\"text\":\"");
            try json_utils.appendJsonEscaped(self.allocator, &json, msg.content);
            try json.appendSlice(self.allocator, "\"}]}");
        }
        try json.appendSlice(self.allocator, "],\"generationConfig\":{");
        try json.print(self.allocator, "\"temperature\":{d:.2},\"topP\":{d:.2}", .{ request.temperature, request.top_p });
        if (request.max_output_tokens) |tokens| {
            try json.print(self.allocator, ",\"maxOutputTokens\":{d}", .{tokens});
        }
        try json.appendSlice(self.allocator, "}}");

        return json.toOwnedSlice(self.allocator);
    }

    fn decodeGenerateResponse(self: *Client, body: []const u8, fallback_model: []const u8) !GenerateResponse {
        const parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, body, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();

        const root = try json_utils.getRequiredObject(parsed.value);
        const candidates = try json_utils.parseArrayField(root, "candidates");
        if (candidates.items.len == 0) return GeminiError.InvalidResponse;

        const first = try json_utils.getRequiredObject(candidates.items[0]);
        const content_obj = try json_utils.parseObjectField(first, "content");
        const parts = try json_utils.parseArrayField(content_obj, "parts");
        if (parts.items.len == 0) return GeminiError.InvalidResponse;

        var text = std.ArrayListUnmanaged(u8).empty;
        errdefer text.deinit(self.allocator);

        for (parts.items) |part| {
            const part_obj = try json_utils.getRequiredObject(part);
            const piece = try json_utils.parseStringField(part_obj, "text", self.allocator);
            defer self.allocator.free(piece);
            try text.appendSlice(self.allocator, piece);
        }

        const model = if (root.get("modelVersion")) |v| blk: {
            if (v != .string) break :blk try self.allocator.dupe(u8, fallback_model);
            break :blk try self.allocator.dupe(u8, v.string);
        } else try self.allocator.dupe(u8, fallback_model);

        return .{
            .model = model,
            .text = try text.toOwnedSlice(self.allocator),
        };
    }
};

pub fn loadFromEnv(allocator: std.mem.Allocator) !Config {
    const api_key_raw = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_GEMINI_API_KEY",
        "GEMINI_API_KEY",
    });
    const api_key = api_key_raw orelse return GeminiError.MissingApiKey;
    if (api_key.len == 0) {
        allocator.free(api_key);
        return GeminiError.MissingApiKey;
    }
    errdefer allocator.free(api_key);

    const base_url_raw = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_GEMINI_BASE_URL",
        "GEMINI_BASE_URL",
    });
    const base_url = if (base_url_raw) |u| blk: {
        if (u.len == 0) {
            allocator.free(u);
            break :blk try allocator.dupe(u8, "https://generativelanguage.googleapis.com/v1beta");
        }
        break :blk u;
    } else try allocator.dupe(u8, "https://generativelanguage.googleapis.com/v1beta");
    errdefer allocator.free(base_url);

    const model_raw = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_GEMINI_MODEL",
        "GEMINI_MODEL",
    });
    const model = if (model_raw) |m| blk: {
        if (m.len == 0) {
            allocator.free(m);
            break :blk try allocator.dupe(u8, "gemini-1.5-pro");
        }
        break :blk m;
    } else try allocator.dupe(u8, "gemini-1.5-pro");

    return .{
        .api_key = api_key,
        .base_url = base_url,
        .model = model,
        .model_owned = true,
        .timeout_ms = 120_000,
    };
}

pub fn createClient(allocator: std.mem.Allocator) !Client {
    const config = try loadFromEnv(allocator);
    return try Client.init(allocator, config);
}

pub fn isAvailable() bool {
    return shared.anyEnvIsSet(&.{
        "ABI_GEMINI_API_KEY",
        "GEMINI_API_KEY",
    });
}

test "gemini availability returns bool" {
    _ = isAvailable();
}
