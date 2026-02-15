//! Cohere API connector.
//!
//! Provides integration with Cohere's models for chat, embeddings, and reranking.
//! Cohere uses a different API format from OpenAI-style APIs.

const std = @import("std");
const connectors = @import("mod.zig");
const shared = @import("shared.zig");
const async_http = @import("../shared/utils.zig").async_http;
const json_utils = @import("../shared/utils.zig").json;

/// Errors that can occur when interacting with the Cohere API.
pub const CohereError = error{
    /// API key was not provided via environment variable.
    MissingApiKey,
    /// The API request failed (network error or non-2xx status).
    ApiRequestFailed,
    /// The API response could not be parsed.
    InvalidResponse,
    /// Rate limit exceeded (HTTP 429). Retry after backoff.
    RateLimitExceeded,
};

pub const Config = struct {
    api_key: []u8,
    base_url: []u8,
    model: []const u8 = "command-r-plus",
    model_owned: bool = false,
    timeout_ms: u32 = 60_000,

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        shared.deinitConfig(allocator, self.api_key, self.base_url);
        if (self.model_owned) allocator.free(@constCast(self.model));
        self.* = undefined;
    }
};

pub const ChatRole = enum {
    user,
    assistant,
    system,
    tool,

    pub fn toString(self: ChatRole) []const u8 {
        return switch (self) {
            .user => "USER",
            .assistant => "CHATBOT",
            .system => "SYSTEM",
            .tool => "TOOL",
        };
    }
};

pub const ChatMessage = struct {
    role: ChatRole,
    message: []const u8,
};

pub const ChatRequest = struct {
    model: []const u8,
    message: []const u8,
    chat_history: []const ChatMessage = &.{},
    preamble: ?[]const u8 = null,
    temperature: f32 = 0.3,
    max_tokens: ?u32 = null,
    stream: bool = false,
};

pub const ChatResponse = struct {
    response_id: []const u8,
    text: []const u8,
    generation_id: []const u8,
    finish_reason: []const u8,
    meta: ChatMeta,
};

pub const ChatMeta = struct {
    api_version: ApiVersion,
    billed_units: BilledUnits,
    tokens: TokenUsage,
};

pub const ApiVersion = struct {
    version: []const u8,
};

pub const BilledUnits = struct {
    input_tokens: u32,
    output_tokens: u32,
};

pub const TokenUsage = struct {
    input_tokens: u32,
    output_tokens: u32,
};

pub const EmbedRequest = struct {
    model: []const u8 = "embed-english-v3.0",
    texts: []const []const u8,
    input_type: []const u8 = "search_document",
    truncate: []const u8 = "END",
};

pub const EmbedResponse = struct {
    id: []const u8,
    embeddings: [][]f32,
    texts: [][]const u8,
    meta: EmbedMeta,
};

pub const EmbedMeta = struct {
    api_version: ApiVersion,
    billed_units: EmbedBilledUnits,
};

pub const EmbedBilledUnits = struct {
    input_tokens: u32,
};

pub const RerankRequest = struct {
    model: []const u8 = "rerank-english-v3.0",
    query: []const u8,
    documents: []const []const u8,
    top_n: ?u32 = null,
    return_documents: bool = false,
};

pub const RerankResponse = struct {
    id: []const u8,
    results: []RerankResult,
    meta: RerankMeta,
};

pub const RerankResult = struct {
    index: u32,
    relevance_score: f32,
    document: ?[]const u8 = null,
};

pub const RerankMeta = struct {
    api_version: ApiVersion,
    billed_units: RerankBilledUnits,
};

pub const RerankBilledUnits = struct {
    search_units: u32,
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

    pub fn chat(self: *Client, request: ChatRequest) !ChatResponse {
        const url = try std.fmt.allocPrint(self.allocator, "{s}/chat", .{self.config.base_url});
        defer self.allocator.free(url);

        const json = try self.encodeChatRequest(request);
        defer self.allocator.free(json);

        var http_req = try async_http.HttpRequest.init(self.allocator, .POST, url);
        defer http_req.deinit();

        try http_req.setBearerToken(self.config.api_key);
        try http_req.setJsonBody(json);

        const http_res = try self.http.fetchJsonWithRetry(&http_req, shared.DEFAULT_RETRY_OPTIONS);
        defer http_res.deinit();

        if (!http_res.isSuccess()) {
            if (http_res.status_code == 429) {
                return CohereError.RateLimitExceeded;
            }
            return CohereError.ApiRequestFailed;
        }

        return try self.decodeChatResponse(http_res.body);
    }

    pub fn chatSimple(self: *Client, message: []const u8) !ChatResponse {
        return try self.chat(.{
            .model = self.config.model,
            .message = message,
        });
    }

    pub fn chatWithHistory(self: *Client, message: []const u8, history: []const ChatMessage) !ChatResponse {
        return try self.chat(.{
            .model = self.config.model,
            .message = message,
            .chat_history = history,
        });
    }

    pub fn chatStreaming(self: *Client, request: ChatRequest) !async_http.StreamingResponse {
        var req = request;
        req.stream = true;

        const url = try std.fmt.allocPrint(self.allocator, "{s}/chat", .{self.config.base_url});
        defer self.allocator.free(url);

        const json = try self.encodeChatRequest(req);
        defer self.allocator.free(json);

        var http_req = try async_http.HttpRequest.init(self.allocator, .POST, url);
        defer http_req.deinit();

        try http_req.setBearerToken(self.config.api_key);
        try http_req.setJsonBody(json);

        return try self.http.fetchStreaming(&http_req);
    }

    pub fn embed(self: *Client, request: EmbedRequest) !EmbedResponse {
        const url = try std.fmt.allocPrint(self.allocator, "{s}/embed", .{self.config.base_url});
        defer self.allocator.free(url);

        const json = try self.encodeEmbedRequest(request);
        defer self.allocator.free(json);

        var http_req = try async_http.HttpRequest.init(self.allocator, .POST, url);
        defer http_req.deinit();

        try http_req.setBearerToken(self.config.api_key);
        try http_req.setJsonBody(json);

        const http_res = try self.http.fetchJsonWithRetry(&http_req, shared.DEFAULT_RETRY_OPTIONS);
        defer http_res.deinit();

        if (!http_res.isSuccess()) {
            return CohereError.ApiRequestFailed;
        }

        return try self.decodeEmbedResponse(http_res.body);
    }

    pub fn embedTexts(self: *Client, texts: []const []const u8) !EmbedResponse {
        return try self.embed(.{
            .texts = texts,
        });
    }

    pub fn embedSingle(self: *Client, text: []const u8) ![]f32 {
        const texts = [_][]const u8{text};
        const response = try self.embedTexts(&texts);
        defer {
            for (response.embeddings) |emb| {
                self.allocator.free(emb);
            }
            self.allocator.free(response.embeddings);
        }

        if (response.embeddings.len == 0) {
            return CohereError.InvalidResponse;
        }

        return try self.allocator.dupe(f32, response.embeddings[0]);
    }

    pub fn rerank(self: *Client, request: RerankRequest) !RerankResponse {
        const url = try std.fmt.allocPrint(self.allocator, "{s}/rerank", .{self.config.base_url});
        defer self.allocator.free(url);

        const json = try self.encodeRerankRequest(request);
        defer self.allocator.free(json);

        var http_req = try async_http.HttpRequest.init(self.allocator, .POST, url);
        defer http_req.deinit();

        try http_req.setBearerToken(self.config.api_key);
        try http_req.setJsonBody(json);

        const http_res = try self.http.fetchJsonWithRetry(&http_req, shared.DEFAULT_RETRY_OPTIONS);
        defer http_res.deinit();

        if (!http_res.isSuccess()) {
            return CohereError.ApiRequestFailed;
        }

        return try self.decodeRerankResponse(http_res.body);
    }

    pub fn rerankDocuments(self: *Client, query: []const u8, documents: []const []const u8, top_n: ?u32) !RerankResponse {
        return try self.rerank(.{
            .query = query,
            .documents = documents,
            .top_n = top_n,
            .return_documents = true,
        });
    }

    fn encodeChatRequest(self: *Client, request: ChatRequest) ![]u8 {
        var json_str = std.ArrayListUnmanaged(u8).empty;
        errdefer json_str.deinit(self.allocator);

        try json_str.appendSlice(self.allocator, "{\"model\":\"");
        try json_utils.appendJsonEscaped(self.allocator, &json_str, request.model);
        try json_str.appendSlice(self.allocator, "\",\"message\":\"");
        try json_utils.appendJsonEscaped(self.allocator, &json_str, request.message);
        try json_str.append(self.allocator, '"');

        if (request.chat_history.len > 0) {
            try json_str.appendSlice(self.allocator, ",\"chat_history\":[");
            for (request.chat_history, 0..) |msg, i| {
                if (i > 0) try json_str.append(self.allocator, ',');
                try json_str.appendSlice(self.allocator, "{\"role\":\"");
                try json_str.appendSlice(self.allocator, msg.role.toString());
                try json_str.appendSlice(self.allocator, "\",\"message\":\"");
                try json_utils.appendJsonEscaped(self.allocator, &json_str, msg.message);
                try json_str.appendSlice(self.allocator, "\"}");
            }
            try json_str.append(self.allocator, ']');
        }

        if (request.preamble) |preamble| {
            try json_str.print(self.allocator, ",\"preamble\":\"{}\"", .{json_utils.jsonEscape(preamble)});
        }

        try json_str.print(self.allocator, ",\"temperature\":{d:.2}", .{request.temperature});

        if (request.max_tokens) |max_tokens| {
            try json_str.print(self.allocator, ",\"max_tokens\":{d}", .{max_tokens});
        }

        if (request.stream) {
            try json_str.appendSlice(self.allocator, ",\"stream\":true");
        }

        try json_str.append(self.allocator, '}');

        return json_str.toOwnedSlice(self.allocator);
    }

    fn encodeEmbedRequest(self: *Client, request: EmbedRequest) ![]u8 {
        var json_str = std.ArrayListUnmanaged(u8).empty;
        errdefer json_str.deinit(self.allocator);

        try json_str.appendSlice(self.allocator, "{\"model\":\"");
        try json_utils.appendJsonEscaped(self.allocator, &json_str, request.model);
        try json_str.appendSlice(self.allocator, "\",\"texts\":[");

        try shared.encodeStringArray(self.allocator, &json_str, request.texts);

        try json_str.appendSlice(self.allocator, "],\"input_type\":\"");
        try json_str.appendSlice(self.allocator, request.input_type);
        try json_str.appendSlice(self.allocator, "\",\"truncate\":\"");
        try json_str.appendSlice(self.allocator, request.truncate);
        try json_str.appendSlice(self.allocator, "\"}");

        return json_str.toOwnedSlice(self.allocator);
    }

    fn encodeRerankRequest(self: *Client, request: RerankRequest) ![]u8 {
        var json_str = std.ArrayListUnmanaged(u8).empty;
        errdefer json_str.deinit(self.allocator);

        try json_str.appendSlice(self.allocator, "{\"model\":\"");
        try json_utils.appendJsonEscaped(self.allocator, &json_str, request.model);
        try json_str.appendSlice(self.allocator, "\",\"query\":\"");
        try json_utils.appendJsonEscaped(self.allocator, &json_str, request.query);
        try json_str.appendSlice(self.allocator, "\",\"documents\":[");

        try shared.encodeStringArray(self.allocator, &json_str, request.documents);

        try json_str.append(self.allocator, ']');

        if (request.top_n) |top_n| {
            try json_str.print(self.allocator, ",\"top_n\":{d}", .{top_n});
        }

        if (request.return_documents) {
            try json_str.appendSlice(self.allocator, ",\"return_documents\":true");
        }

        try json_str.append(self.allocator, '}');

        return json_str.toOwnedSlice(self.allocator);
    }

    fn decodeChatResponse(self: *Client, json: []const u8) !ChatResponse {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const object = try json_utils.getRequiredObject(parsed.value);

        const response_id = try json_utils.parseStringField(object, "response_id", self.allocator);
        errdefer self.allocator.free(response_id);

        const text = try json_utils.parseStringField(object, "text", self.allocator);
        errdefer self.allocator.free(text);

        const generation_id = try json_utils.parseStringField(object, "generation_id", self.allocator);
        errdefer self.allocator.free(generation_id);

        const finish_reason = try json_utils.parseStringField(object, "finish_reason", self.allocator);
        errdefer self.allocator.free(finish_reason);

        const meta_obj = try json_utils.parseObjectField(object, "meta");

        const api_version_obj = try json_utils.parseObjectField(meta_obj, "api_version");
        const api_version_str = try json_utils.parseStringField(api_version_obj, "version", self.allocator);

        const billed_obj = try json_utils.parseObjectField(meta_obj, "billed_units");
        const billed_units = BilledUnits{
            .input_tokens = @intCast(try json_utils.parseIntField(billed_obj, "input_tokens")),
            .output_tokens = @intCast(try json_utils.parseIntField(billed_obj, "output_tokens")),
        };

        const tokens_obj = try json_utils.parseObjectField(meta_obj, "tokens");
        const tokens = TokenUsage{
            .input_tokens = @intCast(try json_utils.parseIntField(tokens_obj, "input_tokens")),
            .output_tokens = @intCast(try json_utils.parseIntField(tokens_obj, "output_tokens")),
        };

        return ChatResponse{
            .response_id = response_id,
            .text = text,
            .generation_id = generation_id,
            .finish_reason = finish_reason,
            .meta = .{
                .api_version = .{ .version = api_version_str },
                .billed_units = billed_units,
                .tokens = tokens,
            },
        };
    }

    fn decodeEmbedResponse(self: *Client, json: []const u8) !EmbedResponse {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const object = try json_utils.getRequiredObject(parsed.value);

        const id = try json_utils.parseStringField(object, "id", self.allocator);
        errdefer self.allocator.free(id);

        const embeddings_array = try json_utils.parseArrayField(object, "embeddings");
        var embeddings = try self.allocator.alloc([]f32, embeddings_array.items.len);
        errdefer self.allocator.free(embeddings);

        for (embeddings_array.items, 0..) |emb_array, i| {
            const inner_array = emb_array.array;
            var embedding = try self.allocator.alloc(f32, inner_array.items.len);
            for (inner_array.items, 0..) |val, j| {
                embedding[j] = @floatCast(val.float);
            }
            embeddings[i] = embedding;
        }

        const texts_array = try json_utils.parseArrayField(object, "texts");
        var texts = try self.allocator.alloc([]const u8, texts_array.items.len);
        for (texts_array.items, 0..) |text_val, i| {
            texts[i] = try json_utils.parseString(text_val, self.allocator);
        }

        const meta_obj = try json_utils.parseObjectField(object, "meta");
        const api_version_obj = try json_utils.parseObjectField(meta_obj, "api_version");
        const api_version_str = try json_utils.parseStringField(api_version_obj, "version", self.allocator);

        const billed_obj = try json_utils.parseObjectField(meta_obj, "billed_units");
        const billed_units = EmbedBilledUnits{
            .input_tokens = @intCast(try json_utils.parseIntField(billed_obj, "input_tokens")),
        };

        return EmbedResponse{
            .id = id,
            .embeddings = embeddings,
            .texts = texts,
            .meta = .{
                .api_version = .{ .version = api_version_str },
                .billed_units = billed_units,
            },
        };
    }

    fn decodeRerankResponse(self: *Client, json: []const u8) !RerankResponse {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const object = try json_utils.getRequiredObject(parsed.value);

        const id = try json_utils.parseStringField(object, "id", self.allocator);
        errdefer self.allocator.free(id);

        const results_array = try json_utils.parseArrayField(object, "results");
        var results = try self.allocator.alloc(RerankResult, results_array.items.len);
        errdefer self.allocator.free(results);

        for (results_array.items, 0..) |result_val, i| {
            const result_obj = try json_utils.getRequiredObject(result_val);
            const index: u32 = @intCast(try json_utils.parseIntField(result_obj, "index"));
            const relevance_score: f32 = @floatCast(try json_utils.parseFloatField(result_obj, "relevance_score"));

            const document = json_utils.parseOptionalStringField(result_obj, "document", self.allocator) catch null;

            results[i] = .{
                .index = index,
                .relevance_score = relevance_score,
                .document = document,
            };
        }

        const meta_obj = try json_utils.parseObjectField(object, "meta");
        const api_version_obj = try json_utils.parseObjectField(meta_obj, "api_version");
        const api_version_str = try json_utils.parseStringField(api_version_obj, "version", self.allocator);

        const billed_obj = try json_utils.parseObjectField(meta_obj, "billed_units");
        const billed_units = RerankBilledUnits{
            .search_units = @intCast(try json_utils.parseIntField(billed_obj, "search_units")),
        };

        return RerankResponse{
            .id = id,
            .results = results,
            .meta = .{
                .api_version = .{ .version = api_version_str },
                .billed_units = billed_units,
            },
        };
    }
};

pub fn loadFromEnv(allocator: std.mem.Allocator) !Config {
    const api_key_raw = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_COHERE_API_KEY",
        "COHERE_API_KEY",
        "CO_API_KEY",
    });
    const api_key = api_key_raw orelse return CohereError.MissingApiKey;
    // Treat empty string as missing (e.g., COHERE_API_KEY="")
    if (api_key.len == 0) {
        allocator.free(api_key);
        return CohereError.MissingApiKey;
    }
    errdefer allocator.free(api_key);

    const base_url_raw = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_COHERE_BASE_URL",
        "COHERE_BASE_URL",
    });
    // Treat empty base URL as unset — fall through to default
    const base_url = if (base_url_raw) |u| blk: {
        if (u.len == 0) {
            allocator.free(u);
            break :blk try allocator.dupe(u8, "https://api.cohere.ai/v1");
        }
        break :blk u;
    } else try allocator.dupe(u8, "https://api.cohere.ai/v1");
    errdefer allocator.free(base_url);

    const model_raw = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_COHERE_MODEL",
        "COHERE_MODEL",
    });
    // Treat empty model as unset — fall through to default
    const model = if (model_raw) |m| blk: {
        if (m.len == 0) {
            allocator.free(m);
            break :blk try allocator.dupe(u8, "command-r-plus");
        }
        break :blk m;
    } else try allocator.dupe(u8, "command-r-plus");

    return .{
        .api_key = api_key,
        .base_url = base_url,
        .model = model,
        .model_owned = true,
        .timeout_ms = 60_000,
    };
}

pub fn createClient(allocator: std.mem.Allocator) !Client {
    const config = try loadFromEnv(allocator);
    return try Client.init(allocator, config);
}

/// Check if the Cohere connector is available (API key env var is set).
/// This is a zero-allocation health check suitable for status dashboards.
pub fn isAvailable() bool {
    return shared.anyEnvIsSet(&.{
        "ABI_COHERE_API_KEY",
        "COHERE_API_KEY",
    });
}

test "cohere config deinit" {
    const allocator = std.testing.allocator;
    var config = Config{
        .api_key = try allocator.dupe(u8, "test-key"),
        .base_url = try allocator.dupe(u8, "https://api.cohere.ai/v1"),
    };
    config.deinit(allocator);
}

test "cohere chat request encoding" {
    const allocator = std.testing.allocator;

    const config = Config{
        .api_key = try allocator.dupe(u8, "test-key"),
        .base_url = try allocator.dupe(u8, "https://api.cohere.ai/v1"),
    };
    var client = try Client.init(allocator, config);
    defer client.deinit();

    const json = try client.encodeChatRequest(.{
        .model = "command-r-plus",
        .message = "Hello",
    });
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"model\":\"command-r-plus\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"message\":\"Hello\"") != null);
}

test "cohere embed request encoding" {
    const allocator = std.testing.allocator;

    const config = Config{
        .api_key = try allocator.dupe(u8, "test-key"),
        .base_url = try allocator.dupe(u8, "https://api.cohere.ai/v1"),
    };
    var client = try Client.init(allocator, config);
    defer client.deinit();

    const texts = [_][]const u8{ "Hello", "World" };
    const json = try client.encodeEmbedRequest(.{
        .texts = &texts,
    });
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"model\":\"embed-english-v3.0\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"Hello\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"World\"") != null);
}

test "cohere rerank request encoding" {
    const allocator = std.testing.allocator;

    const config = Config{
        .api_key = try allocator.dupe(u8, "test-key"),
        .base_url = try allocator.dupe(u8, "https://api.cohere.ai/v1"),
    };
    var client = try Client.init(allocator, config);
    defer client.deinit();

    const docs = [_][]const u8{ "Doc 1", "Doc 2" };
    const json = try client.encodeRerankRequest(.{
        .query = "test query",
        .documents = &docs,
        .top_n = 5,
    });
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"query\":\"test query\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"top_n\":5") != null);
}

test "cohere chat role to string" {
    try std.testing.expectEqualStrings("USER", ChatRole.user.toString());
    try std.testing.expectEqualStrings("CHATBOT", ChatRole.assistant.toString());
    try std.testing.expectEqualStrings("SYSTEM", ChatRole.system.toString());
    try std.testing.expectEqualStrings("TOOL", ChatRole.tool.toString());
}

test "isAvailable returns bool" {
    // Just verify it returns without crashing - actual availability depends on env
    _ = isAvailable();
}
