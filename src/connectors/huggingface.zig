const std = @import("std");
const connectors = @import("mod.zig");
const async_http = @import("../shared/utils.zig").async_http;
const json_utils = @import("../shared/utils.zig").json;

pub const HuggingFaceError = error{
    MissingApiToken,
    ModelLoading,
    ApiRequestFailed,
    InvalidResponse,
};

pub const Config = struct {
    api_token: []u8,
    base_url: []u8,
    model: []const u8 = "gpt2",
    timeout_ms: u32 = 60_000,

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        allocator.free(self.api_token);
        allocator.free(self.base_url);
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

    pub fn inference(self: *Client, request: InferenceRequest) !InferenceResponse {
        const url = try std.fmt.allocPrint(self.allocator, "{s}/models/{s}", .{ self.config.base_url, self.config.model });
        defer self.allocator.free(url);

        const json = try self.encodeInferenceRequest(request);
        defer self.allocator.free(json);

        var http_req = try async_http.HttpRequest.init(self.allocator, .POST, url);
        defer http_req.deinit();

        try http_req.setBearerToken(self.config.api_token);
        try http_req.setJsonBody(json);

        const http_res = try self.http.fetchJson(&http_req);
        defer http_res.deinit();

        if (http_res.status_code == 503) {
            return HuggingFaceError.ModelLoading;
        }

        if (!http_res.isSuccess()) {
            return HuggingFaceError.ApiRequestFailed;
        }

        return try self.decodeInferenceResponse(http_res.body);
    }

    pub fn generateText(self: *Client, prompt: []const u8, params: ?Parameters) !InferenceResponse {
        return try self.inference(.{
            .inputs = prompt,
            .parameters = params,
        });
    }

    pub fn generateTextSimple(self: *Client, prompt: []const u8) !InferenceResponse {
        return try self.generateText(prompt, null);
    }

    pub fn encodeInferenceRequest(self: *Client, request: InferenceRequest) ![]u8 {
        var json_str = std.ArrayListUnmanaged(u8){};
        errdefer json_str.deinit(self.allocator);

        if (request.inputs) |inputs| {
            try json_str.print(self.allocator, "{{\"inputs\":\"{}\"", .{json_utils.jsonEscape(inputs)});
        } else {
            try json_str.append(self.allocator, '{');
        }

        if (request.parameters) |params| {
            try json_str.append(self.allocator, ',');
            try self.encodeParameters(&json_str, params);
        }

        try json_str.append(self.allocator, '}');

        return json_str.toOwnedSlice(self.allocator);
    }

    pub fn encodeParameters(self: *Client, json_str: *std.ArrayListUnmanaged(u8), params: Parameters) !void {
        var first = true;

        if (params.top_k) |top_k| {
            if (!first) try json_str.append(self.allocator, ',');
            try json_str.print(self.allocator, "\"top_k\":{d}", .{top_k});
            first = false;
        }

        if (params.top_p) |top_p| {
            if (!first) try json_str.append(self.allocator, ',');
            try json_str.print(self.allocator, "\"top_p\":{d:.2}", .{top_p});
            first = false;
        }

        if (params.temperature) |temp| {
            if (!first) try json_str.append(self.allocator, ',');
            try json_str.print(self.allocator, "\"temperature\":{d:.2}", .{temp});
            first = false;
        }

        if (params.max_new_tokens) |max_tokens| {
            if (!first) try json_str.append(self.allocator, ',');
            try json_str.print(self.allocator, "\"max_new_tokens\":{d}", .{max_tokens});
            first = false;
        }

        if (params.return_full_text) |return_full_text| {
            if (!first) try json_str.append(self.allocator, ',');
            try json_str.print(self.allocator, "\"return_full_text\":{d}", .{@intFromBool(return_full_text)});
        }
    }

    pub fn decodeInferenceResponse(self: *Client, json: []const u8) !InferenceResponse {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const generated_text = try json_utils.parseStringField(
            try json_utils.getRequiredObject(parsed.value),
            "generated_text",
            self.allocator,
        );

        return InferenceResponse{
            .generated_text = generated_text,
        };
    }
};

pub fn loadFromEnv(allocator: std.mem.Allocator) !Config {
    const api_token = (try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_HF_API_TOKEN",
        "HF_API_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    })) orelse return HuggingFaceError.MissingApiToken;

    const base_url = (try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_HF_BASE_URL",
    })) orelse try allocator.dupe(u8, "https://api-inference.huggingface.co");

    const model = (try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_HF_MODEL",
        "HF_MODEL",
    })) orelse try allocator.dupe(u8, "gpt2");
    errdefer allocator.free(model);

    return .{
        .api_token = api_token,
        .base_url = base_url,
        .model = model,
        .timeout_ms = 60_000,
    };
}

pub fn createClient(allocator: std.mem.Allocator) !Client {
    const config = try loadFromEnv(allocator);
    return try Client.init(allocator, config);
}
