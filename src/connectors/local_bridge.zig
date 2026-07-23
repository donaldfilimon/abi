//! Local inference bridge — a Zig HTTP client that dispatches completions to a
//! user-run local OpenAI-compatible server (llama-server, ollama, mlx-server,
//! lm-server, etc.). ABI does not embed or bundle any external inference engine;
//! the user starts the server separately and ABI talks to it over loopback HTTP.
//!
//! The bridge endpoint is configured via environment variables:
//!   `ABI_LLAMA_CPP_ENDPOINT` (default `http://127.0.0.1:8080`)
//!   `ABI_MLX_ENDPOINT`       (default `http://127.0.0.1:8081`)
//!
//! Model id prefixes routed to this bridge: `llama-cpp/`, `llama/`, `ollama/`,
//! `ollama-`, `lmstudio/`, `vllm/`, `mlx/`, `mlx-`.

const std = @import("std");
const connector = @import("connector.zig");
const http = @import("http.zig");
const json = @import("json.zig");

const ConnectorError = connector.ConnectorError;
const ConnectorConfig = connector.ConnectorConfig;
const Response = connector.Response;

/// Default endpoints for known local inference servers.
pub const LLAMA_CPP_DEFAULT_ENDPOINT = "http://127.0.0.1:8080";
pub const MLX_DEFAULT_ENDPOINT = "http://127.0.0.1:8081";

/// Re-export HTTP stream types for callers
pub const StreamCallback = http.StreamCallback;
pub const StreamChunk = http.StreamChunk;

/// Determine whether a model id should dispatch to a local inference bridge
/// rather than the in-process persona router. Matches the same prefixes that
/// `models.providerOf` classifies as `.local`.
pub fn isLocalBridgeModel(model: []const u8) bool {
    const prefixes = [_][]const u8{
        "llama-cpp/", "llama/", "ollama/", "ollama-",
        "lmstudio/",  "vllm/",  "mlx/",    "mlx-",
    };
    for (prefixes) |prefix| {
        if (std.mem.startsWith(u8, model, prefix)) return true;
    }
    return false;
}

/// Resolve the bridge endpoint for a model id. `mlx/` and `mlx-` prefixes use
/// the MLX endpoint; all other local-bridge prefixes use the llama-cpp endpoint.
/// Callers can pass an override endpoint (e.g. from env) via `override`.
pub fn endpointFor(model: []const u8, override: ?[]const u8) []const u8 {
    if (override) |ep| return ep;
    if (std.mem.startsWith(u8, model, "mlx/") or std.mem.startsWith(u8, model, "mlx-")) {
        return MLX_DEFAULT_ENDPOINT;
    }
    return LLAMA_CPP_DEFAULT_ENDPOINT;
}

/// Check whether the local inference server is reachable. Returns `false`
/// (not an error) when the server is down — callers should fall back to the
/// in-process persona router with a warning.
pub fn healthCheck(io: std.Io, allocator: std.mem.Allocator, endpoint: []const u8) bool {
    const config = ConnectorConfig{
        .api_key = "",
        .base_url = endpoint,
        .timeout_ms = 3000,
        .transport = .live,
    };
    var response = http.httpGetJson(io, allocator, config, "/health") catch return false;
    defer response.deinit(allocator);
    return response.status == 200;
}

/// Send a completion request to the local inference server. The request body
/// is OpenAI-compatible (`/v1/chat/completions`). Returns the response body
/// (owned by the caller). The caller is responsible for parsing the
/// `choices[0].message.content` field.
pub fn completeLive(
    io: std.Io,
    allocator: std.mem.Allocator,
    model: []const u8,
    input: []const u8,
) !Response {
    const endpoint = endpointFor(model, null);
    const config = ConnectorConfig{
        .api_key = "",
        .base_url = endpoint,
        .timeout_ms = 30000,
        .transport = .live,
    };

    // Build OpenAI-compatible messages array
    var messages = std.ArrayListUnmanaged(u8).empty;
    defer messages.deinit(allocator);
    try messages.appendSlice(allocator, "[{\"role\":\"user\",\"content\":");
    try json.appendJsonString(&messages, allocator, input);
    try messages.appendSlice(allocator, "}]");

    const body = try json.buildOpenAiBody(allocator, model, messages.items, false);
    defer allocator.free(body);

    return try http.httpPostJson(io, allocator, config, "/v1/chat/completions", body, &.{});
}

/// Send a streaming completion request to the local inference server via SSE.
/// The server must support OpenAI-compatible `/v1/chat/completions` with
/// `stream=true`. The callback is invoked for each token delta.
pub fn completeLiveStreaming(
    io: std.Io,
    allocator: std.mem.Allocator,
    model: []const u8,
    input: []const u8,
    on_chunk: http.StreamCallback,
    callback_ctx: *anyopaque,
) ![]const u8 {
    const endpoint = endpointFor(model, null);
    const config = ConnectorConfig{
        .api_key = "",
        .base_url = endpoint,
        .timeout_ms = 60000,
        .transport = .live,
    };

    // Build OpenAI-compatible messages array with stream=true
    var messages = std.ArrayListUnmanaged(u8).empty;
    defer messages.deinit(allocator);
    try messages.appendSlice(allocator, "[{\"role\":\"user\",\"content\":");
    try json.appendJsonString(&messages, allocator, input);
    try messages.appendSlice(allocator, "}]");

    const body = try json.buildOpenAiBody(allocator, model, messages.items, true);
    defer allocator.free(body);

    return try http.httpPostJsonStreamingIncremental(io, allocator, config, "/v1/chat/completions", body, on_chunk, callback_ctx, &.{});
}

/// Extract the completion text from an OpenAI-compatible JSON response.
/// Returns an owned slice. Returns `error.InvalidResponse` if the response
/// is not valid OpenAI JSON or the `choices` array is empty.
pub fn extractCompletion(allocator: std.mem.Allocator, body: []const u8) ![]u8 {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch return ConnectorError.InvalidResponse;
    defer parsed.deinit();
    const root = parsed.value.object;
    const choices = root.get("choices") orelse return ConnectorError.InvalidResponse;
    if (choices.array.items.len == 0) return ConnectorError.InvalidResponse;
    const choice = choices.array.items[0].object;
    const message = choice.get("message") orelse return ConnectorError.InvalidResponse;
    const content = message.object.get("content") orelse return ConnectorError.InvalidResponse;
    return try allocator.dupe(u8, content.string);
}

test "isLocalBridgeModel matches known prefixes" {
    try std.testing.expect(isLocalBridgeModel("llama-cpp/mistral"));
    try std.testing.expect(isLocalBridgeModel("llama/llama3"));
    try std.testing.expect(isLocalBridgeModel("ollama/qwen2"));
    try std.testing.expect(isLocalBridgeModel("ollama-qwen2"));
    try std.testing.expect(isLocalBridgeModel("lmstudio/phi3"));
    try std.testing.expect(isLocalBridgeModel("vllm/mixtral"));
    try std.testing.expect(isLocalBridgeModel("mlx/gemma"));
    try std.testing.expect(isLocalBridgeModel("mlx-gemma"));
    try std.testing.expect(!isLocalBridgeModel("claude-fable-5"));
    try std.testing.expect(!isLocalBridgeModel("abi-local"));
    try std.testing.expect(!isLocalBridgeModel("gpt-4"));
}

test "fable-5 alias is not a local bridge model" {
    // The catalog alias `fable-5` (-> `claude-fable-5`) must never match the
    // local-bridge prefix table — it routes to the anthropic provider, not here.
    try std.testing.expect(!isLocalBridgeModel("fable-5"));
    try std.testing.expect(!isLocalBridgeModel("fable5"));
}

test "endpointFor routes mlx to MLX endpoint" {
    try std.testing.expectEqualStrings(LLAMA_CPP_DEFAULT_ENDPOINT, endpointFor("llama-cpp/mistral", null));
    try std.testing.expectEqualStrings(MLX_DEFAULT_ENDPOINT, endpointFor("mlx/gemma", null));
    try std.testing.expectEqualStrings(MLX_DEFAULT_ENDPOINT, endpointFor("mlx-gemma", null));
    try std.testing.expectEqualStrings(LLAMA_CPP_DEFAULT_ENDPOINT, endpointFor("ollama/qwen2", null));
    // Override takes precedence
    try std.testing.expectEqualStrings("http://localhost:9999", endpointFor("mlx/gemma", "http://localhost:9999"));
}

test "extractCompletion parses OpenAI response" {
    const allocator = std.testing.allocator;
    const body =
        \\{"id":"chatcmpl-1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"Hello world"},"finish_reason":"stop"}]}
    ;
    const result = try extractCompletion(allocator, body);
    defer allocator.free(result);
    try std.testing.expectEqualStrings("Hello world", result);
}

test "extractCompletion rejects empty choices" {
    const allocator = std.testing.allocator;
    const body = "{\"choices\":[]}";
    try std.testing.expectError(ConnectorError.InvalidResponse, extractCompletion(allocator, body));
}

test "extractCompletion rejects invalid JSON" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(ConnectorError.InvalidResponse, extractCompletion(allocator, "not json"));
}

test {
    std.testing.refAllDecls(@This());
}
