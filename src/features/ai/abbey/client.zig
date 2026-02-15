//! Abbey LLM Client Abstraction
//!
//! Unified interface to multiple LLM backends:
//! - OpenAI, Anthropic, Ollama, HuggingFace
//! - Local GGUF models
//! - Streaming support
//! - Retry and fallback handling

const std = @import("std");
const time = @import("../../../services/shared/time.zig");
const sync = @import("../../../services/shared/sync.zig");
const types = @import("../core/types.zig");
const config = @import("../core/config.zig");
const build_options = @import("build_options");
const shared_utils = @import("../../../services/shared/utils.zig");

// Import web client if web feature is enabled
const web_client = if (build_options.enable_web) @import("../../web/client.zig") else @as(?void, null);
const web_enabled = build_options.enable_web;

// ============================================================================
// LLM Request/Response Types
// ============================================================================

/// Message for LLM conversation
pub const ChatMessage = struct {
    role: []const u8,
    content: []const u8,
    name: ?[]const u8 = null,

    pub fn fromMessage(msg: types.Message) ChatMessage {
        return .{
            .role = msg.role.toString(),
            .content = msg.content,
            .name = msg.name,
        };
    }
};

/// LLM completion request
pub const CompletionRequest = struct {
    messages: []const ChatMessage,
    model: []const u8 = "gpt-4",
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    max_tokens: u32 = 2048,
    stop: ?[]const []const u8 = null,
    stream: bool = false,
    tools: ?[]const Tool = null,
    tool_choice: ?[]const u8 = null,

    pub const Tool = struct {
        name: []const u8,
        description: []const u8,
        parameters: []const u8, // JSON schema
    };
};

/// LLM completion response
pub const CompletionResponse = struct {
    content: []const u8,
    finish_reason: FinishReason,
    tool_calls: ?[]const ToolCall = null,
    usage: Usage,
    model: []const u8,
    latency_ms: i64,

    pub const FinishReason = enum {
        stop,
        length,
        tool_calls,
        content_filter,
        unknown,
    };

    pub const ToolCall = struct {
        id: []const u8,
        name: []const u8,
        arguments: []const u8,
    };

    pub const Usage = struct {
        prompt_tokens: usize,
        completion_tokens: usize,
        total_tokens: usize,
    };

    pub fn deinit(self: *CompletionResponse, allocator: std.mem.Allocator) void {
        if (self.tool_calls) |calls| {
            for (calls) |call| {
                allocator.free(call.id);
                allocator.free(call.name);
                allocator.free(call.arguments);
            }
            allocator.free(calls);
        }
        allocator.free(self.content);
        allocator.free(self.model);
    }
};

/// Streaming chunk
pub const StreamChunk = struct {
    content: []const u8,
    is_final: bool,
    finish_reason: ?CompletionResponse.FinishReason = null,
};

// ============================================================================
// LLM Client Interface
// ============================================================================

/// Error set for LLM client operations.
/// Covers the known backend error surface used by the client interface.
pub const ClientError = error{
    /// The requested operation is not implemented
    NotImplemented,
    /// Connection to the backend failed
    ConnectionRefused,
    /// Connection timed out
    ConnectionTimedOut,
    /// Temporary failure that may be retried
    TemporaryFailure,
    /// Invalid argument provided
    InvalidArgument,
    /// Rate limit exceeded
    RateLimitExceeded,
    /// Authentication failed
    AuthenticationFailed,
    /// Backend returned an error
    BackendError,
    /// Request was malformed
    MalformedRequest,
    /// Response could not be parsed
    ResponseParseError,
} || std.mem.Allocator.Error;

/// Abstract LLM client interface
pub const LLMClient = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    /// VTable uses ClientError so the polymorphic interface stays explicit and stable.
    const VTable = struct {
        complete: *const fn (*anyopaque, CompletionRequest) ClientError!CompletionResponse,
        streamComplete: *const fn (*anyopaque, CompletionRequest, *StreamCallback) ClientError!void,
        isAvailable: *const fn (*anyopaque) bool,
        getBackendName: *const fn (*anyopaque) []const u8,
        deinit: *const fn (*anyopaque) void,
    };

    pub const StreamCallback = fn (chunk: StreamChunk) void;

    pub fn complete(self: LLMClient, request: CompletionRequest) ClientError!CompletionResponse {
        return self.vtable.complete(self.ptr, request);
    }

    pub fn streamComplete(
        self: LLMClient,
        request: CompletionRequest,
        callback: *StreamCallback,
    ) ClientError!void {
        return self.vtable.streamComplete(self.ptr, request, callback);
    }

    pub fn isAvailable(self: LLMClient) bool {
        return self.vtable.isAvailable(self.ptr);
    }

    pub fn getBackendName(self: LLMClient) []const u8 {
        return self.vtable.getBackendName(self.ptr);
    }

    pub fn deinit(self: LLMClient) void {
        self.vtable.deinit(self.ptr);
    }
};

// ============================================================================
// Echo Backend (for testing)
// ============================================================================

pub const EchoBackend = struct {
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{ .allocator = allocator };
    }

    pub fn deinit(self: *Self) void {
        _ = self;
    }

    pub fn complete(self: *Self, request: CompletionRequest) ClientError!CompletionResponse {
        const start = types.getTimestampMs();

        // Build echo response
        var response = std.ArrayListUnmanaged(u8).empty;
        try response.appendSlice(self.allocator, "[Abbey Echo] Received ");

        const len_str = try std.fmt.allocPrint(self.allocator, "{}", .{request.messages.len});
        defer self.allocator.free(len_str);
        try response.appendSlice(self.allocator, len_str);

        try response.appendSlice(self.allocator, " messages. Last: ");

        if (request.messages.len > 0) {
            const last = request.messages[request.messages.len - 1];
            const preview_len = @min(100, last.content.len);
            try response.appendSlice(self.allocator, last.content[0..preview_len]);
            if (last.content.len > 100) {
                try response.appendSlice(self.allocator, "...");
            }
        }

        const content = try response.toOwnedSlice(self.allocator);
        errdefer self.allocator.free(content);
        const model = try self.allocator.dupe(u8, "echo");
        const end = types.getTimestampMs();

        return CompletionResponse{
            .content = content,
            .finish_reason = .stop,
            .usage = .{
                .prompt_tokens = estimateTokens(request.messages),
                .completion_tokens = (content.len + 3) / 4,
                .total_tokens = estimateTokens(request.messages) + (content.len + 3) / 4,
            },
            .model = model,
            .latency_ms = end - start,
        };
    }

    pub fn streamComplete(
        self: *Self,
        request: CompletionRequest,
        callback: *LLMClient.StreamCallback,
    ) ClientError!void {
        var response = try self.complete(request);
        defer response.deinit(self.allocator);

        callback(.{
            .content = response.content,
            .is_final = true,
            .finish_reason = .stop,
        });
    }

    pub fn isAvailable(self: *Self) bool {
        _ = self;
        return true;
    }

    pub fn getBackendName(self: *Self) []const u8 {
        _ = self;
        return "echo";
    }

    pub fn client(self: *Self) LLMClient {
        return .{
            .ptr = self,
            .vtable = &.{
                .complete = @ptrCast(&complete),
                .streamComplete = @ptrCast(&streamComplete),
                .isAvailable = @ptrCast(&isAvailable),
                .getBackendName = @ptrCast(&getBackendName),
                .deinit = @ptrCast(&deinit),
            },
        };
    }

    fn estimateTokens(messages: []const ChatMessage) usize {
        var total: usize = 0;
        for (messages) |msg| {
            total += (msg.content.len + 3) / 4;
        }
        return total;
    }
};

// ============================================================================
// OpenAI Backend (Stub - would use HTTP client)
// ============================================================================

pub const OpenAIBackend = struct {
    allocator: std.mem.Allocator,
    api_key: ?[]const u8,
    base_url: []const u8,
    model: []const u8,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, llm_config: config.LLMConfig) !Self {
        return Self{
            .allocator = allocator,
            .api_key = llm_config.api_key,
            .base_url = llm_config.base_url orelse "https://api.openai.com/v1",
            .model = llm_config.model,
        };
    }

    pub fn deinit(self: *Self) void {
        _ = self;
    }

    pub fn complete(self: *Self, request: CompletionRequest) ClientError!CompletionResponse {
        // Check if web feature is enabled
        if (!web_enabled) {
            return error.NotImplemented;
        }

        const api_key = self.api_key orelse return error.AuthenticationFailed;
        var timer = time.Timer.start() catch return error.BackendError;

        // Build URL
        const url = try std.fmt.allocPrint(self.allocator, "{s}/chat/completions", .{self.base_url});
        defer self.allocator.free(url);

        // Build JSON request using allocPrint
        var json_buffer = std.ArrayListUnmanaged(u8).empty;
        defer json_buffer.deinit(self.allocator);

        try json_buffer.appendSlice(self.allocator, "{\"model\":\"");
        try json_buffer.appendSlice(self.allocator, request.model);
        try json_buffer.appendSlice(self.allocator, "\",\"messages\":[");

        for (request.messages, 0..) |msg, i| {
            if (i > 0) try json_buffer.appendSlice(self.allocator, ",");
            try json_buffer.appendSlice(self.allocator, "{\"role\":\"");
            try json_buffer.appendSlice(self.allocator, msg.role);
            try json_buffer.appendSlice(self.allocator, "\",\"content\":\"");
            try appendJsonEscaped(&json_buffer, self.allocator, msg.content);
            try json_buffer.appendSlice(self.allocator, "\"}");
        }

        const temp_str = try std.fmt.allocPrint(self.allocator, "],\"temperature\":{d},\"max_tokens\":{d},\"stream\":false", .{ request.temperature, request.max_tokens });
        defer self.allocator.free(temp_str);
        try json_buffer.appendSlice(self.allocator, temp_str);

        if (request.top_p < 1.0) {
            const top_p_str = try std.fmt.allocPrint(self.allocator, ",\"top_p\":{d}", .{request.top_p});
            defer self.allocator.free(top_p_str);
            try json_buffer.appendSlice(self.allocator, top_p_str);
        }

        try json_buffer.appendSlice(self.allocator, "}");

        // Make HTTP request with Authorization header
        var http_client = web_client.HttpClient.init(self.allocator) catch return error.ConnectionRefused;
        defer http_client.deinit();

        // Build Authorization header
        const auth_header = try std.fmt.allocPrint(self.allocator, "Bearer {s}", .{api_key});
        defer self.allocator.free(auth_header);

        const headers = [_]std.http.Header{
            .{ .name = "Authorization", .value = auth_header },
        };

        const response = http_client.requestWithOptions(
            .POST,
            url,
            json_buffer.items,
            .{
                .content_type = "application/json",
                .extra_headers = &headers,
            },
        ) catch |err| {
            return switch (err) {
                error.ConnectionRefused => error.ConnectionRefused,
                error.InvalidUrl => error.MalformedRequest,
                else => error.BackendError,
            };
        };
        defer http_client.freeResponse(response);

        // Handle HTTP status codes
        if (response.status == 401) return error.AuthenticationFailed;
        if (response.status == 429) return error.RateLimitExceeded;
        if (response.status >= 500) return error.BackendError;
        if (response.status != 200) return error.BackendError;

        // Parse JSON response
        const parsed = std.json.parseFromSlice(
            OpenAIResponse,
            self.allocator,
            response.body,
            .{ .ignore_unknown_fields = true },
        ) catch return error.ResponseParseError;
        defer parsed.deinit();

        const elapsed_ns = timer.read();
        const elapsed_ms: i64 = @intCast(elapsed_ns / std.time.ns_per_ms);

        if (parsed.value.choices.len == 0) {
            return error.ResponseParseError;
        }

        const choice = parsed.value.choices[0];

        // Build response
        const content = try self.allocator.dupe(u8, choice.message.content);
        const model = try self.allocator.dupe(u8, parsed.value.model);

        const finish_reason: CompletionResponse.FinishReason = if (std.mem.eql(u8, choice.finish_reason, "stop"))
            .stop
        else if (std.mem.eql(u8, choice.finish_reason, "length"))
            .length
        else
            .unknown;

        return CompletionResponse{
            .content = content,
            .finish_reason = finish_reason,
            .tool_calls = null,
            .usage = .{
                .prompt_tokens = parsed.value.usage.prompt_tokens,
                .completion_tokens = parsed.value.usage.completion_tokens,
                .total_tokens = parsed.value.usage.total_tokens,
            },
            .model = model,
            .latency_ms = elapsed_ms,
        };
    }

    const OpenAIResponse = struct {
        choices: []struct {
            message: struct {
                role: []const u8,
                content: []const u8,
            },
            finish_reason: []const u8,
        },
        usage: struct {
            prompt_tokens: usize,
            completion_tokens: usize,
            total_tokens: usize,
        },
        model: []const u8,
    };

    pub fn isAvailable(self: *Self) bool {
        return self.api_key != null;
    }

    pub fn getBackendName(self: *Self) []const u8 {
        _ = self;
        return "openai";
    }

    pub fn streamComplete(
        self: *Self,
        request: CompletionRequest,
        callback: *LLMClient.StreamCallback,
    ) ClientError!void {
        // Fallback: call complete() and emit result as single chunk
        // Real implementation would use SSE streaming with "stream":true
        var response = try self.complete(request);
        defer response.deinit(self.allocator);

        callback(.{
            .content = response.content,
            .is_final = true,
            .finish_reason = .stop,
        });
    }

    pub fn client(self: *Self) LLMClient {
        return .{
            .ptr = self,
            .vtable = &.{
                .complete = @ptrCast(&complete),
                .streamComplete = @ptrCast(&streamComplete),
                .isAvailable = @ptrCast(&isAvailable),
                .getBackendName = @ptrCast(&getBackendName),
                .deinit = @ptrCast(&deinit),
            },
        };
    }
};

// ============================================================================
// Ollama Backend (Stub - would use HTTP client)
// ============================================================================

pub const OllamaBackend = struct {
    allocator: std.mem.Allocator,
    host: []const u8,
    model: []const u8,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, llm_config: config.LLMConfig) Self {
        return Self{
            .allocator = allocator,
            .host = llm_config.base_url orelse "http://127.0.0.1:11434",
            .model = llm_config.model,
        };
    }

    pub fn deinit(self: *Self) void {
        _ = self;
    }

    pub fn complete(self: *Self, request: CompletionRequest) ClientError!CompletionResponse {
        // Check if web feature is enabled
        if (!web_enabled) {
            return error.NotImplemented;
        }

        var timer = time.Timer.start() catch return error.BackendError;

        // Build URL
        const url = try std.fmt.allocPrint(self.allocator, "{s}/api/chat", .{self.host});
        defer self.allocator.free(url);

        // Build JSON request using allocPrint
        var json_buffer = std.ArrayListUnmanaged(u8).empty;
        defer json_buffer.deinit(self.allocator);

        try json_buffer.appendSlice(self.allocator, "{\"model\":\"");
        try json_buffer.appendSlice(self.allocator, request.model);
        try json_buffer.appendSlice(self.allocator, "\",\"messages\":[");

        for (request.messages, 0..) |msg, i| {
            if (i > 0) try json_buffer.appendSlice(self.allocator, ",");
            try json_buffer.appendSlice(self.allocator, "{\"role\":\"");
            try json_buffer.appendSlice(self.allocator, msg.role);
            try json_buffer.appendSlice(self.allocator, "\",\"content\":\"");
            try appendJsonEscaped(&json_buffer, self.allocator, msg.content);
            try json_buffer.appendSlice(self.allocator, "\"}");
        }

        const opts_str = try std.fmt.allocPrint(self.allocator, "],\"stream\":false,\"options\":{{\"temperature\":{d},\"top_p\":{d}}}", .{ request.temperature, request.top_p });
        defer self.allocator.free(opts_str);
        try json_buffer.appendSlice(self.allocator, opts_str);

        if (request.max_tokens > 0) {
            const predict_str = try std.fmt.allocPrint(self.allocator, ",\"num_predict\":{d}", .{request.max_tokens});
            defer self.allocator.free(predict_str);
            try json_buffer.appendSlice(self.allocator, predict_str);
        }

        try json_buffer.appendSlice(self.allocator, "}");

        // Make HTTP request
        var http_client = web_client.HttpClient.init(self.allocator) catch return error.ConnectionRefused;
        defer http_client.deinit();

        const response = http_client.postJson(url, json_buffer.items) catch |err| {
            return switch (err) {
                error.ConnectionRefused => error.ConnectionRefused,
                error.InvalidUrl => error.MalformedRequest,
                else => error.BackendError,
            };
        };
        defer http_client.freeResponse(response);

        if (response.status != 200) {
            return error.BackendError;
        }

        // Parse JSON response
        const parsed = std.json.parseFromSlice(
            OllamaResponse,
            self.allocator,
            response.body,
            .{ .ignore_unknown_fields = true },
        ) catch return error.ResponseParseError;
        defer parsed.deinit();

        const elapsed_ns = timer.read();
        const elapsed_ms: i64 = @intCast(elapsed_ns / std.time.ns_per_ms);

        // Build response
        const content = try self.allocator.dupe(u8, parsed.value.message.content);
        const model = try self.allocator.dupe(u8, request.model);

        return CompletionResponse{
            .content = content,
            .finish_reason = if (parsed.value.done) .stop else .unknown,
            .tool_calls = null,
            .usage = .{
                .prompt_tokens = parsed.value.prompt_eval_count orelse 0,
                .completion_tokens = parsed.value.eval_count orelse 0,
                .total_tokens = (parsed.value.prompt_eval_count orelse 0) + (parsed.value.eval_count orelse 0),
            },
            .model = model,
            .latency_ms = elapsed_ms,
        };
    }

    const OllamaResponse = struct {
        message: struct {
            role: []const u8,
            content: []const u8,
        },
        done: bool,
        total_duration: ?i64 = null,
        prompt_eval_count: ?usize = null,
        eval_count: ?usize = null,
    };

    pub fn isAvailable(self: *Self) bool {
        if (!web_enabled) return false;

        var http_client = web_client.HttpClient.init(self.allocator) catch return false;
        defer http_client.deinit();

        const url = std.fmt.allocPrint(self.allocator, "{s}/api/tags", .{self.host}) catch return false;
        defer self.allocator.free(url);

        const response = http_client.get(url) catch return false;
        defer http_client.freeResponse(response);

        return response.status == 200;
    }

    pub fn getBackendName(self: *Self) []const u8 {
        _ = self;
        return "ollama";
    }

    pub fn streamComplete(
        self: *Self,
        request: CompletionRequest,
        callback: *LLMClient.StreamCallback,
    ) ClientError!void {
        // Fallback: call complete() and emit result as single chunk
        // Real implementation would use Ollama streaming with "stream":true
        var response = try self.complete(request);
        defer response.deinit(self.allocator);

        callback(.{
            .content = response.content,
            .is_final = true,
            .finish_reason = .stop,
        });
    }

    pub fn client(self: *Self) LLMClient {
        return .{
            .ptr = self,
            .vtable = &.{
                .complete = @ptrCast(&complete),
                .streamComplete = @ptrCast(&streamComplete),
                .isAvailable = @ptrCast(&isAvailable),
                .getBackendName = @ptrCast(&getBackendName),
                .deinit = @ptrCast(&deinit),
            },
        };
    }
};

// ============================================================================
// Client Factory
// ============================================================================

/// Create an LLM client based on configuration
pub fn createClient(allocator: std.mem.Allocator, llm_config: config.LLMConfig) !ClientWrapper {
    switch (llm_config.backend) {
        .echo => {
            const backend = try allocator.create(EchoBackend);
            backend.* = EchoBackend.init(allocator);
            return .{ .echo = backend };
        },
        .openai => {
            const backend = try allocator.create(OpenAIBackend);
            backend.* = try OpenAIBackend.init(allocator, llm_config);
            return .{ .openai = backend };
        },
        .ollama => {
            const backend = try allocator.create(OllamaBackend);
            backend.* = OllamaBackend.init(allocator, llm_config);
            return .{ .ollama = backend };
        },
        else => {
            // Default to echo for unsupported backends
            const backend = try allocator.create(EchoBackend);
            backend.* = EchoBackend.init(allocator);
            return .{ .echo = backend };
        },
    }
}

pub const ClientWrapper = union(enum) {
    echo: *EchoBackend,
    openai: *OpenAIBackend,
    ollama: *OllamaBackend,

    pub fn deinit(self: *ClientWrapper, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .echo => |e| {
                e.deinit();
                allocator.destroy(e);
            },
            .openai => |o| {
                o.deinit();
                allocator.destroy(o);
            },
            .ollama => |ol| {
                ol.deinit();
                allocator.destroy(ol);
            },
        }
    }

    pub fn complete(self: *ClientWrapper, request: CompletionRequest) ClientError!CompletionResponse {
        return switch (self.*) {
            .echo => |e| e.complete(request),
            .openai => |o| o.complete(request),
            .ollama => |ol| ol.complete(request),
        };
    }

    pub fn isAvailable(self: *ClientWrapper) bool {
        return switch (self.*) {
            .echo => |e| e.isAvailable(),
            .openai => |o| o.isAvailable(),
            .ollama => |ol| ol.isAvailable(),
        };
    }

    pub fn getBackendName(self: *ClientWrapper) []const u8 {
        return switch (self.*) {
            .echo => |e| e.getBackendName(),
            .openai => |o| o.getBackendName(),
            .ollama => |ol| ol.getBackendName(),
        };
    }

    pub fn streamComplete(
        self: *ClientWrapper,
        request: CompletionRequest,
        callback: *LLMClient.StreamCallback,
    ) ClientError!void {
        return switch (self.*) {
            .echo => |e| e.streamComplete(request, callback),
            .openai => |o| o.streamComplete(request, callback),
            .ollama => |ol| ol.streamComplete(request, callback),
        };
    }
};

// ============================================================================
// Retry Handler
// ============================================================================

pub const RetryHandler = struct {
    max_retries: u32,
    base_delay_ms: u32,
    max_delay_ms: u32,
    retry_count: u32 = 0,

    pub fn init(max_retries: u32, base_delay_ms: u32) RetryHandler {
        return .{
            .max_retries = max_retries,
            .base_delay_ms = base_delay_ms,
            .max_delay_ms = base_delay_ms * 16, // Max 16x base
        };
    }

    /// Determines if an error should trigger a retry.
    /// Only known transient ClientError values are retried.
    pub fn shouldRetry(self: *RetryHandler, err: ClientError) bool {
        if (self.retry_count >= self.max_retries) return false;

        // Retry on transient errors.
        return switch (err) {
            error.ConnectionRefused,
            error.ConnectionTimedOut,
            error.TemporaryFailure,
            => true,
            else => false,
        };
    }

    pub fn getDelay(self: *RetryHandler) u32 {
        const delay = self.base_delay_ms * std.math.pow(u32, 2, self.retry_count);
        self.retry_count += 1;
        return @min(delay, self.max_delay_ms);
    }

    pub fn reset(self: *RetryHandler) void {
        self.retry_count = 0;
    }
};

// ============================================================================
// Helper Functions
// ============================================================================

/// Append a JSON-escaped string to an ArrayListUnmanaged
fn appendJsonEscaped(buffer: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, s: []const u8) !void {
    for (s) |c| {
        switch (c) {
            '"' => try buffer.appendSlice(allocator, "\\\""),
            '\\' => try buffer.appendSlice(allocator, "\\\\"),
            '\n' => try buffer.appendSlice(allocator, "\\n"),
            '\r' => try buffer.appendSlice(allocator, "\\r"),
            '\t' => try buffer.appendSlice(allocator, "\\t"),
            else => try buffer.append(allocator, c),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

test "echo backend" {
    const allocator = std.testing.allocator;

    var backend = EchoBackend.init(allocator);
    defer backend.deinit();

    const messages = [_]ChatMessage{
        .{ .role = "user", .content = "Hello, Abbey!" },
    };

    var response = try backend.complete(.{
        .messages = &messages,
    });
    defer response.deinit(allocator);

    try std.testing.expect(response.content.len > 0);
    try std.testing.expectEqual(CompletionResponse.FinishReason.stop, response.finish_reason);
}

test "client factory echo" {
    const allocator = std.testing.allocator;

    var client = try createClient(allocator, .{ .backend = .echo });
    defer client.deinit(allocator);

    try std.testing.expect(client.isAvailable());
    try std.testing.expectEqualStrings("echo", client.getBackendName());
}

test "retry handler" {
    var handler = RetryHandler.init(3, 100);

    try std.testing.expect(handler.shouldRetry(error.ConnectionTimedOut));
    try std.testing.expect(!handler.shouldRetry(error.InvalidArgument));

    const delay1 = handler.getDelay();
    const delay2 = handler.getDelay();

    try std.testing.expect(delay2 > delay1);
}
