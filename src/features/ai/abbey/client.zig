//! Abbey LLM Client Abstraction
//!
//! Unified interface to multiple LLM backends:
//! - OpenAI, Anthropic, Ollama, HuggingFace
//! - Local GGUF models
//! - Streaming support
//! - Retry and fallback handling

const std = @import("std");
const types = @import("core/types.zig");
const config = @import("core/config.zig");

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
/// Uses anyerror in VTable because different backends may return different errors,
/// and the polymorphic interface needs to accommodate all possible backend errors.
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

    /// VTable uses anyerror because this is a polymorphic interface where
    /// different backend implementations may return different error sets.
    /// Callers should handle errors generically or check for specific known errors.
    const VTable = struct {
        complete: *const fn (*anyopaque, CompletionRequest) anyerror!CompletionResponse,
        streamComplete: *const fn (*anyopaque, CompletionRequest, *StreamCallback) anyerror!void,
        isAvailable: *const fn (*anyopaque) bool,
        getBackendName: *const fn (*anyopaque) []const u8,
        deinit: *const fn (*anyopaque) void,
    };

    pub const StreamCallback = fn (chunk: StreamChunk) void;

    pub fn complete(self: LLMClient, request: CompletionRequest) !CompletionResponse {
        return self.vtable.complete(self.ptr, request);
    }

    pub fn streamComplete(self: LLMClient, request: CompletionRequest, callback: *StreamCallback) !void {
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

    pub fn complete(self: *Self, request: CompletionRequest) !CompletionResponse {
        const start = types.getTimestampMs();

        // Build echo response
        var response = std.ArrayListUnmanaged(u8){};
        try response.appendSlice(self.allocator, "[Abbey Echo] Received ");
        try response.appendSlice(self.allocator, std.fmt.comptimePrint("{}", .{request.messages.len}));
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
        const end = types.getTimestampMs();

        return CompletionResponse{
            .content = content,
            .finish_reason = .stop,
            .usage = .{
                .prompt_tokens = estimateTokens(request.messages),
                .completion_tokens = (content.len + 3) / 4,
                .total_tokens = estimateTokens(request.messages) + (content.len + 3) / 4,
            },
            .model = "echo",
            .latency_ms = end - start,
        };
    }

    pub fn streamComplete(self: *Self, request: CompletionRequest, callback: *LLMClient.StreamCallback) !void {
        const response = try self.complete(request);
        defer self.allocator.free(response.content);

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

    pub fn complete(self: *Self, request: CompletionRequest) !CompletionResponse {
        _ = self;
        _ = request;
        // In production, would make HTTP request to OpenAI API
        return error.NotImplemented;
    }

    pub fn isAvailable(self: *Self) bool {
        return self.api_key != null;
    }

    pub fn getBackendName(self: *Self) []const u8 {
        _ = self;
        return "openai";
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

    pub fn complete(self: *Self, request: CompletionRequest) !CompletionResponse {
        _ = self;
        _ = request;
        // In production, would make HTTP request to Ollama
        return error.NotImplemented;
    }

    pub fn isAvailable(self: *Self) bool {
        _ = self;
        // Would check if Ollama is running
        return false;
    }

    pub fn getBackendName(self: *Self) []const u8 {
        _ = self;
        return "ollama";
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

    pub fn complete(self: *ClientWrapper, request: CompletionRequest) !CompletionResponse {
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
    /// Uses anyerror because this function needs to handle errors from any backend,
    /// which may return different error sets. Only known transient errors are retried.
    pub fn shouldRetry(self: *RetryHandler, err: anyerror) bool {
        if (self.retry_count >= self.max_retries) return false;

        // Retry on transient errors - anyerror is intentional here because
        // we need to handle errors from various backends with different error sets
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
// Tests
// ============================================================================

test "echo backend" {
    const allocator = std.testing.allocator;

    var backend = EchoBackend.init(allocator);
    defer backend.deinit();

    const messages = [_]ChatMessage{
        .{ .role = "user", .content = "Hello, Abbey!" },
    };

    const response = try backend.complete(.{
        .messages = &messages,
    });
    defer allocator.free(response.content);

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
