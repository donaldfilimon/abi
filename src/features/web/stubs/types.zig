//! Type definitions for the web module stub.
//!
//! These types mirror the real implementations to maintain API compatibility
//! when the web feature is disabled.

const std = @import("std");

/// Web module error type.
pub const WebError = error{
    /// The web feature is disabled in the build configuration.
    WebDisabled,
};

/// HTTP response structure.
pub const Response = struct {
    /// HTTP status code.
    status: u16,
    /// Response body content.
    body: []const u8,
};

/// HTTP request configuration options.
pub const RequestOptions = struct {
    /// Maximum bytes to read from response body.
    max_response_bytes: usize = 1024 * 1024,
    /// User-Agent header value.
    user_agent: []const u8 = "abi-http",
    /// Whether to follow HTTP redirects.
    follow_redirects: bool = true,
    /// Maximum number of redirects to follow.
    redirect_limit: u16 = 3,
    /// Content-Type header for requests with a body.
    content_type: ?[]const u8 = null,
    /// Additional headers to include in the request.
    extra_headers: []const std.http.Header = &.{},

    /// Hard upper limit for response size (100MB).
    pub const MAX_ALLOWED_RESPONSE_BYTES: usize = 100 * 1024 * 1024;

    /// Returns the effective max response bytes, capped at the hard limit.
    pub fn effectiveMaxResponseBytes(self: RequestOptions) usize {
        return @min(self.max_response_bytes, MAX_ALLOWED_RESPONSE_BYTES);
    }
};

/// Weather client configuration.
pub const WeatherConfig = struct {
    /// Base URL for the weather API.
    base_url: []const u8 = "https://api.open-meteo.com/v1/forecast",
    /// Whether to include current weather conditions.
    include_current: bool = true,
};

/// JSON value type for response parsing.
pub const JsonValue = std.json.Value;
/// Parsed JSON with ownership.
pub const ParsedJson = std.json.Parsed(JsonValue);

// ============================================================================
// Stub-only types (chat, routing, server, middleware)
// ============================================================================

pub const ProfileType = enum { assistant, coder, writer, analyst, companion, docs, reviewer, minimal, abbey, aviva, abi, ralph, ava };

pub const ChatRequest = struct {
    content: []const u8,
    user_id: ?[]const u8 = null,
    session_id: ?[]const u8 = null,
    profile: ?[]const u8 = null,
    context: ?[]const u8 = null,
    max_tokens: ?u32 = null,
    temperature: ?f32 = null,
    pub fn deinit(_: *ChatRequest, _: std.mem.Allocator) void {}
    pub fn dupe(_: std.mem.Allocator, other: ChatRequest) !ChatRequest {
        return other;
    }
};

pub const ChatResponse = struct {
    content: []const u8,
    profile: []const u8,
    confidence: f32,
    latency_ms: u64,
    code_blocks: ?[]const CodeBlock = null,
    references: ?[]const Source = null,
    request_id: ?[]const u8 = null,
};

pub const ChatResult = struct { status: u16, body: []const u8 };
pub const CodeBlock = struct { language: []const u8, code: []const u8 };
pub const Source = struct { title: []const u8, url: ?[]const u8 = null, confidence: f32 };

pub const ChatHandler = struct {
    allocator: std.mem.Allocator,
    pub fn init(allocator: std.mem.Allocator) ChatHandler {
        return .{ .allocator = allocator };
    }
    pub fn handleChat(_: *ChatHandler, _: []const u8) ![]const u8 {
        return error.FeatureDisabled;
    }
    pub fn handleAbbeyChat(_: *ChatHandler, _: []const u8) ![]const u8 {
        return error.FeatureDisabled;
    }
    pub fn handleAvivaChat(_: *ChatHandler, _: []const u8) ![]const u8 {
        return error.FeatureDisabled;
    }
    pub fn handleChatWithProfileResult(_: *ChatHandler, _: []const u8, _: ?ProfileType) !ChatResult {
        return error.FeatureDisabled;
    }
    pub fn listProfiles(_: *ChatHandler) ![]const u8 {
        return error.FeatureDisabled;
    }
    pub fn getMetrics(_: *ChatHandler) ![]const u8 {
        return error.FeatureDisabled;
    }
    pub fn formatError(_: *ChatHandler, _: []const u8, _: []const u8, _: ?[]const u8) ![]const u8 {
        return error.FeatureDisabled;
    }
};

pub const Method = enum { GET, POST, PUT, DELETE, PATCH, OPTIONS, HEAD };

pub const Route = struct {
    path: []const u8,
    method: Method,
    description: []const u8,
    requires_auth: bool = false,
};

pub const RouteContext = struct {
    allocator: std.mem.Allocator,
    body: []const u8 = "",
    response_status: u16 = 200,
    response_content_type: []const u8 = "application/json",
    pub fn init(allocator: std.mem.Allocator, _: *ChatHandler) RouteContext {
        return .{ .allocator = allocator };
    }
    pub fn deinit(_: *RouteContext) void {}
    pub fn write(_: *RouteContext, _: []const u8) !void {}
    pub fn setStatus(self: *RouteContext, status: u16) void {
        self.response_status = status;
    }
    pub fn setContentType(self: *RouteContext, content_type: []const u8) void {
        self.response_content_type = content_type;
    }
};

pub const Router = struct {
    allocator: std.mem.Allocator,
    pub fn init(allocator: std.mem.Allocator, _: *ChatHandler) Router {
        return .{ .allocator = allocator };
    }
    pub fn match(_: *const Router, _: []const u8, _: Method) ?Route {
        return null;
    }
    pub fn getRouteDefinitions(_: *const Router) []const Route {
        return &.{};
    }
};

pub const ServerConfig = struct {
    host: []const u8 = "127.0.0.1",
    port: u16 = 8080,
    max_connections: usize = 1024,
    read_timeout_ms: u32 = 30_000,
    write_timeout_ms: u32 = 30_000,
    keep_alive: bool = true,
    keep_alive_timeout_ms: u32 = 5_000,
    worker_threads: u32 = 4,
};
pub const ServerState = enum { stopped, starting, running, stopping };
pub const ServerStats = struct {};
pub const ServerError = error{FeatureDisabled};

pub const Server = struct {
    pub fn init(_: std.mem.Allocator, _: anytype) Server {
        return .{};
    }
    pub fn deinit(_: *Server) void {}
};

pub const BUCKET_COUNT: usize = 8;
pub const bucket_bounds_us: [BUCKET_COUNT]u64 = .{
    100,
    500,
    1_000,
    5_000,
    50_000,
    200_000,
    1_000_000,
    std.math.maxInt(u64),
};

pub const RequestMetrics = struct {
    start_ns: i128,
};

pub const MetricsSnapshot = struct {
    total_requests: u64,
    total_errors: u64,
    active_requests: u64,
    request_durations_us: [BUCKET_COUNT]u64,
    status_counts: [6]u64,
};

pub const MetricsMiddleware = struct {
    total_requests: u64 = 0,
    total_errors: u64 = 0,
    active_requests: u64 = 0,
    request_durations_us: [BUCKET_COUNT]u64 = .{0} ** BUCKET_COUNT,
    status_counts: [6]u64 = .{0} ** 6,

    pub fn init() MetricsMiddleware {
        return .{};
    }
    pub fn processRequest(_: *MetricsMiddleware) RequestMetrics {
        return .{ .start_ns = 0 };
    }
    pub fn recordResponse(_: *MetricsMiddleware, _: RequestMetrics, _: u16) void {}
    pub fn getSnapshot(_: *const MetricsMiddleware) MetricsSnapshot {
        return .{
            .total_requests = 0,
            .total_errors = 0,
            .active_requests = 0,
            .request_durations_us = .{0} ** BUCKET_COUNT,
            .status_counts = .{0} ** 6,
        };
    }
    pub fn formatPrometheus(_: *const MetricsMiddleware, _: std.mem.Allocator) ![]u8 {
        return error.FeatureDisabled;
    }
    pub fn reset(_: *MetricsMiddleware) void {}
};

test {
    std.testing.refAllDecls(@This());
}
