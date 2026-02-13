//! Cloud Function Types
//!
//! Common types and structures for cloud function adapters.
//! Provides a unified interface for AWS Lambda, GCP Functions, and Azure Functions.

const std = @import("std");
const time = @import("../../services/shared/time.zig");

/// Supported cloud providers.
pub const CloudProvider = enum {
    aws_lambda,
    gcp_functions,
    azure_functions,

    pub fn name(self: CloudProvider) []const u8 {
        return switch (self) {
            .aws_lambda => "AWS Lambda",
            .gcp_functions => "Google Cloud Functions",
            .azure_functions => "Azure Functions",
        };
    }

    pub fn runtimeIdentifier(self: CloudProvider) []const u8 {
        return switch (self) {
            .aws_lambda => "provided.al2023",
            .gcp_functions => "zig-runtime",
            .azure_functions => "custom",
        };
    }
};

/// HTTP method for cloud events.
pub const HttpMethod = enum {
    GET,
    POST,
    PUT,
    DELETE,
    PATCH,
    HEAD,
    OPTIONS,

    pub fn fromString(s: []const u8) ?HttpMethod {
        return std.StaticStringMap(HttpMethod).initComptime(.{
            .{ "GET", .GET },
            .{ "POST", .POST },
            .{ "PUT", .PUT },
            .{ "DELETE", .DELETE },
            .{ "PATCH", .PATCH },
            .{ "HEAD", .HEAD },
            .{ "OPTIONS", .OPTIONS },
        }).get(s);
    }

    pub fn toString(self: HttpMethod) []const u8 {
        return @tagName(self);
    }
};

/// Cloud event representing an incoming request.
/// Unified across all cloud providers.
pub const CloudEvent = struct {
    /// Unique identifier for this event/request.
    request_id: []const u8,

    /// The cloud provider this event originated from.
    provider: CloudProvider,

    /// HTTP method (for HTTP-triggered functions).
    method: ?HttpMethod = null,

    /// Request path (for HTTP-triggered functions).
    path: ?[]const u8 = null,

    /// Query parameters as key-value pairs.
    query_params: ?std.StringHashMap([]const u8) = null,

    /// Request headers as key-value pairs.
    headers: ?std.StringHashMap([]const u8) = null,

    /// Request body (raw bytes).
    body: ?[]const u8 = null,

    /// Parsed JSON body (if content-type is application/json).
    json_body: ?std.json.Value = null,

    /// Source ARN/resource (for event-triggered functions).
    source: ?[]const u8 = null,

    /// Event type (e.g., "s3:ObjectCreated", "pubsub:message").
    event_type: ?[]const u8 = null,

    /// Timestamp of the event.
    timestamp: i64 = 0,

    /// Provider-specific context/metadata.
    context: ProviderContext = .{},

    /// Memory allocator used for this event.
    allocator: std.mem.Allocator,

    pub const ProviderContext = struct {
        /// AWS-specific: function ARN
        function_arn: ?[]const u8 = null,
        /// AWS-specific: log group name
        log_group: ?[]const u8 = null,
        /// AWS-specific: log stream name
        log_stream: ?[]const u8 = null,
        /// AWS-specific: remaining time in milliseconds
        remaining_time_ms: ?u64 = null,
        /// GCP-specific: project ID
        project_id: ?[]const u8 = null,
        /// GCP-specific: region
        region: ?[]const u8 = null,
        /// Azure-specific: invocation ID
        invocation_id: ?[]const u8 = null,
        /// Azure-specific: function name
        function_name: ?[]const u8 = null,
    };

    /// Initialize a new cloud event.
    pub fn init(allocator: std.mem.Allocator, provider: CloudProvider, request_id: []const u8) CloudEvent {
        return .{
            .allocator = allocator,
            .provider = provider,
            .request_id = request_id,
            .timestamp = time.unixSeconds(),
        };
    }

    /// Get a header value (case-insensitive lookup).
    pub fn getHeader(self: *const CloudEvent, key: []const u8) ?[]const u8 {
        if (self.headers) |hdrs| {
            if (hdrs.get(key)) |value| return value;

            var it = hdrs.iterator();
            while (it.next()) |entry| {
                if (std.ascii.eqlIgnoreCase(entry.key_ptr.*, key)) {
                    return entry.value_ptr.*;
                }
            }
        }
        return null;
    }

    /// Get a query parameter value.
    pub fn getQueryParam(self: *const CloudEvent, key: []const u8) ?[]const u8 {
        if (self.query_params) |params| {
            return params.get(key);
        }
        return null;
    }

    /// Check if the request is an HTTP request.
    pub fn isHttpRequest(self: *const CloudEvent) bool {
        return self.method != null;
    }

    /// Get content type from headers.
    pub fn getContentType(self: *const CloudEvent) ?[]const u8 {
        return self.getHeader("content-type");
    }

    /// Check if content type is JSON.
    pub fn isJsonRequest(self: *const CloudEvent) bool {
        if (self.getContentType()) |ct| {
            return std.mem.indexOf(u8, ct, "application/json") != null;
        }
        return false;
    }

    /// Free allocated resources.
    pub fn deinit(self: *CloudEvent) void {
        if (self.query_params) |*params| {
            params.deinit();
        }
        if (self.headers) |*hdrs| {
            hdrs.deinit();
        }
    }
};

/// Cloud response to return from a function.
pub const CloudResponse = struct {
    /// HTTP status code.
    status_code: u16 = 200,

    /// Response headers.
    headers: std.StringHashMap([]const u8),

    /// Response body.
    body: []const u8 = "",

    /// Whether the body is base64 encoded.
    is_base64_encoded: bool = false,

    /// Memory allocator used for this response.
    allocator: std.mem.Allocator,

    /// Initialize a new cloud response.
    pub fn init(allocator: std.mem.Allocator) CloudResponse {
        return .{
            .allocator = allocator,
            .headers = std.StringHashMap([]const u8).init(allocator),
        };
    }

    /// Create a success response with JSON body.
    pub fn json(allocator: std.mem.Allocator, body: []const u8) !CloudResponse {
        var resp = CloudResponse.init(allocator);
        try resp.headers.put("Content-Type", "application/json");
        resp.body = body;
        return resp;
    }

    /// Create a success response with plain text body.
    pub fn text(allocator: std.mem.Allocator, body: []const u8) !CloudResponse {
        var resp = CloudResponse.init(allocator);
        try resp.headers.put("Content-Type", "text/plain");
        resp.body = body;
        return resp;
    }

    /// Create an error response.
    pub fn err(allocator: std.mem.Allocator, status_code: u16, message: []const u8) !CloudResponse {
        var resp = CloudResponse.init(allocator);
        resp.status_code = status_code;
        try resp.headers.put("Content-Type", "application/json");

        // Format error as JSON using Zig 0.16 API
        var out: std.Io.Writer.Allocating = .init(allocator);
        errdefer out.deinit();
        var writer = std.json.Stringify{ .writer = &out.writer, .options = .{} };
        try writer.beginObject();
        try writer.objectField("error");
        try writer.beginObject();
        try writer.objectField("code");
        try writer.write(status_code);
        try writer.objectField("message");
        try writer.write(message);
        try writer.endObject();
        try writer.endObject();
        // Get owned slice so allocator ownership is clear
        resp.body = try out.toOwnedSlice();

        return resp;
    }

    /// Set a header.
    pub fn setHeader(self: *CloudResponse, key: []const u8, value: []const u8) !void {
        try self.headers.put(key, value);
    }

    /// Set the response body.
    pub fn setBody(self: *CloudResponse, body: []const u8) void {
        self.body = body;
    }

    /// Set the status code.
    pub fn setStatus(self: *CloudResponse, code: u16) void {
        self.status_code = code;
    }

    /// Free allocated resources.
    pub fn deinit(self: *CloudResponse) void {
        self.headers.deinit();
    }
};

/// Cloud function handler type.
pub const CloudHandler = *const fn (event: *CloudEvent, allocator: std.mem.Allocator) anyerror!CloudResponse;

/// Cloud function configuration.
pub const CloudConfig = struct {
    /// Memory allocation in MB.
    memory_mb: u32 = 256,

    /// Timeout in seconds.
    timeout_seconds: u32 = 30,

    /// Environment variables.
    environment: std.StringHashMap([]const u8) = undefined,

    /// Whether to enable tracing.
    tracing_enabled: bool = false,

    /// Whether to enable logging.
    logging_enabled: bool = true,

    /// Log level.
    log_level: LogLevel = .info,

    /// CORS configuration (for HTTP functions).
    cors: ?CorsConfig = null,

    pub const LogLevel = enum {
        debug,
        info,
        warn,
        @"error",

        pub fn toString(self: LogLevel) []const u8 {
            return @tagName(self);
        }
    };

    pub const CorsConfig = struct {
        allowed_origins: []const []const u8 = &.{"*"},
        allowed_methods: []const HttpMethod = &.{ .GET, .POST, .PUT, .DELETE, .OPTIONS },
        allowed_headers: []const []const u8 = &.{ "Content-Type", "Authorization" },
        max_age_seconds: u32 = 86400,
    };

    pub fn defaults() CloudConfig {
        return .{};
    }
};

/// Cloud function error types.
pub const CloudError = error{
    /// The cloud feature is disabled.
    CloudDisabled,
    /// Invalid event format.
    InvalidEvent,
    /// Failed to parse event.
    EventParseFailed,
    /// Failed to serialize response.
    ResponseSerializeFailed,
    /// Handler execution failed.
    HandlerFailed,
    /// Timeout exceeded.
    TimeoutExceeded,
    /// Invalid configuration.
    InvalidConfig,
    /// Provider-specific error.
    ProviderError,
    /// Memory allocation failed.
    OutOfMemory,
};

/// Invocation metadata for logging and tracing.
pub const InvocationMetadata = struct {
    request_id: []const u8,
    provider: CloudProvider,
    start_time: i64,
    end_time: ?i64 = null,
    status_code: ?u16 = null,
    error_message: ?[]const u8 = null,
    cold_start: bool = false,
    memory_used_mb: ?u32 = null,

    pub fn duration_ms(self: *const InvocationMetadata) ?i64 {
        if (self.end_time) |end| {
            return end - self.start_time;
        }
        return null;
    }
};

// ============================================================================
// Shared Helpers
// ============================================================================

/// Extract a string value from a JSON value, returning null for non-strings.
pub fn jsonStringOrNull(value: std.json.Value) ?[]const u8 {
    return switch (value) {
        .string => |s| s,
        else => null,
    };
}

/// Parse a JSON object into a StringHashMap (string values only).
pub fn parseJsonStringMap(
    allocator: std.mem.Allocator,
    object_value: std.json.Value,
) !std.StringHashMap([]const u8) {
    var out = std.StringHashMap([]const u8).init(allocator);
    errdefer out.deinit();

    var iter = object_value.object.iterator();
    while (iter.next()) |entry| {
        if (entry.value_ptr.* == .string) {
            try out.put(entry.key_ptr.*, entry.value_ptr.string);
        }
    }

    return out;
}

/// Parse a JSON object of headers into a StringHashMap.
/// Handles both string values and array-of-string values (takes the first).
pub fn parseJsonHeaderMap(
    allocator: std.mem.Allocator,
    object_value: std.json.Value,
) !std.StringHashMap([]const u8) {
    var out = std.StringHashMap([]const u8).init(allocator);
    errdefer out.deinit();

    var iter = object_value.object.iterator();
    while (iter.next()) |entry| {
        const header_value = switch (entry.value_ptr.*) {
            .string => |s| s,
            .array => |arr| if (arr.items.len > 0 and arr.items[0] == .string) arr.items[0].string else continue,
            else => continue,
        };
        // Keep original key storage; CloudEvent.getHeader performs case-insensitive lookup.
        try out.put(entry.key_ptr.*, header_value);
    }

    return out;
}

/// Clone a StringHashMap, preserving original keys.
/// CloudEvent.getHeader performs case-insensitive lookup at query time.
pub fn cloneStringMap(
    allocator: std.mem.Allocator,
    source: std.StringHashMap([]const u8),
) !std.StringHashMap([]const u8) {
    var clone = std.StringHashMap([]const u8).init(allocator);
    errdefer clone.deinit();

    var iter = source.iterator();
    while (iter.next()) |entry| {
        // Keep original key storage; CloudEvent.getHeader performs case-insensitive lookup.
        try clone.put(entry.key_ptr.*, entry.value_ptr.*);
    }

    return clone;
}

/// Parse raw JSON into a parsed value, returning CloudError on failure.
pub fn parseJsonRoot(
    allocator: std.mem.Allocator,
    raw_event: []const u8,
) !std.json.Parsed(std.json.Value) {
    return std.json.parseFromSlice(std.json.Value, allocator, raw_event, .{}) catch {
        return CloudError.EventParseFailed;
    };
}

// ============================================================================
// Tests
// ============================================================================

test "CloudProvider names" {
    try std.testing.expectEqualStrings("AWS Lambda", CloudProvider.aws_lambda.name());
    try std.testing.expectEqualStrings("Google Cloud Functions", CloudProvider.gcp_functions.name());
    try std.testing.expectEqualStrings("Azure Functions", CloudProvider.azure_functions.name());
}

test "HttpMethod fromString" {
    try std.testing.expectEqual(HttpMethod.GET, HttpMethod.fromString("GET").?);
    try std.testing.expectEqual(HttpMethod.POST, HttpMethod.fromString("POST").?);
    try std.testing.expect(HttpMethod.fromString("INVALID") == null);
}

test "CloudEvent initialization" {
    const allocator = std.testing.allocator;
    var event = CloudEvent.init(allocator, .aws_lambda, "test-request-id");
    defer event.deinit();

    try std.testing.expectEqualStrings("test-request-id", event.request_id);
    try std.testing.expectEqual(CloudProvider.aws_lambda, event.provider);
    try std.testing.expect(!event.isHttpRequest());
}

test "CloudResponse JSON creation" {
    const allocator = std.testing.allocator;
    var resp = try CloudResponse.json(allocator, "{\"ok\":true}");
    defer resp.deinit();

    try std.testing.expectEqual(@as(u16, 200), resp.status_code);
    try std.testing.expectEqualStrings("application/json", resp.headers.get("Content-Type").?);
    try std.testing.expectEqualStrings("{\"ok\":true}", resp.body);
}

test "CloudResponse error creation" {
    const allocator = std.testing.allocator;
    var resp = try CloudResponse.err(allocator, 404, "Not Found");
    defer {
        allocator.free(resp.body);
        resp.deinit();
    }

    try std.testing.expectEqual(@as(u16, 404), resp.status_code);
}
