const std = @import("std");

/// Cloud module errors.
pub const Error = error{
    CloudDisabled,
    UnsupportedProvider,
    InitializationFailed,
    InvalidEvent,
    EventParseFailed,
    ResponseSerializeFailed,
    HandlerFailed,
    TimeoutExceeded,
    InvalidConfig,
    ProviderError,
    OutOfMemory,
};

pub const CloudError = Error;

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
        _ = s;
        return null;
    }

    pub fn toString(self: HttpMethod) []const u8 {
        return @tagName(self);
    }
};

/// Cloud event stub.
pub const CloudEvent = struct {
    request_id: []const u8,
    provider: CloudProvider,
    method: ?HttpMethod = null,
    path: ?[]const u8 = null,
    query_params: ?std.StringHashMapUnmanaged([]const u8) = null,
    headers: ?std.StringHashMapUnmanaged([]const u8) = null,
    body: ?[]const u8 = null,
    json_body: ?std.json.Value = null,
    source: ?[]const u8 = null,
    event_type: ?[]const u8 = null,
    timestamp: i64 = 0,
    context: ProviderContext = .{},
    allocator: std.mem.Allocator,

    pub const ProviderContext = struct {
        function_arn: ?[]const u8 = null,
        log_group: ?[]const u8 = null,
        log_stream: ?[]const u8 = null,
        remaining_time_ms: ?u64 = null,
        project_id: ?[]const u8 = null,
        region: ?[]const u8 = null,
        invocation_id: ?[]const u8 = null,
        function_name: ?[]const u8 = null,
    };

    pub fn init(allocator: std.mem.Allocator, provider: CloudProvider, request_id: []const u8) CloudEvent {
        return .{
            .allocator = allocator,
            .provider = provider,
            .request_id = request_id,
        };
    }

    pub fn getHeader(self: *const CloudEvent, key: []const u8) ?[]const u8 {
        _ = self;
        _ = key;
        return null;
    }

    pub fn getQueryParam(self: *const CloudEvent, key: []const u8) ?[]const u8 {
        _ = self;
        _ = key;
        return null;
    }

    pub fn isHttpRequest(self: *const CloudEvent) bool {
        _ = self;
        return false;
    }

    pub fn getContentType(self: *const CloudEvent) ?[]const u8 {
        _ = self;
        return null;
    }

    pub fn isJsonRequest(self: *const CloudEvent) bool {
        _ = self;
        return false;
    }

    pub fn deinit(self: *CloudEvent) void {
        _ = self;
    }
};

/// Cloud response stub.
pub const CloudResponse = struct {
    status_code: u16 = 200,
    headers: std.StringHashMapUnmanaged([]const u8),
    body: []const u8 = "",
    is_base64_encoded: bool = false,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) CloudResponse {
        return .{
            .allocator = allocator,
            .headers = .empty,
        };
    }

    pub fn json(allocator: std.mem.Allocator, body: []const u8) !CloudResponse {
        _ = body;
        _ = allocator;
        return Error.CloudDisabled;
    }

    pub fn text(allocator: std.mem.Allocator, body: []const u8) !CloudResponse {
        _ = body;
        _ = allocator;
        return Error.CloudDisabled;
    }

    pub fn err(allocator: std.mem.Allocator, status_code: u16, message: []const u8) !CloudResponse {
        _ = status_code;
        _ = message;
        _ = allocator;
        return Error.CloudDisabled;
    }

    pub fn setHeader(self: *CloudResponse, key: []const u8, value: []const u8) !void {
        _ = self;
        _ = key;
        _ = value;
        return Error.CloudDisabled;
    }

    pub fn setBody(self: *CloudResponse, body: []const u8) void {
        _ = self;
        _ = body;
    }

    pub fn setStatus(self: *CloudResponse, code: u16) void {
        _ = self;
        _ = code;
    }

    pub fn deinit(self: *CloudResponse) void {
        self.headers.deinit(self.allocator);
    }
};

/// Cloud function handler type.
pub const CloudHandler = *const fn (event: *CloudEvent, allocator: std.mem.Allocator) anyerror!CloudResponse;

/// Cloud function configuration.
pub const CloudConfig = struct {
    memory_mb: u32 = 256,
    timeout_seconds: u32 = 30,
    tracing_enabled: bool = false,
    logging_enabled: bool = true,
    log_level: LogLevel = .info,

    pub const LogLevel = enum {
        debug,
        info,
        warn,
        @"error",

        pub fn toString(self: LogLevel) []const u8 {
            return @tagName(self);
        }
    };

    pub fn defaults() CloudConfig {
        return .{};
    }
};

/// Invocation metadata.
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
        _ = self;
        return null;
    }
};
