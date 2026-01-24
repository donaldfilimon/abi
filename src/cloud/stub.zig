//! Stub for Cloud Functions module when disabled.
//!
//! Mirrors the full API of mod.zig, returning error.CloudDisabled for all operations.

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
    query_params: ?std.StringHashMap([]const u8) = null,
    headers: ?std.StringHashMap([]const u8) = null,
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
    headers: std.StringHashMap([]const u8),
    body: []const u8 = "",
    is_base64_encoded: bool = false,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) CloudResponse {
        return .{
            .allocator = allocator,
            .headers = std.StringHashMap([]const u8).init(allocator),
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
        self.headers.deinit();
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

// Re-export types namespace
pub const types = struct {
    pub const CloudEvent = @This().CloudEvent;
    pub const CloudResponse = @This().CloudResponse;
    pub const CloudProvider = @This().CloudProvider;
    pub const CloudHandler = @This().CloudHandler;
    pub const CloudConfig = @This().CloudConfig;
    pub const CloudError = @This().CloudError;
    pub const HttpMethod = @This().HttpMethod;
    pub const InvocationMetadata = @This().InvocationMetadata;
};

/// AWS Lambda adapter stub.
pub const aws_lambda = struct {
    pub const LambdaRuntime = struct {
        pub fn init(allocator: std.mem.Allocator, handler: CloudHandler) !LambdaRuntime {
            _ = allocator;
            _ = handler;
            return Error.CloudDisabled;
        }

        pub fn run(self: *LambdaRuntime) !void {
            _ = self;
            return Error.CloudDisabled;
        }
    };

    pub fn parseEvent(allocator: std.mem.Allocator, raw_event: []const u8, request_id: []const u8) !CloudEvent {
        _ = allocator;
        _ = raw_event;
        _ = request_id;
        return Error.CloudDisabled;
    }

    pub fn formatResponse(allocator: std.mem.Allocator, response: *const CloudResponse) ![]const u8 {
        _ = allocator;
        _ = response;
        return Error.CloudDisabled;
    }

    pub fn runHandler(allocator: std.mem.Allocator, handler: CloudHandler) !void {
        _ = allocator;
        _ = handler;
        return Error.CloudDisabled;
    }
};

/// GCP Functions adapter stub.
pub const gcp_functions = struct {
    pub const GcpConfig = struct {
        port: u16 = 8080,
        project_id: ?[]const u8 = null,
        region: ?[]const u8 = null,
        function_name: ?[]const u8 = null,

        pub fn fromEnvironment() GcpConfig {
            return .{};
        }
    };

    pub const GcpRuntime = struct {
        pub fn init(allocator: std.mem.Allocator, handler: CloudHandler) GcpRuntime {
            _ = allocator;
            _ = handler;
            return .{};
        }

        pub fn run(self: *GcpRuntime) !void {
            _ = self;
            return Error.CloudDisabled;
        }
    };

    pub fn parseHttpRequest(
        allocator: std.mem.Allocator,
        method: []const u8,
        path: []const u8,
        headers: std.StringHashMap([]const u8),
        body: ?[]const u8,
        request_id: []const u8,
    ) !CloudEvent {
        _ = allocator;
        _ = method;
        _ = path;
        _ = headers;
        _ = body;
        _ = request_id;
        return Error.CloudDisabled;
    }

    pub fn parseCloudEvent(allocator: std.mem.Allocator, raw_event: []const u8) !CloudEvent {
        _ = allocator;
        _ = raw_event;
        return Error.CloudDisabled;
    }

    pub fn runHandler(allocator: std.mem.Allocator, handler: CloudHandler, port: u16) !void {
        _ = allocator;
        _ = handler;
        _ = port;
        return Error.CloudDisabled;
    }
};

/// Azure Functions adapter stub.
pub const azure_functions = struct {
    pub const AzureConfig = struct {
        port: u16 = 7071,
        function_name: ?[]const u8 = null,

        pub fn fromEnvironment() AzureConfig {
            return .{};
        }
    };

    pub const TriggerType = enum {
        http,
        timer,
        blob,
        queue,
        event_hub,
        service_bus,
        cosmos_db,
        event_grid,
        unknown,
    };

    pub const AzureRuntime = struct {
        pub fn init(allocator: std.mem.Allocator, handler: CloudHandler) AzureRuntime {
            _ = allocator;
            _ = handler;
            return .{};
        }

        pub fn run(self: *AzureRuntime) !void {
            _ = self;
            return Error.CloudDisabled;
        }
    };

    pub fn parseInvocationRequest(allocator: std.mem.Allocator, raw_request: []const u8) !CloudEvent {
        _ = allocator;
        _ = raw_request;
        return Error.CloudDisabled;
    }

    pub fn formatInvocationResponse(allocator: std.mem.Allocator, response: *const CloudResponse) ![]const u8 {
        _ = allocator;
        _ = response;
        return Error.CloudDisabled;
    }

    pub fn runHandler(allocator: std.mem.Allocator, handler: CloudHandler) !void {
        _ = allocator;
        _ = handler;
        return Error.CloudDisabled;
    }
};

/// Cloud module context stub.
pub const Context = struct {
    allocator: std.mem.Allocator,
    config: CloudConfig,
    provider: ?CloudProvider = null,

    pub fn init(allocator: std.mem.Allocator, cfg: CloudConfig) !*Context {
        _ = allocator;
        _ = cfg;
        return Error.CloudDisabled;
    }

    pub fn deinit(self: *Context) void {
        _ = self;
    }

    pub fn getProvider(self: *const Context) ?CloudProvider {
        _ = self;
        return null;
    }

    pub fn isCloudEnvironment(self: *const Context) bool {
        _ = self;
        return false;
    }
};

/// Response builder stub.
pub const ResponseBuilder = struct {
    allocator: std.mem.Allocator,
    response: CloudResponse,

    pub fn init(allocator: std.mem.Allocator) ResponseBuilder {
        return .{
            .allocator = allocator,
            .response = CloudResponse.init(allocator),
        };
    }

    pub fn status(self: *ResponseBuilder, code: u16) *ResponseBuilder {
        _ = code;
        return self;
    }

    pub fn header(self: *ResponseBuilder, key: []const u8, value: []const u8) *ResponseBuilder {
        _ = key;
        _ = value;
        return self;
    }

    pub fn json(self: *ResponseBuilder) *ResponseBuilder {
        return self;
    }

    pub fn text(self: *ResponseBuilder) *ResponseBuilder {
        return self;
    }

    pub fn html(self: *ResponseBuilder) *ResponseBuilder {
        return self;
    }

    pub fn body(self: *ResponseBuilder, content: []const u8) *ResponseBuilder {
        _ = content;
        return self;
    }

    pub fn cors(self: *ResponseBuilder, origin: []const u8) *ResponseBuilder {
        _ = origin;
        return self;
    }

    pub fn build(self: *ResponseBuilder) CloudResponse {
        return self.response;
    }
};

/// Detect which cloud provider environment we're running in.
pub fn detectProvider() ?CloudProvider {
    return null;
}

/// Run a handler on the detected cloud provider.
pub fn runHandler(allocator: std.mem.Allocator, handler: CloudHandler) !void {
    _ = allocator;
    _ = handler;
    return Error.CloudDisabled;
}

/// Module lifecycle.
var initialized: bool = false;

pub fn init(allocator: std.mem.Allocator) !void {
    _ = allocator;
    return Error.CloudDisabled;
}

pub fn deinit() void {
    initialized = false;
}

pub fn isEnabled() bool {
    return false;
}

pub fn isInitialized() bool {
    return initialized;
}
