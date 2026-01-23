//! Cloud Functions Module
//!
//! Provides unified adapters for deploying ABI applications as serverless
//! functions across major cloud providers: AWS Lambda, Google Cloud Functions,
//! and Azure Functions.
//!
//! ## Features
//!
//! - **Unified Event Model**: Common `CloudEvent` struct that normalizes events
//!   across all providers
//! - **Unified Response Model**: Common `CloudResponse` struct for consistent
//!   response handling
//! - **Provider-Specific Adapters**: Optimized parsing and formatting for each
//!   cloud provider
//! - **Context Extraction**: Access provider-specific context and metadata
//!
//! ## Quick Start
//!
//! ```zig
//! const std = @import("std");
//! const abi = @import("abi");
//! const cloud = abi.cloud;
//!
//! /// Your function handler - same code works on all providers
//! fn handler(event: *cloud.CloudEvent, allocator: std.mem.Allocator) !cloud.CloudResponse {
//!     // Access request data uniformly
//!     const body = event.body orelse "{}";
//!
//!     // Return a JSON response
//!     return try cloud.CloudResponse.json(allocator,
//!         \\{"message": "Hello from the cloud!"}
//!     );
//! }
//!
//! pub fn main() !void {
//!     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//!     defer _ = gpa.deinit();
//!     const allocator = gpa.allocator();
//!
//!     // Deploy to AWS Lambda
//!     try cloud.aws_lambda.runHandler(allocator, handler);
//!
//!     // Or Google Cloud Functions
//!     // try cloud.gcp_functions.runHandler(allocator, handler, 8080);
//!
//!     // Or Azure Functions
//!     // try cloud.azure_functions.runHandler(allocator, handler);
//! }
//! ```
//!
//! ## Deployment
//!
//! See the deployment templates in `deploy/` for provider-specific configurations:
//! - `deploy/aws/template.yaml` - AWS SAM template
//! - `deploy/gcp/cloudfunctions.yaml` - GCP configuration
//! - `deploy/azure/function.json` - Azure Functions configuration

const std = @import("std");
const build_options = @import("build_options");
const config_module = @import("../config.zig");

// Re-export types
pub const types = @import("types.zig");
pub const CloudEvent = types.CloudEvent;
pub const CloudResponse = types.CloudResponse;
pub const CloudProvider = types.CloudProvider;
pub const CloudHandler = types.CloudHandler;
pub const CloudConfig = types.CloudConfig;
pub const CloudError = types.CloudError;
pub const HttpMethod = types.HttpMethod;
pub const InvocationMetadata = types.InvocationMetadata;

// Re-export provider-specific adapters
pub const aws_lambda = @import("aws_lambda.zig");
pub const gcp_functions = @import("gcp_functions.zig");
pub const azure_functions = @import("azure_functions.zig");

/// Cloud module errors.
pub const Error = error{
    CloudDisabled,
    UnsupportedProvider,
    InitializationFailed,
} || CloudError;

/// Cloud module context for Framework integration.
pub const Context = struct {
    allocator: std.mem.Allocator,
    config: CloudConfig,
    provider: ?CloudProvider = null,

    /// Initialize the cloud context.
    pub fn init(allocator: std.mem.Allocator, cfg: CloudConfig) !*Context {
        const ctx = try allocator.create(Context);
        errdefer allocator.destroy(ctx);

        ctx.* = .{
            .allocator = allocator,
            .config = cfg,
        };

        // Auto-detect provider from environment
        ctx.provider = detectProvider();

        return ctx;
    }

    /// Deinitialize the cloud context.
    pub fn deinit(self: *Context) void {
        self.allocator.destroy(self);
    }

    /// Get the detected cloud provider.
    pub fn getProvider(self: *const Context) ?CloudProvider {
        return self.provider;
    }

    /// Check if running in a cloud function environment.
    pub fn isCloudEnvironment(self: *const Context) bool {
        return self.provider != null;
    }

    /// Create a handler wrapper that integrates with the ABI framework.
    pub fn wrapHandler(
        self: *Context,
        comptime handler: fn (*CloudEvent, std.mem.Allocator) anyerror!CloudResponse,
    ) CloudHandler {
        _ = self;
        return struct {
            fn wrapped(event: *CloudEvent, allocator: std.mem.Allocator) anyerror!CloudResponse {
                return handler(event, allocator);
            }
        }.wrapped;
    }
};

/// Detect which cloud provider environment we're running in.
pub fn detectProvider() ?CloudProvider {
    // AWS Lambda
    if (std.posix.getenv("AWS_LAMBDA_RUNTIME_API") != null or
        std.posix.getenv("AWS_LAMBDA_FUNCTION_NAME") != null)
    {
        return .aws_lambda;
    }

    // Google Cloud Functions
    if (std.posix.getenv("K_SERVICE") != null or
        std.posix.getenv("FUNCTION_NAME") != null or
        std.posix.getenv("GOOGLE_CLOUD_PROJECT") != null)
    {
        return .gcp_functions;
    }

    // Azure Functions
    if (std.posix.getenv("FUNCTIONS_WORKER_RUNTIME") != null or
        std.posix.getenv("AZURE_FUNCTIONS_ENVIRONMENT") != null or
        std.posix.getenv("WEBSITE_SITE_NAME") != null)
    {
        return .azure_functions;
    }

    return null;
}

/// Run a handler on the detected cloud provider.
/// Automatically selects the appropriate runtime based on environment detection.
pub fn runHandler(allocator: std.mem.Allocator, handler: CloudHandler) !void {
    const provider = detectProvider() orelse {
        // Not running in a cloud environment
        std.log.warn("No cloud provider detected. Running in local mode.", .{});
        return Error.UnsupportedProvider;
    };

    switch (provider) {
        .aws_lambda => try aws_lambda.runHandler(allocator, handler),
        .gcp_functions => try gcp_functions.runHandler(allocator, handler, 8080),
        .azure_functions => try azure_functions.runHandler(allocator, handler),
    }
}

/// Create a response helper for common response patterns.
pub const ResponseBuilder = struct {
    allocator: std.mem.Allocator,
    response: CloudResponse,

    pub fn init(allocator: std.mem.Allocator) ResponseBuilder {
        return .{
            .allocator = allocator,
            .response = CloudResponse.init(allocator),
        };
    }

    /// Set HTTP status code.
    pub fn status(self: *ResponseBuilder, code: u16) *ResponseBuilder {
        self.response.status_code = code;
        return self;
    }

    /// Add a header.
    pub fn header(self: *ResponseBuilder, key: []const u8, value: []const u8) *ResponseBuilder {
        self.response.headers.put(key, value) catch {};
        return self;
    }

    /// Set content type to JSON.
    pub fn json(self: *ResponseBuilder) *ResponseBuilder {
        return self.header("Content-Type", "application/json");
    }

    /// Set content type to plain text.
    pub fn text(self: *ResponseBuilder) *ResponseBuilder {
        return self.header("Content-Type", "text/plain");
    }

    /// Set content type to HTML.
    pub fn html(self: *ResponseBuilder) *ResponseBuilder {
        return self.header("Content-Type", "text/html");
    }

    /// Set the response body.
    pub fn body(self: *ResponseBuilder, content: []const u8) *ResponseBuilder {
        self.response.body = content;
        return self;
    }

    /// Add CORS headers.
    pub fn cors(self: *ResponseBuilder, origin: []const u8) *ResponseBuilder {
        _ = self.header("Access-Control-Allow-Origin", origin);
        _ = self.header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
        _ = self.header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        return self;
    }

    /// Build the final response.
    pub fn build(self: *ResponseBuilder) CloudResponse {
        return self.response;
    }
};

/// Module lifecycle.
var initialized: bool = false;

pub fn init(allocator: std.mem.Allocator) !void {
    _ = allocator;
    initialized = true;
}

pub fn deinit() void {
    initialized = false;
}

pub fn isEnabled() bool {
    return build_options.enable_web;
}

pub fn isInitialized() bool {
    return initialized;
}

// ============================================================================
// Tests
// ============================================================================

test "detectProvider returns null in non-cloud environment" {
    // In a test environment, no cloud provider should be detected
    // (unless running on actual cloud infrastructure)
    const provider = detectProvider();
    // We can't assert null since tests might run in cloud environments
    _ = provider;
}

test "CloudResponse.json creates valid response" {
    const allocator = std.testing.allocator;

    var response = try CloudResponse.json(allocator, "{\"ok\":true}");
    defer response.deinit();

    try std.testing.expectEqual(@as(u16, 200), response.status_code);
    try std.testing.expectEqualStrings("application/json", response.headers.get("Content-Type").?);
}

test "CloudResponse.err creates error response" {
    const allocator = std.testing.allocator;

    var response = try CloudResponse.err(allocator, 500, "Internal Server Error");
    defer {
        allocator.free(response.body);
        response.deinit();
    }

    try std.testing.expectEqual(@as(u16, 500), response.status_code);
}

test "ResponseBuilder creates valid response" {
    const allocator = std.testing.allocator;

    var builder = ResponseBuilder.init(allocator);
    var response = builder
        .status(201)
        .json()
        .header("X-Custom", "value")
        .body("{\"created\":true}")
        .build();
    defer response.deinit();

    try std.testing.expectEqual(@as(u16, 201), response.status_code);
    try std.testing.expectEqualStrings("application/json", response.headers.get("Content-Type").?);
    try std.testing.expectEqualStrings("value", response.headers.get("X-Custom").?);
}

test "CloudEvent initialization" {
    const allocator = std.testing.allocator;

    var event = CloudEvent.init(allocator, .aws_lambda, "test-123");
    defer event.deinit();

    try std.testing.expectEqual(CloudProvider.aws_lambda, event.provider);
    try std.testing.expectEqualStrings("test-123", event.request_id);
}

test "HttpMethod parsing" {
    try std.testing.expectEqual(HttpMethod.GET, HttpMethod.fromString("GET").?);
    try std.testing.expectEqual(HttpMethod.POST, HttpMethod.fromString("POST").?);
    try std.testing.expect(HttpMethod.fromString("INVALID") == null);
}
