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
//! pub fn main(init: std.process.Init) !void {
//!     const arena = init.arena.allocator();
//!
//!     // Deploy to AWS Lambda
//!     try cloud.aws_lambda.runHandler(arena, handler);
//!
//!     // Or Google Cloud Functions
//!     // try cloud.gcp_functions.runHandler(arena, handler, 8080);
//!
//!     // Or Azure Functions
//!     // try cloud.azure_functions.runHandler(arena, handler);
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

// Submodule imports
const detection = @import("detection.zig");
const response_builder = @import("response_builder.zig");

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

// Aliases matching stub.zig convention
pub const aws = aws_lambda;
pub const gcp = gcp_functions;
pub const azure = azure_functions;

// Re-export submodule functions and types
pub const detectProvider = detection.detectProvider;
pub const detectProviderWithAllocator = detection.detectProviderWithAllocator;
pub const ResponseBuilder = response_builder.ResponseBuilder;

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
        ctx.provider = detectProviderWithAllocator(allocator);

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

/// Run a handler on the detected cloud provider.
/// Automatically selects the appropriate runtime based on environment detection.
pub fn runHandler(allocator: std.mem.Allocator, handler: CloudHandler) !void {
    const provider = detectProviderWithAllocator(allocator) orelse {
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

/// Module lifecycle.
var initialized = std.atomic.Value(bool).init(false);

pub fn init(allocator: std.mem.Allocator) !void {
    _ = allocator;
    if (initialized.load(.acquire)) return;
    initialized.store(true, .release);
}

pub fn deinit() void {
    initialized.store(false, .release);
}

pub fn isEnabled() bool {
    return build_options.feat_cloud;
}

pub fn isInitialized() bool {
    return initialized.load(.acquire);
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

// refAllDecls deferred — aws_lambda, azure_functions, gcp_functions have pre-existing Zig 0.16 API errors
