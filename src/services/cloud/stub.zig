//! Stub for Cloud Functions module when disabled.
//!
//! Mirrors the full API of mod.zig, returning error.CloudDisabled for all operations.

const std = @import("std");

// ============================================================================
// Local Stubs Imports
// ============================================================================

pub const types = @import("stubs/types.zig");
pub const aws = @import("stubs/aws.zig");
pub const gcp = @import("stubs/gcp.zig");
pub const azure = @import("stubs/azure.zig");

// ============================================================================
// Re-exports
// ============================================================================

pub const Error = types.Error;
pub const CloudError = types.CloudError;
pub const CloudProvider = types.CloudProvider;
pub const HttpMethod = types.HttpMethod;
pub const CloudEvent = types.CloudEvent;
pub const CloudResponse = types.CloudResponse;
pub const CloudHandler = types.CloudHandler;
pub const CloudConfig = types.CloudConfig;
pub const InvocationMetadata = types.InvocationMetadata;

pub const aws_lambda = aws;
pub const gcp_functions = gcp;
pub const azure_functions = azure;

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
