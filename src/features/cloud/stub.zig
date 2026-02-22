//! Cloud Functions stub — disabled at compile time.

const std = @import("std");

// --- Local Stubs Imports ---

pub const types = @import("stubs/types.zig");
pub const aws = @import("stubs/aws.zig");
pub const gcp = @import("stubs/gcp.zig");
pub const azure = @import("stubs/azure.zig");

// --- Re-exports ---

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

// --- Context ---

pub const Context = struct {
    allocator: std.mem.Allocator,
    config: CloudConfig,
    provider: ?CloudProvider = null,
    pub fn init(_: std.mem.Allocator, _: CloudConfig) !*Context {
        return Error.CloudDisabled;
    }
    pub fn deinit(_: *Context) void {}
    pub fn getProvider(_: *const Context) ?CloudProvider {
        return null;
    }
    pub fn isCloudEnvironment(_: *const Context) bool {
        return false;
    }
    /// Note: returns anyerror!CloudResponse to match the CloudHandler function pointer type.
    pub fn wrapHandler(_: *Context, comptime handler: fn (*CloudEvent, std.mem.Allocator) anyerror!CloudResponse) CloudHandler {
        _ = handler;
        return struct {
            fn wrapped(_: *CloudEvent, _: std.mem.Allocator) anyerror!CloudResponse {
                return Error.CloudDisabled;
            }
        }.wrapped;
    }
};

// --- Response Builder ---

pub const ResponseBuilder = struct {
    allocator: std.mem.Allocator,
    response: CloudResponse,
    pub fn init(allocator: std.mem.Allocator) ResponseBuilder {
        return .{ .allocator = allocator, .response = CloudResponse.init(allocator) };
    }
    pub fn status(self: *ResponseBuilder, _: u16) *ResponseBuilder {
        return self;
    }
    pub fn header(self: *ResponseBuilder, _: []const u8, _: []const u8) *ResponseBuilder {
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
    pub fn body(self: *ResponseBuilder, _: []const u8) *ResponseBuilder {
        return self;
    }
    pub fn cors(self: *ResponseBuilder, _: []const u8) *ResponseBuilder {
        return self;
    }
    pub fn build(self: *ResponseBuilder) CloudResponse {
        return self.response;
    }
    pub fn deinit(self: *ResponseBuilder) void {
        self.response.deinit();
    }
};

// --- Module Functions ---

pub fn detectProvider() ?CloudProvider {
    return null;
}
pub fn detectProviderWithAllocator(_: std.mem.Allocator) ?CloudProvider {
    return null;
}
pub fn runHandler(_: std.mem.Allocator, _: CloudHandler) !void {
    return Error.CloudDisabled;
}

var initialized: bool = false;
pub fn init(_: std.mem.Allocator) !void {
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
