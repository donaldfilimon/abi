//! Stub for Web feature when disabled.
//!
//! Mirrors the full API of mod.zig, returning error.WebDisabled for all operations.

const std = @import("std");
const config_module = @import("../config/mod.zig");

// ============================================================================
// Local Stubs Imports
// ============================================================================

pub const types = @import("stubs/types.zig");
pub const client = @import("stubs/client.zig");

// ============================================================================
// Re-exports
// ============================================================================

pub const WebError = types.WebError;
pub const Response = types.Response;
pub const RequestOptions = types.RequestOptions;
pub const WeatherConfig = types.WeatherConfig;
pub const JsonValue = types.JsonValue;
pub const ParsedJson = types.ParsedJson;

pub const HttpClient = client.HttpClient;
pub const WeatherClient = client.WeatherClient;

/// Web Context stub for Framework integration.
pub const Context = struct {
    allocator: std.mem.Allocator,
    config: config_module.WebConfig,
    http_client: ?HttpClient = null,

    pub fn init(allocator: std.mem.Allocator, cfg: config_module.WebConfig) !*Context {
        _ = allocator;
        _ = cfg;
        return error.WebDisabled;
    }

    pub fn deinit(self: *Context) void {
        _ = self;
    }

    pub fn get(self: *Context, url: []const u8) !Response {
        _ = self;
        _ = url;
        return error.WebDisabled;
    }

    pub fn getWithOptions(self: *Context, url: []const u8, options: RequestOptions) !Response {
        _ = self;
        _ = url;
        _ = options;
        return error.WebDisabled;
    }

    pub fn postJson(self: *Context, url: []const u8, body: []const u8) !Response {
        _ = self;
        _ = url;
        _ = body;
        return error.WebDisabled;
    }

    pub fn freeResponse(self: *Context, response: Response) void {
        _ = self;
        _ = response;
    }

    pub fn parseJsonValue(self: *Context, response: Response) !ParsedJson {
        _ = self;
        _ = response;
        return error.WebDisabled;
    }
};

// HTTP utilities stub
pub const http = struct {
    pub fn isSuccess(status: u16) bool {
        return status >= 200 and status < 300;
    }

    pub fn isRedirect(status: u16) bool {
        return status >= 300 and status < 400;
    }

    pub fn isClientError(status: u16) bool {
        return status >= 400 and status < 500;
    }

    pub fn isServerError(status: u16) bool {
        return status >= 500 and status < 600;
    }
};

// Module lifecycle
var initialized: bool = false;

pub fn init(_: std.mem.Allocator) !void {
    return error.WebDisabled;
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

// Convenience functions
pub fn get(allocator: std.mem.Allocator, url: []const u8) !Response {
    _ = allocator;
    _ = url;
    return error.WebDisabled;
}

pub fn getWithOptions(
    allocator: std.mem.Allocator,
    url: []const u8,
    options: RequestOptions,
) !Response {
    _ = allocator;
    _ = url;
    _ = options;
    return error.WebDisabled;
}

pub fn postJson(allocator: std.mem.Allocator, url: []const u8, body: []const u8) !Response {
    _ = allocator;
    _ = url;
    _ = body;
    return error.WebDisabled;
}

pub fn freeResponse(allocator: std.mem.Allocator, response: Response) void {
    _ = allocator;
    _ = response;
}

pub fn parseJsonValue(allocator: std.mem.Allocator, response: Response) !ParsedJson {
    _ = allocator;
    _ = response;
    return error.WebDisabled;
}

pub fn isSuccessStatus(status: u16) bool {
    return http.isSuccess(status);
}
