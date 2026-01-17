//! Stub for Web feature when disabled.
//!
//! Mirrors the full API of mod.zig, returning error.WebDisabled for all operations.

const std = @import("std");
const config_module = @import("../config.zig");

pub const WebError = error{
    WebDisabled,
};

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

// Type stubs
pub const JsonValue = std.json.Value;
pub const ParsedJson = std.json.Parsed(JsonValue);

pub const Response = struct {
    status: u16,
    body: []const u8,
};

pub const HttpClient = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !HttpClient {
        _ = allocator;
        return error.WebDisabled;
    }

    pub fn deinit(self: *HttpClient) void {
        _ = self;
    }

    pub fn get(self: *HttpClient, url: []const u8) !Response {
        _ = self;
        _ = url;
        return error.WebDisabled;
    }

    pub fn getWithOptions(self: *HttpClient, url: []const u8, options: RequestOptions) !Response {
        _ = self;
        _ = url;
        _ = options;
        return error.WebDisabled;
    }

    pub fn postJson(self: *HttpClient, url: []const u8, body: []const u8) !Response {
        _ = self;
        _ = url;
        _ = body;
        return error.WebDisabled;
    }

    pub fn requestWithOptions(
        self: *HttpClient,
        method: std.http.Method,
        url: []const u8,
        body: ?[]const u8,
        options: RequestOptions,
    ) !Response {
        _ = self;
        _ = method;
        _ = url;
        _ = body;
        _ = options;
        return error.WebDisabled;
    }

    pub fn freeResponse(self: *HttpClient, response: Response) void {
        _ = self;
        _ = response;
    }
};

pub const RequestOptions = struct {
    max_response_bytes: usize = 1024 * 1024,
    user_agent: []const u8 = "abi-http",
    follow_redirects: bool = true,
    redirect_limit: u16 = 3,
    content_type: ?[]const u8 = null,
    extra_headers: []const std.http.Header = &.{},
};

pub const WeatherClient = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: WeatherConfig) !WeatherClient {
        _ = config;
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *WeatherClient) void {
        _ = self;
    }

    pub fn forecast(self: *WeatherClient, location: []const u8) !Response {
        _ = self;
        _ = location;
        return error.WebDisabled;
    }

    pub fn freeResponse(self: *WeatherClient, response: Response) void {
        _ = self;
        _ = response;
    }
};

pub const WeatherConfig = struct {
    base_url: []const u8 = "https://api.open-meteo.com/v1/forecast",
    include_current: bool = true,
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
