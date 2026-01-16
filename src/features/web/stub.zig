//! Stub for Web feature when disabled.
//!
//! Mirrors the full API of mod.zig, returning error.WebDisabled for all operations.

const std = @import("std");

pub const WebError = error{
    WebDisabled,
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

    pub fn init(allocator: std.mem.Allocator) WebError!HttpClient {
        _ = allocator;
        return error.WebDisabled;
    }

    pub fn deinit(self: *HttpClient) void {
        _ = self;
    }

    pub fn get(self: *HttpClient, url: []const u8) WebError!Response {
        _ = self;
        _ = url;
        return error.WebDisabled;
    }

    pub fn getWithOptions(self: *HttpClient, url: []const u8, options: RequestOptions) WebError!Response {
        _ = self;
        _ = url;
        _ = options;
        return error.WebDisabled;
    }

    pub fn postJson(self: *HttpClient, url: []const u8, body: []const u8) WebError!Response {
        _ = self;
        _ = url;
        _ = body;
        return error.WebDisabled;
    }
};

pub const RequestOptions = struct {
    headers: ?std.StringHashMap([]const u8) = null,
    timeout_ms: u32 = 30_000,
    follow_redirects: bool = true,
};

pub const WeatherClient = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: WeatherConfig) WeatherClient {
        _ = config;
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *WeatherClient) void {
        _ = self;
    }

    pub fn getCurrentWeather(self: *WeatherClient, location: []const u8) WebError!WeatherData {
        _ = self;
        _ = location;
        return error.WebDisabled;
    }
};

pub const WeatherConfig = struct {
    api_key: ?[]const u8 = null,
    base_url: []const u8 = "https://api.weather.example",
};

pub const WeatherData = struct {
    temperature: f32 = 0.0,
    humidity: f32 = 0.0,
    description: []const u8 = "",
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

pub fn init(_: std.mem.Allocator) WebError!void {
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
pub fn get(allocator: std.mem.Allocator, url: []const u8) WebError!Response {
    _ = allocator;
    _ = url;
    return error.WebDisabled;
}

pub fn getWithOptions(
    allocator: std.mem.Allocator,
    url: []const u8,
    options: RequestOptions,
) WebError!Response {
    _ = allocator;
    _ = url;
    _ = options;
    return error.WebDisabled;
}

pub fn postJson(allocator: std.mem.Allocator, url: []const u8, body: []const u8) WebError!Response {
    _ = allocator;
    _ = url;
    _ = body;
    return error.WebDisabled;
}

pub fn freeResponse(allocator: std.mem.Allocator, response: Response) void {
    _ = allocator;
    _ = response;
}

pub fn parseJsonValue(allocator: std.mem.Allocator, response: Response) WebError!ParsedJson {
    _ = allocator;
    _ = response;
    return error.WebDisabled;
}

pub fn isSuccessStatus(status: u16) bool {
    return http.isSuccess(status);
}
