//! Client stubs for the web module when disabled.
//!
//! These stubs provide the same API surface as the real implementations
//! but return `error.WebDisabled` for all operations that would perform
//! network I/O.

const std = @import("std");
const types = @import("types.zig");

/// Stub HTTP client.
///
/// All operations return `error.WebDisabled` to indicate the feature
/// is not available in this build configuration.
pub const HttpClient = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !HttpClient {
        _ = allocator;
        return error.WebDisabled;
    }

    pub fn deinit(self: *HttpClient) void {
        _ = self;
    }

    pub fn get(self: *HttpClient, url: []const u8) !types.Response {
        _ = self;
        _ = url;
        return error.WebDisabled;
    }

    pub fn getWithOptions(self: *HttpClient, url: []const u8, options: types.RequestOptions) !types.Response {
        _ = self;
        _ = url;
        _ = options;
        return error.WebDisabled;
    }

    pub fn postJson(self: *HttpClient, url: []const u8, body: []const u8) !types.Response {
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
        options: types.RequestOptions,
    ) !types.Response {
        _ = self;
        _ = method;
        _ = url;
        _ = body;
        _ = options;
        return error.WebDisabled;
    }

    pub fn freeResponse(self: *HttpClient, response: types.Response) void {
        _ = self;
        _ = response;
    }
};

/// Stub weather client.
///
/// All forecast operations return `error.WebDisabled` to indicate the
/// feature is not available in this build configuration.
pub const WeatherClient = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: types.WeatherConfig) !WeatherClient {
        _ = config;
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *WeatherClient) void {
        _ = self;
    }

    pub fn forecast(self: *WeatherClient, location: []const u8) !types.Response {
        _ = self;
        _ = location;
        return error.WebDisabled;
    }

    pub fn freeResponse(self: *WeatherClient, response: types.Response) void {
        _ = self;
        _ = response;
    }
};

test {
    std.testing.refAllDecls(@This());
}
