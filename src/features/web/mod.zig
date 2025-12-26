//! Web feature helpers for HTTP and weather client access.
const std = @import("std");
const build_options = @import("build_options");

const client = @import("client.zig");
const weather = @import("weather.zig");

pub const JsonValue = std.json.Value;
pub const ParsedJson = std.json.Parsed(JsonValue);
pub const Response = client.Response;
pub const HttpClient = client.HttpClient;
pub const RequestOptions = client.RequestOptions;
pub const WeatherClient = weather.WeatherClient;
pub const WeatherConfig = weather.WeatherConfig;
pub const http = @import("../../shared/utils/http/mod.zig");

pub const WebError = error{
    WebDisabled,
};

var initialized: bool = false;

pub fn init(_: std.mem.Allocator) !void {
    if (!isEnabled()) return WebError.WebDisabled;
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

pub fn get(allocator: std.mem.Allocator, url: []const u8) !Response {
    var client_instance = try HttpClient.init(allocator);
    defer client_instance.deinit();
    return client_instance.get(url);
}

pub fn getWithOptions(
    allocator: std.mem.Allocator,
    url: []const u8,
    options: RequestOptions,
) !Response {
    var client_instance = try HttpClient.init(allocator);
    defer client_instance.deinit();
    return client_instance.getWithOptions(url, options);
}

pub fn postJson(allocator: std.mem.Allocator, url: []const u8, body: []const u8) !Response {
    var client_instance = try HttpClient.init(allocator);
    defer client_instance.deinit();
    return client_instance.postJson(url, body);
}

pub fn postJsonWithOptions(
    allocator: std.mem.Allocator,
    url: []const u8,
    body: []const u8,
    options: RequestOptions,
) !Response {
    var client_instance = try HttpClient.init(allocator);
    defer client_instance.deinit();
    var request_options = options;
    if (request_options.content_type == null) {
        request_options.content_type = "application/json";
    }
    return client_instance.requestWithOptions(.POST, url, body, request_options);
}

pub fn freeResponse(allocator: std.mem.Allocator, response: Response) void {
    allocator.free(response.body);
}

pub fn parseJsonValue(allocator: std.mem.Allocator, response: Response) !ParsedJson {
    return std.json.parseFromSlice(JsonValue, allocator, response.body, .{});
}

pub fn isSuccessStatus(status: u16) bool {
    return status >= 200 and status < 300;
}

pub fn forecast(allocator: std.mem.Allocator, location: []const u8) !Response {
    var client_instance = try WeatherClient.init(allocator, .{});
    defer client_instance.deinit();
    return client_instance.forecast(location);
}

test "web module init gating" {
    if (!isEnabled()) return;
    try init(std.testing.allocator);
    try std.testing.expect(isInitialized());
    deinit();
    try std.testing.expect(!isInitialized());
}

test "web helpers parse json and status" {
    const response = Response{ .status = 200, .body = "{\"ok\":true}" };
    var parsed = try parseJsonValue(std.testing.allocator, response);
    defer parsed.deinit();
    try std.testing.expect(isSuccessStatus(response.status));
    try std.testing.expect(parsed.value.object.get("ok").?.bool == true);
}
