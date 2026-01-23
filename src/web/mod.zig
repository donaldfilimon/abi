//! Web feature helpers for HTTP and weather client access.
//!
//! This module provides:
//! - HTTP client for making requests
//! - Weather API client
//! - Persona API handlers and routes
const std = @import("std");
const build_options = @import("build_options");
const config_module = @import("../config.zig");

const client = @import("client.zig");
const weather = @import("weather.zig");

// Handlers and routes for persona API
pub const handlers = struct {
    pub const chat = @import("handlers/chat.zig");
};
pub const routes = struct {
    pub const personas = @import("routes/personas.zig");
};

// Re-export handler types
pub const ChatHandler = handlers.chat.ChatHandler;
pub const ChatRequest = handlers.chat.ChatRequest;
pub const ChatResponse = handlers.chat.ChatResponse;

// Re-export route types
pub const PersonaRouter = routes.personas.Router;
pub const Route = routes.personas.Route;
pub const RouteContext = routes.personas.RouteContext;

pub const JsonValue = std.json.Value;
pub const ParsedJson = std.json.Parsed(JsonValue);
pub const Response = client.Response;
pub const HttpClient = client.HttpClient;
pub const RequestOptions = client.RequestOptions;
pub const WeatherClient = weather.WeatherClient;
pub const WeatherConfig = weather.WeatherConfig;
pub const http = @import("../shared/utils_combined.zig").http;

pub const WebError = error{
    WebDisabled,
};

/// Web Context for Framework integration.
/// Wraps the HTTP client functionality to provide a consistent interface with other modules.
pub const Context = struct {
    allocator: std.mem.Allocator,
    config: config_module.WebConfig,
    http_client: ?HttpClient = null,

    pub fn init(allocator: std.mem.Allocator, cfg: config_module.WebConfig) !*Context {
        if (!isEnabled()) return error.WebDisabled;

        const ctx = try allocator.create(Context);
        errdefer allocator.destroy(ctx);

        ctx.* = .{
            .allocator = allocator,
            .config = cfg,
            .http_client = try HttpClient.init(allocator),
        };

        return ctx;
    }

    pub fn deinit(self: *Context) void {
        if (self.http_client) |*c| {
            c.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Perform an HTTP GET request.
    pub fn get(self: *Context, url: []const u8) !Response {
        if (self.http_client) |*c| {
            return c.get(url);
        }
        return error.WebDisabled;
    }

    /// Perform an HTTP GET request with options.
    pub fn getWithOptions(self: *Context, url: []const u8, options: RequestOptions) !Response {
        if (self.http_client) |*c| {
            return c.getWithOptions(url, options);
        }
        return error.WebDisabled;
    }

    /// Perform an HTTP POST request with JSON body.
    pub fn postJson(self: *Context, url: []const u8, body: []const u8) !Response {
        if (self.http_client) |*c| {
            return c.postJson(url, body);
        }
        return error.WebDisabled;
    }

    /// Free a response body.
    pub fn freeResponse(self: *Context, response: Response) void {
        self.allocator.free(response.body);
    }

    /// Parse a JSON response.
    pub fn parseJsonValue(self: *Context, response: Response) !ParsedJson {
        return std.json.parseFromSlice(JsonValue, self.allocator, response.body, .{});
    }
};

var initialized: bool = false;
var client_mutex = std.Thread.Mutex{};
var default_client: ?HttpClient = null;

pub fn init(allocator: std.mem.Allocator) !void {
    if (!isEnabled()) return WebError.WebDisabled;

    client_mutex.lock();
    defer client_mutex.unlock();

    if (default_client == null) {
        default_client = try HttpClient.init(allocator);
    }
    initialized = true;
}

pub fn deinit() void {
    client_mutex.lock();
    defer client_mutex.unlock();

    if (default_client) |*http_client| {
        http_client.deinit();
        default_client = null;
    }
    initialized = false;
}

pub fn isEnabled() bool {
    return build_options.enable_web;
}

pub fn isInitialized() bool {
    return initialized;
}

pub fn get(allocator: std.mem.Allocator, url: []const u8) !Response {
    client_mutex.lock();
    defer client_mutex.unlock();

    if (default_client) |*http_client| {
        return http_client.get(url);
    }

    var client_instance = try HttpClient.init(allocator);
    defer client_instance.deinit();

    return client_instance.get(url);
}

pub fn getWithOptions(
    allocator: std.mem.Allocator,
    url: []const u8,
    options: RequestOptions,
) !Response {
    client_mutex.lock();
    defer client_mutex.unlock();

    if (default_client) |*http_client| {
        return http_client.getWithOptions(url, options);
    }

    var client_instance = try HttpClient.init(allocator);
    defer client_instance.deinit();

    return client_instance.getWithOptions(url, options);
}

pub fn postJson(allocator: std.mem.Allocator, url: []const u8, body: []const u8) !Response {
    client_mutex.lock();
    defer client_mutex.unlock();

    if (default_client) |*http_client| {
        return http_client.postJson(url, body);
    }

    var client_instance = try HttpClient.init(allocator);
    defer client_instance.deinit();

    return client_instance.postJson(url, body);
}

pub fn freeResponse(allocator: std.mem.Allocator, response: Response) void {
    allocator.free(response.body);
}

pub fn parseJsonValue(allocator: std.mem.Allocator, response: Response) !ParsedJson {
    return std.json.parseFromSlice(JsonValue, allocator, response.body, .{});
}

pub fn isSuccessStatus(status: u16) bool {
    return http.isSuccess(status);
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
