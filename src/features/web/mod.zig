//! Web Module - HTTP Client and Web Utilities
//!
//! This module provides HTTP client functionality, weather API integration,
//! and persona API handlers for the ABI framework. It wraps Zig's standard
//! library HTTP client with convenient utilities for common web operations.
//!
//! ## Features
//!
//! - **HTTP Client**: Synchronous HTTP client with configurable options
//!   - GET and POST requests with JSON support
//!   - Configurable timeouts, redirects, and response size limits
//!   - Thread-safe global client with mutex protection
//!
//! - **Weather Client**: Integration with Open-Meteo weather API
//!   - Coordinate-based weather forecasts
//!   - Location validation and URL building
//!
//! - **Persona API**: HTTP handlers and routes for AI persona system
//!   - Chat request/response handlers
//!   - REST API routes with OpenAPI documentation
//!   - Health check and metrics endpoints
//!
//! ## Usage Example
//!
//! ```zig
//! const web = @import("abi").web;
//!
//! // Initialize the web module
//! try web.init(allocator);
//! defer web.deinit();
//!
//! // Make an HTTP GET request
//! const response = try web.get(allocator, "https://api.example.com/data");
//! defer web.freeResponse(allocator, response);
//!
//! if (web.isSuccessStatus(response.status)) {
//!     // Parse JSON response
//!     var parsed = try web.parseJsonValue(allocator, response);
//!     defer parsed.deinit();
//!     // Use parsed.value...
//! }
//! ```
//!
//! ## Using the Context API
//!
//! For Framework integration, use the Context struct:
//!
//! ```zig
//! const cfg = config_module.WebConfig{};
//! var ctx = try web.Context.init(allocator, cfg);
//! defer ctx.deinit();
//!
//! const response = try ctx.get("https://api.example.com/data");
//! defer ctx.freeResponse(response);
//! ```
//!
//! ## POST Request with JSON
//!
//! ```zig
//! const body = "{\"message\": \"hello\"}";
//! const response = try web.postJson(allocator, "https://api.example.com/chat", body);
//! defer web.freeResponse(allocator, response);
//! ```
//!
//! ## Request Options
//!
//! ```zig
//! const response = try web.getWithOptions(allocator, url, .{
//!     .max_response_bytes = 10 * 1024 * 1024,  // 10MB limit
//!     .user_agent = "my-app/1.0",
//!     .follow_redirects = true,
//!     .redirect_limit = 5,
//!     .extra_headers = &.{
//!         .{ .name = "Authorization", .value = "Bearer token" },
//!     },
//! });
//! ```
//!
//! ## Error Handling
//!
//! The module uses `HttpError` for HTTP-specific errors:
//! - `InvalidUrl`: URL parsing failed
//! - `InvalidRequest`: Request configuration is invalid
//! - `RequestFailed`: HTTP request failed
//! - `ConnectionFailed`: Network connection failed
//! - `ResponseTooLarge`: Response exceeds max_response_bytes
//! - `Timeout`: Request timed out
//! - `ReadFailed`: Error reading response body
//!
//! ## Feature Flag
//!
//! This module is controlled by `-Denable-web=true` (default: enabled).
//! When disabled, all operations return `error.WebDisabled`.
//!
//! ## Thread Safety
//!
//! The global `init()`/`deinit()` functions use mutex protection for
//! thread-safe access to the default client. The `Context` struct should
//! be used per-thread or with external synchronization.

const std = @import("std");
const time = @import("../../services/shared/time.zig");
const sync = @import("../../services/shared/sync.zig");
const build_options = @import("build_options");
const config_module = @import("../../core/config/mod.zig");

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
pub const ChatResult = handlers.chat.ChatResult;

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
pub const http = @import("../../services/shared/utils.zig").http;

/// Errors specific to the web module.
pub const WebError = error{
    /// The web feature is disabled in the build configuration.
    /// Enable with `-Denable-web=true`.
    WebDisabled,
};

/// Web Context for Framework integration.
///
/// Wraps the HTTP client functionality to provide a consistent interface
/// with other ABI modules. This is the preferred API for Framework users
/// as it integrates with the unified configuration system.
///
/// ## Example
///
/// ```zig
/// var ctx = try web.Context.init(allocator, config);
/// defer ctx.deinit();
///
/// const response = try ctx.get("https://api.example.com/data");
/// defer ctx.freeResponse(response);
///
/// var json = try ctx.parseJsonValue(response);
/// defer json.deinit();
/// ```
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
var client_mutex = sync.Mutex{};
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

test "isEnabled returns build option" {
    try std.testing.expectEqual(build_options.enable_web, isEnabled());
}

test "isSuccessStatus for 2xx codes" {
    try std.testing.expect(isSuccessStatus(200));
    try std.testing.expect(isSuccessStatus(201));
    try std.testing.expect(isSuccessStatus(204));
    try std.testing.expect(!isSuccessStatus(301));
    try std.testing.expect(!isSuccessStatus(404));
    try std.testing.expect(!isSuccessStatus(500));
}

test "parseJsonValue handles objects and arrays" {
    const json_str = "{\"items\":[1,2,3]}";
    const response = Response{ .status = 200, .body = json_str };
    var parsed = try parseJsonValue(std.testing.allocator, response);
    defer parsed.deinit();

    const items = parsed.value.object.get("items").?.array;
    try std.testing.expectEqual(@as(usize, 3), items.items.len);
}

test "parseJsonValue rejects invalid json" {
    const response = Response{ .status = 200, .body = "not json{" };
    const result = parseJsonValue(std.testing.allocator, response);
    try std.testing.expect(if (result) |_| false else |_| true);
}

test "freeResponse releases body memory" {
    const body = try std.testing.allocator.dupe(u8, "test body");
    const response = Response{ .status = 200, .body = body };
    freeResponse(std.testing.allocator, response);
    // No leak = test passes (testing.allocator detects leaks)
}
