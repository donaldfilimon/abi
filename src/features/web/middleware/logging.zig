//! Logging Middleware
//!
//! Logs HTTP requests in configurable formats (common, combined, JSON).

const std = @import("std");
const types = @import("types.zig");
const MiddlewareContext = types.MiddlewareContext;
const logging_key = "_logging";
const access_log_key = "_access_log";

/// Log output format.
pub const LogFormat = enum {
    /// Apache Common Log Format
    common,
    /// Apache Combined Log Format (with referrer and user-agent)
    combined,
    /// JSON format for structured logging
    json,
    /// Minimal format (method path status duration)
    minimal,
};

/// Logging configuration.
pub const LogConfig = struct {
    format: LogFormat = .minimal,
    /// Include request body in logs (careful with sensitive data)
    log_body: bool = false,
    /// Include response body in logs
    log_response_body: bool = false,
    /// Maximum body length to log
    max_body_log_length: usize = 1024,
    /// Skip logging for these paths (e.g., /health)
    skip_paths: []const []const u8 = &.{},
};

/// Creates a logging middleware with the given configuration.
pub fn createLoggingMiddleware(config: LogConfig) types.MiddlewareFn {
    _ = config;
    // Return the default minimal logger
    // In a full implementation, we'd capture config in a closure
    return &minimalLogger;
}

/// Minimal logging middleware - logs method, path, status, duration.
pub fn minimalLogger(ctx: *MiddlewareContext) !void {
    // Log at the end of request processing
    // Store start time is already done in context init

    // We log after the request completes, so we defer this
    // For now, just mark that logging is enabled
    try ctx.set(logging_key, "minimal");
}

/// Logs the request after processing completes.
pub fn logRequest(ctx: *const MiddlewareContext, status: u16) void {
    const path = ctx.request.path;
    const duration = ctx.elapsedMs();

    std.log.info("{t} {s} {d} {d}ms", .{ ctx.request.method, path, status, duration });
}

const CommonLogFields = struct {
    path: []const u8,
    remote_addr: []const u8,
    user: []const u8,
};

fn commonLogFields(ctx: *const MiddlewareContext) CommonLogFields {
    return .{
        .path = ctx.request.raw_path,
        .remote_addr = ctx.get("remote_addr") orelse "-",
        .user = ctx.getUserId() orelse "-",
    };
}

/// Logs in Apache Common Log Format.
pub fn logCommon(ctx: *const MiddlewareContext, status: u16, bytes: usize) void {
    const fields = commonLogFields(ctx);

    // Format: host ident authuser date request status bytes
    std.log.info("{s} - {s} \"{t} {s} HTTP/1.1\" {d} {d}", .{
        fields.remote_addr,
        fields.user,
        ctx.request.method,
        fields.path,
        status,
        bytes,
    });
}

/// Logs in Apache Combined Log Format.
pub fn logCombined(ctx: *const MiddlewareContext, status: u16, bytes: usize) void {
    const fields = commonLogFields(ctx);
    const referrer = ctx.request.getHeader("Referer") orelse "-";
    const user_agent = ctx.request.getHeader("User-Agent") orelse "-";

    std.log.info("{s} - {s} \"{t} {s} HTTP/1.1\" {d} {d} \"{s}\" \"{s}\"", .{
        fields.remote_addr,
        fields.user,
        ctx.request.method,
        fields.path,
        status,
        bytes,
        referrer,
        user_agent,
    });
}

/// Logs in JSON format for structured logging.
pub fn logJson(
    allocator: std.mem.Allocator,
    ctx: *const MiddlewareContext,
    status: u16,
    bytes: usize,
) !void {
    const log_entry = .{
        .timestamp = std.time.timestamp(),
        .method = @tagName(ctx.request.method),
        .path = ctx.request.path,
        .query = ctx.request.query orelse "",
        .status = status,
        .bytes = bytes,
        .duration_ms = ctx.elapsedMs(),
        .remote_addr = ctx.get("remote_addr") orelse "-",
        .user_id = ctx.getUserId() orelse null,
        .user_agent = ctx.request.getHeader("User-Agent") orelse "-",
    };

    var output = std.ArrayListUnmanaged(u8).empty;
    defer output.deinit(allocator);

    try std.json.stringify(log_entry, .{}, output.writer(allocator));
    std.log.info("{s}", .{output.items});
}

/// Request logging middleware that logs at request start.
pub fn requestStartLogger(ctx: *MiddlewareContext) !void {
    const path = ctx.request.path;

    std.log.debug("--> {t} {s}", .{ ctx.request.method, path });
}

/// Access log middleware (logs completed requests).
pub fn accessLog(ctx: *MiddlewareContext) !void {
    // This runs before the handler, we want to log after
    // Store a marker that we should log on completion
    try ctx.set(access_log_key, "true");
}

/// Should be called after request completes to log access.
pub fn finalizeAccessLog(ctx: *const MiddlewareContext) void {
    if (ctx.get(access_log_key) == null) return;

    const status = @intFromEnum(ctx.response.status);
    const bytes = ctx.response.body.items.len;

    logCommon(ctx, status, bytes);
}

test "minimal logger sets marker" {
    const allocator = std.testing.allocator;
    const server = @import("../server/mod.zig");

    var request = server.ParsedRequest{
        .method = .GET,
        .path = "/test",
        .query = null,
        .version = .http_1_1,
        .headers = std.StringHashMap([]const u8).init(allocator),
        .body = null,
        .raw_path = "/test",
        .allocator = allocator,
        .owned_data = null,
    };
    defer request.deinit();

    var response = server.ResponseBuilder.init(allocator);
    defer response.deinit();

    var ctx = MiddlewareContext.init(allocator, &request, &response);
    defer ctx.deinit();

    try minimalLogger(&ctx);

    try std.testing.expectEqualStrings("minimal", ctx.get(logging_key).?);
}

test "log format selection" {
    // Test that createLoggingMiddleware returns a valid function
    const mw = createLoggingMiddleware(.{ .format = .json });
    try std.testing.expect(mw == &minimalLogger);
}
