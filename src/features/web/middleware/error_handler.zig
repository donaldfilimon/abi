//! Error Handling Middleware
//!
//! Catches errors and converts them to appropriate HTTP responses.

const std = @import("std");
const types = @import("types.zig");
const server = @import("../server/mod.zig");
const MiddlewareContext = types.MiddlewareContext;

/// Error response format.
pub const ErrorFormat = enum {
    /// JSON error response
    json,
    /// Plain text error
    text,
    /// HTML error page
    html,
};

/// Error handler configuration.
pub const ErrorConfig = struct {
    /// Response format for errors.
    format: ErrorFormat = .json,
    /// Include stack traces in development.
    include_stack_trace: bool = false,
    /// Include internal error details.
    include_details: bool = false,
    /// Custom error page template for HTML format.
    html_template: ?[]const u8 = null,
};

/// Standard error response structure.
pub const ErrorResponse = struct {
    status: u16,
    message: []const u8,
    code: ?[]const u8 = null,
    details: ?[]const u8 = null,
    request_id: ?[]const u8 = null,
};

/// Creates an error handler middleware.
pub fn createErrorHandler(config: ErrorConfig) types.MiddlewareFn {
    _ = config;
    return &jsonErrorHandler;
}

/// JSON error handler middleware.
pub fn jsonErrorHandler(ctx: *MiddlewareContext) !void {
    // This middleware sets up error handling context
    // Actual error handling happens in handleError
    try ctx.set("_error_format", "json");
}

/// Handles an error and sends appropriate response.
pub fn handleError(
    ctx: *MiddlewareContext,
    err: anyerror,
    config: ErrorConfig,
) !void {
    const status = errorToStatus(err);
    const message = errorToMessage(err);

    _ = ctx.response.setStatus(status);

    switch (config.format) {
        .json => try sendJsonError(ctx, status, message, config),
        .text => try sendTextError(ctx, status, message),
        .html => try sendHtmlError(ctx, status, message, config),
    }

    ctx.abort();
}

/// Sends a JSON error response.
pub fn sendJsonError(
    ctx: *MiddlewareContext,
    status: server.Status,
    message: []const u8,
    config: ErrorConfig,
) !void {
    _ = config;
    ctx.response.body.clearRetainingCapacity();

    const status_code = @intFromEnum(status);
    const request_id = ctx.get("request_id") orelse "";

    try ctx.response.body.writer().print(
        \\{{"error":{{"status":{d},"message":"{s}","request_id":"{s}"}}}}
    , .{ status_code, message, request_id });

    _ = try ctx.response.setHeader(server.Header.content_type, server.MimeType.json);
}

/// Sends a plain text error response.
pub fn sendTextError(
    ctx: *MiddlewareContext,
    status: server.Status,
    message: []const u8,
) !void {
    _ = try ctx.response.text(message);
    _ = ctx.response.setStatus(status);
}

/// Sends an HTML error response.
pub fn sendHtmlError(
    ctx: *MiddlewareContext,
    status: server.Status,
    message: []const u8,
    config: ErrorConfig,
) !void {
    const status_code = @intFromEnum(status);
    const phrase = status.phrase() orelse "Error";

    if (config.html_template) |_| {
        // Custom template would be used here
        _ = try ctx.response.html(message);
    } else {
        // Default error page
        ctx.response.body.clearRetainingCapacity();
        try ctx.response.body.writer().print(
            \\<!DOCTYPE html>
            \\<html>
            \\<head><title>{d} {s}</title></head>
            \\<body>
            \\<h1>{d} {s}</h1>
            \\<p>{s}</p>
            \\</body>
            \\</html>
        , .{ status_code, phrase, status_code, phrase, message });
        _ = try ctx.response.setHeader(server.Header.content_type, server.MimeType.html);
    }
    _ = ctx.response.setStatus(status);
}

/// Maps Zig errors to HTTP status codes.
/// Note: accepts anyerror intentionally — this is a catch-all error mapper
/// that must handle any error from any handler in the middleware pipeline.
pub fn errorToStatus(err: anyerror) server.Status {
    return switch (err) {
        // Client errors (4xx)
        error.InvalidRequest, error.MalformedRequest => .bad_request,
        error.Unauthorized, error.InvalidToken => .unauthorized,
        error.Forbidden, error.AccessDenied => .forbidden,
        error.NotFound, error.FileNotFound => .not_found,
        error.MethodNotAllowed => .method_not_allowed,
        error.Conflict => .conflict,
        error.Gone => .gone,
        error.PayloadTooLarge, error.RequestTooLarge, error.BodyTooLarge => .payload_too_large,
        error.UnsupportedMediaType => .unsupported_media_type,
        error.TooManyRequests, error.RateLimited => .too_many_requests,

        // Server errors (5xx)
        error.InternalError, error.Unexpected => .internal_server_error,
        error.NotImplemented => .not_implemented,
        error.BadGateway => .bad_gateway,
        error.ServiceUnavailable, error.Unavailable => .service_unavailable,
        error.GatewayTimeout, error.Timeout => .gateway_timeout,

        // Default to internal server error
        else => .internal_server_error,
    };
}

/// Maps errors to human-readable messages.
/// Note: accepts anyerror intentionally — this is a catch-all error mapper
/// that must handle any error from any handler in the middleware pipeline.
pub fn errorToMessage(err: anyerror) []const u8 {
    return switch (err) {
        error.InvalidRequest, error.MalformedRequest => "Invalid request",
        error.Unauthorized, error.InvalidToken => "Authentication required",
        error.Forbidden, error.AccessDenied => "Access denied",
        error.NotFound, error.FileNotFound => "Resource not found",
        error.MethodNotAllowed => "Method not allowed",
        error.Conflict => "Resource conflict",
        error.PayloadTooLarge, error.RequestTooLarge, error.BodyTooLarge => "Request too large",
        error.TooManyRequests, error.RateLimited => "Too many requests",
        error.InternalError => "Internal server error",
        error.NotImplemented => "Not implemented",
        error.ServiceUnavailable, error.Unavailable => "Service unavailable",
        error.Timeout, error.GatewayTimeout => "Request timeout",
        else => "An error occurred",
    };
}

/// Convenience function to send a 400 Bad Request.
pub fn badRequest(ctx: *MiddlewareContext, message: []const u8) !void {
    _ = ctx.response.setStatus(.bad_request);
    try sendJsonError(ctx, .bad_request, message, .{});
    ctx.abort();
}

/// Convenience function to send a 401 Unauthorized.
pub fn unauthorized(ctx: *MiddlewareContext, message: []const u8) !void {
    _ = ctx.response.setStatus(.unauthorized);
    _ = try ctx.response.setHeader("WWW-Authenticate", "Bearer");
    try sendJsonError(ctx, .unauthorized, message, .{});
    ctx.abort();
}

/// Convenience function to send a 403 Forbidden.
pub fn forbidden(ctx: *MiddlewareContext, message: []const u8) !void {
    _ = ctx.response.setStatus(.forbidden);
    try sendJsonError(ctx, .forbidden, message, .{});
    ctx.abort();
}

/// Convenience function to send a 404 Not Found.
pub fn notFound(ctx: *MiddlewareContext, message: []const u8) !void {
    _ = ctx.response.setStatus(.not_found);
    try sendJsonError(ctx, .not_found, message, .{});
    ctx.abort();
}

/// Convenience function to send a 500 Internal Server Error.
pub fn internalError(ctx: *MiddlewareContext, message: []const u8) !void {
    _ = ctx.response.setStatus(.internal_server_error);
    try sendJsonError(ctx, .internal_server_error, message, .{});
    ctx.abort();
}

test "errorToStatus mapping" {
    try std.testing.expectEqual(server.Status.bad_request, errorToStatus(error.InvalidRequest));
    try std.testing.expectEqual(server.Status.unauthorized, errorToStatus(error.Unauthorized));
    try std.testing.expectEqual(server.Status.forbidden, errorToStatus(error.Forbidden));
    try std.testing.expectEqual(server.Status.not_found, errorToStatus(error.NotFound));
    try std.testing.expectEqual(server.Status.internal_server_error, errorToStatus(error.InternalError));
    try std.testing.expectEqual(server.Status.too_many_requests, errorToStatus(error.TooManyRequests));
}

test "errorToMessage" {
    try std.testing.expectEqualStrings("Invalid request", errorToMessage(error.InvalidRequest));
    try std.testing.expectEqualStrings("Resource not found", errorToMessage(error.NotFound));
    try std.testing.expectEqualStrings("Authentication required", errorToMessage(error.Unauthorized));
}
