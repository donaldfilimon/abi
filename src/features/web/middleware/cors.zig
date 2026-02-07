//! CORS Middleware
//!
//! Handles Cross-Origin Resource Sharing headers and preflight requests.

const std = @import("std");
const types = @import("types.zig");
const server = @import("../server/mod.zig");
const MiddlewareContext = types.MiddlewareContext;

/// CORS configuration.
pub const CorsConfig = struct {
    /// Allowed origins ("*" for any, or specific origins).
    allowed_origins: []const []const u8 = &.{"*"},
    /// Allowed HTTP methods.
    allowed_methods: []const server.Method = &.{ .GET, .POST, .PUT, .DELETE, .PATCH, .OPTIONS },
    /// Allowed request headers.
    allowed_headers: []const []const u8 = &.{ "Content-Type", "Authorization", "X-Request-Id" },
    /// Headers to expose to the browser.
    exposed_headers: []const []const u8 = &.{},
    /// Allow credentials (cookies, auth headers).
    allow_credentials: bool = false,
    /// Preflight cache duration in seconds.
    max_age: u32 = 86400,
};

/// Default CORS configuration (permissive for development).
pub const default_config = CorsConfig{};

/// Strict CORS configuration (no cross-origin by default).
pub const strict_config = CorsConfig{
    .allowed_origins = &.{},
    .allow_credentials = false,
};

/// Creates a CORS middleware with the given configuration.
/// Note: Zig function pointers cannot capture state, so this returns
/// a handler that uses the provided config at the call site.
/// For custom configs, call addCorsHeaders/handlePreflight directly.
pub fn createCorsMiddleware(config: CorsConfig) types.MiddlewareFn {
    _ = config;
    return &permissiveCors;
}

/// Permissive CORS middleware (allows all origins).
pub fn permissiveCors(ctx: *MiddlewareContext) !void {
    try addCorsHeaders(ctx, default_config);

    // Handle preflight
    if (ctx.request.method == .OPTIONS) {
        try handlePreflight(ctx, default_config);
    }
}

/// Adds CORS headers to the response.
pub fn addCorsHeaders(ctx: *MiddlewareContext, config: CorsConfig) !void {
    const origin = ctx.request.getHeader("Origin");

    if (origin) |req_origin| {
        // Check if origin is allowed
        if (isOriginAllowed(req_origin, config.allowed_origins)) {
            _ = try ctx.response.setHeader("Access-Control-Allow-Origin", req_origin);

            if (config.allow_credentials) {
                _ = try ctx.response.setHeader("Access-Control-Allow-Credentials", "true");
            }

            // Add Vary header for proper caching
            _ = try ctx.response.setHeader("Vary", "Origin");
        }
    } else if (config.allowed_origins.len > 0 and std.mem.eql(u8, config.allowed_origins[0], "*")) {
        // No Origin header but we allow all
        _ = try ctx.response.setHeader("Access-Control-Allow-Origin", "*");
    }

    // Add exposed headers
    if (config.exposed_headers.len > 0) {
        var buf: [1024]u8 = undefined;
        var stream = std.io.fixedBufferStream(&buf);
        for (config.exposed_headers, 0..) |header, i| {
            if (i > 0) stream.writer().writeAll(", ") catch break;
            stream.writer().writeAll(header) catch break;
        }
        _ = try ctx.response.setHeader("Access-Control-Expose-Headers", stream.getWritten());
    }
}

/// Handles CORS preflight (OPTIONS) requests.
pub fn handlePreflight(ctx: *MiddlewareContext, config: CorsConfig) !void {
    // Add allowed methods
    var methods_buf: [256]u8 = undefined;
    var methods_stream = std.io.fixedBufferStream(&methods_buf);
    for (config.allowed_methods, 0..) |method, i| {
        if (i > 0) methods_stream.writer().writeAll(", ") catch break;
        methods_stream.writer().writeAll(@tagName(method)) catch break;
    }
    _ = try ctx.response.setHeader("Access-Control-Allow-Methods", methods_stream.getWritten());

    // Add allowed headers
    var headers_buf: [1024]u8 = undefined;
    var headers_stream = std.io.fixedBufferStream(&headers_buf);
    for (config.allowed_headers, 0..) |header, i| {
        if (i > 0) headers_stream.writer().writeAll(", ") catch break;
        headers_stream.writer().writeAll(header) catch break;
    }
    _ = try ctx.response.setHeader("Access-Control-Allow-Headers", headers_stream.getWritten());

    // Add max age
    var age_buf: [16]u8 = undefined;
    const age_str = std.fmt.bufPrint(&age_buf, "{d}", .{config.max_age}) catch "86400";
    _ = try ctx.response.setHeader("Access-Control-Max-Age", age_str);

    // Preflight response is 204 No Content
    _ = ctx.response.setStatus(.no_content);
    ctx.abort(); // Don't continue to handler
}

/// Checks if an origin is in the allowed list.
pub fn isOriginAllowed(origin: []const u8, allowed: []const []const u8) bool {
    for (allowed) |allowed_origin| {
        // Wildcard allows all
        if (std.mem.eql(u8, allowed_origin, "*")) {
            return true;
        }
        // Exact match
        if (std.mem.eql(u8, allowed_origin, origin)) {
            return true;
        }
        // Subdomain wildcard (e.g., "*.example.com")
        if (allowed_origin.len > 2 and std.mem.startsWith(u8, allowed_origin, "*.")) {
            const domain = allowed_origin[1..]; // ".example.com" (keep the dot)
            if (std.mem.endsWith(u8, origin, domain) and origin.len > domain.len) {
                return true;
            }
        }
    }
    return false;
}

/// Validates the request method against allowed methods.
pub fn isMethodAllowed(method: server.Method, allowed: []const server.Method) bool {
    for (allowed) |allowed_method| {
        if (method == allowed_method) {
            return true;
        }
    }
    return false;
}

/// Validates request headers against allowed headers.
pub fn areHeadersAllowed(request_headers: []const u8, allowed: []const []const u8) bool {
    var headers = std.mem.splitScalar(u8, request_headers, ',');
    while (headers.next()) |header| {
        const trimmed = std.mem.trim(u8, header, " \t");
        var found = false;
        for (allowed) |allowed_header| {
            if (std.ascii.eqlIgnoreCase(trimmed, allowed_header)) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

test "isOriginAllowed" {
    // Wildcard allows all
    try std.testing.expect(isOriginAllowed("https://example.com", &.{"*"}));

    // Exact match
    try std.testing.expect(isOriginAllowed("https://example.com", &.{"https://example.com"}));
    try std.testing.expect(!isOriginAllowed("https://other.com", &.{"https://example.com"}));

    // Multiple allowed origins
    try std.testing.expect(isOriginAllowed("https://a.com", &.{ "https://a.com", "https://b.com" }));
    try std.testing.expect(isOriginAllowed("https://b.com", &.{ "https://a.com", "https://b.com" }));
    try std.testing.expect(!isOriginAllowed("https://c.com", &.{ "https://a.com", "https://b.com" }));

    // Subdomain wildcard
    try std.testing.expect(isOriginAllowed("https://sub.example.com", &.{"*.example.com"}));
    try std.testing.expect(isOriginAllowed("https://deep.sub.example.com", &.{"*.example.com"}));

    // Must not match non-subdomains (security: prevent evil-example.com bypass)
    try std.testing.expect(!isOriginAllowed("https://evil-example.com", &.{"*.example.com"}));
    try std.testing.expect(!isOriginAllowed("https://notexample.com", &.{"*.example.com"}));
    try std.testing.expect(!isOriginAllowed(".example.com", &.{"*.example.com"}));
}

test "isMethodAllowed" {
    const allowed = [_]server.Method{ .GET, .POST };
    try std.testing.expect(isMethodAllowed(.GET, &allowed));
    try std.testing.expect(isMethodAllowed(.POST, &allowed));
    try std.testing.expect(!isMethodAllowed(.DELETE, &allowed));
}

test "areHeadersAllowed" {
    const allowed = [_][]const u8{ "Content-Type", "Authorization" };
    try std.testing.expect(areHeadersAllowed("Content-Type", &allowed));
    try std.testing.expect(areHeadersAllowed("content-type", &allowed)); // Case insensitive
    try std.testing.expect(areHeadersAllowed("Content-Type, Authorization", &allowed));
    try std.testing.expect(!areHeadersAllowed("X-Custom-Header", &allowed));
}
