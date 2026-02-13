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

/// CORS middleware that captures its own configuration.
///
/// Unlike `createCorsMiddleware` (which discards config due to Zig fn pointer
/// limitations), this struct properly stores config and applies it per-request.
///
/// Usage:
/// ```zig
/// const cors = CorsMiddleware.init(.{ .allowed_origins = &.{"https://mysite.com"} });
/// cors.handle(&ctx);
/// ```
pub const CorsMiddleware = struct {
    config: CorsConfig,

    pub fn init(config: CorsConfig) CorsMiddleware {
        return .{ .config = config };
    }

    /// Apply CORS headers and handle preflight for this request.
    pub fn handle(self: *const CorsMiddleware, ctx: *MiddlewareContext) !void {
        try addCorsHeaders(ctx, self.config);
        if (ctx.request.method == .OPTIONS) {
            try handlePreflight(ctx, self.config);
        }
    }
};

/// Creates a CORS middleware function pointer (always permissive).
///
/// Zig function pointers cannot capture state, so this ignores the config
/// parameter and returns a permissive handler. For config-aware CORS,
/// use `CorsMiddleware.init(config)` instead.
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
        var exposed_headers_buf: [1024]u8 = undefined;
        const exposed_headers = joinStringList(&exposed_headers_buf, config.exposed_headers);
        _ = try ctx.response.setHeader("Access-Control-Expose-Headers", exposed_headers);
    }
}

fn joinMethodList(buf: []u8, methods: []const server.Method) []const u8 {
    var stream = std.io.fixedBufferStream(buf);
    const writer = stream.writer();
    for (methods, 0..) |method, i| {
        if (i > 0) writer.writeAll(", ") catch break;
        writer.print("{t}", .{method}) catch break;
    }
    return stream.getWritten();
}

fn joinStringList(buf: []u8, items: []const []const u8) []const u8 {
    var stream = std.io.fixedBufferStream(buf);
    const writer = stream.writer();
    for (items, 0..) |item, i| {
        if (i > 0) writer.writeAll(", ") catch break;
        writer.writeAll(item) catch break;
    }
    return stream.getWritten();
}

fn setStringListHeader(
    ctx: *MiddlewareContext,
    name: []const u8,
    items: []const []const u8,
    buf: []u8,
) !void {
    const joined = joinStringList(buf, items);
    _ = try ctx.response.setHeader(name, joined);
}

/// Handles CORS preflight (OPTIONS) requests.
pub fn handlePreflight(ctx: *MiddlewareContext, config: CorsConfig) !void {
    // Add allowed methods
    var methods_buf: [256]u8 = undefined;
    const allowed_methods = joinMethodList(&methods_buf, config.allowed_methods);
    _ = try ctx.response.setHeader("Access-Control-Allow-Methods", allowed_methods);

    // Add allowed headers
    var headers_buf: [1024]u8 = undefined;
    try setStringListHeader(ctx, "Access-Control-Allow-Headers", config.allowed_headers, &headers_buf);

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

test "CorsMiddleware applies custom config" {
    const allocator = std.testing.allocator;

    const cors = CorsMiddleware.init(.{
        .allowed_origins = &.{"https://mysite.com"},
        .allow_credentials = true,
    });

    var request = server.ParsedRequest{
        .method = .GET,
        .path = "/api/data",
        .query = null,
        .version = .http_1_1,
        .headers = std.StringHashMap([]const u8).init(allocator),
        .body = null,
        .raw_path = "/api/data",
        .allocator = allocator,
        .owned_data = null,
    };
    defer request.deinit();
    try request.headers.put("Origin", "https://mysite.com");

    var response = server.ResponseBuilder.init(allocator);
    defer response.deinit();

    var ctx = MiddlewareContext.init(allocator, &request, &response);
    defer ctx.deinit();

    try cors.handle(&ctx);

    try std.testing.expectEqualStrings(
        "https://mysite.com",
        response.getHeader("Access-Control-Allow-Origin").?,
    );
    try std.testing.expectEqualStrings(
        "true",
        response.getHeader("Access-Control-Allow-Credentials").?,
    );
}

test "handlePreflight sets headers and aborts" {
    const allocator = std.testing.allocator;

    var request = server.ParsedRequest{
        .method = .OPTIONS,
        .path = "/",
        .query = null,
        .version = .http_1_1,
        .headers = std.StringHashMap([]const u8).init(allocator),
        .body = null,
        .raw_path = "/",
        .allocator = allocator,
        .owned_data = null,
    };
    defer request.deinit();

    var response = server.ResponseBuilder.init(allocator);
    defer response.deinit();

    var ctx = MiddlewareContext.init(allocator, &request, &response);
    defer ctx.deinit();

    try handlePreflight(&ctx, default_config);

    try std.testing.expectEqual(server.Status.no_content, response.status);
    try std.testing.expectEqualStrings(
        "GET, POST, PUT, DELETE, PATCH, OPTIONS",
        response.getHeader("Access-Control-Allow-Methods").?,
    );
    try std.testing.expectEqualStrings(
        "Content-Type, Authorization, X-Request-Id",
        response.getHeader("Access-Control-Allow-Headers").?,
    );
    try std.testing.expectEqualStrings("86400", response.getHeader("Access-Control-Max-Age").?);
    try std.testing.expect(!ctx.should_continue);
}
