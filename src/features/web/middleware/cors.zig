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
/// const cors = CorsMiddleware.init(.{
///     .allowed_origins = &.{"https://mysite.com"},
///     .allow_credentials = true,
/// });
/// cors.handle(&ctx);
/// ```
pub const CorsMiddleware = struct {
    config: CorsConfig,

    pub fn init(config: CorsConfig) CorsMiddleware {
        return .{ .config = config };
    }

    /// Apply CORS headers and handle preflight for this request.
    ///
    /// For simple requests: adds `Access-Control-Allow-Origin` and credential
    /// headers when the request origin matches the config.
    ///
    /// For preflight (OPTIONS) requests: additionally validates the requested
    /// method (from `Access-Control-Request-Method`) and headers (from
    /// `Access-Control-Request-Headers`) against the config before setting
    /// the corresponding response headers. If the requested method is not
    /// allowed, the preflight response omits the allow-methods header.
    pub fn handle(self: *const CorsMiddleware, ctx: *MiddlewareContext) !void {
        try addCorsHeaders(ctx, self.config);
        if (ctx.request.method == .OPTIONS) {
            try self.handleCheckedPreflight(ctx);
        }
    }

    /// Handles preflight with validation of the request method and headers
    /// against the stored config.
    fn handleCheckedPreflight(
        self: *const CorsMiddleware,
        ctx: *MiddlewareContext,
    ) !void {
        const config = self.config;

        // Validate Access-Control-Request-Method if present.
        if (ctx.request.getHeader("Access-Control-Request-Method")) |rm| {
            const requested_method = parseMethod(rm);
            if (requested_method) |method| {
                if (!isMethodAllowed(method, config.allowed_methods)) {
                    // Method not allowed — respond without allow headers.
                    _ = ctx.response.setStatus(.forbidden);
                    ctx.abort();
                    return;
                }
            }
        }

        // Validate Access-Control-Request-Headers if present.
        if (ctx.request.getHeader("Access-Control-Request-Headers")) |rh| {
            if (!areHeadersAllowed(rh, config.allowed_headers)) {
                _ = ctx.response.setStatus(.forbidden);
                ctx.abort();
                return;
            }
        }

        // Validation passed — set the standard preflight headers.
        try handlePreflight(ctx, config);
    }
};

/// Creates a CORS middleware function pointer (always permissive).
///
/// **Deprecated**: Zig function pointers cannot capture state, so this
/// ignores the `config` parameter and always returns a permissive handler
/// that allows all origins. Use `CorsMiddleware.init(config)` instead for
/// config-aware CORS enforcement.
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
    var writer = std.Io.Writer.fixed(buf);
    for (methods, 0..) |method, i| {
        if (i > 0) writer.writeAll(", ") catch break;
        writer.print("{t}", .{method}) catch break;
    }
    return buf[0..writer.end];
}

fn joinStringList(buf: []u8, items: []const []const u8) []const u8 {
    var writer = std.Io.Writer.fixed(buf);
    for (items, 0..) |item, i| {
        if (i > 0) writer.writeAll(", ") catch break;
        writer.writeAll(item) catch break;
    }
    return buf[0..writer.end];
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

/// Parses an HTTP method string (e.g. from Access-Control-Request-Method)
/// into a `server.Method`. Returns `null` for unrecognized methods.
fn parseMethod(method_str: []const u8) ?server.Method {
    const trimmed = std.mem.trim(u8, method_str, " \t");
    const methods = [_]struct { name: []const u8, value: server.Method }{
        .{ .name = "GET", .value = .GET },
        .{ .name = "HEAD", .value = .HEAD },
        .{ .name = "POST", .value = .POST },
        .{ .name = "PUT", .value = .PUT },
        .{ .name = "DELETE", .value = .DELETE },
        .{ .name = "CONNECT", .value = .CONNECT },
        .{ .name = "OPTIONS", .value = .OPTIONS },
        .{ .name = "TRACE", .value = .TRACE },
        .{ .name = "PATCH", .value = .PATCH },
    };
    for (methods) |entry| {
        if (std.ascii.eqlIgnoreCase(trimmed, entry.name)) {
            return entry.value;
        }
    }
    return null;
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

test "CorsMiddleware rejects disallowed origin" {
    const allocator = std.testing.allocator;

    const cors = CorsMiddleware.init(.{
        .allowed_origins = &.{"https://allowed.com"},
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
    try request.headers.put("Origin", "https://evil.com");

    var response = server.ResponseBuilder.init(allocator);
    defer response.deinit();

    var ctx = MiddlewareContext.init(allocator, &request, &response);
    defer ctx.deinit();

    try cors.handle(&ctx);

    // Origin not allowed — no Access-Control-Allow-Origin header set.
    try std.testing.expect(response.getHeader("Access-Control-Allow-Origin") == null);
}

test "CorsMiddleware preflight validates request method" {
    const allocator = std.testing.allocator;

    const cors = CorsMiddleware.init(.{
        .allowed_origins = &.{"*"},
        .allowed_methods = &.{ .GET, .POST },
    });

    // Preflight requesting DELETE (not in allowed list).
    var request = server.ParsedRequest{
        .method = .OPTIONS,
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
    try request.headers.put("Origin", "https://example.com");
    try request.headers.put("Access-Control-Request-Method", "DELETE");

    var response = server.ResponseBuilder.init(allocator);
    defer response.deinit();

    var ctx = MiddlewareContext.init(allocator, &request, &response);
    defer ctx.deinit();

    try cors.handle(&ctx);

    // Should be forbidden and aborted.
    try std.testing.expectEqual(server.Status.forbidden, response.status);
    try std.testing.expect(!ctx.should_continue);
    // No allow-methods header should be set.
    try std.testing.expect(
        response.getHeader("Access-Control-Allow-Methods") == null,
    );
}

test "CorsMiddleware preflight validates request headers" {
    const allocator = std.testing.allocator;

    const cors = CorsMiddleware.init(.{
        .allowed_origins = &.{"*"},
        .allowed_headers = &.{"Content-Type"},
    });

    var request = server.ParsedRequest{
        .method = .OPTIONS,
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
    try request.headers.put("Origin", "https://example.com");
    try request.headers.put(
        "Access-Control-Request-Headers",
        "X-Custom-Secret",
    );

    var response = server.ResponseBuilder.init(allocator);
    defer response.deinit();

    var ctx = MiddlewareContext.init(allocator, &request, &response);
    defer ctx.deinit();

    try cors.handle(&ctx);

    // Should be forbidden — requested header not in allowed list.
    try std.testing.expectEqual(server.Status.forbidden, response.status);
    try std.testing.expect(!ctx.should_continue);
}

test "CorsMiddleware preflight succeeds with valid method and headers" {
    const allocator = std.testing.allocator;

    const cors = CorsMiddleware.init(.{
        .allowed_origins = &.{"https://app.example.com"},
        .allowed_methods = &.{ .GET, .POST, .PUT },
        .allowed_headers = &.{ "Content-Type", "Authorization" },
        .max_age = 3600,
    });

    var request = server.ParsedRequest{
        .method = .OPTIONS,
        .path = "/api/resource",
        .query = null,
        .version = .http_1_1,
        .headers = std.StringHashMap([]const u8).init(allocator),
        .body = null,
        .raw_path = "/api/resource",
        .allocator = allocator,
        .owned_data = null,
    };
    defer request.deinit();
    try request.headers.put("Origin", "https://app.example.com");
    try request.headers.put("Access-Control-Request-Method", "PUT");
    try request.headers.put("Access-Control-Request-Headers", "Authorization");

    var response = server.ResponseBuilder.init(allocator);
    defer response.deinit();

    var ctx = MiddlewareContext.init(allocator, &request, &response);
    defer ctx.deinit();

    try cors.handle(&ctx);

    // Preflight succeeded — 204 with proper headers.
    try std.testing.expectEqual(server.Status.no_content, response.status);
    try std.testing.expect(!ctx.should_continue);
    try std.testing.expectEqualStrings(
        "https://app.example.com",
        response.getHeader("Access-Control-Allow-Origin").?,
    );
    try std.testing.expectEqualStrings(
        "GET, POST, PUT",
        response.getHeader("Access-Control-Allow-Methods").?,
    );
    try std.testing.expectEqualStrings(
        "Content-Type, Authorization",
        response.getHeader("Access-Control-Allow-Headers").?,
    );
    try std.testing.expectEqualStrings(
        "3600",
        response.getHeader("Access-Control-Max-Age").?,
    );
}

test "CorsMiddleware strict config rejects all origins" {
    const allocator = std.testing.allocator;

    const cors = CorsMiddleware.init(strict_config);

    var request = server.ParsedRequest{
        .method = .GET,
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
    try request.headers.put("Origin", "https://any-origin.com");

    var response = server.ResponseBuilder.init(allocator);
    defer response.deinit();

    var ctx = MiddlewareContext.init(allocator, &request, &response);
    defer ctx.deinit();

    try cors.handle(&ctx);

    // Strict config has empty allowed_origins — nothing should be set.
    try std.testing.expect(response.getHeader("Access-Control-Allow-Origin") == null);
    try std.testing.expect(response.getHeader("Access-Control-Allow-Credentials") == null);
}

test "CorsMiddleware credentials not set when allow_credentials is false" {
    const allocator = std.testing.allocator;

    const cors = CorsMiddleware.init(.{
        .allowed_origins = &.{"https://example.com"},
        .allow_credentials = false,
    });

    var request = server.ParsedRequest{
        .method = .GET,
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
    try request.headers.put("Origin", "https://example.com");

    var response = server.ResponseBuilder.init(allocator);
    defer response.deinit();

    var ctx = MiddlewareContext.init(allocator, &request, &response);
    defer ctx.deinit();

    try cors.handle(&ctx);

    // Origin should be allowed.
    try std.testing.expectEqualStrings(
        "https://example.com",
        response.getHeader("Access-Control-Allow-Origin").?,
    );
    // Credentials header should NOT be set.
    try std.testing.expect(
        response.getHeader("Access-Control-Allow-Credentials") == null,
    );
}

test "parseMethod recognizes standard methods" {
    try std.testing.expectEqual(server.Method.GET, parseMethod("GET").?);
    try std.testing.expectEqual(server.Method.POST, parseMethod("POST").?);
    try std.testing.expectEqual(server.Method.PUT, parseMethod("PUT").?);
    try std.testing.expectEqual(server.Method.DELETE, parseMethod("DELETE").?);
    try std.testing.expectEqual(server.Method.PATCH, parseMethod("PATCH").?);
    try std.testing.expectEqual(server.Method.OPTIONS, parseMethod("OPTIONS").?);
    try std.testing.expectEqual(server.Method.HEAD, parseMethod("HEAD").?);

    // Case insensitive
    try std.testing.expectEqual(server.Method.GET, parseMethod("get").?);
    try std.testing.expectEqual(server.Method.POST, parseMethod("post").?);

    // Whitespace trimming
    try std.testing.expectEqual(server.Method.PUT, parseMethod("  PUT  ").?);

    // Unknown method
    try std.testing.expect(parseMethod("UNKNOWN") == null);
    try std.testing.expect(parseMethod("") == null);
}

test "createCorsMiddleware returns permissive handler" {
    // Verify the deprecated fn still works for backward compatibility.
    const handler = createCorsMiddleware(.{
        .allowed_origins = &.{"https://specific.com"},
    });

    // The returned handler should be the permissive one (ignores config).
    try std.testing.expectEqual(@as(types.MiddlewareFn, &permissiveCors), handler);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

test "isOriginAllowed: empty origin vs wildcard" {
    try std.testing.expect(isOriginAllowed("", &.{"*"}));
    try std.testing.expect(!isOriginAllowed("", &.{"https://example.com"}));
}

test "isOriginAllowed: empty allowed list rejects all" {
    try std.testing.expect(!isOriginAllowed("https://example.com", &.{}));
    try std.testing.expect(!isOriginAllowed("", &.{}));
}

test "isOriginAllowed: wildcard subdomain rejects base domain" {
    // *.example.com must NOT match example.com itself
    try std.testing.expect(!isOriginAllowed("https://example.com", &.{"*.example.com"}));
}

test "isOriginAllowed: deeply nested subdomains" {
    try std.testing.expect(isOriginAllowed("https://a.b.c.d.example.com", &.{"*.example.com"}));
}

test "isOriginAllowed: multiple wildcard patterns" {
    const allowed = [_][]const u8{ "*.example.com", "*.test.com" };
    try std.testing.expect(isOriginAllowed("https://app.example.com", &allowed));
    try std.testing.expect(isOriginAllowed("https://app.test.com", &allowed));
    try std.testing.expect(!isOriginAllowed("https://app.other.com", &allowed));
}

test "areHeadersAllowed: extra whitespace" {
    const allowed = [_][]const u8{ "Content-Type", "Authorization" };
    try std.testing.expect(areHeadersAllowed("  Content-Type  ,  Authorization  ", &allowed));
}

test "areHeadersAllowed: one bad header rejects all" {
    const allowed = [_][]const u8{ "Content-Type", "Authorization" };
    try std.testing.expect(!areHeadersAllowed("Content-Type, X-Evil", &allowed));
}

test "areHeadersAllowed: empty allowed list" {
    try std.testing.expect(!areHeadersAllowed("Content-Type", &.{}));
}

test "parseMethod: all standard methods" {
    try std.testing.expectEqual(server.Method.GET, parseMethod("GET").?);
    try std.testing.expectEqual(server.Method.HEAD, parseMethod("HEAD").?);
    try std.testing.expectEqual(server.Method.POST, parseMethod("POST").?);
    try std.testing.expectEqual(server.Method.PUT, parseMethod("PUT").?);
    try std.testing.expectEqual(server.Method.DELETE, parseMethod("DELETE").?);
    try std.testing.expectEqual(server.Method.CONNECT, parseMethod("CONNECT").?);
    try std.testing.expectEqual(server.Method.OPTIONS, parseMethod("OPTIONS").?);
    try std.testing.expectEqual(server.Method.TRACE, parseMethod("TRACE").?);
    try std.testing.expectEqual(server.Method.PATCH, parseMethod("PATCH").?);
}

test "parseMethod: case insensitive" {
    try std.testing.expectEqual(server.Method.GET, parseMethod("get").?);
    try std.testing.expectEqual(server.Method.POST, parseMethod("post").?);
    try std.testing.expectEqual(server.Method.DELETE, parseMethod("Delete").?);
}

test "CorsConfig defaults are sensible" {
    const config = CorsConfig{};
    try std.testing.expect(isOriginAllowed("https://anything.com", config.allowed_origins));
    try std.testing.expect(isMethodAllowed(.GET, config.allowed_methods));
    try std.testing.expect(isMethodAllowed(.POST, config.allowed_methods));
    try std.testing.expect(!config.allow_credentials);
    try std.testing.expectEqual(@as(u32, 86400), config.max_age);
}

test "strict_config rejects all origins" {
    try std.testing.expectEqual(@as(usize, 0), strict_config.allowed_origins.len);
    try std.testing.expect(!strict_config.allow_credentials);
}
