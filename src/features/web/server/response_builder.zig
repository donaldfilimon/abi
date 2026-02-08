//! HTTP Response Builder
//!
//! Provides a fluent interface for building HTTP responses with support
//! for JSON, HTML, text, and binary content.

const std = @import("std");
const types = @import("types.zig");

/// Fluent response builder for constructing HTTP responses.
pub const ResponseBuilder = struct {
    /// Response status code.
    status: types.Status,
    /// Response headers.
    headers: std.StringHashMap([]const u8),
    /// Response body.
    body: std.ArrayListUnmanaged(u8),
    /// Memory allocator.
    allocator: std.mem.Allocator,
    /// Whether response has been finalized.
    finalized: bool,
    /// Owned header values that need cleanup.
    owned_values: std.ArrayListUnmanaged([]u8),

    /// Creates a new response builder.
    pub fn init(allocator: std.mem.Allocator) ResponseBuilder {
        return .{
            .status = .ok,
            .headers = std.StringHashMap([]const u8).init(allocator),
            .body = .empty,
            .allocator = allocator,
            .finalized = false,
            .owned_values = .empty,
        };
    }

    /// Cleans up all resources.
    pub fn deinit(self: *ResponseBuilder) void {
        self.headers.deinit();
        self.body.deinit(self.allocator);
        self.freeOwnedValues();
        self.owned_values.deinit(self.allocator);
    }

    /// Resets the builder for reuse.
    pub fn reset(self: *ResponseBuilder) void {
        self.status = .ok;
        self.headers.clearRetainingCapacity();
        self.body.clearRetainingCapacity();
        self.finalized = false;
        self.freeOwnedValues();
    }

    fn freeOwnedValues(self: *ResponseBuilder) void {
        for (self.owned_values.items) |value| {
            self.allocator.free(value);
        }
        self.owned_values.clearRetainingCapacity();
    }

    // --- Status Methods ---

    /// Sets the response status.
    pub fn setStatus(self: *ResponseBuilder, status: types.Status) *ResponseBuilder {
        self.status = status;
        return self;
    }

    /// Sets status to 200 OK.
    pub fn ok(self: *ResponseBuilder) *ResponseBuilder {
        return self.setStatus(.ok);
    }

    /// Sets status to 201 Created.
    pub fn created(self: *ResponseBuilder) *ResponseBuilder {
        return self.setStatus(.created);
    }

    /// Sets status to 204 No Content.
    pub fn noContent(self: *ResponseBuilder) *ResponseBuilder {
        return self.setStatus(.no_content);
    }

    /// Sets status to 400 Bad Request.
    pub fn badRequest(self: *ResponseBuilder) *ResponseBuilder {
        return self.setStatus(.bad_request);
    }

    /// Sets status to 401 Unauthorized.
    pub fn unauthorized(self: *ResponseBuilder) *ResponseBuilder {
        return self.setStatus(.unauthorized);
    }

    /// Sets status to 403 Forbidden.
    pub fn forbidden(self: *ResponseBuilder) *ResponseBuilder {
        return self.setStatus(.forbidden);
    }

    /// Sets status to 404 Not Found.
    pub fn notFound(self: *ResponseBuilder) *ResponseBuilder {
        return self.setStatus(.not_found);
    }

    /// Sets status to 500 Internal Server Error.
    pub fn internalError(self: *ResponseBuilder) *ResponseBuilder {
        return self.setStatus(.internal_server_error);
    }

    // --- Header Methods ---

    /// Sets a response header.
    pub fn setHeader(self: *ResponseBuilder, name: []const u8, value: []const u8) !*ResponseBuilder {
        try self.headers.put(name, value);
        return self;
    }

    /// Sets a header with a formatted value.
    pub fn setHeaderFmt(self: *ResponseBuilder, name: []const u8, comptime fmt: []const u8, args: anytype) !*ResponseBuilder {
        const value = try std.fmt.allocPrint(self.allocator, fmt, args);
        try self.owned_values.append(self.allocator, value);
        try self.headers.put(name, value);
        return self;
    }

    /// Sets the Content-Type header.
    pub fn setContentType(self: *ResponseBuilder, content_type: []const u8) !*ResponseBuilder {
        return self.setHeader(types.Header.content_type, content_type);
    }

    /// Sets the Location header (for redirects).
    pub fn setLocation(self: *ResponseBuilder, location: []const u8) !*ResponseBuilder {
        return self.setHeader(types.Header.location, location);
    }

    /// Sets cache control headers.
    pub fn setCacheControl(self: *ResponseBuilder, directive: []const u8) !*ResponseBuilder {
        return self.setHeader(types.Header.cache_control, directive);
    }

    /// Disables caching.
    pub fn noCache(self: *ResponseBuilder) !*ResponseBuilder {
        return self.setCacheControl("no-store, no-cache, must-revalidate");
    }

    /// Adds a Set-Cookie header.
    pub fn setCookie(self: *ResponseBuilder, name: []const u8, value: []const u8, options: CookieOptions) !*ResponseBuilder {
        var cookie = std.ArrayListUnmanaged(u8).empty;
        defer cookie.deinit(self.allocator);

        try cookie.writer(self.allocator).print("{s}={s}", .{ name, value });

        if (options.path) |path| {
            try cookie.writer(self.allocator).print("; Path={s}", .{path});
        }
        if (options.domain) |domain| {
            try cookie.writer(self.allocator).print("; Domain={s}", .{domain});
        }
        if (options.max_age) |max_age| {
            try cookie.writer(self.allocator).print("; Max-Age={d}", .{max_age});
        }
        if (options.secure) {
            try cookie.appendSlice(self.allocator, "; Secure");
        }
        if (options.http_only) {
            try cookie.appendSlice(self.allocator, "; HttpOnly");
        }
        if (options.same_site) |same_site| {
            try cookie.writer(self.allocator).print("; SameSite={s}", .{same_site});
        }

        const owned_cookie = try self.allocator.dupe(u8, cookie.items);
        try self.owned_values.append(self.allocator, owned_cookie);
        return self.setHeader(types.Header.set_cookie, owned_cookie);
    }

    // --- Body Methods ---

    /// Sets the response body as raw bytes.
    pub fn setBody(self: *ResponseBuilder, body: []const u8) !*ResponseBuilder {
        self.body.clearRetainingCapacity();
        try self.body.appendSlice(self.allocator, body);
        return self;
    }

    /// Appends to the response body.
    pub fn appendBody(self: *ResponseBuilder, data: []const u8) !*ResponseBuilder {
        try self.body.appendSlice(self.allocator, data);
        return self;
    }

    fn setJsonBodyWithOptions(self: *ResponseBuilder, value: anytype, options: anytype) !*ResponseBuilder {
        self.body.clearRetainingCapacity();
        try std.json.stringify(value, options, self.body.writer(self.allocator));
        _ = try self.setContentType(types.MimeType.json);
        return self;
    }

    /// Sets the body as JSON.
    pub fn json(self: *ResponseBuilder, value: anytype) !*ResponseBuilder {
        return self.setJsonBodyWithOptions(value, .{});
    }

    /// Sets the body as pretty-printed JSON.
    pub fn jsonPretty(self: *ResponseBuilder, value: anytype) !*ResponseBuilder {
        return self.setJsonBodyWithOptions(value, .{ .whitespace = .indent_2 });
    }

    fn setTypedBody(self: *ResponseBuilder, content: []const u8, mime_type: []const u8) !*ResponseBuilder {
        _ = try self.setBody(content);
        _ = try self.setContentType(mime_type);
        return self;
    }

    /// Sets the body as plain text.
    pub fn text(self: *ResponseBuilder, content: []const u8) !*ResponseBuilder {
        return self.setTypedBody(content, types.MimeType.plain);
    }

    /// Sets the body as HTML.
    pub fn html(self: *ResponseBuilder, content: []const u8) !*ResponseBuilder {
        return self.setTypedBody(content, types.MimeType.html);
    }

    /// Sets the body as XML.
    pub fn xml(self: *ResponseBuilder, content: []const u8) !*ResponseBuilder {
        return self.setTypedBody(content, types.MimeType.xml);
    }

    fn redirectWithStatus(self: *ResponseBuilder, status: types.Status, url: []const u8) !*ResponseBuilder {
        _ = self.setStatus(status);
        _ = try self.setLocation(url);
        return self;
    }

    // --- Redirect Methods ---

    /// Redirects to a URL (302 Found).
    pub fn redirect(self: *ResponseBuilder, url: []const u8) !*ResponseBuilder {
        return self.redirectWithStatus(.found, url);
    }

    /// Permanent redirect (301).
    pub fn redirectPermanent(self: *ResponseBuilder, url: []const u8) !*ResponseBuilder {
        return self.redirectWithStatus(.moved_permanently, url);
    }

    /// Temporary redirect (307).
    pub fn redirectTemporary(self: *ResponseBuilder, url: []const u8) !*ResponseBuilder {
        return self.redirectWithStatus(.temporary_redirect, url);
    }

    // --- Error Response Methods ---

    /// Creates a JSON error response.
    pub fn jsonError(self: *ResponseBuilder, status: types.Status, message: []const u8) !*ResponseBuilder {
        _ = self.setStatus(status);
        // Build error JSON manually since "error" is a reserved keyword
        self.body.clearRetainingCapacity();
        try self.body.writer(self.allocator).print(
            \\{{"error":{{"status":{d},"message":"{s}"}}}}
        , .{ @intFromEnum(status), message });
        _ = try self.setContentType(types.MimeType.json);
        return self;
    }

    // --- Output Methods ---

    fn writeHttpResponse(self: *ResponseBuilder, writer: anytype) !void {
        // Status line
        try writer.print("HTTP/1.1 {d} {s}\r\n", .{
            @intFromEnum(self.status),
            self.status.phrase() orelse "Unknown",
        });

        // Content-Length (if body present)
        if (self.body.items.len > 0) {
            try writer.print("Content-Length: {d}\r\n", .{self.body.items.len});
        }

        // Headers
        var it = self.headers.iterator();
        while (it.next()) |entry| {
            try writer.print("{s}: {s}\r\n", .{ entry.key_ptr.*, entry.value_ptr.* });
        }

        // End headers
        try writer.writeAll("\r\n");

        // Body
        if (self.body.items.len > 0) {
            try writer.writeAll(self.body.items);
        }

        self.finalized = true;
    }

    /// Builds the complete HTTP response as bytes.
    pub fn build(self: *ResponseBuilder) ![]const u8 {
        var result = std.ArrayListUnmanaged(u8).empty;
        errdefer result.deinit(self.allocator);

        const writer = result.writer(self.allocator);
        try self.writeHttpResponse(writer);
        return result.toOwnedSlice(self.allocator);
    }

    /// Writes the response directly to a writer.
    pub fn writeTo(self: *ResponseBuilder, writer: anytype) !void {
        try self.writeHttpResponse(writer);
    }

    /// Returns the response body.
    pub fn getBody(self: *const ResponseBuilder) []const u8 {
        return self.body.items;
    }

    /// Returns a header value.
    pub fn getHeader(self: *const ResponseBuilder, name: []const u8) ?[]const u8 {
        return self.headers.get(name);
    }
};

/// Options for Set-Cookie header.
pub const CookieOptions = struct {
    path: ?[]const u8 = "/",
    domain: ?[]const u8 = null,
    max_age: ?i64 = null,
    secure: bool = false,
    http_only: bool = true,
    same_site: ?[]const u8 = "Lax",
};

/// Creates a pre-configured error response.
pub fn errorResponse(allocator: std.mem.Allocator, status: types.Status, message: []const u8) !ResponseBuilder {
    var builder = ResponseBuilder.init(allocator);
    _ = try builder.jsonError(status, message);
    return builder;
}

/// Creates a simple JSON success response.
pub fn successResponse(allocator: std.mem.Allocator, data: anytype) !ResponseBuilder {
    var builder = ResponseBuilder.init(allocator);
    _ = builder.ok();
    _ = try builder.json(.{ .success = true, .data = data });
    return builder;
}

test "response builder basic usage" {
    const allocator = std.testing.allocator;

    var builder = ResponseBuilder.init(allocator);
    defer builder.deinit();

    _ = builder.ok();
    _ = try builder.text("Hello, World!");

    try std.testing.expectEqual(types.Status.ok, builder.status);
    try std.testing.expectEqualStrings("Hello, World!", builder.getBody());
    try std.testing.expectEqualStrings(types.MimeType.plain, builder.getHeader(types.Header.content_type).?);
}

test "response builder json" {
    const allocator = std.testing.allocator;

    var builder = ResponseBuilder.init(allocator);
    defer builder.deinit();

    const data = .{ .name = "John", .age = @as(u32, 30) };
    _ = try builder.json(data);

    try std.testing.expectEqualStrings(types.MimeType.json, builder.getHeader(types.Header.content_type).?);
    try std.testing.expect(std.mem.indexOf(u8, builder.getBody(), "\"name\":\"John\"") != null);
}

test "response builder redirect" {
    const allocator = std.testing.allocator;

    var builder = ResponseBuilder.init(allocator);
    defer builder.deinit();

    _ = try builder.redirect("https://example.com");

    try std.testing.expectEqual(types.Status.found, builder.status);
    try std.testing.expectEqualStrings("https://example.com", builder.getHeader(types.Header.location).?);
}

test "response builder error response" {
    const allocator = std.testing.allocator;

    var builder = ResponseBuilder.init(allocator);
    defer builder.deinit();

    _ = try builder.jsonError(.not_found, "Resource not found");

    try std.testing.expectEqual(types.Status.not_found, builder.status);
    try std.testing.expect(std.mem.indexOf(u8, builder.getBody(), "\"message\":\"Resource not found\"") != null);
}

test "response builder build output" {
    const allocator = std.testing.allocator;

    var builder = ResponseBuilder.init(allocator);
    defer builder.deinit();

    _ = builder.ok();
    _ = try builder.text("Test");

    const output = try builder.build();
    defer allocator.free(output);

    try std.testing.expect(std.mem.startsWith(u8, output, "HTTP/1.1 200 OK\r\n"));
    try std.testing.expect(std.mem.indexOf(u8, output, "Content-Length: 4\r\n") != null);
    try std.testing.expect(std.mem.endsWith(u8, output, "Test"));
}

test "response builder cookie" {
    const allocator = std.testing.allocator;

    var builder = ResponseBuilder.init(allocator);
    defer builder.deinit();

    _ = try builder.setCookie("session", "abc123", .{
        .path = "/",
        .http_only = true,
        .secure = true,
        .max_age = 3600,
    });

    const cookie = builder.getHeader(types.Header.set_cookie).?;
    try std.testing.expect(std.mem.indexOf(u8, cookie, "session=abc123") != null);
    try std.testing.expect(std.mem.indexOf(u8, cookie, "HttpOnly") != null);
    try std.testing.expect(std.mem.indexOf(u8, cookie, "Secure") != null);
    try std.testing.expect(std.mem.indexOf(u8, cookie, "Max-Age=3600") != null);
}

test "response builder reset" {
    const allocator = std.testing.allocator;

    var builder = ResponseBuilder.init(allocator);
    defer builder.deinit();

    _ = builder.notFound();
    _ = try builder.text("Not found");

    builder.reset();

    try std.testing.expectEqual(types.Status.ok, builder.status);
    try std.testing.expectEqual(@as(usize, 0), builder.body.items.len);
}
