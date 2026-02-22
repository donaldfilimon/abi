//! HTTP Request Parser
//!
//! Parses HTTP/1.1 requests from raw bytes into structured request objects.
//! Supports path parameter extraction for route matching.

const std = @import("std");
const types = @import("types.zig");
const router_types = @import("../router/types.zig");

/// Parsed HTTP request.
pub const ParsedRequest = struct {
    /// HTTP method.
    method: types.Method,
    /// Request path (without query string).
    path: []const u8,
    /// Query string (without leading '?').
    query: ?[]const u8,
    /// HTTP version.
    version: HttpVersion,
    /// Request headers.
    headers: std.StringHashMapUnmanaged([]const u8),
    /// Request body (if present).
    body: ?[]const u8,
    /// Raw request line.
    raw_path: []const u8,
    /// Allocator for cleanup.
    allocator: std.mem.Allocator,
    /// Owned memory that needs freeing.
    owned_data: ?[]u8,

    pub fn deinit(self: *ParsedRequest) void {
        self.headers.deinit(self.allocator);
        if (self.owned_data) |data| {
            self.allocator.free(data);
        }
    }

    /// Gets a header value by name (case-insensitive lookup).
    pub fn getHeader(self: *const ParsedRequest, name: []const u8) ?[]const u8 {
        // Try exact match first
        if (self.headers.get(name)) |value| {
            return value;
        }

        // Try case-insensitive match
        var it = self.headers.iterator();
        while (it.next()) |entry| {
            if (std.ascii.eqlIgnoreCase(entry.key_ptr.*, name)) {
                return entry.value_ptr.*;
            }
        }

        return null;
    }

    /// Gets the Content-Length header value.
    pub fn getContentLength(self: *const ParsedRequest) ?usize {
        const value = self.getHeader(types.Header.content_length) orelse return null;
        return std.fmt.parseInt(usize, value, 10) catch null;
    }

    /// Gets the Content-Type header value.
    pub fn getContentType(self: *const ParsedRequest) ?[]const u8 {
        return self.getHeader(types.Header.content_type);
    }

    /// Checks if the request wants keep-alive.
    pub fn wantsKeepAlive(self: *const ParsedRequest) bool {
        const connection = self.getHeader(types.Header.connection) orelse {
            // HTTP/1.1 defaults to keep-alive
            return self.version == .http_1_1;
        };

        return std.ascii.eqlIgnoreCase(connection, "keep-alive");
    }

    /// Gets a query parameter by name.
    pub fn getQueryParam(self: *const ParsedRequest, name: []const u8) ?[]const u8 {
        const query = self.query orelse return null;
        return parseQueryParam(query, name);
    }
};

/// HTTP version.
pub const HttpVersion = enum {
    http_1_0,
    http_1_1,
    http_2_0,
    unknown,

    pub fn fromString(s: []const u8) HttpVersion {
        if (std.mem.eql(u8, s, "HTTP/1.0")) return .http_1_0;
        if (std.mem.eql(u8, s, "HTTP/1.1")) return .http_1_1;
        if (std.mem.eql(u8, s, "HTTP/2.0") or std.mem.eql(u8, s, "HTTP/2")) return .http_2_0;
        return .unknown;
    }
};

/// Parser errors.
pub const ParseError = error{
    InvalidMethod,
    InvalidPath,
    InvalidVersion,
    InvalidHeader,
    HeaderTooLarge,
    BodyTooLarge,
    MalformedRequest,
    UnexpectedEof,
    OutOfMemory,
};

/// Parses an HTTP request from raw bytes.
pub fn parseRequest(
    allocator: std.mem.Allocator,
    data: []const u8,
    max_header_size: usize,
    max_body_size: usize,
) ParseError!ParsedRequest {
    if (data.len > max_header_size + max_body_size) {
        return ParseError.HeaderTooLarge;
    }

    // Find end of headers (double CRLF)
    const header_end = std.mem.indexOf(u8, data, "\r\n\r\n") orelse {
        return ParseError.MalformedRequest;
    };

    if (header_end > max_header_size) {
        return ParseError.HeaderTooLarge;
    }

    const header_section = data[0..header_end];
    const body_start = header_end + 4;

    // Parse request line
    const first_line_end = std.mem.indexOf(u8, header_section, "\r\n") orelse {
        return ParseError.MalformedRequest;
    };

    const request_line = header_section[0..first_line_end];
    const parsed_line = try parseRequestLine(request_line);
    const method = parsed_line.method;
    const raw_path = parsed_line.raw_path;
    const version = parsed_line.version;

    // Parse path and query
    const path, const query = splitPathQuery(raw_path);

    // Parse headers
    var headers: std.StringHashMapUnmanaged([]const u8) = .empty;
    errdefer headers.deinit(allocator);

    const headers_section = header_section[first_line_end + 2 ..];
    try parseHeaders(allocator, headers_section, &headers);

    // Get body if present
    var body: ?[]const u8 = null;
    if (body_start < data.len) {
        const body_data = data[body_start..];
        if (body_data.len > max_body_size) {
            return ParseError.BodyTooLarge;
        }
        if (body_data.len > 0) {
            body = body_data;
        }
    }

    return ParsedRequest{
        .method = method,
        .path = path,
        .query = query,
        .version = version,
        .headers = headers,
        .body = body,
        .raw_path = raw_path,
        .allocator = allocator,
        .owned_data = null,
    };
}

const RequestLine = struct {
    method: types.Method,
    raw_path: []const u8,
    version: HttpVersion,
};

/// Parses the HTTP request line (e.g., "GET /path HTTP/1.1").
fn parseRequestLine(line: []const u8) ParseError!RequestLine {
    var parts = std.mem.splitScalar(u8, line, ' ');

    const method_str = parts.next() orelse return ParseError.InvalidMethod;
    const path = parts.next() orelse return ParseError.InvalidPath;
    const version_str = parts.next() orelse return ParseError.InvalidVersion;

    const method = methodFromString(method_str) orelse return ParseError.InvalidMethod;
    const version = HttpVersion.fromString(version_str);

    return .{
        .method = method,
        .raw_path = path,
        .version = version,
    };
}

const method_map = std.StaticStringMap(types.Method).initComptime(.{
    .{ "GET", .GET },
    .{ "POST", .POST },
    .{ "PUT", .PUT },
    .{ "DELETE", .DELETE },
    .{ "PATCH", .PATCH },
    .{ "HEAD", .HEAD },
    .{ "OPTIONS", .OPTIONS },
    .{ "CONNECT", .CONNECT },
    .{ "TRACE", .TRACE },
});

/// Converts a method string to enum.
fn methodFromString(s: []const u8) ?types.Method {
    return method_map.get(s);
}

/// Splits path and query string.
fn splitPathQuery(raw_path: []const u8) struct { []const u8, ?[]const u8 } {
    if (std.mem.indexOf(u8, raw_path, "?")) |query_start| {
        const path = raw_path[0..query_start];
        const query = if (query_start + 1 < raw_path.len) raw_path[query_start + 1 ..] else null;
        return .{ path, query };
    }
    return .{ raw_path, null };
}

/// Parses HTTP headers.
fn parseHeaders(allocator: std.mem.Allocator, data: []const u8, headers: *std.StringHashMapUnmanaged([]const u8)) ParseError!void {
    var lines = std.mem.splitSequence(u8, data, "\r\n");

    while (lines.next()) |line| {
        if (line.len == 0) continue;

        const colon_pos = std.mem.indexOf(u8, line, ":") orelse {
            return ParseError.InvalidHeader;
        };

        const name = std.mem.trim(u8, line[0..colon_pos], " \t");
        const value = std.mem.trim(u8, line[colon_pos + 1 ..], " \t");

        headers.put(allocator, name, value) catch return ParseError.OutOfMemory;
    }
}

/// Parses a single query parameter.
pub fn parseQueryParam(query: []const u8, name: []const u8) ?[]const u8 {
    var pairs = std.mem.splitScalar(u8, query, '&');

    while (pairs.next()) |pair| {
        if (std.mem.indexOf(u8, pair, "=")) |eq_pos| {
            const key = pair[0..eq_pos];
            const value = pair[eq_pos + 1 ..];

            if (std.mem.eql(u8, key, name)) {
                return value;
            }
        } else {
            // Key without value
            if (std.mem.eql(u8, pair, name)) {
                return "";
            }
        }
    }

    return null;
}

/// Extracts path parameters from a URL pattern.
/// Pattern: "/users/:id/posts/:post_id"
/// Path: "/users/123/posts/456"
/// Returns: {id: "123", post_id: "456"}
pub fn extractPathParams(
    allocator: std.mem.Allocator,
    pattern: []const u8,
    path: []const u8,
) !std.StringHashMapUnmanaged([]const u8) {
    var params: std.StringHashMapUnmanaged([]const u8) = .empty;
    errdefer params.deinit(allocator);

    var pattern_parts = std.mem.splitScalar(u8, pattern, '/');
    var path_parts = std.mem.splitScalar(u8, path, '/');

    while (pattern_parts.next()) |pattern_part| {
        const path_part = path_parts.next() orelse break;

        if (pattern_part.len > 0 and pattern_part[0] == ':') {
            // This is a parameter
            const param_name = pattern_part[1..];
            try params.put(allocator, param_name, path_part);
        }
    }

    return params;
}

/// Checks if a path matches a pattern.
/// Delegates to the canonical implementation in router/types.zig.
pub const matchesPattern = router_types.matchPattern;

/// URL decodes a string.
pub fn urlDecode(allocator: std.mem.Allocator, encoded: []const u8) ![]u8 {
    var result = std.ArrayListUnmanaged(u8).empty;
    errdefer result.deinit(allocator);

    var i: usize = 0;
    while (i < encoded.len) {
        if (encoded[i] == '%' and i + 2 < encoded.len) {
            const hex = encoded[i + 1 .. i + 3];
            const byte = std.fmt.parseInt(u8, hex, 16) catch {
                try result.append(allocator, encoded[i]);
                i += 1;
                continue;
            };
            try result.append(allocator, byte);
            i += 3;
        } else if (encoded[i] == '+') {
            try result.append(allocator, ' ');
            i += 1;
        } else {
            try result.append(allocator, encoded[i]);
            i += 1;
        }
    }

    return result.toOwnedSlice(allocator);
}

test "parse simple GET request" {
    const allocator = std.testing.allocator;
    const raw_request =
        "GET /api/users?page=1 HTTP/1.1\r\n" ++
        "Host: localhost:8080\r\n" ++
        "Accept: application/json\r\n" ++
        "\r\n";

    var request = try parseRequest(allocator, raw_request, 8192, 1024 * 1024);
    defer request.deinit();

    try std.testing.expectEqual(types.Method.GET, request.method);
    try std.testing.expectEqualStrings("/api/users", request.path);
    try std.testing.expectEqualStrings("page=1", request.query.?);
    try std.testing.expectEqual(HttpVersion.http_1_1, request.version);
    try std.testing.expectEqualStrings("localhost:8080", request.getHeader("Host").?);
    try std.testing.expectEqualStrings("1", request.getQueryParam("page").?);
}

test "parse POST request with body" {
    const allocator = std.testing.allocator;
    const raw_request =
        "POST /api/users HTTP/1.1\r\n" ++
        "Host: localhost:8080\r\n" ++
        "Content-Type: application/json\r\n" ++
        "Content-Length: 27\r\n" ++
        "\r\n" ++
        "{\"name\":\"John\",\"age\":30}";

    var request = try parseRequest(allocator, raw_request, 8192, 1024 * 1024);
    defer request.deinit();

    try std.testing.expectEqual(types.Method.POST, request.method);
    try std.testing.expectEqualStrings("/api/users", request.path);
    try std.testing.expectEqual(@as(?usize, 27), request.getContentLength());
    try std.testing.expectEqualStrings("{\"name\":\"John\",\"age\":30}", request.body.?);
}

test "extract path parameters" {
    const allocator = std.testing.allocator;

    var params = try extractPathParams(allocator, "/users/:id/posts/:post_id", "/users/123/posts/456");
    defer params.deinit(allocator);

    try std.testing.expectEqualStrings("123", params.get("id").?);
    try std.testing.expectEqualStrings("456", params.get("post_id").?);
}

test "pattern matching" {
    try std.testing.expect(matchesPattern("/users/:id", "/users/123"));
    try std.testing.expect(matchesPattern("/users/:id/posts", "/users/123/posts"));
    try std.testing.expect(!matchesPattern("/users/:id", "/users/123/extra"));
    try std.testing.expect(!matchesPattern("/users/:id/posts", "/users/123"));
    try std.testing.expect(matchesPattern("/api/*", "/api/anything"));
}

test "url decode" {
    const allocator = std.testing.allocator;

    const decoded = try urlDecode(allocator, "hello%20world%21");
    defer allocator.free(decoded);

    try std.testing.expectEqualStrings("hello world!", decoded);
}

test "keep-alive detection" {
    const allocator = std.testing.allocator;

    // HTTP/1.1 defaults to keep-alive
    const raw_1 = "GET / HTTP/1.1\r\nHost: localhost\r\n\r\n";
    var req1 = try parseRequest(allocator, raw_1, 8192, 1024);
    defer req1.deinit();
    try std.testing.expect(req1.wantsKeepAlive());

    // HTTP/1.0 defaults to close
    const raw_2 = "GET / HTTP/1.0\r\nHost: localhost\r\n\r\n";
    var req2 = try parseRequest(allocator, raw_2, 8192, 1024);
    defer req2.deinit();
    try std.testing.expect(!req2.wantsKeepAlive());

    // Explicit Connection: close
    const raw_3 = "GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
    var req3 = try parseRequest(allocator, raw_3, 8192, 1024);
    defer req3.deinit();
    try std.testing.expect(!req3.wantsKeepAlive());
}

test "header lookup is case-insensitive" {
    const allocator = std.testing.allocator;
    const raw_request =
        "GET / HTTP/1.1\r\n" ++
        "content-type: text/plain\r\n" ++
        "\r\n";

    var request = try parseRequest(allocator, raw_request, 8192, 1024);
    defer request.deinit();

    try std.testing.expectEqualStrings("text/plain", request.getHeader("Content-Type").?);
}
