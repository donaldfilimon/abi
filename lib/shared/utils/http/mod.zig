//! HTTP Utilities Module
//! Contains HTTP-related types, status codes, methods, and request/response handling

const std = @import("std");

// =============================================================================
// HTTP STATUS CODES
// =============================================================================

/// HTTP status codes as defined in RFC 7231
pub const HttpStatus = enum(u16) {
    // 2xx Success
    ok = 200,
    created = 201,
    accepted = 202,
    no_content = 204,

    // 3xx Redirection
    moved_permanently = 301,
    found = 302,
    not_modified = 304,

    // 4xx Client Error
    bad_request = 400,
    unauthorized = 401,
    forbidden = 403,
    not_found = 404,
    method_not_allowed = 405,
    conflict = 409,
    unprocessable_entity = 422,
    too_many_requests = 429,

    // 5xx Server Error
    internal_server_error = 500,
    not_implemented = 501,
    bad_gateway = 502,
    service_unavailable = 503,
    gateway_timeout = 504,

    /// Get the human-readable phrase for this status code
    pub fn phrase(self: HttpStatus) []const u8 {
        return switch (self) {
            .ok => "OK",
            .created => "Created",
            .accepted => "Accepted",
            .no_content => "No Content",
            .moved_permanently => "Moved Permanently",
            .found => "Found",
            .not_modified => "Not Modified",
            .bad_request => "Bad Request",
            .unauthorized => "Unauthorized",
            .forbidden => "Forbidden",
            .not_found => "Not Found",
            .method_not_allowed => "Method Not Allowed",
            .conflict => "Conflict",
            .unprocessable_entity => "Unprocessable Entity",
            .too_many_requests => "Too Many Requests",
            .internal_server_error => "Internal Server Error",
            .not_implemented => "Not Implemented",
            .bad_gateway => "Bad Gateway",
            .service_unavailable => "Service Unavailable",
            .gateway_timeout => "Gateway Timeout",
        };
    }

    /// Get the numeric code for this status
    pub fn code(self: HttpStatus) u16 {
        return @intFromEnum(self);
    }

    /// Check if this is a 2xx success status
    pub fn isSuccess(self: HttpStatus) bool {
        const c = @intFromEnum(self);
        return c >= 200 and c < 300;
    }

    /// Check if this is a 3xx redirection status
    pub fn isRedirect(self: HttpStatus) bool {
        const c = @intFromEnum(self);
        return c >= 300 and c < 400;
    }

    /// Check if this is a 4xx client error status
    pub fn isClientError(self: HttpStatus) bool {
        const c = @intFromEnum(self);
        return c >= 400 and c < 500;
    }

    /// Check if this is a 5xx server error status
    pub fn isServerError(self: HttpStatus) bool {
        const c = @intFromEnum(self);
        return c >= 500 and c < 600;
    }

    /// Check if this is any kind of error status (4xx or 5xx)
    pub fn isError(self: HttpStatus) bool {
        return self.isClientError() or self.isServerError();
    }
};

// =============================================================================
// HTTP METHODS
// =============================================================================

/// HTTP method types as defined in RFC 7231
pub const HttpMethod = enum {
    GET,
    POST,
    PUT,
    DELETE,
    PATCH,
    OPTIONS,
    HEAD,
    TRACE,
    CONNECT,

    /// Parse an HTTP method from a string (case-insensitive)
    pub fn fromString(method: []const u8) ?HttpMethod {
        if (std.ascii.eqlIgnoreCase(method, "GET")) return .GET;
        if (std.ascii.eqlIgnoreCase(method, "POST")) return .POST;
        if (std.ascii.eqlIgnoreCase(method, "PUT")) return .PUT;
        if (std.ascii.eqlIgnoreCase(method, "DELETE")) return .DELETE;
        if (std.ascii.eqlIgnoreCase(method, "PATCH")) return .PATCH;
        if (std.ascii.eqlIgnoreCase(method, "OPTIONS")) return .OPTIONS;
        if (std.ascii.eqlIgnoreCase(method, "HEAD")) return .HEAD;
        if (std.ascii.eqlIgnoreCase(method, "TRACE")) return .TRACE;
        if (std.ascii.eqlIgnoreCase(method, "CONNECT")) return .CONNECT;
        return null;
    }

    /// Convert HTTP method to string representation
    pub fn toString(self: HttpMethod) []const u8 {
        return switch (self) {
            .GET => "GET",
            .POST => "POST",
            .PUT => "PUT",
            .DELETE => "DELETE",
            .PATCH => "PATCH",
            .OPTIONS => "OPTIONS",
            .HEAD => "HEAD",
            .TRACE => "TRACE",
            .CONNECT => "CONNECT",
        };
    }

    /// Check if this method is considered "safe" (RFC 7231)
    /// Safe methods do not modify server state
    pub fn isSafe(self: HttpMethod) bool {
        return switch (self) {
            .GET, .HEAD, .OPTIONS, .TRACE => true,
            else => false,
        };
    }

    /// Check if this method is idempotent (RFC 7231)
    /// Idempotent methods can be called multiple times with the same effect
    pub fn isIdempotent(self: HttpMethod) bool {
        return switch (self) {
            .GET, .PUT, .DELETE, .HEAD, .OPTIONS, .TRACE => true,
            else => false,
        };
    }

    /// Check if this method typically allows a request body
    pub fn allowsBody(self: HttpMethod) bool {
        return switch (self) {
            .POST, .PUT, .PATCH, .DELETE => true,
            else => false,
        };
    }
};

// =============================================================================
// HTTP HEADERS
// =============================================================================

/// HTTP header management with case-insensitive operations
pub const Headers = struct {
    map: std.StringHashMap([]const u8),
    allocator: std.mem.Allocator,

    /// Initialize a new Headers instance
    pub fn init(allocator: std.mem.Allocator) Headers {
        return .{
            .map = std.StringHashMap([]const u8).init(allocator),
            .allocator = allocator,
        };
    }

    /// Clean up resources used by Headers
    pub fn deinit(self: *Headers) void {
        self.map.deinit();
    }

    /// Set a header value (overwrites existing values)
    pub fn set(self: *Headers, name: []const u8, value: []const u8) !void {
        try self.map.put(name, value);
    }

    /// Get a header value
    pub fn get(self: *Headers, name: []const u8) ?[]const u8 {
        return self.map.get(name);
    }

    /// Remove a header
    pub fn remove(self: *Headers, name: []const u8) bool {
        return self.map.remove(name);
    }

    /// Get a header value or return a default if not found
    pub fn getOr(self: *Headers, name: []const u8, default_value: []const u8) []const u8 {
        return self.get(name) orelse default_value;
    }
};

// =============================================================================
// HTTP REQUEST/RESPONSE STRUCTURES
// =============================================================================

/// HTTP request structure
pub const HttpRequest = struct {
    method: HttpMethod,
    path: []const u8,
    headers: Headers,
    body: ?[]const u8 = null,
    query_params: std.StringHashMap([]const u8),

    /// Initialize a new HTTP request
    pub fn init(allocator: std.mem.Allocator, method: HttpMethod, path: []const u8) HttpRequest {
        return .{
            .method = method,
            .path = path,
            .headers = Headers.init(allocator),
            .query_params = std.StringHashMap([]const u8).init(allocator),
        };
    }

    /// Clean up resources used by the request
    pub fn deinit(self: *HttpRequest) void {
        self.headers.deinit();
        self.query_params.deinit();
    }
};

/// HTTP response structure
pub const HttpResponse = struct {
    status: HttpStatus,
    headers: Headers,
    body: ?[]const u8 = null,

    /// Initialize a new HTTP response
    pub fn init(allocator: std.mem.Allocator, status: HttpStatus) HttpResponse {
        return .{
            .status = status,
            .headers = Headers.init(allocator),
        };
    }

    /// Clean up resources used by the response
    pub fn deinit(self: *HttpResponse) void {
        self.headers.deinit();
    }

    /// Set the Content-Type header
    pub fn setContentType(self: *HttpResponse, content_type: []const u8) !void {
        try self.headers.set("Content-Type", content_type);
    }

    /// Set Content-Type to application/json
    pub fn setJson(self: *HttpResponse) !void {
        try self.setContentType("application/json");
    }

    /// Set Content-Type to text/plain
    pub fn setText(self: *HttpResponse) !void {
        try self.setContentType("text/plain");
    }

    /// Set Content-Type to text/html
    pub fn setHtml(self: *HttpResponse) !void {
        try self.setContentType("text/html");
    }
};

// =============================================================================
// TESTS
// =============================================================================

test "HttpStatus phrase" {
    try std.testing.expectEqualStrings("OK", HttpStatus.ok.phrase());
    try std.testing.expectEqualStrings("Not Found", HttpStatus.not_found.phrase());
    try std.testing.expectEqualStrings("Internal Server Error", HttpStatus.internal_server_error.phrase());
}

test "HttpStatus classification" {
    try std.testing.expect(HttpStatus.ok.isSuccess());
    try std.testing.expect(HttpStatus.not_found.isClientError());
    try std.testing.expect(HttpStatus.internal_server_error.isServerError());
    try std.testing.expect(HttpStatus.not_found.isError());
}

test "HttpMethod fromString" {
    try std.testing.expectEqual(HttpMethod.GET, HttpMethod.fromString("GET").?);
    try std.testing.expectEqual(HttpMethod.POST, HttpMethod.fromString("post").?);
    try std.testing.expectEqual(@as(?HttpMethod, null), HttpMethod.fromString("INVALID"));
}

test "HttpMethod properties" {
    try std.testing.expect(HttpMethod.GET.isSafe());
    try std.testing.expect(HttpMethod.POST.isIdempotent() == false);
    try std.testing.expect(HttpMethod.POST.allowsBody());
}

test "Headers basic operations" {
    var headers = Headers.init(std.testing.allocator);
    defer headers.deinit();

    try headers.set("Content-Type", "application/json");
    try std.testing.expectEqualStrings("application/json", headers.get("Content-Type").?);

    try std.testing.expectEqualStrings("default", headers.getOr("Missing", "default"));
}
