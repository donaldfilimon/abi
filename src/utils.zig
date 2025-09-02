//! Utilities Module
//! This module contains common utilities, definitions, and types used across the project

const std = @import("std");

// =============================================================================
// COMMON DEFINITIONS
// =============================================================================

/// Project version information
pub const VERSION = .{
    .major = 1,
    .minor = 0,
    .patch = 0,
    .pre_release = "alpha",
};

/// Common configuration struct
pub const Config = struct {
    name: []const u8 = "abi-ai",
    version: u32 = 1,
    debug_mode: bool = false,

    pub fn init(name: []const u8) Config {
        return .{ .name = name };
    }
};

/// Definition types used throughout the project
pub const DefinitionType = enum {
    core,
    database,
    neural,
    web,
    cli,

    pub fn toString(self: DefinitionType) []const u8 {
        return switch (self) {
            .core => "core",
            .database => "database",
            .neural => "neural",
            .web => "web",
            .cli => "cli",
        };
    }
};

// =============================================================================
// WEB UTILITIES
// =============================================================================

/// HTTP status codes
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
};

/// HTTP method types
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
};

/// HTTP header management
pub const Headers = struct {
    map: std.StringHashMap([]const u8),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Headers {
        return .{
            .map = std.StringHashMap([]const u8).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Headers) void {
        self.map.deinit();
    }

    pub fn set(self: *Headers, name: []const u8, value: []const u8) !void {
        try self.map.put(name, value);
    }

    pub fn get(self: *Headers, name: []const u8) ?[]const u8 {
        return self.map.get(name);
    }

    pub fn remove(self: *Headers, name: []const u8) bool {
        return self.map.remove(name);
    }
};

/// HTTP request structure
pub const HttpRequest = struct {
    method: HttpMethod,
    path: []const u8,
    headers: Headers,
    body: ?[]const u8 = null,
    query_params: std.StringHashMap([]const u8),

    pub fn init(allocator: std.mem.Allocator, method: HttpMethod, path: []const u8) HttpRequest {
        return .{
            .method = method,
            .path = path,
            .headers = Headers.init(allocator),
            .query_params = std.StringHashMap([]const u8).init(allocator),
        };
    }

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

    pub fn init(allocator: std.mem.Allocator, status: HttpStatus) HttpResponse {
        return .{
            .status = status,
            .headers = Headers.init(allocator),
        };
    }

    pub fn deinit(self: *HttpResponse) void {
        self.headers.deinit();
    }

    pub fn setContentType(self: *HttpResponse, content_type: []const u8) !void {
        try self.headers.set("Content-Type", content_type);
    }

    pub fn setJson(self: *HttpResponse) !void {
        try self.setContentType("application/json");
    }

    pub fn setText(self: *HttpResponse) !void {
        try self.setContentType("text/plain");
    }

    pub fn setHtml(self: *HttpResponse) !void {
        try self.setContentType("text/html");
    }
};

// =============================================================================
// COMMON UTILITIES
// =============================================================================

/// String utilities
pub const StringUtils = struct {
    /// Check if string is empty or whitespace only
    pub fn isEmptyOrWhitespace(str: []const u8) bool {
        return std.mem.trim(u8, str, " \t\r\n").len == 0;
    }

    /// Convert string to lowercase (allocates)
    pub fn toLower(allocator: std.mem.Allocator, str: []const u8) ![]u8 {
        const result = try allocator.alloc(u8, str.len);
        for (str, 0..) |c, i| {
            result[i] = std.ascii.toLower(c);
        }
        return result;
    }

    /// Convert string to uppercase (allocates)
    pub fn toUpper(allocator: std.mem.Allocator, str: []const u8) ![]u8 {
        const result = try allocator.alloc(u8, str.len);
        for (str, 0..) |c, i| {
            result[i] = std.ascii.toUpper(c);
        }
        return result;
    }
};

/// Array utilities
pub const ArrayUtils = struct {
    /// Check if array contains element
    pub fn contains(comptime T: type, haystack: []const T, needle: T) bool {
        for (haystack) |item| {
            if (item == needle) return true;
        }
        return false;
    }

    /// Find index of element in array
    pub fn indexOf(comptime T: type, haystack: []const T, needle: T) ?usize {
        for (haystack, 0..) |item, i| {
            if (item == needle) return i;
        }
        return null;
    }
};

/// Time utilities
pub const TimeUtils = struct {
    /// Get current timestamp in milliseconds
    pub fn nowMs() i64 {
        return std.time.milliTimestamp();
    }

    /// Get current timestamp in microseconds
    pub fn nowUs() i64 {
        return std.time.microTimestamp();
    }

    /// Get current timestamp in nanoseconds
    pub fn nowNs() i64 {
        return std.time.nanoTimestamp();
    }

    /// Format duration in human readable format
    pub fn formatDuration(allocator: std.mem.Allocator, duration_ns: u64) ![]u8 {
        const ms = duration_ns / std.time.ns_per_ms;
        const us = (duration_ns % std.time.ns_per_ms) / std.time.ns_per_us;
        const ns = duration_ns % std.time.ns_per_us;

        if (ms > 0) {
            return std.fmt.allocPrint(allocator, "{d}.{d:0>3}ms", .{ ms, us });
        } else if (us > 0) {
            return std.fmt.allocPrint(allocator, "{d}.{d:0>3}μs", .{ us, ns / 1000 });
        } else {
            return std.fmt.allocPrint(allocator, "{d}ns", .{ns});
        }
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

test "HttpMethod fromString" {
    try std.testing.expectEqual(HttpMethod.GET, HttpMethod.fromString("GET").?);
    try std.testing.expectEqual(HttpMethod.POST, HttpMethod.fromString("post").?);
    try std.testing.expectEqual(@as(?HttpMethod, null), HttpMethod.fromString("INVALID"));
}

test "StringUtils" {
    try std.testing.expect(StringUtils.isEmptyOrWhitespace(""));
    try std.testing.expect(StringUtils.isEmptyOrWhitespace("   "));
    try std.testing.expect(!StringUtils.isEmptyOrWhitespace("hello"));
}

test "ArrayUtils" {
    const arr = [_]i32{ 1, 2, 3, 4, 5 };
    try std.testing.expect(ArrayUtils.contains(i32, &arr, 3));
    try std.testing.expect(!ArrayUtils.contains(i32, &arr, 6));
    try std.testing.expectEqual(@as(?usize, 2), ArrayUtils.indexOf(i32, &arr, 3));
    try std.testing.expectEqual(@as(?usize, null), ArrayUtils.indexOf(i32, &arr, 6));
}
