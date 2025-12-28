//! HTTP utility functions for method parsing, status codes, and response handling.
//!
//! Provides helpers for working with HTTP including method parsing, status code
//! text lookups, success status detection, and error handling.

const std = @import("std");

pub const HttpError = error{
    InvalidUrl,
    InvalidRequest,
    RequestFailed,
    ConnectionFailed,
    ResponseTooLarge,
    Timeout,
    ReadFailed,
    InvalidResponse,
    RedirectExceeded,
};

pub const Method = enum {
    get,
    post,
    put,
    delete,
    patch,
    head,
    options,
};

pub const Status = struct {
    code: u16,
    reason: []const u8,
};

pub const Request = struct {
    method: Method,
    path: []const u8,
};

pub const Response = struct {
    status: Status,
    body: []const u8,
};

pub fn parseMethod(text: []const u8) ?Method {
    comptime std.debug.assert(@typeInfo(Method) == .Enum);
    comptime std.debug.assert(std.enums.values(Method).len == 7); // All HTTP methods covered

    if (std.ascii.eqlIgnoreCase(text, "GET")) return .get;
    if (std.ascii.eqlIgnoreCase(text, "POST")) return .post;
    if (std.ascii.eqlIgnoreCase(text, "PUT")) return .put;
    if (std.ascii.eqlIgnoreCase(text, "DELETE")) return .delete;
    if (std.ascii.eqlIgnoreCase(text, "PATCH")) return .patch;
    if (std.ascii.eqlIgnoreCase(text, "HEAD")) return .head;
    if (std.ascii.eqlIgnoreCase(text, "OPTIONS")) return .options;
    return null;
}

const HTTP_STATUS_OK = 200;
const HTTP_STATUS_CREATED = 201;
const HTTP_STATUS_NO_CONTENT = 204;
const HTTP_STATUS_BAD_REQUEST = 400;
const HTTP_STATUS_UNAUTHORIZED = 401;
const HTTP_STATUS_FORBIDDEN = 403;
const HTTP_STATUS_NOT_FOUND = 404;
const HTTP_STATUS_CONFLICT = 409;
const HTTP_STATUS_TOO_MANY_REQUESTS = 429;
const HTTP_STATUS_INTERNAL_SERVER_ERROR = 500;
const HTTP_STATUS_NOT_IMPLEMENTED = 501;
const HTTP_STATUS_SERVICE_UNAVAILABLE = 503;

const HTTP_STATUS_SUCCESS_MIN: u16 = 200;
const HTTP_STATUS_SUCCESS_MAX: u16 = 299;

pub fn statusText(code: u16) []const u8 {
    return switch (code) {
        HTTP_STATUS_OK => "OK",
        HTTP_STATUS_CREATED => "Created",
        HTTP_STATUS_NO_CONTENT => "No Content",
        HTTP_STATUS_BAD_REQUEST => "Bad Request",
        HTTP_STATUS_UNAUTHORIZED => "Unauthorized",
        HTTP_STATUS_FORBIDDEN => "Forbidden",
        HTTP_STATUS_NOT_FOUND => "Not Found",
        HTTP_STATUS_CONFLICT => "Conflict",
        HTTP_STATUS_TOO_MANY_REQUESTS => "Too Many Requests",
        HTTP_STATUS_INTERNAL_SERVER_ERROR => "Internal Server Error",
        HTTP_STATUS_NOT_IMPLEMENTED => "Not Implemented",
        HTTP_STATUS_SERVICE_UNAVAILABLE => "Service Unavailable",
        else => "Unknown",
    };
}

pub fn isSuccess(code: u16) bool {
    return code >= HTTP_STATUS_SUCCESS_MIN and code <= HTTP_STATUS_SUCCESS_MAX;
}

test "http helpers" {
    try std.testing.expectEqual(@as(?Method, .get), parseMethod("GET"));
    try std.testing.expectEqual(@as(?Method, .post), parseMethod("post"));
    try std.testing.expect(parseMethod("unknown") == null);

    try std.testing.expectEqualStrings("Not Found", statusText(404));
    try std.testing.expect(isSuccess(200));
    try std.testing.expect(!isSuccess(404));
}
