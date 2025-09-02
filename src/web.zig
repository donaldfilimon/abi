//! Web module
//! This module contains web-related definitions and utilities

const std = @import("std");

/// Placeholder for web definitions
pub const PLACEHOLDER = true;

/// HTTP status codes
pub const StatusCode = enum(u16) {
    ok = 200,
    not_found = 404,
    internal_server_error = 500,
};

/// HTTP method types
pub const Method = enum {
    GET,
    POST,
    PUT,
    DELETE,
    PATCH,
    OPTIONS,
    HEAD,
};

/// Basic HTTP request structure
pub const Request = struct {
    method: Method,
    path: []const u8,
    headers: std.StringHashMap([]const u8),
    body: ?[]const u8 = null,
};

/// Basic HTTP response structure
pub const Response = struct {
    status: StatusCode,
    headers: std.StringHashMap([]const u8),
    body: ?[]const u8 = null,
};
