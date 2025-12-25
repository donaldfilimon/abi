const std = @import("std");

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
    if (std.ascii.eqlIgnoreCase(text, "GET")) return .get;
    if (std.ascii.eqlIgnoreCase(text, "POST")) return .post;
    if (std.ascii.eqlIgnoreCase(text, "PUT")) return .put;
    if (std.ascii.eqlIgnoreCase(text, "DELETE")) return .delete;
    if (std.ascii.eqlIgnoreCase(text, "PATCH")) return .patch;
    if (std.ascii.eqlIgnoreCase(text, "HEAD")) return .head;
    if (std.ascii.eqlIgnoreCase(text, "OPTIONS")) return .options;
    return null;
}

pub fn statusText(code: u16) []const u8 {
    return switch (code) {
        200 => "OK",
        201 => "Created",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        409 => "Conflict",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        else => "Unknown",
    };
}

pub fn isSuccess(code: u16) bool {
    return code >= 200 and code < 300;
}
