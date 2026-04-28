const std = @import("std");

pub const Endpoint = struct {
    path: []const u8,
    method: std.http.Method,
    handler: *const fn (*anyopaque, *std.http.Server.Request, *anyopaque) anyerror!void,
};
