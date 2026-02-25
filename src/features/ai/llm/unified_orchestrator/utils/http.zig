//! HTTP utilities stub.

const std = @import("std");

pub fn get(allocator: std.mem.Allocator, url: []const u8) ![]const u8 {
    _ = allocator;
    _ = url;
    return "";
}
