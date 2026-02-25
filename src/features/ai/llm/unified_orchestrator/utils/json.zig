//! JSON utilities stub.

const std = @import("std");

pub fn parse(allocator: std.mem.Allocator, input: []const u8) !void {
    _ = allocator;
    _ = input;
}

pub fn stringify(allocator: std.mem.Allocator) ![]const u8 {
    return allocator.dupe(u8, "{}");
}
