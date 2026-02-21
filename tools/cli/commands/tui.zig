//! Compatibility shim for the v2 TUI module split.

const std = @import("std");
const v2 = @import("tui/mod.zig");

pub fn run(allocator: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    try v2.run(allocator, io, args);
}

pub fn printHelp() void {
    v2.printHelp();
}
