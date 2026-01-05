//! Stub for Monitoring feature when disabled
const std = @import("std");

pub fn init(allocator: std.mem.Allocator) !void {
    _ = allocator;
}

pub fn deinit() void {}
