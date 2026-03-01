//! Hello World Example
//!
//! Minimal example showing framework initialization.
//! Use this as a starting point for new ABI projects.
//!
//! Run with: `zig build run-hello`

const std = @import("std");
const abi = @import("abi");

pub fn main(_: std.process.Init) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var framework = try abi.App.initMinimal(allocator);
    defer framework.deinit();

    std.debug.print("ABI Framework v{s}\n", .{abi.version()});
    std.debug.print("Framework initialized successfully\n", .{});
}
