//! ABI Framework — Default Entry Point (Template)
//!
//! This file exists as the default `zig init` entry point. The real CLI
//! entry point is `tools/cli/main.zig`. This file demonstrates basic
//! framework usage for contributors.

const std = @import("std");
const Io = std.Io;

const abi = @import("abi");

pub fn main(init: std.process.Init) !void {
    const arena: std.mem.Allocator = init.arena.allocator();
    const io = init.io;

    // Parse CLI arguments.
    const args = try init.minimal.args.toSlice(arena);
    for (args) |arg| {
        std.log.info("arg: {s}", .{arg});
    }

    // Write framework version to stdout.
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const stdout_writer = &stdout_file_writer.interface;

    try stdout_writer.print("ABI Framework v{s}\n", .{abi.version()});
    try stdout_writer.flush();
}

test "simple test" {
    const gpa = std.testing.allocator;
    var list: std.ArrayList(i32) = .empty;
    defer list.deinit(gpa);
    try list.append(gpa, 42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}
