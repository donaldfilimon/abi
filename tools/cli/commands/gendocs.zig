//! CLI command: abi gendocs
//!
//! Runs the API documentation generator (zig build gendocs). Generates
//! docs/api/*.md from src/abi.zig module discovery and Zig doc comments.

const std = @import("std");
const utils = @import("../utils/mod.zig");
const cli_io = utils.io_backend;

const c = @cImport({
    @cInclude("stdlib.h");
});

pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (utils.args.containsHelpArgs(args)) {
        printHelp(allocator);
        return;
    }

    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    var f = std.Io.Dir.cwd().openFile(io, "build.zig", .{}) catch {
        std.debug.print("Error: build.zig not found in current directory. Run from repo root.\n", .{});
        std.process.exit(1);
    };
    defer f.close(io);

    var cmd_buf = std.ArrayListUnmanaged(u8).empty;
    defer cmd_buf.deinit(allocator);
    try cmd_buf.appendSlice(allocator, "zig build gendocs");
    for (args) |arg| {
        const a = std.mem.sliceTo(arg, 0);
        try cmd_buf.append(allocator, ' ');
        if (std.mem.indexOf(u8, a, " ") != null) {
            try cmd_buf.append(allocator, '"');
            try cmd_buf.appendSlice(allocator, a);
            try cmd_buf.append(allocator, '"');
        } else {
            try cmd_buf.appendSlice(allocator, a);
        }
    }
    try cmd_buf.append(allocator, 0);

    const ret = c.system(@ptrCast(cmd_buf.items.ptr));
    if (ret != 0) std.process.exit(1);
}

fn printHelp(allocator: std.mem.Allocator) void {
    _ = allocator;
    std.debug.print(
        \\Usage: abi gendocs [-- zig-build-args]
        \\
        \\Generate API documentation from source (runs zig build gendocs).
        \\
        \\  Run from repo root. Creates docs/api/index.md and docs/api/<module>.md
        \\  from src/abi.zig and Zig doc comments (/// and //!).
        \\
        \\  Optional: pass extra args to the build step after --, e.g.:
        \\    abi gendocs -- -Doptimize=ReleaseSafe
        \\
        \\See also: zig build gendocs
        \\
    , .{});
}
