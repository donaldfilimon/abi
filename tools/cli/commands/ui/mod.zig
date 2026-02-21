//! UI command family (CLI/TUI v2).

const std = @import("std");
const utils = @import("../../utils/mod.zig");
const launch = @import("launch.zig");
const gpu = @import("gpu.zig");
const train = @import("train.zig");
const neural = @import("neural.zig");

pub fn run(allocator: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        try launch.run(allocator, io, args);
        return;
    }

    const sub = std.mem.sliceTo(args[0], 0);
    const rest = args[1..];

    if (utils.args.matchesAny(sub, &.{ "help", "--help", "-h" })) {
        printHelp();
        return;
    }

    if (std.mem.eql(u8, sub, "launch")) {
        try launch.run(allocator, io, rest);
        return;
    }

    if (std.mem.eql(u8, sub, "gpu")) {
        try gpu.run(allocator, io, rest);
        return;
    }

    if (std.mem.eql(u8, sub, "train")) {
        try train.run(allocator, io, rest);
        return;
    }

    if (std.mem.eql(u8, sub, "neural")) {
        try neural.run(allocator, io, rest);
        return;
    }

    std.debug.print("Unknown ui command: {s}\n", .{sub});
    printHelp();
}

pub fn printHelp() void {
    const help_text =
        \\Usage: abi ui <command> [options]
        \\
        \\Interactive UI command family.
        \\
        \\Commands:
        \\  launch               Open command launcher TUI
        \\  gpu                  Open GPU dashboard TUI
        \\  train [monitor args] Open training monitor TUI
        \\  neural [options]     Render dynamic 3D neural network view
        \\  help                 Show this help
        \\
        \\Examples:
        \\  abi ui launch
        \\  abi ui gpu
        \\  abi ui train --refresh-ms 250
        \\  abi ui neural --layers 12,24,24,12,4 --frames 0
        \\
    ;
    std.debug.print("{s}", .{help_text});
}
