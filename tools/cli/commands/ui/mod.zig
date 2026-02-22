//! UI command family (CLI/TUI v2).

const std = @import("std");
const command_mod = @import("../../command.zig");
const utils = @import("../../utils/mod.zig");
const launch = @import("launch.zig");
const gpu_cmd = @import("gpu.zig");
const train_cmd = @import("train.zig");
const neural = @import("neural.zig");

pub const meta: command_mod.Meta = .{
    .name = "ui",
    .description = "UI command family (launch, gpu, train, neural)",
    .subcommands = &.{ "launch", "gpu", "train", "neural", "help" },
    .io_mode = .io,
    .kind = .group,
    .children = &.{
        .{ .name = "launch", .description = "Open command launcher TUI", .handler = .{ .io = launch.run } },
        .{ .name = "gpu", .description = "Open GPU dashboard TUI", .handler = .{ .io = gpu_cmd.run } },
        .{ .name = "train", .description = "Open training monitor TUI", .handler = .{ .io = train_cmd.run } },
        .{ .name = "neural", .description = "Render dynamic 3D neural network view", .handler = .{ .io = neural.run } },
    },
};

/// Run the UI command with the provided arguments.
/// Known subcommands (launch, gpu, train, neural) are dispatched
/// by the framework router via meta.children. This function handles:
/// - No args: default to launch
/// - Unknown subcommands: print error + help
/// - Explicit help request
pub fn run(allocator: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        try launch.run(allocator, io, args);
        return;
    }
    const sub = std.mem.sliceTo(args[0], 0);
    if (utils.args.matchesAny(sub, &.{ "help", "--help", "-h" })) {
        printHelp();
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
