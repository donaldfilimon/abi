//! UI command family (CLI/TUI v2).

const std = @import("std");
const command_mod = @import("../../command.zig");
const context_mod = @import("../../framework/context.zig");
const utils = @import("../../utils/mod.zig");
const launch = @import("launch.zig");
const gpu_cmd = @import("gpu.zig");
const train_cmd = @import("train.zig");
const neural = @import("neural.zig");

const ui_subcommands = [_][]const u8{
    "launch", "gpu", "train", "neural", "help",
};

pub const meta: command_mod.Meta = .{
    .name = "ui",
    .description = "UI command family (launch, gpu, train, neural)",
    .subcommands = &.{ "launch", "gpu", "train", "neural", "help" },
    .kind = .group,
    .children = &.{
        .{ .name = "launch", .description = "Open command launcher TUI", .handler = launch.run },
        .{ .name = "gpu", .description = "Open GPU dashboard TUI", .handler = gpu_cmd.run },
        .{ .name = "train", .description = "Open training monitor TUI", .handler = train_cmd.run },
        .{ .name = "neural", .description = "Render dynamic 3D neural network view", .handler = neural.run },
    },
};

/// Run the UI command with the provided arguments.
/// Known subcommands (launch, gpu, train, neural) are dispatched
/// by the framework router via meta.children. This function handles:
/// - No args: default to launch
/// - Unknown subcommands: print error + help
/// - Explicit help request
pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        try launch.run(ctx, args);
        return;
    }
    const sub = std.mem.sliceTo(args[0], 0);
    if (utils.args.matchesAny(sub, &.{ "help", "--help", "-h" })) {
        printHelp();
        return;
    }
    utils.output.printError("Unknown ui subcommand: {s}", .{sub});
    if (utils.args.suggestCommand(sub, &ui_subcommands)) |suggestion| {
        utils.output.printInfo("Did you mean: {s}", .{suggestion});
    }
    utils.output.printInfo("Run 'abi ui help' for usage.", .{});
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
