//! UI command family (CLI/TUI v2).

const std = @import("std");
const command_mod = @import("../../command.zig");
const context_mod = @import("../../framework/context.zig");
const utils = @import("../../utils/mod.zig");
const launch = @import("launch.zig");
const gpu_cmd = @import("gpu.zig");
const train_cmd = @import("train.zig");
const neural = @import("neural.zig");
const model_cmd = @import("model.zig");
const streaming_cmd = @import("streaming.zig");
const db_cmd = @import("db.zig");
const network_cmd = @import("network.zig");
const bench_cmd = @import("bench.zig");
const brain_cmd = @import("brain.zig");
const dashboard_cmd = @import("dashboard.zig");

const ui_subcommands = [_][]const u8{
    "launch", "gpu", "train", "neural", "model", "streaming", "db", "network", "bench", "brain", "dashboard", "help",
};

pub const meta: command_mod.Meta = .{
    .name = "ui",
    .description = "UI command family (launch, gpu, train, model, streaming, db, network, bench, brain, dashboard)",
    .aliases = &.{ "launch", "start" },
    .subcommands = &.{ "launch", "gpu", "train", "neural", "model", "streaming", "db", "network", "bench", "brain", "dashboard", "help" },
    .kind = .group,
    .children = &.{
        .{ .name = "launch", .description = "Open command launcher TUI", .handler = launch.run },
        .{ .name = "gpu", .description = "Open GPU dashboard TUI", .handler = gpu_cmd.run },
        .{ .name = "train", .description = "Open training monitor TUI", .handler = train_cmd.run },
        .{ .name = "neural", .description = "Render dynamic 3D neural network view", .handler = neural.run },
        .{ .name = "model", .description = "Open model management dashboard", .handler = model_cmd.run },
        .{ .name = "streaming", .description = "Open streaming inference dashboard", .handler = streaming_cmd.run },
        .{ .name = "db", .description = "Open database monitoring dashboard", .handler = db_cmd.run },
        .{ .name = "network", .description = "Open network status dashboard", .handler = network_cmd.run },
        .{ .name = "bench", .description = "Open benchmark results dashboard", .handler = bench_cmd.run },
        .{ .name = "brain", .description = "Open brain visualization dashboard", .handler = brain_cmd.run },
        .{ .name = "dashboard", .description = "Open unified tabbed dashboard", .handler = dashboard_cmd.run },
    },
};

/// Run the UI command with the provided arguments.
/// Known subcommands are dispatched by the framework router via meta.children.
/// This function handles: no args → launch, unknown subcommands, help.
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
    // Allow `abi ui --theme ...` style launcher options.
    if (sub.len > 0 and sub[0] == '-') {
        try launch.run(ctx, args);
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
        \\  model                Open model management dashboard
        \\  streaming [endpoint] Open streaming inference dashboard
        \\  db                   Open database monitoring dashboard
        \\  network              Open network status dashboard
        \\  bench                Open benchmark results dashboard
        \\  brain                Open brain visualization dashboard
        \\  dashboard            Open unified tabbed dashboard
        \\  help                 Show this help
        \\
        \\Examples:
        \\  abi ui launch
        \\  abi launch
        \\  abi start
        \\  abi ui gpu
        \\  abi ui model
        \\  abi ui streaming http://localhost:8080
        \\  abi ui db
        \\  abi ui network
        \\  abi ui bench
        \\  abi ui brain
        \\  abi ui dashboard
        \\  abi ui dashboard --theme nord
        \\  abi ui train --refresh-ms 250
        \\  abi ui neural --layers 12,24,24,12,4 --frames 0
        \\
    ;
    utils.output.print("{s}", .{help_text});
}

test {
    std.testing.refAllDecls(@This());
}
