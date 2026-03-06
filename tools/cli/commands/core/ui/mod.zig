//! UI command family (CLI/TUI v2).

const std = @import("std");
const command_mod = @import("../../../command.zig");
const context_mod = @import("../../../framework/context.zig");
const utils = @import("../../../utils/mod.zig");
const gpu_cmd = @import("./gpu.zig");
const train_cmd = @import("./train.zig");
const neural = @import("./neural.zig");
const model_cmd = @import("./model.zig");
const streaming_cmd = @import("./streaming.zig");
const db_cmd = @import("./db.zig");
const network_cmd = @import("./network.zig");
const bench_cmd = @import("./bench.zig");
const brain_cmd = @import("./brain.zig");
const dashboard_cmd = @import("./dashboard.zig");
const editor_cmd = @import("./editor.zig");
const chat_cmd = @import("./chat.zig");

pub const meta: command_mod.Meta = .{
    .name = "ui",
    .description = "Shared UI shell and focused terminal views (editor, gpu, train, model, streaming, db, network, bench, brain, chat)",
    .aliases = &.{},
    .subcommands = &.{ "gpu", "train", "neural", "model", "streaming", "db", "network", "bench", "brain", "chat", "editor", "help" },
    .kind = .group,
    .children = &.{
        .{ .name = "gpu", .description = "Open GPU dashboard TUI", .handler = gpu_cmd.run },
        .{ .name = "train", .description = "Open training monitor TUI", .handler = train_cmd.run },
        .{ .name = "neural", .description = "Render dynamic 3D neural network view", .handler = neural.run },
        .{ .name = "model", .description = "Open model management dashboard", .handler = model_cmd.run },
        .{ .name = "streaming", .description = "Open streaming inference dashboard", .handler = streaming_cmd.run },
        .{ .name = "db", .description = "Open database monitoring dashboard", .handler = db_cmd.run },
        .{ .name = "network", .description = "Open network status dashboard", .handler = network_cmd.run },
        .{ .name = "bench", .description = "Open benchmark results dashboard", .handler = bench_cmd.run },
        .{ .name = "brain", .description = "Open brain visualization dashboard", .handler = brain_cmd.run },
        .{ .name = "chat", .description = "Open multi-persona chat dashboard", .handler = chat_cmd.run },
        .{ .name = "editor", .description = "Open an inline terminal text editor", .handler = editor_cmd.run },
    },
};

/// Run the UI command with the provided arguments.
/// Known subcommands are dispatched by the framework router via meta.children.
/// This function handles: no args → shared shell, removed legacy modes,
/// unknown subcommands, and help.
pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        try dashboard_cmd.run(ctx, args);
        return;
    }
    const sub = std.mem.sliceTo(args[0], 0);
    if (utils.args.matchesAny(sub, &.{ "help", "--help", "-h" })) {
        printHelp();
        return;
    }
    if (isRemovedLegacyMode(sub)) {
        if (std.mem.eql(u8, sub, "launch")) {
            utils.output.printError("'abi ui launch' was removed. Use 'abi ui' for the shared shell.", .{});
        } else {
            utils.output.printError("'abi ui dashboard' was removed. Use 'abi ui' for the shared shell.", .{});
        }
        return error.InvalidArgument;
    }
    // Allow `abi ui --theme ...` style shared-shell options.
    if (sub.len > 0 and sub[0] == '-') {
        try dashboard_cmd.run(ctx, args);
        return;
    }
    utils.output.printError("Unknown ui subcommand: {s}", .{sub});
    if (command_mod.suggestSubcommand(meta, sub)) |suggestion| {
        utils.output.printInfo("Did you mean: {s}", .{suggestion});
    }
    utils.output.printInfo("Run 'abi ui help' for usage.", .{});
    printHelp();
}

fn isRemovedLegacyMode(sub: []const u8) bool {
    return std.mem.eql(u8, sub, "launch") or std.mem.eql(u8, sub, "dashboard");
}

pub fn printHelp() void {
    const help_text =
        \\Usage: abi ui [options]
        \\       abi ui <view> [options]
        \\
        \\Shared UI shell with focused terminal views.
        \\
        \\Views:
        \\  gpu                  Open GPU dashboard TUI
        \\  train [monitor args] Open training monitor TUI
        \\  neural [options]     Render dynamic 3D neural network view
        \\  model                Open model management dashboard
        \\  streaming [endpoint] Open streaming inference dashboard
        \\  db                   Open database monitoring dashboard
        \\  network              Open network status dashboard
        \\  bench                Open benchmark results dashboard
        \\  brain                Open brain visualization dashboard
        \\  chat                 Open multi-persona chat dashboard
        \\  editor [file]        Open the inline terminal editor
        \\  help                 Show this help
        \\
        \\Default:
        \\  abi ui               Open the shared tabbed shell
        \\
        \\Examples:
        \\  abi ui
        \\  abi ui gpu
        \\  abi ui model
        \\  abi ui streaming http://localhost:8080
        \\  abi ui db
        \\  abi ui network
        \\  abi ui bench
        \\  abi ui brain
        \\  abi ui chat
        \\  abi ui --theme nord
        \\  abi ui editor build.zig
        \\  abi ui train --refresh-ms 250
        \\  abi ui neural --layers 12,24,24,12,4 --frames 0
        \\
    ;
    utils.output.print("{s}", .{help_text});
}

test "ui meta exposes every public child command" {
    const expected = [_][]const u8{
        "gpu",
        "train",
        "neural",
        "model",
        "streaming",
        "db",
        "network",
        "bench",
        "brain",
        "chat",
        "editor",
    };

    try std.testing.expectEqual(expected.len, meta.children.len);
    try std.testing.expectEqual(expected.len + 1, meta.subcommands.len);

    for (expected, 0..) |name, index| {
        try std.testing.expectEqualStrings(name, meta.children[index].name);
        try std.testing.expectEqualStrings(name, meta.subcommands[index]);
    }
    try std.testing.expectEqualStrings("help", meta.subcommands[meta.subcommands.len - 1]);
}

test "removed ui legacy modes stay rejected" {
    try std.testing.expect(isRemovedLegacyMode("launch"));
    try std.testing.expect(isRemovedLegacyMode("dashboard"));
    try std.testing.expect(!isRemovedLegacyMode("gpu"));
}

test {
    std.testing.refAllDecls(@This());
}
