//! Shared action execution for launcher-style UIs.

const std = @import("std");
const abi = @import("abi");
const context_mod = @import("../../framework/context.zig");
const framework_mod = @import("../../framework/mod.zig");
const tui = @import("../mod.zig");
const utils = @import("../../utils/mod.zig");
const commands_mod = @import("../../commands/mod.zig");
const types = @import("types.zig");

const Action = types.Action;
const CommandRef = types.CommandRef;

pub const ExecuteOptions = struct {
    help_callback: ?*const fn () void = null,
    return_prompt: []const u8 = "Press Enter to return...",
};

pub fn executeWithTerminal(
    allocator: std.mem.Allocator,
    terminal: *tui.Terminal,
    action: Action,
    options: ExecuteOptions,
) !bool {
    switch (action) {
        .quit => return true,
        else => {},
    }

    try terminal.exit();
    errdefer terminal.enter() catch {};

    runAction(allocator, action, options.help_callback) catch {};

    utils.output.printInfo("\n{s}", .{options.return_prompt});
    _ = terminal.readKey() catch {};
    try terminal.enter();
    return false;
}

pub fn runAction(
    allocator: std.mem.Allocator,
    action: Action,
    help_callback: ?*const fn () void,
) !void {
    switch (action) {
        .command => |cmd| runCommand(allocator, cmd) catch |err| {
            utils.output.printError("Command '{s}' failed: {t}", .{ commandLabel(cmd), err });
            return err;
        },
        .version => utils.output.printInfo("ABI Framework v{s}", .{abi.version()}),
        .help => {
            if (help_callback) |callback| {
                callback();
            } else {
                utils.output.printInfo("Run 'abi help' or 'abi ui help' for available commands.", .{});
            }
        },
        .quit => {},
    }
}

pub fn commandLabel(cmd: CommandRef) []const u8 {
    if (std.mem.eql(u8, cmd.command, "train") and cmd.args.len > 0) {
        const first = std.mem.sliceTo(cmd.args[0], 0);
        if (std.mem.eql(u8, first, "monitor")) return "train monitor";
    }
    return cmd.id;
}

pub fn runCommand(allocator: std.mem.Allocator, cmd: CommandRef) !void {
    var io_backend = utils.io_backend.initIoBackend(allocator);
    defer io_backend.deinit();

    const cmd_ctx = context_mod.CommandContext{
        .allocator = allocator,
        .io = io_backend.io(),
    };

    const command_name = framework_mod.completion.resolveAlias(&commands_mod.descriptors, cmd.command);
    const matched = try framework_mod.router.runCommand(cmd_ctx, &commands_mod.descriptors, command_name, cmd.args);
    if (!matched) {
        return framework_mod.errors.Error.UnknownCommand;
    }
}

test {
    std.testing.refAllDecls(@This());
}
