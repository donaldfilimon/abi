const std = @import("std");

const context_mod = @import("context.zig");

pub const CommandKind = enum {
    /// Action command that can execute directly.
    action,
    /// Namespace/group command whose primary behavior is child dispatch.
    group,
};

pub const CommandForward = struct {
    target: []const u8,
    prepend_args: []const [:0]const u8 = &.{},
    warning: ?[]const u8 = null,
};

pub const CommandHandler = *const fn (ctx: *const context_mod.CommandContext, args: []const [:0]const u8) anyerror!void;

pub const CommandDescriptor = struct {
    name: []const u8,
    description: []const u8,
    aliases: []const []const u8 = &.{},
    subcommands: []const []const u8 = &.{},
    children: []const CommandDescriptor = &.{},
    kind: CommandKind = .action,
    help_text: ?[]const u8 = null,
    handler: CommandHandler,
    forward: ?CommandForward = null,
};

pub fn isHelpToken(token: []const u8) bool {
    return std.mem.eql(u8, token, "help") or
        std.mem.eql(u8, token, "--help") or
        std.mem.eql(u8, token, "-h");
}

test {
    std.testing.refAllDecls(@This());
}
