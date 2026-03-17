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

pub const Visibility = enum {
    public,
    hidden,
};

pub const RiskLevel = enum {
    safe,
    caution,
    destructive,
};

pub const UiCategory = enum {
    ai,
    data,
    system,
    tools,
    meta,
};

pub const OptionInfo = struct {
    flag: []const u8,
    description: []const u8 = "",
};

pub const UiMeta = struct {
    include_in_launcher: bool = true,
    include_in_dashboard: bool = false,
    label: ?[]const u8 = null,
    category: ?UiCategory = null,
    shortcut: ?u8 = null,
    usage: ?[]const u8 = null,
    examples: []const []const u8 = &.{},
    related: []const []const u8 = &.{},
    risk_badge: ?[]const u8 = null,
};

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
    options: []const OptionInfo = &.{},
    ui: UiMeta = .{},
    visibility: Visibility = .public,
    risk: RiskLevel = .safe,
    source_id: ?[]const u8 = null,
    default_subcommand: ?[:0]const u8 = null,
    middleware_tags: []const []const u8 = &.{},
};

pub fn isHelpToken(token: []const u8) bool {
    return std.mem.eql(u8, token, "help") or
        std.mem.eql(u8, token, "--help") or
        std.mem.eql(u8, token, "-h");
}

test {
    std.testing.refAllDecls(@This());
}
