const std = @import("std");

pub const CommandIoMode = enum {
    basic,
    io,
};

pub const CommandForward = struct {
    target: []const u8,
    prepend_args: []const [:0]const u8 = &.{},
    warning: ?[]const u8 = null,
};

pub const CommandHandler = union(CommandIoMode) {
    basic: *const fn (std.mem.Allocator, []const [:0]const u8) anyerror!void,
    io: *const fn (std.mem.Allocator, std.Io, []const [:0]const u8) anyerror!void,
};

pub const CommandDescriptor = struct {
    name: []const u8,
    description: []const u8,
    aliases: []const []const u8 = &.{},
    subcommands: []const []const u8 = &.{},
    handler: CommandHandler,
    forward: ?CommandForward = null,
};

pub fn isHelpToken(token: []const u8) bool {
    return std.mem.eql(u8, token, "help") or
        std.mem.eql(u8, token, "--help") or
        std.mem.eql(u8, token, "-h");
}
