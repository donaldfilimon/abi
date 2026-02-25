//! Command catalog — backward-compatible view of command metadata.
//!
//! Now derived from `commands/mod.zig:descriptors` (the single source of truth).
//! Subcommand arrays, children, and all other metadata live in each command
//! module's `pub const meta: command_mod.Meta`.

const std = @import("std");
const commands_mod = @import("../commands/mod.zig");

pub const ChildSpec = struct {
    name: []const u8,
    description: []const u8,
    aliases: []const []const u8 = &.{},
};

pub const CommandSpec = struct {
    name: []const u8,
    description: []const u8,
    aliases: []const []const u8 = &.{},
    /// Structural subcommands (command tree nodes only, no option/value tokens).
    subcommands: []const []const u8 = &.{},
    /// Optional command-specific completion tokens.
    completion_tokens: []const []const u8 = &.{},
};

/// Commands array derived from the comptime-generated descriptors.
pub const commands: [commands_mod.descriptors.len]CommandSpec = blk: {
    var result: [commands_mod.descriptors.len]CommandSpec = undefined;
    for (commands_mod.descriptors, 0..) |desc, i| {
        result[i] = .{
            .name = desc.name,
            .description = desc.description,
            .aliases = desc.aliases,
            .subcommands = desc.subcommands,
        };
    }
    break :blk result;
};

pub fn findCommandByName(raw_name: []const u8) ?CommandSpec {
    for (commands) |spec| {
        if (std.mem.eql(u8, raw_name, spec.name)) return spec;
        for (spec.aliases) |alias| {
            if (std.mem.eql(u8, raw_name, alias)) return spec;
        }
    }
    return null;
}

pub fn findSubcommands(raw_name: []const u8) ?[]const []const u8 {
    const spec = findCommandByName(raw_name) orelse return null;
    return spec.subcommands;
}

test "catalog command names unique" {
    for (commands, 0..) |lhs, i| {
        var idx: usize = 0;
        while (idx < i) : (idx += 1) {
            try std.testing.expect(!std.mem.eql(u8, lhs.name, commands[idx].name));
        }
    }
}

test {
    std.testing.refAllDecls(@This());
}
