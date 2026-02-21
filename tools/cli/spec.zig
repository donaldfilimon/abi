//! CLI command and completion metadata derived from command descriptors.

const std = @import("std");
const commands = @import("commands/mod.zig");

pub const CommandInfo = struct {
    name: []const u8,
    description: []const u8,
};

pub const AliasInfo = struct {
    alias: []const u8,
    target: []const u8,
};

pub const CommandSubcommands = struct {
    command: []const u8,
    subcommands: []const []const u8,
};

const descriptors = commands.descriptors;

const command_infos_array: [descriptors.len + 2]CommandInfo = blk: {
    var infos: [descriptors.len + 2]CommandInfo = undefined;
    var index: usize = 0;

    for (descriptors) |descriptor| {
        infos[index] = .{
            .name = descriptor.name,
            .description = descriptor.description,
        };
        index += 1;
    }

    infos[index] = .{ .name = "version", .description = "Show framework version" };
    index += 1;
    infos[index] = .{ .name = "help", .description = "Show help (use: abi help <command>)" };

    break :blk infos;
};

pub const command_infos = command_infos_array;

const alias_count: usize = blk: {
    var count: usize = 0;
    for (descriptors) |descriptor| {
        count += descriptor.aliases.len;
    }
    break :blk count;
};

const aliases_array: [alias_count]AliasInfo = blk: {
    var out: [alias_count]AliasInfo = undefined;
    var index: usize = 0;

    for (descriptors) |descriptor| {
        for (descriptor.aliases) |alias| {
            out[index] = .{
                .alias = alias,
                .target = descriptor.name,
            };
            index += 1;
        }
    }

    break :blk out;
};

pub const aliases = aliases_array;

const subcommand_count: usize = blk: {
    var count: usize = 0;
    for (descriptors) |descriptor| {
        if (descriptor.subcommands.len > 0) count += 1;
    }
    break :blk count;
};

const command_subcommands_array: [subcommand_count]CommandSubcommands = blk: {
    var out: [subcommand_count]CommandSubcommands = undefined;
    var index: usize = 0;

    for (descriptors) |descriptor| {
        if (descriptor.subcommands.len > 0) {
            out[index] = .{
                .command = descriptor.name,
                .subcommands = descriptor.subcommands,
            };
            index += 1;
        }
    }

    break :blk out;
};

pub const command_subcommands = command_subcommands_array;

const command_names_array: [descriptors.len + 2][]const u8 = blk: {
    var out: [descriptors.len + 2][]const u8 = undefined;
    var index: usize = 0;

    for (descriptors) |descriptor| {
        out[index] = descriptor.name;
        index += 1;
    }

    out[index] = "version";
    index += 1;
    out[index] = "help";

    break :blk out;
};

pub const command_names = command_names_array[0..];

const command_names_with_aliases_array: [command_names_array.len + aliases_array.len][]const u8 = blk: {
    var out: [command_names_array.len + aliases_array.len][]const u8 = undefined;
    var index: usize = 0;

    for (command_names_array) |name| {
        out[index] = name;
        index += 1;
    }

    for (aliases_array) |alias| {
        out[index] = alias.alias;
        index += 1;
    }

    break :blk out;
};

pub const command_names_with_aliases = command_names_with_aliases_array[0..];

pub fn resolveAlias(raw: []const u8) []const u8 {
    for (aliases) |alias| {
        if (std.mem.eql(u8, raw, alias.alias)) {
            return alias.target;
        }
    }
    return raw;
}

pub fn findSubcommands(command: []const u8) ?[]const []const u8 {
    for (command_subcommands) |entry| {
        if (std.mem.eql(u8, command, entry.command)) {
            return entry.subcommands;
        }
    }
    return null;
}

test "alias targets resolve to known command" {
    for (aliases) |alias| {
        var found = false;
        for (command_names) |command| {
            if (std.mem.eql(u8, alias.target, command)) {
                found = true;
                break;
            }
        }
        try std.testing.expect(found);
    }
}

test "command names are unique" {
    for (command_names, 0..) |command, i| {
        var index: usize = 0;
        while (index < i) : (index += 1) {
            try std.testing.expect(!std.mem.eql(u8, command, command_names[index]));
        }
    }
}
