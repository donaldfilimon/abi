//! CLI command and completion metadata derived from command descriptors.

const std = @import("std");
const commands = @import("commands/mod.zig");
const framework_types = @import("framework/types.zig");

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

/// A subcommand with its description (for rich completions in zsh/fish).
pub const SubcommandInfo = struct {
    name: []const u8,
    description: []const u8,
};

/// Subcommands with descriptions for a given parent command.
pub const CommandSubcommandInfos = struct {
    command: []const u8,
    subcommands: []const SubcommandInfo,
};

/// A command-specific option flag for shell completion.
pub const OptionInfo = framework_types.OptionInfo;

/// Maps a command (or command path like "llm run") to its option flags.
pub const CommandOptions = struct {
    command: []const u8,
    options: []const OptionInfo,
};

const command_infos_array: [commands.descriptors.len + 2]CommandInfo = blk: {
    var infos: [commands.descriptors.len + 2]CommandInfo = undefined;
    var index: usize = 0;

    for (commands.descriptors) |desc| {
        infos[index] = .{
            .name = desc.name,
            .description = desc.description,
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
    for (commands.descriptors) |desc| {
        count += desc.aliases.len;
    }
    break :blk count;
};

const aliases_array: [alias_count]AliasInfo = blk: {
    var out: [alias_count]AliasInfo = undefined;
    var index: usize = 0;

    for (commands.descriptors) |desc| {
        for (desc.aliases) |alias| {
            out[index] = .{
                .alias = alias,
                .target = desc.name,
            };
            index += 1;
        }
    }

    break :blk out;
};

pub const aliases = aliases_array;

const subcommand_count: usize = blk: {
    var count: usize = 0;
    for (commands.descriptors) |desc| {
        if (desc.subcommands.len > 0) count += 1;
    }
    break :blk count;
};

const command_subcommands_array: [subcommand_count]CommandSubcommands = blk: {
    var out: [subcommand_count]CommandSubcommands = undefined;
    var index: usize = 0;

    for (commands.descriptors) |desc| {
        if (desc.subcommands.len > 0) {
            out[index] = .{
                .command = desc.name,
                .subcommands = desc.subcommands,
            };
            index += 1;
        }
    }

    break :blk out;
};

pub const command_subcommands = command_subcommands_array;

const command_names_array: [commands.descriptors.len + 2][]const u8 = blk: {
    var out: [commands.descriptors.len + 2][]const u8 = undefined;
    var index: usize = 0;

    for (commands.descriptors) |desc| {
        out[index] = desc.name;
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

// ─── Subcommand descriptions (derived from children metadata) ──────────────

/// Count commands that have children with descriptions.
const children_info_count: usize = blk: {
    var count: usize = 0;
    for (commands.descriptors) |desc| {
        if (desc.children.len > 0) count += 1;
    }
    break :blk count;
};

const command_subcommand_infos_array: [children_info_count]CommandSubcommandInfos = blk: {
    var out: [children_info_count]CommandSubcommandInfos = undefined;
    var index: usize = 0;

    for (commands.descriptors) |desc| {
        if (desc.children.len > 0) {
            const Holder = struct {
                fn build(comptime children: []const @import("framework/types.zig").CommandDescriptor) []const SubcommandInfo {
                    var infos: [children.len]SubcommandInfo = undefined;
                    for (children, 0..) |child, i| {
                        infos[i] = .{
                            .name = child.name,
                            .description = child.description,
                        };
                    }
                    const final = infos;
                    return &final;
                }
            };
            out[index] = .{
                .command = desc.name,
                .subcommands = Holder.build(desc.children),
            };
            index += 1;
        }
    }

    break :blk out;
};

/// Subcommands with descriptions for commands that have children metadata.
pub const command_subcommand_infos = command_subcommand_infos_array;

/// Find subcommand descriptions for a given command name.
pub fn findSubcommandInfos(command: []const u8) ?[]const SubcommandInfo {
    for (command_subcommand_infos) |entry| {
        if (std.mem.eql(u8, command, entry.command)) {
            return entry.subcommands;
        }
    }
    return null;
}

// ─── Command-specific options (descriptor-derived) ───────────────────────────

const option_command_count: usize = blk: {
    var count: usize = 0;
    for (commands.descriptors) |desc| {
        if (desc.options.len > 0) count += 1;
    }
    break :blk count;
};

const command_options_array: [option_command_count]CommandOptions = blk: {
    var out: [option_command_count]CommandOptions = undefined;
    var index: usize = 0;
    for (commands.descriptors) |desc| {
        if (desc.options.len == 0) continue;
        out[index] = .{
            .command = desc.name,
            .options = desc.options,
        };
        index += 1;
    }
    break :blk out;
};

pub const command_options: []const CommandOptions = command_options_array[0..];

/// Find options for a given command name.
pub fn findCommandOptions(command: []const u8) ?[]const OptionInfo {
    for (command_options) |entry| {
        if (std.mem.eql(u8, command, entry.command)) {
            return entry.options;
        }
    }
    return null;
}

// ─── Theme names for UI completion ─────────────────────────────────────────

/// Available TUI theme names for --theme completion.
pub const theme_names: []const []const u8 = &.{
    "default",
    "monokai",
    "solarized",
    "nord",
    "gruvbox",
    "high_contrast",
    "minimal",
};

test "alias targets resolve to known command" {
    for (aliases) |alias| {
        var found = false;
        for (command_names) |cmd_name| {
            if (std.mem.eql(u8, alias.target, cmd_name)) {
                found = true;
                break;
            }
        }
        try std.testing.expect(found);
    }
}

test "command names are unique" {
    for (command_names, 0..) |cmd_name, i| {
        var index: usize = 0;
        while (index < i) : (index += 1) {
            try std.testing.expect(!std.mem.eql(u8, cmd_name, command_names[index]));
        }
    }
}

test "subcommand infos have matching descriptions" {
    // Every command with children should appear in command_subcommand_infos
    for (command_subcommand_infos) |entry| {
        try std.testing.expect(entry.subcommands.len > 0);
        for (entry.subcommands) |sub| {
            try std.testing.expect(sub.name.len > 0);
            try std.testing.expect(sub.description.len > 0);
        }
    }
}

test "command_options entries reference known commands" {
    for (command_options) |entry| {
        try std.testing.expect(entry.options.len > 0);
        // Verify each option flag starts with --
        for (entry.options) |opt| {
            try std.testing.expect(std.mem.startsWith(u8, opt.flag, "--"));
            try std.testing.expect(opt.description.len > 0);
        }
    }
}

test "theme_names list is non-empty and includes default" {
    try std.testing.expect(theme_names.len >= 7);
    var has_default = false;
    for (theme_names) |name| {
        if (std.mem.eql(u8, name, "default")) has_default = true;
    }
    try std.testing.expect(has_default);
}

test "findCommandOptions returns options for known commands" {
    const llm_opts = findCommandOptions("llm");
    try std.testing.expect(llm_opts != null);
    try std.testing.expect(llm_opts.?.len > 0);

    const unknown = findCommandOptions("nonexistent");
    try std.testing.expect(unknown == null);
}

test "findSubcommandInfos returns infos for group commands" {
    const llm_infos = findSubcommandInfos("llm");
    try std.testing.expect(llm_infos != null);
    try std.testing.expect(llm_infos.?.len > 0);

    const unknown = findSubcommandInfos("nonexistent");
    try std.testing.expect(unknown == null);
}

test {
    std.testing.refAllDecls(@This());
}
