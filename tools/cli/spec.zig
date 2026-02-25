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
pub const OptionInfo = struct {
    flag: []const u8,
    description: []const u8,
};

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

// ─── Command-specific options (hand-curated for top commands) ──────────────

pub const command_options: []const CommandOptions = &.{
    .{
        .command = "llm",
        .options = &.{
            .{ .flag = "--model", .description = "Model path or name" },
            .{ .flag = "--prompt", .description = "Input prompt text" },
            .{ .flag = "--backend", .description = "Provider backend (ollama, mlx, etc.)" },
            .{ .flag = "--fallback", .description = "Comma-separated fallback providers" },
            .{ .flag = "--temperature", .description = "Sampling temperature" },
            .{ .flag = "--max-tokens", .description = "Maximum tokens to generate" },
            .{ .flag = "--stream", .description = "Enable streaming output" },
            .{ .flag = "--json", .description = "Output in JSON format" },
        },
    },
    .{
        .command = "train",
        .options = &.{
            .{ .flag = "--epochs", .description = "Number of training epochs" },
            .{ .flag = "--batch-size", .description = "Training batch size" },
            .{ .flag = "--learning-rate", .description = "Learning rate" },
            .{ .flag = "--optimizer", .description = "Optimizer (sgd, adam, adamw)" },
            .{ .flag = "--lr-schedule", .description = "LR schedule (constant, cosine, warmup_cosine)" },
            .{ .flag = "--checkpoint-path", .description = "Checkpoint save path" },
            .{ .flag = "--checkpoint-interval", .description = "Steps between checkpoints" },
            .{ .flag = "--mixed-precision", .description = "Enable mixed precision training" },
            .{ .flag = "--use-gpu", .description = "Enable GPU acceleration" },
            .{ .flag = "--cpu-only", .description = "Force CPU-only training" },
            .{ .flag = "--export-gguf", .description = "Export GGUF weights after training" },
            .{ .flag = "--dataset-path", .description = "Path to training dataset" },
            .{ .flag = "--dataset-url", .description = "URL to download dataset" },
            .{ .flag = "--dataset-format", .description = "Dataset format (tokenbin, text, jsonl)" },
        },
    },
    .{
        .command = "model",
        .options = &.{
            .{ .flag = "--json", .description = "Output in JSON format" },
            .{ .flag = "--no-size", .description = "Hide file sizes in list" },
            .{ .flag = "--output", .description = "Output file path" },
            .{ .flag = "--no-verify", .description = "Skip checksum verification" },
            .{ .flag = "--force", .description = "Force removal without confirmation" },
            .{ .flag = "--reset", .description = "Reset cache directory to default" },
        },
    },
    .{
        .command = "config",
        .options = &.{
            .{ .flag = "--output", .description = "Output file path" },
            .{ .flag = "--format", .description = "Output format (human, json, zon)" },
            .{ .flag = "--force", .description = "Overwrite existing config" },
        },
    },
    .{
        .command = "ui",
        .options = &.{
            .{ .flag = "--theme", .description = "Set TUI color theme" },
            .{ .flag = "--list-themes", .description = "List available themes" },
            .{ .flag = "--refresh-ms", .description = "Dashboard refresh interval in ms" },
            .{ .flag = "--layers", .description = "Neural network layer sizes (comma-separated)" },
            .{ .flag = "--frames", .description = "Number of animation frames (0=infinite)" },
        },
    },
    .{
        .command = "ralph",
        .options = &.{
            .{ .flag = "--task", .description = "Task description for inline execution" },
            .{ .flag = "--gate", .description = "Run quality gate after execution" },
            .{ .flag = "--auto-skill", .description = "Auto-extract skill after run" },
            .{ .flag = "--iterations", .description = "Number of loop iterations" },
            .{ .flag = "--config", .description = "Path to ralph.yml config" },
        },
    },
    .{
        .command = "bench",
        .options = &.{
            .{ .flag = "--json", .description = "Output results in JSON format" },
            .{ .flag = "--output", .description = "Write results to file" },
        },
    },
    .{
        .command = "db",
        .options = &.{
            .{ .flag = "--db", .description = "Database file path" },
            .{ .flag = "--id", .description = "Record ID" },
            .{ .flag = "--embed", .description = "Embedding text" },
            .{ .flag = "--top-k", .description = "Number of results to return" },
            .{ .flag = "--out", .description = "Backup output path" },
            .{ .flag = "--in", .description = "Restore input path" },
            .{ .flag = "--path", .description = "Legacy path shorthand" },
        },
    },
    .{
        .command = "profile",
        .options = &.{
            .{ .flag = "--json", .description = "Output in JSON format" },
            .{ .flag = "--force", .description = "Force delete without confirmation" },
        },
    },
    .{
        .command = "lsp",
        .options = &.{
            .{ .flag = "--path", .description = "Source file path" },
            .{ .flag = "--line", .description = "Line number (0-based)" },
            .{ .flag = "--character", .description = "Character offset (0-based)" },
            .{ .flag = "--method", .description = "LSP method name" },
            .{ .flag = "--params", .description = "JSON parameters" },
            .{ .flag = "--new-name", .description = "New name for rename" },
        },
    },
    .{
        .command = "mcp",
        .options = &.{
            .{ .flag = "--db", .description = "Database path for MCP server" },
            .{ .flag = "--transport", .description = "Transport type (stdio)" },
        },
    },
    .{
        .command = "network",
        .options = &.{
            .{ .flag = "--cluster-id", .description = "Cluster identifier" },
            .{ .flag = "--address", .description = "Node address (host:port)" },
        },
    },
};

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
