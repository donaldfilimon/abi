//! Descriptor-validated launcher catalog for the TUI command launcher.

const std = @import("std");
const framework = @import("../../framework/mod.zig");
const commands = @import("../../mod.zig");
const types = @import("types.zig");

const MenuItem = types.MenuItem;
const CommandRef = types.CommandRef;

const empty_args = &[_][:0]const u8{};
const args_stats = [_][:0]const u8{"stats"};
const args_quick = [_][:0]const u8{"quick"};
const args_show = [_][:0]const u8{"show"};
const args_status = [_][:0]const u8{"status"};
const args_summary = [_][:0]const u8{"summary"};
const args_list = [_][:0]const u8{"list"};
const args_info = [_][:0]const u8{"info"};
const args_monitor = [_][:0]const u8{"monitor"};

pub fn menuItems() []const MenuItem {
    return &catalog_items;
}

pub fn findCommandById(command_id: []const u8) ?CommandRef {
    for (catalog_items) |item| {
        switch (item.action) {
            .command => |cmd| {
                if (std.mem.eql(u8, cmd.id, command_id)) return cmd;
            },
            else => {},
        }
    }
    return null;
}

pub fn commandName(command_id: []const u8) []const u8 {
    if (findCommandById(command_id)) |cmd| return cmd.id;
    return command_id;
}

const catalog_items = [_]MenuItem{
    // AI & ML (shortcuts 1-3)
    .{
        .label = "AI Agent",
        .description = "Interactive AI assistant",
        .action = .{ .command = commandRef("agent", "agent", empty_args) },
        .category = .ai,
        .shortcut = 1,
        .usage = "abi agent [--message \"...\"] [--persona <name>]",
        .examples = &[_][]const u8{ "abi agent", "abi agent --message \"Hello\"" },
        .related = &[_][]const u8{ "llm", "train" },
    },
    .{
        .label = "LLM",
        .description = "Local LLM inference",
        .action = .{ .command = commandRef("llm", "llm", &args_list) },
        .category = .ai,
        .shortcut = 2,
        .usage = "abi llm <subcommand> [options]",
        .examples = &[_][]const u8{ "abi llm run --model llama3 --prompt \"hello\"", "abi llm session --model llama3", "abi llm providers" },
        .related = &[_][]const u8{ "agent", "embed" },
    },
    .{
        .label = "Training",
        .description = "Run training pipelines",
        .action = .{ .command = commandRef("train", "train", &args_info) },
        .category = .ai,
        .shortcut = 3,
        .usage = "abi train <subcommand> [options]",
        .examples = &[_][]const u8{ "abi train run", "abi train resume", "abi train info" },
        .related = &[_][]const u8{ "agent", "llm", "train-monitor" },
    },
    .{
        .label = "Training Monitor",
        .description = "Live training dashboard",
        .action = .{ .command = commandRef("train-monitor", "train", &args_monitor) },
        .category = .ai,
        .usage = "abi train monitor [run-id]",
        .examples = &[_][]const u8{ "abi train monitor", "abi train monitor --log-dir ./logs" },
        .related = &[_][]const u8{ "train", "llm" },
    },
    .{
        .label = "Embeddings",
        .description = "Generate embeddings",
        .action = .{ .command = commandRef("embed", "embed", empty_args) },
        .category = .ai,
        .usage = "abi embed [--provider <name>] <text>",
        .examples = &[_][]const u8{ "abi embed \"hello world\"", "abi embed --provider openai \"text\"" },
        .related = &[_][]const u8{ "db", "llm" },
    },
    .{
        .label = "Model",
        .description = "Model management (download, cache, switch)",
        .action = .{ .command = commandRef("model", "model", &args_list) },
        .category = .ai,
        .usage = "abi model <subcommand> [options]",
        .examples = &[_][]const u8{ "abi model list", "abi model download llama-7b", "abi model info mistral" },
        .related = &[_][]const u8{ "llm", "agent", "embed" },
    },
    .{
        .label = "Ralph",
        .description = "Iterative agent loop (init, run, status, gate, improve, skills)",
        .action = .{ .command = commandRef("ralph", "ralph", &args_status) },
        .category = .ai,
        .usage = "abi ralph <subcommand> [options]",
        .examples = &[_][]const u8{ "abi ralph init", "abi ralph run", "abi ralph run --task \"...\"", "abi ralph status" },
        .related = &[_][]const u8{ "agent", "llm" },
    },

    // Data (shortcuts 4-5)
    .{
        .label = "Database",
        .description = "Manage vector database",
        .action = .{ .command = commandRef("db", "db", &args_stats) },
        .category = .data,
        .shortcut = 4,
        .usage = "abi db <subcommand> [options]",
        .examples = &[_][]const u8{ "abi db stats", "abi db add", "abi db query", "abi db backup" },
        .related = &[_][]const u8{ "embed", "explore" },
    },
    .{
        .label = "Explore",
        .description = "Search the codebase",
        .action = .{ .command = commandRef("explore", "explore", empty_args) },
        .category = .data,
        .shortcut = 5,
        .usage = "abi explore [query]",
        .examples = &[_][]const u8{ "abi explore", "abi explore \"function name\"" },
        .related = &[_][]const u8{ "db", "agent" },
    },

    // System (shortcuts 6-7)
    .{
        .label = "GPU",
        .description = "GPU devices and backends",
        .action = .{ .command = commandRef("gpu", "gpu", &args_summary) },
        .category = .system,
        .shortcut = 6,
        .usage = "abi gpu <subcommand>",
        .examples = &[_][]const u8{ "abi gpu backends", "abi gpu devices", "abi gpu summary" },
        .related = &[_][]const u8{ "bench", "system-info" },
    },
    .{
        .label = "Network",
        .description = "Cluster management",
        .action = .{ .command = commandRef("network", "network", &args_status) },
        .category = .system,
        .shortcut = 7,
        .usage = "abi network <subcommand>",
        .examples = &[_][]const u8{ "abi network list", "abi network status", "abi network register" },
        .related = &[_][]const u8{ "system-info", "config" },
    },
    .{
        .label = "System Info",
        .description = "System and framework status",
        .action = .{ .command = commandRef("system-info", "system-info", empty_args) },
        .category = .system,
        .usage = "abi system-info",
        .examples = &[_][]const u8{"abi system-info"},
        .related = &[_][]const u8{ "gpu", "network" },
    },

    // Tools (shortcuts 8-9)
    .{
        .label = "Benchmarks",
        .description = "Performance benchmarks",
        .action = .{ .command = commandRef("bench", "bench", &args_quick) },
        .category = .tools,
        .shortcut = 8,
        .usage = "abi bench [suite]",
        .examples = &[_][]const u8{ "abi bench", "abi bench all", "abi bench simd" },
        .related = &[_][]const u8{ "simd", "gpu" },
    },
    .{
        .label = "SIMD",
        .description = "SIMD performance demo",
        .action = .{ .command = commandRef("simd", "simd", empty_args) },
        .category = .tools,
        .shortcut = 9,
        .usage = "abi simd",
        .examples = &[_][]const u8{"abi simd"},
        .related = &[_][]const u8{ "bench", "gpu" },
    },
    .{
        .label = "Config",
        .description = "Configuration management",
        .action = .{ .command = commandRef("config", "config", &args_show) },
        .category = .tools,
        .usage = "abi config <subcommand>",
        .examples = &[_][]const u8{ "abi config show", "abi config setup", "abi config validate" },
        .related = &[_][]const u8{ "system-info", "network" },
    },
    .{
        .label = "Tasks",
        .description = "Task management",
        .action = .{ .command = commandRef("task", "task", &args_list) },
        .category = .tools,
        .usage = "abi task <subcommand>",
        .examples = &[_][]const u8{ "abi task list", "abi task add", "abi task done" },
        .related = &[_][]const u8{ "agent", "config" },
    },
    .{
        .label = "Discord",
        .description = "Discord bot integration",
        .action = .{ .command = commandRef("discord", "discord", &args_status) },
        .category = .tools,
        .usage = "abi discord <subcommand>",
        .examples = &[_][]const u8{ "abi discord status", "abi discord guilds" },
        .related = &[_][]const u8{ "agent", "config" },
    },

    // Meta
    .{ .label = "Help", .description = "Show CLI usage", .action = .help, .category = .meta },
    .{ .label = "Version", .description = "Show version", .action = .version, .category = .meta },
    .{ .label = "Quit", .description = "Exit the launcher", .action = .quit, .category = .meta },
};

comptime {
    validateCatalog(&catalog_items);
}

fn commandRef(comptime id: []const u8, comptime command: []const u8, comptime args: []const [:0]const u8) CommandRef {
    return .{
        .id = id,
        .command = command,
        .args = args,
    };
}

fn validateCatalog(comptime items: []const MenuItem) void {
    inline for (items, 0..) |item, index| {
        if (item.shortcut) |shortcut| {
            if (shortcut < 1 or shortcut > 9) {
                @compileError(std.fmt.comptimePrint(
                    "launcher item '{s}' has shortcut {d}; expected 1-9",
                    .{ item.label, shortcut },
                ));
            }

            inline for (items[0..index]) |prev| {
                if (prev.shortcut) |prev_shortcut| {
                    if (prev_shortcut == shortcut) {
                        @compileError(std.fmt.comptimePrint(
                            "duplicate launcher shortcut {d} between '{s}' and '{s}'",
                            .{ shortcut, prev.label, item.label },
                        ));
                    }
                }
            }
        }

        switch (item.action) {
            .command => |cmd| {
                validateCommandRef(cmd, item.label);
                inline for (items[0..index]) |prev| {
                    switch (prev.action) {
                        .command => |prev_cmd| {
                            if (std.mem.eql(u8, prev_cmd.id, cmd.id)) {
                                @compileError(std.fmt.comptimePrint(
                                    "duplicate launcher command id '{s}' in '{s}' and '{s}'",
                                    .{ cmd.id, prev.label, item.label },
                                ));
                            }
                        },
                        else => {},
                    }
                }
            },
            else => {},
        }
    }
}

fn validateCommandRef(comptime cmd: CommandRef, comptime item_label: []const u8) void {
    if (cmd.id.len == 0) {
        @compileError(std.fmt.comptimePrint(
            "launcher item '{s}' has empty command id",
            .{item_label},
        ));
    }

    _ = findTopLevelDescriptor(cmd.command) orelse {
        @compileError(std.fmt.comptimePrint(
            "launcher item '{s}' maps id '{s}' to unknown command '{s}'",
            .{ item_label, cmd.id, cmd.command },
        ));
    };
}

fn findTopLevelDescriptor(comptime raw_command: []const u8) ?*const framework.types.CommandDescriptor {
    inline for (&commands.descriptors) |*descriptor| {
        if (std.mem.eql(u8, raw_command, descriptor.name)) return descriptor;
        inline for (descriptor.aliases) |alias| {
            if (std.mem.eql(u8, raw_command, alias)) return descriptor;
        }
    }
    return null;
}

test {
    std.testing.refAllDecls(@This());
}
