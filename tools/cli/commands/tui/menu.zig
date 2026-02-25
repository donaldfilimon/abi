//! TUI menu item definitions and lookup helpers.

const std = @import("std");
const types = @import("types.zig");
const tui_defaults = @import("smart_defaults.zig");

const MenuItem = types.MenuItem;
const Command = types.Command;

/// Return the full list of menu items with metadata.
pub fn menuItemsExtended() []const MenuItem {
    return &[_]MenuItem{
        // AI & ML (shortcuts 1-3)
        .{
            .label = "AI Agent",
            .description = "Interactive AI assistant",
            .action = .{ .command = .agent },
            .category = .ai,
            .shortcut = 1,
            .usage = "abi agent [--message \"...\"] [--persona <name>]",
            .examples = &[_][]const u8{ "abi agent", "abi agent --message \"Hello\"" },
            .related = &[_][]const u8{ "llm", "train" },
        },
        .{
            .label = "LLM",
            .description = "Local LLM inference",
            .action = .{ .command = .llm },
            .category = .ai,
            .shortcut = 2,
            .usage = "abi llm <subcommand> [options]",
            .examples = &[_][]const u8{ "abi llm run --model llama3 --prompt \"hello\"", "abi llm session --model llama3", "abi llm providers" },
            .related = &[_][]const u8{ "agent", "embed" },
        },
        .{
            .label = "Training",
            .description = "Run training pipelines",
            .action = .{ .command = .train },
            .category = .ai,
            .shortcut = 3,
            .usage = "abi train <subcommand> [options]",
            .examples = &[_][]const u8{ "abi train run", "abi train resume", "abi train info" },
            .related = &[_][]const u8{ "agent", "llm", "train-monitor" },
        },
        .{
            .label = "Training Monitor",
            .description = "Live training dashboard",
            .action = .{ .command = .train_monitor },
            .category = .ai,
            .usage = "abi train monitor [run-id]",
            .examples = &[_][]const u8{ "abi train monitor", "abi train monitor --log-dir ./logs" },
            .related = &[_][]const u8{ "train", "llm" },
        },
        .{
            .label = "Embeddings",
            .description = "Generate embeddings",
            .action = .{ .command = .embed },
            .category = .ai,
            .usage = "abi embed [--provider <name>] <text>",
            .examples = &[_][]const u8{ "abi embed \"hello world\"", "abi embed --provider openai \"text\"" },
            .related = &[_][]const u8{ "db", "llm" },
        },
        .{
            .label = "Model",
            .description = "Model management (download, cache, switch)",
            .action = .{ .command = .model },
            .category = .ai,
            .usage = "abi model <subcommand> [options]",
            .examples = &[_][]const u8{ "abi model list", "abi model download llama-7b", "abi model info mistral" },
            .related = &[_][]const u8{ "llm", "agent", "embed" },
        },
        .{
            .label = "Ralph",
            .description = "Iterative agent loop (init, run, status, gate, improve, skills)",
            .action = .{ .command = .ralph },
            .category = .ai,
            .usage = "abi ralph <subcommand> [options]",
            .examples = &[_][]const u8{ "abi ralph init", "abi ralph run", "abi ralph run --task \"...\"", "abi ralph status" },
            .related = &[_][]const u8{ "agent", "llm" },
        },

        // Data (shortcuts 4-5)
        .{
            .label = "Database",
            .description = "Manage vector database",
            .action = .{ .command = .db },
            .category = .data,
            .shortcut = 4,
            .usage = "abi db <subcommand> [options]",
            .examples = &[_][]const u8{ "abi db stats", "abi db add", "abi db query", "abi db backup" },
            .related = &[_][]const u8{ "embed", "explore" },
        },
        .{
            .label = "Explore",
            .description = "Search the codebase",
            .action = .{ .command = .explore },
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
            .action = .{ .command = .gpu },
            .category = .system,
            .shortcut = 6,
            .usage = "abi gpu <subcommand>",
            .examples = &[_][]const u8{ "abi gpu backends", "abi gpu devices", "abi gpu summary" },
            .related = &[_][]const u8{ "bench", "system-info" },
        },
        .{
            .label = "Network",
            .description = "Cluster management",
            .action = .{ .command = .network },
            .category = .system,
            .shortcut = 7,
            .usage = "abi network <subcommand>",
            .examples = &[_][]const u8{ "abi network list", "abi network status", "abi network register" },
            .related = &[_][]const u8{ "system-info", "config" },
        },
        .{
            .label = "System Info",
            .description = "System and framework status",
            .action = .{ .command = .system_info },
            .category = .system,
            .usage = "abi system-info",
            .examples = &[_][]const u8{"abi system-info"},
            .related = &[_][]const u8{ "gpu", "network" },
        },

        // Tools (shortcuts 8-9)
        .{
            .label = "Benchmarks",
            .description = "Performance benchmarks",
            .action = .{ .command = .bench },
            .category = .tools,
            .shortcut = 8,
            .usage = "abi bench [suite]",
            .examples = &[_][]const u8{ "abi bench", "abi bench all", "abi bench simd" },
            .related = &[_][]const u8{ "simd", "gpu" },
        },
        .{
            .label = "SIMD",
            .description = "SIMD performance demo",
            .action = .{ .command = .simd },
            .category = .tools,
            .shortcut = 9,
            .usage = "abi simd",
            .examples = &[_][]const u8{"abi simd"},
            .related = &[_][]const u8{ "bench", "gpu" },
        },
        .{
            .label = "Config",
            .description = "Configuration management",
            .action = .{ .command = .config },
            .category = .tools,
            .usage = "abi config <subcommand>",
            .examples = &[_][]const u8{ "abi config show", "abi config setup", "abi config validate" },
            .related = &[_][]const u8{ "system-info", "network" },
        },
        .{
            .label = "Tasks",
            .description = "Task management",
            .action = .{ .command = .task },
            .category = .tools,
            .usage = "abi task <subcommand>",
            .examples = &[_][]const u8{ "abi task list", "abi task add", "abi task done" },
            .related = &[_][]const u8{ "agent", "config" },
        },
        .{
            .label = "Discord",
            .description = "Discord bot integration",
            .action = .{ .command = .discord },
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
}

/// Find a menu item by its shortcut number (1-9).
pub fn findByShortcut(items: []const MenuItem, num: u8) ?usize {
    for (items, 0..) |item, i| {
        if (item.shortcut) |s| {
            if (s == num) return i;
        }
    }
    return null;
}

/// Get the display name for a command.
pub fn commandName(cmd: Command) []const u8 {
    return switch (cmd) {
        .db => "db",
        .agent => "agent",
        .bench => "bench",
        .config => "config",
        .discord => "discord",
        .embed => "embed",
        .explore => "explore",
        .gpu => "gpu",
        .llm => "llm",
        .model => "model",
        .network => "network",
        .ralph => "ralph",
        .simd => "simd",
        .system_info => "system-info",
        .train => "train",
        .train_monitor => "train-monitor",
        .task => "task",
    };
}

/// Get default arguments for a command (delegates to smart_defaults).
pub fn commandDefaultArgs(cmd: Command) []const [:0]const u8 {
    return tui_defaults.commandDefaultArgs(cmd);
}

test "command default args mapping" {
    try std.testing.expectEqualStrings("status", commandDefaultArgs(.ralph)[0]);
    try std.testing.expectEqualStrings("summary", commandDefaultArgs(.gpu)[0]);
    try std.testing.expectEqualStrings("monitor", commandDefaultArgs(.train_monitor)[0]);
    try std.testing.expectEqual(@as(usize, 0), commandDefaultArgs(.agent).len);
}

test {
    std.testing.refAllDecls(@This());
}
