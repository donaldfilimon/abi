//! CLI command and completion metadata used across ABI CLI modules.

const std = @import("std");

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

/// Canonical top-level CLI commands and help text used by `abi help`.
pub const command_infos = [_]CommandInfo{
    .{ .name = "db", .description = "Database operations (add, query, stats, optimize, backup, restore)" },
    .{ .name = "agent", .description = "Run AI agent (interactive or one-shot)" },
    .{ .name = "bench", .description = "Run performance benchmarks (all, simd, memory, ai, quick)" },
    .{ .name = "gpu", .description = "GPU commands (backends, devices, summary, default)" },
    .{ .name = "gpu-dashboard", .description = "Interactive GPU + Agent monitoring dashboard" },
    .{ .name = "network", .description = "Manage network registry (list, register, status)" },
    .{ .name = "system-info", .description = "Show system and framework status" },
    .{ .name = "multi-agent", .description = "Run multi-agent workflows" },
    .{ .name = "explore", .description = "Search and explore codebase" },
    .{ .name = "simd", .description = "Run SIMD performance demo" },
    .{ .name = "config", .description = "Configuration management (init, show, validate)" },
    .{ .name = "discord", .description = "Discord bot operations (status, guilds, send, commands)" },
    .{ .name = "llm", .description = "LLM inference (info, generate, chat, bench, download, serve)" },
    .{ .name = "model", .description = "Model management (list, download, remove, search)" },
    .{ .name = "embed", .description = "Generate embeddings from text (openai, mistral, cohere, ollama)" },
    .{ .name = "train", .description = "Training pipeline (run, llm, vision, auto, resume, info)" },
    .{ .name = "convert", .description = "Dataset conversion tools (tokenbin, text, jsonl, wdbx)" },
    .{ .name = "task", .description = "Task management (add, list, done, stats)" },
    .{ .name = "tui", .description = "Launch interactive TUI command menu" },
    .{ .name = "plugins", .description = "Plugin management (list, enable, disable, info)" },
    .{ .name = "profile", .description = "User profile and settings management" },
    .{ .name = "completions", .description = "Generate shell completions (bash, zsh, fish, powershell)" },
    .{ .name = "status", .description = "Show framework health and component status" },
    .{ .name = "toolchain", .description = "Build and install Zig/ZLS from master (install, update, status)" },
    .{ .name = "mcp", .description = "MCP server for WDBX database (serve, tools)" },
    .{ .name = "acp", .description = "Agent Communication Protocol (card, serve)" },
    .{ .name = "ralph", .description = "Ralph orchestrator (init, run, super, multi, status, gate, improve, skills)" },
    .{ .name = "gendocs", .description = "Generate API docs (runs zig build gendocs)" },
    .{ .name = "version", .description = "Show framework version" },
    .{ .name = "help", .description = "Show help (use: abi help <command>)" },
};

/// Top-level command aliases kept for discovery and completion parity.
pub const aliases = [_]AliasInfo{
    .{ .alias = "info", .target = "system-info" },
    .{ .alias = "sysinfo", .target = "system-info" },
    .{ .alias = "ls", .target = "db" },
    .{ .alias = "run", .target = "bench" },
    .{ .alias = "dashboard", .target = "gpu-dashboard" },
    .{ .alias = "chat", .target = "llm" },
    .{ .alias = "reasoning", .target = "llm" },
    .{ .alias = "serve", .target = "llm" },
};

/// Command-specific completions for command families that expose subcommands.
pub const command_subcommands = [_]CommandSubcommands{
    .{ .command = "bench", .subcommands = &.{ "all", "simd", "memory", "ai", "quick", "compare-training", "list", "micro" } },
    .{ .command = "config", .subcommands = &.{ "init", "show", "validate", "env", "help" } },
    .{ .command = "convert", .subcommands = &.{ "dataset", "model", "embeddings" } },
    .{ .command = "db", .subcommands = &.{ "add", "query", "stats", "optimize", "backup", "restore", "serve", "help" } },
    .{ .command = "discord", .subcommands = &.{ "status", "info", "guilds", "send", "commands", "webhook", "channel", "help" } },
    .{ .command = "embed", .subcommands = &.{ "--provider", "openai", "mistral", "cohere", "ollama", "--text", "--file", "--format", "json", "csv", "raw" } },
    .{ .command = "gpu", .subcommands = &.{ "backends", "devices", "list", "summary", "default", "status" } },
    .{ .command = "llm", .subcommands = &.{ "info", "generate", "chat", "bench", "list", "list-local", "demo", "download", "serve", "help" } },
    .{ .command = "model", .subcommands = &.{ "list", "info", "download", "remove", "search", "path" } },
    .{ .command = "multi-agent", .subcommands = &.{ "info", "run", "list", "create", "status" } },
    .{ .command = "network", .subcommands = &.{ "status", "list", "nodes", "register", "unregister", "touch", "set-status" } },
    .{ .command = "plugins", .subcommands = &.{ "list", "info", "enable", "disable", "search" } },
    .{ .command = "profile", .subcommands = &.{ "show", "list", "create", "switch", "delete", "set", "get", "api-key", "export", "import", "help" } },
    .{ .command = "status", .subcommands = &.{"help"} },
    .{ .command = "task", .subcommands = &.{ "add", "list", "ls", "show", "done", "start", "cancel", "delete", "rm", "stats", "import-roadmap", "edit", "block", "unblock", "due", "help" } },
    .{ .command = "toolchain", .subcommands = &.{ "install", "zig", "zls", "status", "update", "path", "help" } },
    .{ .command = "train", .subcommands = &.{ "run", "new", "llm", "vision", "clip", "auto", "resume", "monitor", "info", "generate-data", "help" } },
    .{ .command = "mcp", .subcommands = &.{ "serve", "tools", "help" } },
    .{ .command = "acp", .subcommands = &.{ "card", "serve", "help" } },
    .{ .command = "completions", .subcommands = &.{ "bash", "zsh", "fish", "powershell", "help" } },
    .{ .command = "ralph", .subcommands = &.{ "init", "run", "super", "multi", "status", "gate", "improve", "skills", "help" } },
};

/// All canonical top-level command names in stable order.
pub const command_names: []const []const u8 = blk: {
    break :blk &.{
        "db",
        "agent",
        "bench",
        "gpu",
        "gpu-dashboard",
        "network",
        "system-info",
        "multi-agent",
        "explore",
        "simd",
        "config",
        "discord",
        "llm",
        "model",
        "embed",
        "train",
        "convert",
        "task",
        "tui",
        "plugins",
        "profile",
        "completions",
        "status",
        "toolchain",
        "mcp",
        "acp",
        "ralph",
        "gendocs",
        "version",
        "help",
    };
};

/// Canonical names + aliases in stable order for completion generation.
pub const command_names_with_aliases: []const []const u8 = blk: {
    break :blk &.{
        "db",
        "agent",
        "bench",
        "gpu",
        "gpu-dashboard",
        "network",
        "system-info",
        "multi-agent",
        "explore",
        "simd",
        "config",
        "discord",
        "llm",
        "model",
        "embed",
        "train",
        "convert",
        "task",
        "tui",
        "plugins",
        "profile",
        "completions",
        "status",
        "toolchain",
        "mcp",
        "acp",
        "ralph",
        "gendocs",
        "version",
        "help",
        "info",
        "sysinfo",
        "ls",
        "run",
        "dashboard",
        "chat",
        "reasoning",
        "serve",
    };
};

/// Resolve an alias to its canonical target.
pub fn resolveAlias(raw: []const u8) []const u8 {
    for (aliases) |alias| {
        if (std.mem.eql(u8, raw, alias.alias)) {
            return alias.target;
        }
    }
    return raw;
}

/// Return completion metadata for a top-level command if present.
pub fn findSubcommands(command: []const u8) ?[]const []const u8 {
    for (command_subcommands) |entry| {
        if (std.mem.eql(u8, command, entry.command)) {
            return entry.subcommands;
        }
    }
    return null;
}

// ============================================================================
// Tests
// ============================================================================

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
