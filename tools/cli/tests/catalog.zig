const std = @import("std");

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

pub const bench_subcommands = [_][]const u8{ "all", "simd", "memory", "ai", "quick", "compare-training", "list", "micro" };
pub const config_subcommands = [_][]const u8{ "init", "show", "validate", "env", "help" };
pub const convert_subcommands = [_][]const u8{ "dataset", "model", "embeddings" };
pub const db_subcommands = [_][]const u8{ "add", "query", "stats", "optimize", "backup", "restore", "serve", "help" };
pub const discord_subcommands = [_][]const u8{ "status", "info", "guilds", "send", "commands", "webhook", "channel", "help" };
pub const gpu_subcommands = [_][]const u8{ "backends", "devices", "list", "summary", "default", "status" };
pub const llm_subcommands = [_][]const u8{ "run", "session", "serve", "providers", "plugins", "help" };
pub const model_subcommands = [_][]const u8{ "list", "info", "download", "remove", "search", "path" };
pub const multi_agent_subcommands = [_][]const u8{ "info", "run", "list", "create", "status" };
pub const network_subcommands = [_][]const u8{ "status", "list", "nodes", "register", "unregister", "touch", "set-status" };
pub const plugins_subcommands = [_][]const u8{ "list", "info", "enable", "disable", "search" };
pub const profile_subcommands = [_][]const u8{ "show", "list", "create", "switch", "delete", "set", "get", "api-key", "export", "import", "help" };
pub const status_subcommands = [_][]const u8{"help"};
pub const task_subcommands = [_][]const u8{ "add", "list", "ls", "show", "done", "start", "cancel", "delete", "rm", "stats", "import-roadmap", "seed-self-improve", "edit", "block", "unblock", "due", "help" };
pub const toolchain_subcommands = [_][]const u8{ "install", "zig", "zls", "status", "update", "path", "help" };
pub const train_subcommands = [_][]const u8{ "run", "new", "llm", "vision", "clip", "auto", "self", "resume", "monitor", "info", "generate-data", "help" };
pub const ui_subcommands = [_][]const u8{ "launch", "gpu", "train", "neural", "help" };
pub const mcp_subcommands = [_][]const u8{ "serve", "tools", "help" };
pub const acp_subcommands = [_][]const u8{ "card", "serve", "help" };
pub const completions_subcommands = [_][]const u8{ "bash", "zsh", "fish", "powershell", "help" };
pub const ralph_subcommands = [_][]const u8{ "init", "run", "super", "multi", "status", "gate", "improve", "skills", "help" };

pub const llm_children = [_]ChildSpec{
    .{ .name = "run", .description = "One-shot generation through provider router" },
    .{ .name = "session", .description = "Interactive session through provider router" },
    .{ .name = "serve", .description = "Start streaming HTTP server" },
    .{ .name = "providers", .description = "Show provider availability and routing order" },
    .{ .name = "plugins", .description = "Manage HTTP/native provider plugins" },
};

pub const ui_children = [_]ChildSpec{
    .{ .name = "launch", .description = "Open command launcher TUI" },
    .{ .name = "gpu", .description = "Open GPU dashboard TUI" },
    .{ .name = "train", .description = "Open training monitor TUI" },
    .{ .name = "neural", .description = "Render dynamic 3D neural network view" },
};

pub const commands = [_]CommandSpec{
    .{ .name = "db", .description = "Database operations (add, query, stats, optimize, backup, restore)", .aliases = &.{"ls"}, .subcommands = &db_subcommands },
    .{ .name = "agent", .description = "Run AI agent (interactive or one-shot)" },
    .{ .name = "bench", .description = "Run performance benchmarks (all, simd, memory, ai, quick)", .aliases = &.{"run"}, .subcommands = &bench_subcommands },
    .{ .name = "gpu", .description = "GPU commands (backends, devices, summary, default)", .subcommands = &gpu_subcommands },
    .{ .name = "gpu-dashboard", .description = "Interactive GPU + Agent monitoring dashboard", .aliases = &.{"dashboard"} },
    .{ .name = "network", .description = "Manage network registry (list, register, status)", .subcommands = &network_subcommands },
    .{ .name = "system-info", .description = "Show system and framework status", .aliases = &.{ "info", "sysinfo" } },
    .{ .name = "multi-agent", .description = "Run multi-agent workflows", .subcommands = &multi_agent_subcommands },
    .{ .name = "explore", .description = "Search and explore codebase" },
    .{ .name = "simd", .description = "Run SIMD performance demo" },
    .{ .name = "config", .description = "Configuration management (init, show, validate)", .subcommands = &config_subcommands },
    .{ .name = "discord", .description = "Discord bot operations (status, guilds, send, commands)", .subcommands = &discord_subcommands },
    .{ .name = "llm", .description = "LLM inference (run, session, serve, providers, plugins)", .aliases = &.{ "chat", "reasoning", "serve" }, .subcommands = &llm_subcommands },
    .{ .name = "model", .description = "Model management (list, download, remove, search)", .subcommands = &model_subcommands },
    .{ .name = "embed", .description = "Generate embeddings from text (openai, mistral, cohere, ollama)" },
    .{ .name = "train", .description = "Training pipeline (run, llm, vision, auto, self, resume, info)", .subcommands = &train_subcommands },
    .{ .name = "convert", .description = "Dataset conversion tools (tokenbin, text, jsonl, wdbx)", .subcommands = &convert_subcommands },
    .{ .name = "task", .description = "Task management (add, list, done, stats, seed-self-improve)", .subcommands = &task_subcommands },
    .{ .name = "tui", .description = "Launch interactive TUI command menu" },
    .{ .name = "ui", .description = "UI command family (launch, gpu, train, neural)", .subcommands = &ui_subcommands },
    .{ .name = "plugins", .description = "Plugin management (list, enable, disable, info)", .subcommands = &plugins_subcommands },
    .{ .name = "profile", .description = "User profile and settings management", .subcommands = &profile_subcommands },
    .{ .name = "completions", .description = "Generate shell completions (bash, zsh, fish, powershell)", .subcommands = &completions_subcommands },
    .{ .name = "status", .description = "Show framework health and component status", .subcommands = &status_subcommands },
    .{ .name = "toolchain", .description = "Build and install Zig/ZLS from master (install, update, status)", .subcommands = &toolchain_subcommands },
    .{ .name = "mcp", .description = "MCP server for WDBX database (serve, tools)", .subcommands = &mcp_subcommands },
    .{ .name = "acp", .description = "Agent Communication Protocol (card, serve)", .subcommands = &acp_subcommands },
    .{ .name = "ralph", .description = "Ralph orchestrator (init, run, super, multi, status, gate, improve, skills)", .subcommands = &ralph_subcommands },
    .{ .name = "gendocs", .description = "Generate API docs (runs zig build gendocs)" },
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
