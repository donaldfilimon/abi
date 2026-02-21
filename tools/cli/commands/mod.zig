//! CLI command modules.
//!
//! Each command is implemented in its own module for maintainability.

const std = @import("std");
const framework = @import("../framework/mod.zig");

pub const db = @import("db.zig");
pub const agent = @import("agent.zig");
pub const bench = @import("bench/mod.zig");
pub const config = @import("config.zig");
pub const convert = @import("convert.zig");
pub const discord = @import("discord.zig");
pub const embed = @import("embed.zig");
pub const explore = @import("explore.zig");
pub const gpu = @import("gpu.zig");
pub const gpu_dashboard = @import("gpu_dashboard.zig");
pub const llm = @import("llm/mod.zig");
pub const model = @import("model.zig");
pub const network = @import("network.zig");
pub const simd = @import("simd.zig");
pub const system_info = @import("system_info.zig");
pub const task = @import("task.zig");
pub const tui = @import("tui/mod.zig");
pub const ui = @import("ui/mod.zig");
pub const train = @import("train/mod.zig");
pub const completions = @import("completions.zig");
pub const multi_agent = @import("multi_agent.zig");
pub const plugins = @import("plugins.zig");
pub const profile = @import("profile.zig");
pub const os_agent = @import("os_agent.zig");
pub const status = @import("status.zig");
pub const toolchain = @import("toolchain.zig");
pub const mcp = @import("mcp.zig");
pub const acp = @import("acp.zig");
pub const ralph = @import("ralph/mod.zig");
pub const gendocs = @import("gendocs.zig");

const CommandDescriptor = framework.types.CommandDescriptor;
const CommandHandler = framework.types.CommandHandler;
const CommandForward = framework.types.CommandForward;

const io_forward_launch = [_][:0]const u8{"launch"};
const io_forward_gpu = [_][:0]const u8{"gpu"};

const bench_subcommands = [_][]const u8{ "all", "simd", "memory", "ai", "quick", "compare-training", "list", "micro" };
const config_subcommands = [_][]const u8{ "init", "show", "validate", "env", "help" };
const convert_subcommands = [_][]const u8{ "dataset", "model", "embeddings" };
const db_subcommands = [_][]const u8{ "add", "query", "stats", "optimize", "backup", "restore", "serve", "help" };
const discord_subcommands = [_][]const u8{ "status", "info", "guilds", "send", "commands", "webhook", "channel", "help" };
const embed_subcommands = [_][]const u8{ "--provider", "openai", "mistral", "cohere", "ollama", "--text", "--file", "--format", "json", "csv", "raw" };
const gpu_subcommands = [_][]const u8{ "backends", "devices", "list", "summary", "default", "status" };
const llm_subcommands = [_][]const u8{ "run", "session", "serve", "providers", "plugins", "list", "info", "bench", "download", "help" };
const model_subcommands = [_][]const u8{ "list", "info", "download", "remove", "search", "path" };
const multi_agent_subcommands = [_][]const u8{ "info", "run", "list", "create", "status" };
const network_subcommands = [_][]const u8{ "status", "list", "nodes", "register", "unregister", "touch", "set-status" };
const plugins_subcommands = [_][]const u8{ "list", "info", "enable", "disable", "search" };
const profile_subcommands = [_][]const u8{ "show", "list", "create", "switch", "delete", "set", "get", "api-key", "export", "import", "help" };
const status_subcommands = [_][]const u8{"help"};
const task_subcommands = [_][]const u8{ "add", "list", "ls", "show", "done", "start", "cancel", "delete", "rm", "stats", "import-roadmap", "seed-self-improve", "edit", "block", "unblock", "due", "help" };
const toolchain_subcommands = [_][]const u8{ "install", "zig", "zls", "status", "update", "path", "help" };
const train_subcommands = [_][]const u8{ "run", "new", "llm", "vision", "clip", "auto", "self", "resume", "monitor", "info", "generate-data", "help" };
const ui_subcommands = [_][]const u8{ "launch", "gpu", "train", "neural", "help" };
const mcp_subcommands = [_][]const u8{ "serve", "tools", "help" };
const acp_subcommands = [_][]const u8{ "card", "serve", "help" };
const completions_subcommands = [_][]const u8{ "bash", "zsh", "fish", "powershell", "help" };
const ralph_subcommands = [_][]const u8{ "init", "run", "super", "multi", "status", "gate", "improve", "skills", "help" };

fn basicHandler(comptime module: type) CommandHandler {
    return .{ .basic = module.run };
}

fn ioHandler(comptime module: type) CommandHandler {
    return .{ .io = module.run };
}

pub const descriptors = [_]CommandDescriptor{
    .{ .name = "db", .description = "Database operations (add, query, stats, optimize, backup, restore)", .aliases = &.{"ls"}, .subcommands = &db_subcommands, .handler = basicHandler(db) },
    .{ .name = "agent", .description = "Run AI agent (interactive or one-shot)", .handler = basicHandler(agent) },
    .{ .name = "bench", .description = "Run performance benchmarks (all, simd, memory, ai, quick)", .aliases = &.{"run"}, .subcommands = &bench_subcommands, .handler = basicHandler(bench) },
    .{ .name = "gpu", .description = "GPU commands (backends, devices, summary, default)", .subcommands = &gpu_subcommands, .handler = basicHandler(gpu) },
    .{
        .name = "gpu-dashboard",
        .description = "Interactive GPU + Agent monitoring dashboard",
        .aliases = &.{"dashboard"},
        .handler = ioHandler(gpu_dashboard),
        .forward = CommandForward{
            .target = "ui",
            .prepend_args = &io_forward_gpu,
            .warning = "'abi gpu-dashboard' is deprecated; use 'abi ui gpu'.",
        },
    },
    .{ .name = "network", .description = "Manage network registry (list, register, status)", .subcommands = &network_subcommands, .handler = basicHandler(network) },
    .{ .name = "system-info", .description = "Show system and framework status", .aliases = &.{ "info", "sysinfo" }, .handler = basicHandler(system_info) },
    .{ .name = "multi-agent", .description = "Run multi-agent workflows", .subcommands = &multi_agent_subcommands, .handler = basicHandler(multi_agent) },
    .{ .name = "explore", .description = "Search and explore codebase", .handler = basicHandler(explore) },
    .{ .name = "simd", .description = "Run SIMD performance demo", .handler = basicHandler(simd) },
    .{ .name = "config", .description = "Configuration management (init, show, validate)", .subcommands = &config_subcommands, .handler = basicHandler(config) },
    .{ .name = "discord", .description = "Discord bot operations (status, guilds, send, commands)", .subcommands = &discord_subcommands, .handler = basicHandler(discord) },
    .{ .name = "llm", .description = "LLM inference (run, session, serve, providers, plugins)", .aliases = &.{ "chat", "reasoning", "serve" }, .subcommands = &llm_subcommands, .handler = basicHandler(llm) },
    .{ .name = "model", .description = "Model management (list, download, remove, search)", .subcommands = &model_subcommands, .handler = basicHandler(model) },
    .{ .name = "embed", .description = "Generate embeddings from text (openai, mistral, cohere, ollama)", .subcommands = &embed_subcommands, .handler = basicHandler(embed) },
    .{ .name = "train", .description = "Training pipeline (run, llm, vision, auto, self, resume, info)", .subcommands = &train_subcommands, .handler = basicHandler(train) },
    .{ .name = "convert", .description = "Dataset conversion tools (tokenbin, text, jsonl, wdbx)", .subcommands = &convert_subcommands, .handler = basicHandler(convert) },
    .{ .name = "task", .description = "Task management (add, list, done, stats, seed-self-improve)", .subcommands = &task_subcommands, .handler = basicHandler(task) },
    .{
        .name = "tui",
        .description = "Launch interactive TUI command menu",
        .handler = ioHandler(tui),
        .forward = CommandForward{
            .target = "ui",
            .prepend_args = &io_forward_launch,
            .warning = "'abi tui' is deprecated; use 'abi ui launch'.",
        },
    },
    .{ .name = "ui", .description = "UI command family (launch, gpu, train, neural)", .subcommands = &ui_subcommands, .handler = ioHandler(ui) },
    .{ .name = "plugins", .description = "Plugin management (list, enable, disable, info)", .subcommands = &plugins_subcommands, .handler = basicHandler(plugins) },
    .{ .name = "profile", .description = "User profile and settings management", .subcommands = &profile_subcommands, .handler = basicHandler(profile) },
    .{ .name = "completions", .description = "Generate shell completions (bash, zsh, fish, powershell)", .subcommands = &completions_subcommands, .handler = basicHandler(completions) },
    .{ .name = "status", .description = "Show framework health and component status", .subcommands = &status_subcommands, .handler = basicHandler(status) },
    .{ .name = "toolchain", .description = "Build and install Zig/ZLS from master (install, update, status)", .subcommands = &toolchain_subcommands, .handler = basicHandler(toolchain) },
    .{ .name = "mcp", .description = "MCP server for WDBX database (serve, tools)", .subcommands = &mcp_subcommands, .handler = basicHandler(mcp) },
    .{ .name = "acp", .description = "Agent Communication Protocol (card, serve)", .subcommands = &acp_subcommands, .handler = basicHandler(acp) },
    .{ .name = "ralph", .description = "Ralph orchestrator (init, run, super, multi, status, gate, improve, skills)", .subcommands = &ralph_subcommands, .handler = basicHandler(ralph) },
    .{ .name = "gendocs", .description = "Generate API docs (runs zig build gendocs)", .handler = basicHandler(gendocs) },
};

pub fn findDescriptor(raw_name: []const u8) ?*const CommandDescriptor {
    for (descriptors) |*descriptor| {
        if (std.mem.eql(u8, raw_name, descriptor.name)) return descriptor;
        for (descriptor.aliases) |alias| {
            if (std.mem.eql(u8, raw_name, alias)) return descriptor;
        }
    }
    return null;
}
