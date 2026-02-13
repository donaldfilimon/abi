//! ABI Framework Command-Line Interface
//!
//! Provides a comprehensive CLI for interacting with all ABI framework features.
//! The CLI supports commands for database operations, GPU management, AI agent
//! interaction, network configuration, and system information.
//!
//! ## Usage
//! ```bash
//! abi <command> [options]
//! ```
//!
//! ## Commands (26 total)
//! - `db` - Database operations (add, query, stats, optimize, backup)
//! - `agent` - Run AI agent (interactive or one-shot)
//! - `bench` - Run performance benchmarks (all, simd, memory, ai, quick)
//! - `gpu` - GPU commands (backends, devices, summary, default)
//! - `gpu-dashboard` - Interactive GPU + Agent monitoring dashboard
//! - `network` - Manage network registry (list, register, status)
//! - `system-info` - Show system and framework status
//! - `multi-agent` - Run multi-agent workflows
//! - `explore` - Search and explore codebase
//! - `simd` - Run SIMD performance demo
//! - `config` - Configuration management (init, show, validate)
//! - `discord` - Discord bot operations (status, guilds, send, commands)
//! - `llm` - LLM inference (info, generate, chat, bench, download)
//! - `model` - Model management (list, download, remove, search)
//! - `embed` - Generate embeddings (openai, mistral, cohere, ollama)
//! - `train` - Training pipeline (run, resume, info)
//! - `convert` - Dataset conversion tools (tokenbin, text, jsonl, wdbx)
//! - `task` - Task management (add, list, done, stats)
//! - `tui` - Launch interactive TUI command menu
//! - `plugins` - Plugin management (list, enable, disable, info)
//! - `profile` - User profile and settings management
//! - `completions` - Shell completions (bash, zsh, fish, powershell)
//! - `status` - Show framework health and component status
//! - `toolchain` - Zig/ZLS toolchain (install, update, status)
//! - `version` - Show framework version
//! - `help` - Show help (use: abi help <command>)

const std = @import("std");
const abi = @import("abi");
const commands = @import("commands/mod.zig");
const utils = @import("utils/mod.zig");
const cli_io = utils.io_backend;

const CommandInfo = struct {
    name: []const u8,
    description: []const u8,
};

const command_infos = [_]CommandInfo{
    .{ .name = "db", .description = "Database operations (add, query, stats, optimize, backup)" },
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
    .{ .name = "llm", .description = "LLM inference (info, generate, chat, bench, download)" },
    .{ .name = "model", .description = "Model management (list, download, remove, search)" },
    .{ .name = "embed", .description = "Generate embeddings from text (openai, mistral, cohere, ollama)" },
    .{ .name = "train", .description = "Training pipeline (run, resume, info)" },
    .{ .name = "convert", .description = "Dataset conversion tools (tokenbin, text, jsonl, wdbx)" },
    .{ .name = "task", .description = "Task management (add, list, done, stats)" },
    .{ .name = "tui", .description = "Launch interactive TUI command menu" },
    .{ .name = "plugins", .description = "Plugin management (list, enable, disable, info)" },
    .{ .name = "profile", .description = "User profile and settings management" },
    .{ .name = "completions", .description = "Generate shell completions (bash, zsh, fish, powershell)" },
    .{ .name = "status", .description = "Show framework health and component status" },
    .{ .name = "toolchain", .description = "Build and install Zig/ZLS from master (install, update, status)" },
    .{ .name = "version", .description = "Show framework version" },
    .{ .name = "help", .description = "Show help (use: abi help <command>)" },
};

const command_names = blk: {
    var names: []const []const u8 = &.{};
    for (command_infos) |info| {
        names = names ++ &[_][]const u8{info.name};
    }
    break :blk names;
};

/// Main entry point with args from Zig 0.16 Init.Minimal
pub fn mainWithArgs(proc_args: std.process.Args, environ: std.process.Environ) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Zig 0.16: Convert Args to slice using arena
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const raw_args = try proc_args.toSlice(arena.allocator());

    // Parse global flags (--enable-*, --disable-*, --list-features)
    var global_flags = try utils.global_flags.parseGlobalFlags(allocator, raw_args);
    defer global_flags.deinit();
    defer allocator.free(global_flags.remaining_args);

    // Handle --list-features
    if (global_flags.show_features) {
        utils.global_flags.printFeaturesToStderr(utils.global_flags.ComptimeStatus);
        return;
    }

    // Validate feature overrides before proceeding
    if (global_flags.validate(utils.global_flags.ComptimeStatus)) |validation_error| {
        validation_error.print();
        std.process.exit(1);
    }

    // Initialize shared I/O backend for Zig 0.16
    var io_backend = cli_io.initIoBackendWithEnv(allocator, environ);
    defer io_backend.deinit();
    const io = io_backend.io();

    const fw_config = (abi.FrameworkOptions{}).toConfig();
    var framework = try abi.Framework.initWithIo(allocator, fw_config, io);
    defer abi.shutdown(&framework);

    // Apply runtime feature overrides to registry
    const registry = framework.getRegistry();
    global_flags.applyToRegistry(registry) catch |err| {
        std.debug.print("Warning: Could not apply feature override: {t}\n", .{err});
    };

    const args = global_flags.remaining_args;

    if (args.len <= 1) {
        printHelp();
        return;
    }

    const command = std.mem.sliceTo(args[1], 0);
    if (utils.args.matchesAny(command, &[_][]const u8{ "help", "--help", "-h" })) {
        if (args.len >= 3) {
            try runHelpTarget(allocator, arena.allocator(), io, std.mem.sliceTo(args[2], 0), args[3..]);
            return;
        }
        printHelp();
        return;
    }

    if (utils.args.matchesAny(command, &[_][]const u8{ "version", "--version", "-v" })) {
        std.debug.print("ABI Framework v{s}\n", .{abi.version()});
        return;
    }

    if (try runCommand(allocator, io, command, args[2..])) {
        return;
    }

    printUnknownCommand(command);
    std.process.exit(1);
}

fn printHelp() void {
    std.debug.print(
        \\Usage: abi [global-flags] <command> [options]
        \\
        \\Global Flags:
        \\  --list-features       List available features and their status
        \\  --enable-<feature>    Enable a feature at runtime
        \\  --disable-<feature>   Disable a feature at runtime
        \\
        \\Features: gpu, ai, llm, embeddings, agents, training, database, network, observability, web
        \\
        \\Commands:
        \\
    , .{});

    for (command_infos) |info| {
        const padding = if (info.name.len < 14) 14 - info.name.len else 2;
        std.debug.print("  {s}", .{info.name});
        for (0..padding) |_| std.debug.print(" ", .{});
        std.debug.print("{s}\n", .{info.description});
    }

    std.debug.print(
        \\
        \\Examples:
        \\  abi --list-features              # Show available features
        \\  abi --disable-gpu db stats       # Run db stats with GPU disabled
        \\  abi --enable-ai llm chat         # Run LLM chat with AI enabled
        \\  abi help llm generate            # Show help for nested subcommand
        \\
        \\Run 'abi <command> help' or 'abi help <command>' for command-specific help.
        \\
    , .{});
}

const CommandFn = *const fn (std.mem.Allocator, []const [:0]const u8) anyerror!void;
const IoCommandFn = *const fn (std.mem.Allocator, std.Io, []const [:0]const u8) anyerror!void;

const command_map = std.StaticStringMap(CommandFn).initComptime(.{
    .{ "db", wrap(commands.db) },
    .{ "agent", wrap(commands.agent) },
    .{ "bench", wrap(commands.bench) },
    .{ "gpu", wrap(commands.gpu) },
    .{ "network", wrap(commands.network) },
    .{ "system-info", wrap(commands.system_info) },
    .{ "multi-agent", wrap(commands.multi_agent) },
    .{ "explore", wrap(commands.explore) },
    .{ "simd", wrap(commands.simd) },
    .{ "config", wrap(commands.config) },
    .{ "discord", wrap(commands.discord) },
    .{ "llm", wrap(commands.llm) },
    .{ "model", wrap(commands.model) },
    .{ "embed", wrap(commands.embed) },
    .{ "train", wrap(commands.train) },
    .{ "convert", wrap(commands.convert) },
    .{ "task", wrap(commands.task) },
    .{ "completions", wrap(commands.completions) },
    .{ "plugins", wrap(commands.plugins) },
    .{ "profile", wrap(commands.profile) },
    .{ "status", wrap(commands.status) },
    .{ "toolchain", wrap(commands.toolchain) },
});

const io_command_map = std.StaticStringMap(IoCommandFn).initComptime(.{
    .{ "gpu-dashboard", wrapIo(commands.gpu_dashboard) },
    .{ "tui", wrapIo(commands.tui) },
});

fn wrap(comptime cmd: type) CommandFn {
    return cmd.run;
}

fn wrapIo(comptime cmd: type) IoCommandFn {
    return cmd.run;
}

fn runCommand(
    allocator: std.mem.Allocator,
    io: std.Io,
    command: []const u8,
    args: []const [:0]const u8,
) !bool {
    if (io_command_map.get(command)) |run_fn| {
        try run_fn(allocator, io, args);
        return true;
    }
    if (command_map.get(command)) |run_fn| {
        try run_fn(allocator, args);
        return true;
    }
    return false;
}

fn runHelpTarget(
    allocator: std.mem.Allocator,
    arena_allocator: std.mem.Allocator,
    io: std.Io,
    command: []const u8,
    extra_args: []const [:0]const u8,
) !void {
    var forwarded = std.ArrayListUnmanaged([:0]const u8){};
    for (extra_args) |arg| {
        try forwarded.append(arena_allocator, arg);
    }

    const help_arg: [:0]const u8 = "help";
    if (forwarded.items.len == 0 or !utils.args.matchesAny(forwarded.items[forwarded.items.len - 1], &[_][]const u8{ "help", "--help", "-h" })) {
        try forwarded.append(arena_allocator, help_arg);
    }

    if (try runCommand(allocator, io, command, forwarded.items)) {
        return;
    }

    printUnknownCommand(command);
    std.process.exit(1);
}

fn printUnknownCommand(command: []const u8) void {
    std.debug.print("Unknown command: {s}\n", .{command});
    if (utils.args.suggestCommand(command, command_names)) |suggestion| {
        std.debug.print("Did you mean: {s}\n", .{suggestion});
    }
    std.debug.print("Use 'help' for usage.\n", .{});
}
