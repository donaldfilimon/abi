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
//! ## Commands
//! - `db` - Database operations (add, query, stats, optimize, backup)
//! - `agent` - Run AI agent (interactive or one-shot)
//! - `gpu` - GPU commands (backends, devices, summary, default)
//! - `network` - Manage network registry (list, register, status)
//! - `system-info` - Show system and framework status
//! - `tui` - Interactive terminal UI for command selection
//! - `version` - Show framework version
//! - `help` - Show help message

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
pub fn mainWithArgs(proc_args: std.process.Args) !void {
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
    var io_backend = cli_io.initIoBackend(allocator);
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

fn runCommand(
    allocator: std.mem.Allocator,
    io: std.Io,
    command: []const u8,
    args: []const [:0]const u8,
) !bool {
    if (std.mem.eql(u8, command, "db")) {
        try commands.db.run(allocator, args);
        return true;
    }
    if (std.mem.eql(u8, command, "agent")) {
        try commands.agent.run(allocator, args);
        return true;
    }
    if (std.mem.eql(u8, command, "bench")) {
        try commands.bench.run(allocator, args);
        return true;
    }
    if (std.mem.eql(u8, command, "gpu")) {
        try commands.gpu.run(allocator, args);
        return true;
    }
    if (std.mem.eql(u8, command, "gpu-dashboard")) {
        try commands.gpu_dashboard.run(allocator, io, args);
        return true;
    }
    if (std.mem.eql(u8, command, "network")) {
        try commands.network.run(allocator, args);
        return true;
    }
    if (std.mem.eql(u8, command, "system-info")) {
        try commands.system_info.run(allocator, args);
        return true;
    }
    if (std.mem.eql(u8, command, "multi-agent")) {
        try commands.multi_agent.run(allocator, args);
        return true;
    }
    if (std.mem.eql(u8, command, "explore")) {
        try commands.explore.run(allocator, args);
        return true;
    }
    if (std.mem.eql(u8, command, "simd")) {
        try commands.simd.run(allocator, args);
        return true;
    }
    if (std.mem.eql(u8, command, "config")) {
        try commands.config.run(allocator, args);
        return true;
    }
    if (std.mem.eql(u8, command, "discord")) {
        try commands.discord.run(allocator, args);
        return true;
    }
    if (std.mem.eql(u8, command, "llm")) {
        try commands.llm.run(allocator, args);
        return true;
    }
    if (std.mem.eql(u8, command, "model")) {
        try commands.model.run(allocator, args);
        return true;
    }
    if (std.mem.eql(u8, command, "embed")) {
        try commands.embed.run(allocator, args);
        return true;
    }
    if (std.mem.eql(u8, command, "train")) {
        try commands.train.run(allocator, args);
        return true;
    }
    if (std.mem.eql(u8, command, "convert")) {
        try commands.convert.run(allocator, args);
        return true;
    }
    if (std.mem.eql(u8, command, "task")) {
        try commands.task.run(allocator, args);
        return true;
    }
    if (std.mem.eql(u8, command, "tui")) {
        try commands.tui.run(allocator, io, args);
        return true;
    }
    if (std.mem.eql(u8, command, "completions")) {
        try commands.completions.run(allocator, args);
        return true;
    }
    if (std.mem.eql(u8, command, "plugins")) {
        try commands.plugins.run(allocator, args);
        return true;
    }
    if (std.mem.eql(u8, command, "profile")) {
        try commands.profile.run(allocator, args);
        return true;
    }
    if (std.mem.eql(u8, command, "toolchain")) {
        try commands.toolchain.run(allocator, args);
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
