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

/// Main entry point with args from Zig 0.16 Init.Minimal
pub fn mainWithArgs(proc_args: std.process.Args) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var framework = try abi.init(allocator, abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

    // Zig 0.16: Convert Args to slice using arena
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const args = try proc_args.toSlice(arena.allocator());

    if (args.len <= 1) {
        printHelp();
        return;
    }

    const command = std.mem.sliceTo(args[1], 0);
    if (utils.args.matchesAny(command, &[_][]const u8{ "help", "--help", "-h" })) {
        printHelp();
        return;
    }

    if (utils.args.matchesAny(command, &[_][]const u8{ "version", "--version", "-v" })) {
        std.debug.print("ABI Framework v{s}\n", .{abi.version()});
        return;
    }

    if (std.mem.eql(u8, command, "db")) {
        try commands.db.run(allocator, args[2..]);
        return;
    }

    if (std.mem.eql(u8, command, "agent")) {
        try commands.agent.run(allocator, args[2..]);
        return;
    }

    if (std.mem.eql(u8, command, "bench")) {
        try commands.bench.run(allocator, args[2..]);
        return;
    }

    if (std.mem.eql(u8, command, "gpu")) {
        try commands.gpu.run(allocator, args[2..]);
        return;
    }

    if (std.mem.eql(u8, command, "network")) {
        try commands.network.run(allocator, args[2..]);
        return;
    }

    if (std.mem.eql(u8, command, "system-info")) {
        try commands.system_info.run(allocator, args[2..]);
        return;
    }

    if (std.mem.eql(u8, command, "explore")) {
        try commands.explore.run(allocator, args[2..]);
        return;
    }

    if (std.mem.eql(u8, command, "simd")) {
        try commands.simd.run(allocator, args[2..]);
        return;
    }

    if (std.mem.eql(u8, command, "config")) {
        try commands.config.run(allocator, args[2..]);
        return;
    }

    if (std.mem.eql(u8, command, "discord")) {
        try commands.discord.run(allocator, args[2..]);
        return;
    }

    if (std.mem.eql(u8, command, "llm")) {
        try commands.llm.run(allocator, args[2..]);
        return;
    }

    if (std.mem.eql(u8, command, "embed")) {
        try commands.embed.run(allocator, args[2..]);
        return;
    }

    if (std.mem.eql(u8, command, "train")) {
        try commands.train.run(allocator, args[2..]);
        return;
    }

    if (std.mem.eql(u8, command, "tui")) {
        try commands.tui.run(allocator, args[2..]);
        return;
    }

    std.debug.print("Unknown command: {s}\nUse 'help' for usage.\n", .{command});
    std.process.exit(1);
}

fn printHelp() void {
    const help_text =
        \\Usage: abi <command> [options]
        \\
        \\Commands:
        \\  db <subcommand>    Database operations (add, query, stats, optimize, backup)
        \\  agent [--message]  Run AI agent (interactive or one-shot)
        \\  bench <suite>      Run performance benchmarks (all, simd, memory, ai, quick)
        \\  config [command]   Configuration management (init, show, validate)
        \\  discord [command]  Discord bot operations (status, guilds, send, commands)
        \\  embed [options]    Generate embeddings from text (openai, mistral, cohere, ollama)
        \\  explore [options]  Search and explore codebase
        \\  gpu [subcommand]   GPU commands (backends, devices, summary, default)
        \\  llm <subcommand>   LLM inference (info, generate, chat, bench, download)
        \\  network [command]  Manage network registry (list, register, status)
        \\  simd               Run SIMD performance demo
        \\  system-info        Show system and framework status
        \\  train <subcommand> Training pipeline (run, resume, info)
        \\  tui                Launch interactive TUI command menu (type to filter)
        \\  version            Show framework version
        \\  help               Show this help message
        \\
        \\Run 'abi <command> help' for command-specific help.
        \\
    ;

    std.debug.print("{s}", .{help_text});
}
