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
//! - `version` - Show framework version
//! - `help` - Show help message

const std = @import("std");
const abi = @import("abi");
const commands = @import("commands/mod.zig");
const utils = @import("utils/mod.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();

    var framework = try abi.init(allocator, abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

    const args = try init.minimal.args.toSlice(allocator);

    if (args.len <= 1) {
        printHelp();
        return;
    }

    const command = args[1];
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

    if (std.mem.eql(u8, command, "gpu")) {
        try commands.gpu.run(allocator, args[2..]);
        return;
    }

    if (std.mem.eql(u8, command, "network")) {
        try commands.network.run(allocator, args[2..]);
        return;
    }

    if (std.mem.eql(u8, command, "system-info")) {
        try commands.system_info.run(allocator, &framework);
        return;
    }

    if (std.mem.eql(u8, command, "explore")) {
        try commands.explore.run(allocator, args[2..]);
        return;
    }

    if (std.mem.eql(u8, command, "simd")) {
        try commands.simd.run(allocator);
        return;
    }

    if (std.mem.eql(u8, command, "config")) {
        try commands.config.run(allocator, args[2..]);
        return;
    }

    if (std.mem.eql(u8, command, "llm")) {
        try commands.llm.run(allocator, args[2..]);
        return;
    }

    std.debug.print("Unknown command: {s}\nUse 'help' for usage.\n", .{command});
    std.process.exit(1);
}

fn printHelp() void {
    const help_text =
        "Usage: abi <command> [options]\n\n" ++
        "Commands:\n" ++
        "  db <subcommand>   Database operations (add, query, stats, optimize, backup)\n" ++
        "  agent [--message] Run AI agent (interactive or one-shot)\n" ++
        "  config [command]  Configuration management (init, show, validate)\n" ++
        "  explore [options] Search and explore codebase\n" ++
        "  gpu [subcommand]  GPU commands (backends, devices, summary, default)\n" ++
        "  llm <subcommand>  LLM inference (info, generate, chat, bench)\n" ++
        "  network [command] Manage network registry (list, register, status)\n" ++
        "  simd              Run SIMD performance demo\n" ++
        "  system-info       Show system and framework status\n" ++
        "  version           Show framework version\n" ++
        "  help              Show this help message\n\n" ++
        "Run 'abi <command> help' for command-specific help.\n";

    std.debug.print("{s}", .{help_text});
}
