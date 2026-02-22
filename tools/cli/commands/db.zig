//! Database CLI command.
//!
//! Delegates to the database feature's built-in CLI handler.

const std = @import("std");
const abi = @import("abi");
const command_mod = @import("../command.zig");
const utils = @import("../utils/mod.zig");

pub const meta: command_mod.Meta = .{
    .name = "db",
    .description = "Database operations (add, query, stats, optimize, backup, restore)",
    .aliases = &.{"ls"},
    .subcommands = &.{ "add", "query", "stats", "optimize", "backup", "restore", "serve", "help" },
};

/// Run the database command with the provided arguments.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    // Handle help at CLI level
    if (args.len > 0 and utils.args.matchesAny(args[0], &.{ "help", "--help", "-h" })) {
        printHelp();
        return;
    }

    // Check if database feature is enabled
    if (!abi.database.isEnabled()) {
        std.debug.print("Error: Database feature is disabled.\n", .{});
        std.debug.print("Rebuild with: zig build -Denable-database=true\n", .{});
        return;
    }

    try abi.database.cli.run(allocator, args);
}

fn printHelp() void {
    const help_text =
        \\Usage: abi db <command> [options]
        \\
        \\Vector database commands for storing and querying embeddings.
        \\
        \\Commands:
        \\  stats                    Show database statistics
        \\  add --id <n> --embed <t> Add embedding with ID
        \\  query --embed <text>     Query for similar embeddings
        \\  optimize                 Optimize database indices
        \\  backup --path <file>     Backup database to file
        \\  restore --path <file>    Restore database from backup
        \\  serve [--addr <h:p>]     Start database server
        \\  help                     Show this help message
        \\
        \\Options:
        \\  --path <path>            Database file path (default: wdbx_data)
        \\  --top-k <n>              Number of results to return (default: 10)
        \\
        \\Examples:
        \\  abi db stats
        \\  abi db add --id 1 --embed "Hello world"
        \\  abi db query --embed "similar text" --top-k 5
        \\  abi db backup --path backup.db
        \\
    ;
    std.debug.print("{s}", .{help_text});
}
