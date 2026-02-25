//! Database CLI command.
//!
//! Delegates to the database feature's built-in CLI handler.

const std = @import("std");
const abi = @import("abi");
const command_mod = @import("../command.zig");
const context_mod = @import("../framework/context.zig");
const utils = @import("../utils/mod.zig");

pub const meta: command_mod.Meta = .{
    .name = "db",
    .description = "Database operations (add, query, stats, optimize, backup, restore)",
    .aliases = &.{"ls"},
    .subcommands = &.{ "add", "query", "stats", "optimize", "backup", "restore", "serve", "help" },
};

/// Run the database command with the provided arguments.
pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    // Handle help at CLI level
    if (args.len > 0 and utils.args.matchesAny(args[0], &.{ "help", "--help", "-h" })) {
        printHelp();
        return;
    }

    // Check if database feature is enabled
    if (!abi.database.isEnabled()) {
        utils.output.printError("Database feature is disabled.", .{});
        utils.output.printInfo("Rebuild with: zig build -Denable-database=true", .{});
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
        \\  stats [--db <path>]            Show database statistics
        \\  add --id <n> --embed <t> [--db <path>] Add embedding with ID
        \\  query --embed <text> [--db <path>]    Query for similar embeddings
        \\  optimize [--db <path>]         Optimize database indices
        \\  backup --db <p> --out <p> Backup database to file
        \\  restore --db <p> --in <p> Restore database from backup
        \\  serve [--addr <h:p>]     Start database server
        \\  help                     Show this help message
        \\
        \\Options:
        \\  --path <path>            Legacy shorthand for both db and backup path
        \\  --db <path>              Database file path
        \\  --out <path>             Backup output path
        \\  --in <path>              Restore input path
        \\  --top-k <n>              Number of results to return (default: 10)
        \\
        \\Examples:
        \\  abi db stats
        \\  abi db add --id 1 --embed "Hello world"
        \\  abi db query --embed "similar text" --top-k 5
        \\  abi db backup --db state.db --out backup.db
        \\  abi db restore --db state.db --in backup.db
        \\
    ;
    std.debug.print("{s}", .{help_text});
}
