//! Database CLI command.
//!
//! Delegates to the database feature's built-in CLI handler.

const std = @import("std");
const abi = @import("abi");
const utils = @import("../utils/mod.zig");

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
        \\  search --embed <text>    Search for similar embeddings
        \\  backup --path <file>     Backup database to file
        \\  restore --path <file>    Restore database from backup
        \\  help                     Show this help message
        \\
        \\Options:
        \\  --path <path>            Database file path (default: wdbx_data)
        \\  --dimensions <n>         Vector dimensions (default: 384)
        \\  --metric <type>          Distance metric: cosine, euclidean, dot
        \\
        \\Examples:
        \\  abi db stats
        \\  abi db add --id 1 --embed "Hello world"
        \\  abi db search --embed "similar text" --top 5
        \\  abi db backup --path backup.db
        \\
    ;
    std.debug.print("{s}", .{help_text});
}
