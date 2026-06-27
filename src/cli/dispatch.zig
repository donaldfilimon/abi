//! CLI dispatch — walks the declarative `registry.commands` table.
//!
//! `help`/`--help`/`-h` are intercepted before the table walk (they have
//! command-specific behavior); every other command is matched by name and
//! invoked through either its typed `handler` (the argument spec is parsed via
//! `arg.parse`, and a parse failure emits the command's frozen `.usage` string
//! with exit code 2) or its legacy `raw_handler`.

const std = @import("std");
const usage_mod = @import("usage.zig");
const registry = @import("registry.zig");
const arg = @import("arg.zig");

pub fn runCli(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len < 2) {
        usage_mod.printUsage();
        return 0;
    }

    const cmd = args[1];

    if (std.mem.eql(u8, cmd, "help") or std.mem.eql(u8, cmd, "--help") or std.mem.eql(u8, cmd, "-h")) {
        if (args.len >= 3) return usage_mod.printCommandHelp(args[2]);
        usage_mod.printUsage();
        return 0;
    }

    for (registry.commands) |command| {
        if (!std.mem.eql(u8, cmd, command.name)) continue;

        if (command.handler) |handler| {
            var parsed = arg.parse(allocator, command.args, args) catch |err| switch (err) {
                error.Usage => return usage_mod.usageError(command.usage),
                else => |e| return e,
            };
            defer parsed.deinit();
            return handler(.{ .io = io, .allocator = allocator }, parsed);
        }
        if (command.raw_handler) |raw_handler| {
            return raw_handler(io, allocator, args);
        }
        // Metadata-only command (e.g. `help`) reaching the table walk: fall
        // through to the unknown-command path below.
        break;
    }

    std.debug.print("error: unknown command '{s}'\n\n", .{cmd});
    usage_mod.printUsage();
    return 2;
}

test {
    std.testing.refAllDecls(@This());
}
