const std = @import("std");
const testing = std.testing;

// Simple CLI with working patterns from existing code
pub const CliParser = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) CliParser {
        return CliParser{
            .allocator = allocator,
        };
    }

    pub fn printHelp(self: *const CliParser) void {
        _ = self;
        std.debug.print("ABI CLI v1.0.0\n", .{});
        std.debug.print("Usage: abi [COMMAND]\n\n", .{});
        std.debug.print("Commands:\n", .{});
        std.debug.print("  server    Start HTTP server\n", .{});
        std.debug.print("  chat      Start chat interface\n", .{});
        std.debug.print("  benchmark Run benchmarks\n", .{});
        std.debug.print("  help      Show this help\n", .{});
    }

    pub fn parseArgs(self: *const CliParser, args: [][:0]u8) !void {
        if (args.len < 2) {
            self.printHelp();
            return;
        }

        const command = args[1];

        if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "--help")) {
            self.printHelp();
        } else if (std.mem.eql(u8, command, "server")) {
            std.debug.print("Starting server...\n", .{});
        } else if (std.mem.eql(u8, command, "chat")) {
            std.debug.print("Starting chat...\n", .{});
        } else if (std.mem.eql(u8, command, "benchmark")) {
            std.debug.print("Running benchmarks...\n", .{});
        } else {
            std.debug.print("Unknown command: {s}\n", .{command});
            self.printHelp();
        }
    }
};

test "cli basic functionality" {
    var cli = CliParser.init(testing.allocator);
    cli.printHelp();
}
