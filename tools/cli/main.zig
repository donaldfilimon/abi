const std = @import("std");
const abi = @import("abi");

fn printHeader() void {
    std.debug.print("ABI Framework CLI\n", .{});
}

fn printVersion() void {
    printHeader();
    std.debug.print("Version: {s}\n", .{abi.version()});
}

fn printHelp(exe: []const u8) void {
    printHeader();
    std.debug.print("Usage: {s} [--help|--version]\n", .{exe});
    std.debug.print("\nCommands:\n", .{});
    std.debug.print("  --help, -h     Show this help message\n", .{});
    std.debug.print("  --version, -v  Show version information\n", .{});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const exe_name = if (args.len > 0) std.fs.path.basename(args[0]) else "abi";

    if (args.len <= 1) {
        printHelp(exe_name);
        return;
    }

    const arg = args[1];
    if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "help")) {
        printHelp(exe_name);
        return;
    }
    if (std.mem.eql(u8, arg, "--version") or std.mem.eql(u8, arg, "-v") or std.mem.eql(u8, arg, "version")) {
        printVersion();
        return;
    }

    std.debug.print("Unknown command: {s}\n\n", .{arg});
    printHelp(exe_name);
    std.process.exit(1);
}
