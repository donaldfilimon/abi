const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const args = try std.process.argsAlloc(gpa.allocator());
    defer std.process.argsFree(gpa.allocator(), args);

    if (args.len <= 1 or std.mem.eql(u8, args[1], "--help")) {
        std.debug.print(
            "ABI CLI\n\n" ++
                "Usage:\n" ++
                "  abi --help\n" ++
                "  abi --version\n",
            .{},
        );
        return;
    }

    if (std.mem.eql(u8, args[1], "--version")) {
        std.debug.print("{s}\n", .{abi.version()});
        return;
    }

    std.debug.print(
        "Unknown argument: {s}\nUse --help for usage.\n",
        .{args[1]},
    );
    std.process.exit(2);
}
