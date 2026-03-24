const std = @import("std");

test "cli: e2e routes" {
    const exe_path = "zig-out/bin/abi";

    // Test that the binary exists
    std.fs.accessAbsolute(exe_path, .{}) catch |err| {
        if (err == error.FileNotFound) {
            std.debug.print("Executable not found, skipping e2e tests. Run 'zig build cli' first.\n", .{});
            return;
        }
    };

    const routes = [_][]const []const u8{
        &.{exe_path}, // status
        &.{ exe_path, "version" },
        &.{ exe_path, "doctor" },
        &.{ exe_path, "features" },
        &.{ exe_path, "platform" },
        &.{ exe_path, "connectors" },
        &.{ exe_path, "info" },
        &.{ exe_path, "chat", "test", "message" },
        &.{ exe_path, "serve", "--help" },
        &.{ exe_path, "acp", "serve", "--help" },
        &.{ exe_path, "db", "stats" }, // Assuming stats is non-blocking
        &.{ exe_path, "help" },
        &.{ exe_path, "--help" },
        &.{ exe_path, "-h" },
        &.{ exe_path, "unknown_command" }, // Should print help and exit 0 (or 1 depending on impl)
    };

    for (routes) |args| {
        const result = try std.process.Child.run(.{
            .allocator = std.testing.allocator,
            .argv = args,
        });
        defer std.testing.allocator.free(result.stdout);
        defer std.testing.allocator.free(result.stderr);

        // We mainly care that it doesn't crash (e.g. segfault).
        // It might exit 0 or 1 for unknown command, both are fine as long as it's a Normal exit.
        switch (result.term) {
            .Exited => |code| {
                _ = code; // Accept any normal exit code for now
            },
            else => {
                std.debug.print("Command {s} failed with abnormal exit term: {any}\n", .{ args, result.term });
                return error.AbnormalExit;
            },
        }
    }
}
