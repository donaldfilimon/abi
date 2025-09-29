const std = @import("std");

test "explore std" {
    // Check available top-level modules
    std.debug.print("Available modules in std:\n", .{});

    // Try different paths
    std.debug.print("std.debug exists: {}\n", .{@hasDecl(std, "debug")});
    std.debug.print("std.fs exists: {}\n", .{@hasDecl(std, "fs")});
    std.debug.print("std.mem exists: {}\n", .{@hasDecl(std, "mem")});

    // Check for stdout/stderr
    std.debug.print("std.io exists: {}\n", .{@hasDecl(std, "io")});

    // Try stdout directly
    const stdout = std.debug.getStdOut();
    std.debug.print("stdout type: {s}\n", .{@typeName(@TypeOf(stdout))});
}
