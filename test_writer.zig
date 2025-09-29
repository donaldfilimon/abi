const std = @import("std");

test "find writer types" {
    // Try old patterns
    std.debug.print("Testing writer patterns in Zig 0.16.0\n", .{});

    // Get stdout - try different patterns
    // Pattern 1: std.io.getStdOut()
    // Pattern 2: std.process.stdout
    // Pattern 3: std.debug.getStdOut()

    comptime {
        if (@hasDecl(std, "process")) {
            std.debug.print("std.process exists\n", .{});
        }
        if (@hasDecl(std, "io")) {
            std.debug.print("std.io exists\n", .{});
        }
    }
}
