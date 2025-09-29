const std = @import("std");
const print = std.debug.print;

test "check io apis" {
    print("std.io exists: {}\n", .{@hasDecl(std, "io")});
    print("std.fs exists: {}\n", .{@hasDecl(std, "fs")});

    // Check what writer types exist
    const stdout = std.io.getStdOut().writer();
    print("stdout writer type: {s}\n", .{@typeName(@TypeOf(stdout))});

    // Check if Writer exists
    print("std.io.Writer exists: {}\n", .{@hasDecl(std.io, "Writer")});
    print("std.io.AnyWriter exists: {}\n", .{@hasDecl(std.io, "AnyWriter")});
}
