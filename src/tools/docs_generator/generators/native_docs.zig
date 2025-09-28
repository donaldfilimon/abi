const std = @import("std");

pub fn generateZigNativeDocs(_: std.mem.Allocator) !void {
    try std.fs.cwd().makePath("docs/zig-docs");

    const html =
        \\<!DOCTYPE html>\n
        \\<html lang="en">\n
        \\  <head>\n
        \\    <meta charset="utf-8">\n
        \\    <title>Zig Native Docs</title>\n
        \\  </head>\n
        \\  <body>\n
        \\    <h1>Zig Native Docs</h1>\n
        \\    <p>The full `zig doc` output can be published here when the build pipeline enables it.</p>\n
        \\  </body>\n
        \\</html>\n
    ;

    var file = try std.fs.cwd().createFile("docs/zig-docs/index.html", .{ .truncate = true });
    defer file.close();
    try file.writeAll(html);
}
