const std = @import("std");

pub fn generateReadmeRedirect(_: std.mem.Allocator) !void {
    const content =
        \\# ABI Documentation\n
        \\This repository contains generated documentation. Use the navigation links below when viewing the site locally.\n\n
        \\- [API Reference](generated/API_REFERENCE.md)\n
        \\- [Module Reference](generated/MODULE_REFERENCE.md)\n
        \\- [Examples](generated/EXAMPLES.md)\n
        \\- [Performance Guide](generated/PERFORMANCE_GUIDE.md)\n
    ;

    var file = try std.fs.cwd().createFile("docs/README.md", .{ .truncate = true });
    defer file.close();
    try file.writeAll(content);
}
