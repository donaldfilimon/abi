const std = @import("std");

pub fn generateDocsIndexHtml(_: std.mem.Allocator) !void {
    const html =
        \\<!DOCTYPE html>\n
        \\<html lang="en">\n
        \\  <head>\n
        \\    <meta charset="utf-8">\n
        \\    <meta name="viewport" content="width=device-width, initial-scale=1">\n
        \\    <title>ABI Documentation</title>\n
        \\    <link rel="stylesheet" href="assets/css/documentation.css">\n
        \\  </head>\n
        \\  <body>\n
        \\    <main class="documentation">\n
        \\      <article>\n
        \\        <h1>ABI Documentation</h1>\n
        \\        <p>Select a reference from the navigation menu to explore API details, examples, and operational guides.</p>\n
        \\      </article>\n
        \\      <nav>\n
        \\        <ul>\n
        \\          <li><a href="generated/API_REFERENCE.md">API Reference</a></li>\n
        \\          <li><a href="generated/MODULE_REFERENCE.md">Module Reference</a></li>\n
        \\          <li><a href="generated/EXAMPLES.md">Examples</a></li>\n
        \\          <li><a href="generated/PERFORMANCE_GUIDE.md">Performance Guide</a></li>\n
        \\          <li><a href="generated/DEFINITIONS_REFERENCE.md">Definitions</a></li>\n
        \\        </ul>\n
        \\      </nav>\n
        \\    </main>\n
        \\  </body>\n
        \\</html>\n
    ;

    var file = try std.fs.cwd().createFile("docs/index.html", .{ .truncate = true });
    defer file.close();
    try file.writeAll(html);
}
