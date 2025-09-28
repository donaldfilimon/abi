const std = @import("std");

pub fn generateExamples(allocator: std.mem.Allocator) !void {
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();
    var writer = buffer.writer();

    try writer.writeAll("# Examples\n\n");
    try writer.writeAll("The snippets below demonstrate how to embed vectors, execute similarity search, and expose the results through the HTTP interface.\n\n");

    try writer.writeAll("## Quick Start\n\n");
    try writer.writeAll("```zig\n");
    try writer.writeAll("const std = @import(\"std\");\nconst abi = @import(\"abi\");\n\npub fn main() !void {\n    var gpa = std.heap.GeneralPurposeAllocator(.{}){};\n    defer _ = gpa.deinit();\n    var db = try abi.database.Db.open(\"vectors.wdbx\", true);\n    defer db.close();\n\n    const embedding = [_]f32{ 0.1, 0.2, 0.3 };\n    _ = try db.addEmbedding(&embedding);\n}\n");
    try writer.writeAll("```\n\n");

    try writer.writeAll("## HTTP Search Endpoint\n\n");
    try writer.writeAll("The bundled HTTP helper exposes a `/search` endpoint. Example curl request:\n\n");
    try writer.writeAll("```bash\n$ curl -X POST http://localhost:8765/search \\\n  -H 'content-type: application/json' \\\n  -d '{ \"vector\": [0.1, 0.2, 0.3], \"limit\": 4 }'\n```\n");

    var file = try std.fs.cwd().createFile("docs/generated/EXAMPLES.md", .{ .truncate = true });
    defer file.close();
    try file.writeAll(buffer.items);
}
