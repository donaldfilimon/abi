const std = @import("std");

const Endpoint = struct {
    name: []const u8,
    signature: []const u8,
    description: []const u8,
};

const endpoints = [_]Endpoint{
    .{ .name = "open", .signature = "fn open(path: []const u8, create: bool) !Db", .description = "Open or create a WDBX database." },
    .{ .name = "addEmbedding", .signature = "fn addEmbedding(self: *Db, embedding: []const f32) !u64", .description = "Insert an embedding and return its identifier." },
    .{ .name = "search", .signature = "fn search(self: *Db, query: []const f32, limit: usize) ![]SearchResult", .description = "Perform a nearest neighbour search." },
};

pub fn generateApiReference(allocator: std.mem.Allocator) !void {
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();
    var writer = buffer.writer();

    try writer.writeAll("---\n");
    try writer.writeAll("layout: documentation\n");
    try writer.writeAll("title: API Reference\n");
    try writer.writeAll("description: Key entry points for interacting with ABI programmatically\n");
    try writer.writeAll("---\n\n");

    try writer.writeAll("# ABI API Reference\n\n");
    try writer.writeAll("The following signatures are extracted from the public database API and serve as the canonical interface surface for client applications.\n\n");

    try writer.writeAll("## Database\n\n");
    for (endpoints) |endpoint| {
        try writer.print("### `{s}`\n\n", .{endpoint.signature});
        try writer.print("{s}\n\n", .{endpoint.description});
    }

    try writer.writeAll("## Error Types\n\n");
    try writer.writeAll("```zig\n");
    try writer.writeAll("pub const DatabaseError = error{\n    OutOfMemory,\n    InvalidConfig,\n    VectorDimensionMismatch,\n    StorageFailure,\n};\n");
    try writer.writeAll("```\n");

    var file = try std.fs.cwd().createFile("docs/generated/API_REFERENCE.md", .{ .truncate = true });
    defer file.close();
    try file.writeAll(buffer.items);
}
