const std = @import("std");

const Term = struct {
    name: []const u8,
    definition: []const u8,
};

const terms = [_]Term{
    .{ .name = "Embedding", .definition = "A floating point vector that represents an input document." },
    .{ .name = "Shard", .definition = "A subset of the database used for distribution and locality." },
    .{ .name = "Persona", .definition = "Preconfigured behaviour profile for the conversational agent." },
};

pub fn generateDefinitionsReference(allocator: std.mem.Allocator) !void {
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();
    var writer = buffer.writer();

    try writer.writeAll("# Definitions Reference\n\n");
    try writer.writeAll("Use this glossary to align terminology between the engineering, research, and product teams.\n\n");

    try writer.writeAll("## 📊 Quick Reference Index\n\n");
    for (terms) |term| {
        try writer.print("### {s}\n\n", .{term.name});
        try writer.print("{s}\n\n", .{term.definition});
    }

    var file = try std.fs.cwd().createFile("docs/generated/DEFINITIONS_REFERENCE.md", .{ .truncate = true });
    defer file.close();
    try file.writeAll(buffer.items);
}
