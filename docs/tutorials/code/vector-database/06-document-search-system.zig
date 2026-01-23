//! Vector Database Tutorial - Example 6: Document Search System
//!
//! Run with: zig run docs/tutorials/code/vector-database/06-document-search-system.zig

const std = @import("std");
const abi = @import("abi");

const Document = struct {
    id: u64,
    title: []const u8,
    content: []const u8,
    embedding: [3]f32,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var framework = try abi.initDefault(allocator);
    defer framework.deinit();

    var db = try abi.database.openOrCreate(allocator, "document_search");
    defer abi.database.close(&db);

    const docs = [_]Document{
        .{ .id = 1, .title = "Zig Programming Guide", .content = "Learn Zig from scratch with hands-on examples", .embedding = [_]f32{ 0.9, 0.1, 0.2 } },
        .{ .id = 2, .title = "Vector Database Tutorial", .content = "Understanding similarity search and embeddings", .embedding = [_]f32{ 0.2, 0.9, 0.3 } },
        .{ .id = 3, .title = "Systems Programming", .content = "Low-level programming for performance", .embedding = [_]f32{ 0.8, 0.2, 0.7 } },
    };

    std.debug.print("Indexing documents...\n", .{});
    for (docs) |doc| {
        const metadata = try std.fmt.allocPrint(allocator, "{s}: {s}", .{ doc.title, doc.content });
        defer allocator.free(metadata);

        try abi.database.insert(&db, doc.id, &doc.embedding, metadata);
        std.debug.print("  Indexed: {s}\n", .{doc.title});
    }

    try abi.database.optimize(&db);

    const queries = [_]struct {
        text: []const u8,
        embedding: [3]f32,
    }{
        .{ .text = "How do I learn Zig?", .embedding = [_]f32{ 0.85, 0.15, 0.25 } },
        .{ .text = "What is a vector database?", .embedding = [_]f32{ 0.25, 0.85, 0.35 } },
    };

    for (queries) |query| {
        std.debug.print("\n--- Query: \"{s}\" ---\n", .{query.text});

        const results = try abi.database.search(&db, allocator, &query.embedding, 2);
        defer allocator.free(results);

        for (results, 0..) |result, i| {
            std.debug.print("  {d}. Score={d:.3}\n", .{ i + 1, result.score });
            if (abi.database.get(&db, result.id)) |view| {
                if (view.metadata) |meta| {
                    std.debug.print("     {s}\n", .{meta});
                }
            }
        }
    }

    try abi.database.backup(&db, "search_system_backup.db");
    std.debug.print("\nBackup created\n", .{});
}
