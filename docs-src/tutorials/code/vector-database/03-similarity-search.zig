//! Vector Database Tutorial - Example 3: Similarity Search
//!
//! Run with: zig run docs/tutorials/code/vector-database/03-similarity-search.zig

const std = @import("std");
// In a real project, you would use: const abi = @import("abi");
// For tutorial purposes, we use a relative path.
const abi = @import("../../../../src/abi.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var framework = try abi.initDefault(allocator);
    defer framework.deinit();

    var db = try abi.database.openOrCreate(allocator, "documents");
    defer abi.database.close(&db);

    const embeddings = [_][3]f32{
        [_]f32{ 0.8, 0.2, 0.1 },
        [_]f32{ 0.2, 0.9, 0.3 },
        [_]f32{ 0.7, 0.3, 0.8 },
        [_]f32{ 0.3, 0.8, 0.4 },
    };
    const texts = [_][]const u8{
        "Zig programming language tutorial",
        "Vector database architecture guide",
        "High-performance systems programming",
        "Machine learning embeddings explained",
    };

    for (embeddings, 0..) |emb, i| {
        try abi.database.insert(&db, @intCast(i + 1), &emb, texts[i]);
    }

    const query_embedding = [_]f32{ 0.75, 0.25, 0.15 };
    const k = 3;

    std.debug.print("\n--- Similarity Search ---\n", .{});
    std.debug.print("Query vector: [{d:.2}, {d:.2}, {d:.2}]\n", .{
        query_embedding[0],
        query_embedding[1],
        query_embedding[2],
    });
    std.debug.print("Finding top {d} matches...\n\n", .{k});

    const results = try abi.database.search(&db, allocator, &query_embedding, k);
    defer allocator.free(results);

    std.debug.print("Results:\n", .{});
    for (results, 0..) |result, i| {
        std.debug.print("  {d}. ID={d}, Score={d:.3}\n", .{ i + 1, result.id, result.score });
        if (abi.database.get(&db, result.id)) |view| {
            if (view.metadata) |meta| {
                std.debug.print("     \"{s}\"\n", .{meta});
            }
        }
    }
}
