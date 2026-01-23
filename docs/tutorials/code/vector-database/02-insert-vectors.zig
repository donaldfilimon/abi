//! Vector Database Tutorial - Example 2: Insert Vectors
//!
//! Run with: zig run docs/tutorials/code/vector-database/02-insert-vectors.zig

const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var framework = try abi.initDefault(allocator);
    defer framework.deinit();

    var db = try abi.database.openOrCreate(allocator, "documents");
    defer abi.database.close(&db);

    const Document = struct {
        id: u64,
        text: []const u8,
        embedding: [3]f32,
    };

    const documents = [_]Document{
        .{ .id = 1, .text = "Zig programming language tutorial", .embedding = [_]f32{ 0.8, 0.2, 0.1 } },
        .{ .id = 2, .text = "Vector database architecture guide", .embedding = [_]f32{ 0.2, 0.9, 0.3 } },
        .{ .id = 3, .text = "High-performance systems programming", .embedding = [_]f32{ 0.7, 0.3, 0.8 } },
        .{ .id = 4, .text = "Machine learning embeddings explained", .embedding = [_]f32{ 0.3, 0.8, 0.4 } },
    };

    for (documents) |doc| {
        try abi.database.insert(&db, doc.id, &doc.embedding, doc.text);
        std.debug.print("Inserted: {s}\n", .{doc.text});
    }

    const stats = abi.database.stats(&db);
    std.debug.print("\nDatabase contains {d} vectors of dimension {d}\n", .{
        stats.count,
        stats.dimension,
    });
}
